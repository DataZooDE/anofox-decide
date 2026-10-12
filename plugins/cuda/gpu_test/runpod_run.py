#!/usr/bin/env python3
"""Run the CUDA plugin's on-hardware verification on a throwaway RunPod GPU pod, then destroy it.

Adapted from anofox-tabfm's tools/gpu_test/runpod_run.py. There is no NVIDIA GPU on the dev box and none in CI,
so this rents a pod by the minute, ships the sources to it, runs plugins/cuda/gpu_test/cuda_pod_verify.sh,
streams the output back and terminates the pod. It costs money and needs a key CI does not have: run it by
hand, from a machine whose RUNPOD_API_KEY is set.

    # the default: an RTX 4090 on RunPod's own datacenters, plugin checks only
    plugins/cuda/gpu_test/runpod_run.py

    # also the real-weights checks (recorded values, CPU-vs-GPU, the 200-ticket evaluation): ship a DuckDB 1.5.6
    # CLI and the linux_amd64 anofox_decide extension built from the SAME commit (the CI artifact)
    plugins/cuda/gpu_test/runpod_run.py --duckdb /path/to/duckdb --extension /path/to/anofox_decide.duckdb_extension

    plugins/cuda/gpu_test/runpod_run.py --list                 # pods that are running
    plugins/cuda/gpu_test/runpod_run.py --terminate <pod id>   # if a run was killed with SIGKILL

The key comes from RUNPOD_API_KEY, or from ~/.bashrc, which exports it but returns early for non-interactive
shells, so it is read out of the file rather than sourced. It is NEVER printed.

COST: the pod is terminated in a finally block, on exception, and on SIGINT/SIGTERM. A SIGKILLed run leaves it
running: --list / --terminate clean up. No network volume is created (weights are downloaded on the pod).
"""

from __future__ import annotations

import argparse
import json
import os
import re
import signal
import subprocess
import sys
import tarfile
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path

API = "https://rest.runpod.io/v1"
GRAPHQL = "https://api.runpod.io/graphql"
# CUDA 12.8 with cuDNN 9: the plugin's floor (the CUDA-12 ORT provider needs a CUDA 12 minor newer than 12.4).
DEFAULT_IMAGE = "runpod/pytorch:2.8.0-py3.11-cuda12.8.1-cudnn-devel-ubuntu22.04"
DEFAULT_GPU = "NVIDIA GeForce RTX 4090"
USER_AGENT = "anofox-decide-gpu-test/1.0"
REPO = Path(__file__).resolve().parents[3]

# What the pod needs from the repository, nothing else: the plugin, the ABI and SONAME headers it is built
# from, the pinned runtime constants, the tiny fixture and the evaluation-sample builder.
SHIPPED = [
    "plugins/cuda",
    "src/include/decide_plugin_abi.h",
    "src/include/decide_soname_patch.hpp",
    "src/decide_accelerate.cpp",
    "test/fixtures/local_tiny",
    "tools/eval/make_sample.py",
]
CAPACITY_ERRORS = (
    "resources to deploy",
    "no instances currently available",
    "no longer any instances available",
    "not enough free gpus",
)


def api_key() -> str:
    key = os.environ.get("RUNPOD_API_KEY")
    if key:
        return key.strip()
    bashrc = Path.home() / ".bashrc"
    if bashrc.exists():
        m = re.search(r"^export RUNPOD_API_KEY=(.*)$", bashrc.read_text(), re.M)
        if m:
            return m.group(1).strip().strip("\"'")
    raise SystemExit("RUNPOD_API_KEY not set and not found in ~/.bashrc")


def call(method: str, path: str, key: str, body=None, timeout=60):
    req = urllib.request.Request(
        f"{API}{path}",
        method=method,
        data=json.dumps(body).encode() if body is not None else None,
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json", "User-Agent": USER_AGENT},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode()
            return json.loads(raw) if raw.strip() else None
    except urllib.error.HTTPError as exc:
        # the body can echo the request; never include headers (the key) in the message
        raise SystemExit(f"RunPod {method} {path} -> {exc.code}: {exc.read().decode()[:400]}")


def ensure_keypair(scratch: Path) -> tuple[Path, str]:
    """An ephemeral keypair for this run; RunPod injects the public half."""
    scratch.mkdir(parents=True, exist_ok=True)
    priv = scratch / "runpod_ed25519"
    if not priv.exists():
        subprocess.run(["ssh-keygen", "-t", "ed25519", "-N", "", "-q", "-f", str(priv), "-C", USER_AGENT], check=True)
    return priv, priv.with_suffix(".pub").read_text().strip()


def ssh_base(priv: Path, ip: str, port: int):
    return [
        "ssh", "-i", str(priv), "-p", str(port),
        # IdentitiesOnly: without it ssh offers every agent key first and sshd drops the connection with
        # "Too many authentication failures", which looks exactly like the pod never booting.
        "-o", "IdentitiesOnly=yes", "-o", "StrictHostKeyChecking=no", "-o", "UserKnownHostsFile=/dev/null",
        "-o", "LogLevel=ERROR", "-o", "ConnectTimeout=10", "-o", "ServerAliveInterval=30",
        f"root@{ip}",
    ]


def wait_for_ssh(pod_id: str, key: str, timeout: int):
    deadline = time.time() + timeout
    last, start = None, time.time()
    while time.time() < deadline:
        pod = call("GET", f"/pods/{pod_id}", key) or {}
        status = f"{pod.get('desiredStatus')}/{pod.get('lastStatus')}"
        ip, ports = pod.get("publicIp"), pod.get("portMappings") or {}
        if status != last:
            print(f"  [{int(time.time() - start):4d}s] {status} ip={ip} ports={list(ports)}", flush=True)
            last = status
        if ip and ports.get("22"):
            return ip, int(ports["22"])
        time.sleep(5)
    raise SystemExit("timed out waiting for the pod to expose SSH")


def wait_for_sshd(priv: Path, ip: str, port: int, timeout: int = 900):
    start = time.time()
    while time.time() < start + timeout:
        if subprocess.run(ssh_base(priv, ip, port) + ["true"], capture_output=True).returncode == 0:
            print(f"  ssh up after {int(time.time() - start)}s")
            return
        print(f"  [{int(time.time() - start):4d}s] waiting for sshd", flush=True)
        time.sleep(15)
    raise SystemExit(f"sshd never came up ({timeout}s)")


def source_tarball(dest: Path) -> Path:
    """The repository subset the pod needs, as it is in the working tree (uncommitted edits included)."""
    with tarfile.open(dest, "w:gz") as tar:
        for rel in SHIPPED:
            tar.add(REPO / rel, arcname=rel, filter=lambda ti: None if "/__pycache__" in ti.name else ti)
    return dest


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gpu", default=DEFAULT_GPU, help="comma-separated RunPod GPU type ids (default: %(default)s)")
    ap.add_argument("--image", default=DEFAULT_IMAGE)
    ap.add_argument("--duckdb", type=Path, help="DuckDB 1.5.6 CLI (linux_amd64) for the real-weights checks")
    ap.add_argument("--extension", type=Path, help="anofox_decide.duckdb_extension (linux_amd64) built from this commit")
    ap.add_argument("--model", default="laya-multilingual", help="catalog model for the real-weights checks")
    ap.add_argument("--timeout", type=int, default=900, help="seconds to wait for the pod")
    ap.add_argument("--community", action="store_true",
                    help="allow RunPod's COMMUNITY tier (cheaper; hosts are other people's machines and some cannot "
                         "initialise CUDA inside the container). The default is the SECURE tier.")
    ap.add_argument("--keep", action="store_true", help="do NOT terminate the pod (debugging; costs money)")
    ap.add_argument("--list", action="store_true", help="list pods and exit")
    ap.add_argument("--terminate", help="terminate a pod id and exit")
    args = ap.parse_args()

    key = api_key()
    if args.list:
        for pod in call("GET", "/pods", key) or []:
            print(f"{pod['id']}\t{pod.get('desiredStatus')}\t{pod.get('name')}\t{pod.get('machine', {}).get('gpuTypeId', '')}")
        return 0
    if args.terminate:
        call("DELETE", f"/pods/{args.terminate}", key)
        print(f"terminated {args.terminate}")
        return 0
    if bool(args.duckdb) != bool(args.extension):
        ap.error("--duckdb and --extension go together")

    scratch = Path(tempfile.gettempdir()) / "anofox-decide-runpod"
    priv, pub = ensure_keypair(scratch)
    tarball = source_tarball(scratch / "decide_gpu_src.tar.gz")
    uploads = [tarball] + [p for p in (args.duckdb, args.extension) if p]

    clouds = ("COMMUNITY", "SECURE") if args.community else ("SECURE",)
    pod = None
    for gpu in [g.strip() for g in args.gpu.split(",") if g.strip()]:
        for cloud in clouds:
            spec = {
                "name": "anofox-decide-cuda-verify", "imageName": args.image, "gpuTypeIds": [gpu], "gpuCount": 1,
                "cloudType": cloud, "containerDiskInGb": 40, "volumeInGb": 0, "ports": ["22/tcp"],
                "supportPublicIp": True, "env": {"PUBLIC_KEY": pub},
            }
            try:
                pod = call("POST", "/pods", key, spec)
                print(f"  got {gpu} on {cloud}")
                break
            except SystemExit as exc:
                if not any(m in str(exc).lower() for m in CAPACITY_ERRORS):
                    raise
        if pod:
            break
        print(f"  no capacity: {gpu}")
    if not pod:
        raise SystemExit("no capacity for any candidate GPU")
    pod_id = pod["id"]
    print(f"pod {pod_id} created")

    terminated = False

    def cleanup(*_):
        nonlocal terminated
        if args.keep:
            print(f"\n--keep: pod {pod_id} LEFT RUNNING - terminate with --terminate {pod_id}")
            return
        if terminated:
            return
        terminated = True
        print(f"\nterminating pod {pod_id}...")
        try:
            call("DELETE", f"/pods/{pod_id}", key)
            print("terminated")
        except SystemExit as exc:
            print(f"TERMINATION FAILED ({exc}) - terminate manually: --terminate {pod_id}")

    signal.signal(signal.SIGINT, lambda *a: (cleanup(), sys.exit(130)))
    signal.signal(signal.SIGTERM, lambda *a: (cleanup(), sys.exit(143)))

    try:
        ip, port = wait_for_ssh(pod_id, key, args.timeout)
        print(f"ssh root@{ip}:{port}")
        wait_for_sshd(priv, ip, port)
        subprocess.run(ssh_base(priv, ip, port) + ["mkdir -p /workspace"], check=True)
        for f in uploads:
            print(f"  upload {f.name}")
            subprocess.run(
                ["scp", "-i", str(priv), "-P", str(port), "-o", "IdentitiesOnly=yes", "-o", "StrictHostKeyChecking=no",
                 "-o", "UserKnownHostsFile=/dev/null", "-o", "LogLevel=ERROR", str(f), f"root@{ip}:/workspace/{f.name}"],
                check=True)
        env = f"DECIDE_MODEL={args.model} "
        if args.duckdb:
            env += f"DUCKDB_BIN=/workspace/{args.duckdb.name} DECIDE_EXTENSION=/workspace/{args.extension.name} "
        command = f"cd /workspace && tar xzf decide_gpu_src.tar.gz && {env}bash plugins/cuda/gpu_test/cuda_pod_verify.sh"
        print(f"\n--- running ---\n{command}\n")
        return subprocess.run(ssh_base(priv, ip, port) + [command]).returncode
    finally:
        cleanup()


if __name__ == "__main__":
    sys.exit(main())
