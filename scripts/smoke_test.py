#!/usr/bin/env python3
"""Post-build smoke test: does the shipped artifact actually load and work?

Everything else in CI exercises ``build/release/duckdb`` -- a shell with the
extension *statically linked in*. That never loads the ``.duckdb_extension``
file users install, so it cannot catch an artifact that builds green but will
not instantiate. quack-oauth's wasm side-module is the standing example: it
"builds green but won't instantiate in the browser".

This downloads the OFFICIAL DuckDB CLI for the exact target version, installs
the artifact into it, loads it, and calls a real function -- which is what a
user does, and the only thing that proves the artifact works on this platform.

Usage:
    python3 scripts/smoke_test.py <extension_path> <duckdb_version> <arch>

    arch is the DuckDB arch name used by the build matrix, e.g. linux_amd64.
"""

from __future__ import annotations

import os
import platform
import subprocess
import sys
import tempfile
import urllib.request
import zipfile

# ── Per-repo configuration ───────────────────────────────────────────────────
# The only two values that change between repositories.
EXTENSION_NAME = "anofox_decide"

# A function that must exist and be callable with no network, no credentials
# and no external service. Loading proves the shared object resolves; it does
# not prove the registered functions work, because symbols can resolve lazily
# and only fail when actually called. Verified against a real build of this
# extension before being committed.
SMOKE_QUERY = "SELECT count(*) AS n FROM decide_models();"

# End-to-end local inference on the shipped artifact. decide_models() proves
# functions are callable; it never touches ONNX Runtime. This registers the
# committed weight-free fixture graph + tiny tokenizer (no network, no license
# gate) and scores through the ORT session -- the code path a user hits first,
# and one an artifact can ship broken (resolve lazily, fail at Run). The
# stub provider line proves the default surface as well.
_FIXTURES = "{fixtures}"  # substituted with an absolute path at runtime
INFERENCE_QUERY = f"""
SELECT decide_register_model('smoke-local', 'local', '{_FIXTURES}/julia1_tiny.onnx', '{_FIXTURES}/tiny_tokenizer.json');
SELECT (p BETWEEN 0.0 AND 1.0)::INTEGER FROM (
  SELECT decide_probability('The bill is wrong.', 'A refund is requested.', model := 'smoke-local') AS p);
SELECT (decide_choice('The bill is wrong.', 'Which team?', ['billing','other'], model := 'smoke-local') IN ('billing','other'))::INTEGER;
"""
# The register call echoes one row (true); the contract is the tail: the
# in-range probability flag, then the valid-choice flag.
INFERENCE_EXPECT = ["1", "1"]
# ─────────────────────────────────────────────────────────────────────────────

ARCH_TO_CLI_ZIP: dict[str, str] = {
    "linux_amd64": "duckdb_cli-linux-amd64.zip",
    "linux_arm64": "duckdb_cli-linux-aarch64.zip",
    "linux_amd64_musl": "duckdb_cli-linux-amd64.zip",
    "osx_amd64": "duckdb_cli-osx-universal.zip",
    "osx_arm64": "duckdb_cli-osx-universal.zip",
    "windows_amd64": "duckdb_cli-windows-amd64.zip",
}


def _download_duckdb_cli(version: str, arch: str, dest_dir: str) -> str:
    zip_name = ARCH_TO_CLI_ZIP.get(arch)
    if zip_name is None:
        raise SystemExit(
            f"Unsupported arch '{arch}'. Supported: {sorted(ARCH_TO_CLI_ZIP)}"
        )

    url = f"https://github.com/duckdb/duckdb/releases/download/{version}/{zip_name}"
    zip_path = os.path.join(dest_dir, "duckdb_cli.zip")

    print(f"Downloading official DuckDB {version} CLI ({arch}):\n  {url}")
    urllib.request.urlretrieve(url, zip_path)

    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(dest_dir)

    bin_name = "duckdb.exe" if platform.system() == "Windows" else "duckdb"
    binary = os.path.join(dest_dir, bin_name)
    if not os.path.isfile(binary):
        raise SystemExit(
            f"DuckDB binary not found after extraction: {binary}\n"
            f"Zip contents: {zipfile.ZipFile(zip_path).namelist()}"
        )

    if platform.system() != "Windows":
        os.chmod(binary, 0o755)
    return binary


def _run_sql(duckdb_bin: str, sql: str, home: str, extra_env: dict | None = None,
             timeout: int = 300) -> subprocess.CompletedProcess:
    # An isolated HOME keeps the developer's real ~/.duckdb untouched, and
    # -unsigned is required because a locally built artifact is not signed.
    env = dict(os.environ)
    env["HOME"] = home
    env["USERPROFILE"] = home
    # No telemetry from CI, and no network dependence in the smoke path.
    env["DATAZOO_DISABLE_TELEMETRY"] = "1"
    env.update(extra_env or {})
    return subprocess.run(
        [duckdb_bin, "-unsigned", "-noheader", "-list", "-c", sql],
        capture_output=True,
        text=True,
        env=env,
        timeout=timeout,
    )


def _fail(message: str, proc: subprocess.CompletedProcess | None = None) -> None:
    out = ""
    if proc is not None:
        out = f"\n  exit code : {proc.returncode}\n"
        if proc.stdout:
            out += f"  stdout    : {proc.stdout.strip()}\n"
        if proc.stderr:
            out += f"  stderr    : {proc.stderr.strip()}\n"
    raise SystemExit(f"Smoke test FAILED: {message}{out}")


def run_smoke_test(extension_path: str, duckdb_version: str, arch: str) -> None:
    if not os.path.isfile(extension_path):
        raise SystemExit(f"Smoke test FAILED: artifact not found: {extension_path}")

    # Forward slashes work in DuckDB SQL on every platform, Windows included.
    ext = extension_path.replace("\\", "/")

    with tempfile.TemporaryDirectory() as tmpdir:
        duckdb_bin = _download_duckdb_cli(duckdb_version, arch, tmpdir)
        home = os.path.join(tmpdir, "home")
        os.makedirs(home, exist_ok=True)

        print(
            f"\nSmoke test\n"
            f"  extension : {extension_path}\n"
            f"  duckdb    : {duckdb_version} ({arch})\n"
        )

        # 1. Install -- catches a truncated or wrong-platform artifact.
        print("[1/6] Installing the artifact into a stock CLI")
        proc = _run_sql(duckdb_bin, f"INSTALL '{ext}';", home)
        if proc.returncode != 0:
            _fail("the artifact is not installable on this platform", proc)

        # 2. Load, and confirm DuckDB itself agrees it loaded.
        print("[2/6] Loading and checking duckdb_extensions()")
        proc = _run_sql(
            duckdb_bin,
            f"LOAD {EXTENSION_NAME};\n"
            f"SELECT loaded FROM duckdb_extensions() "
            f"WHERE extension_name = '{EXTENSION_NAME}';",
            home,
        )
        if proc.returncode != 0:
            _fail(f"{EXTENSION_NAME} did not load", proc)
        if "true" not in (proc.stdout or "").lower():
            _fail(
                f"{EXTENSION_NAME} does not report loaded=true. An artifact that "
                f"builds green but will not instantiate is exactly what this "
                f"test exists to catch",
                proc,
            )

        # 3. Call a real function -- loading alone does not prove the registered
        #    functions work.
        print(f"[3/6] Calling a real function:\n      {SMOKE_QUERY}")
        proc = _run_sql(duckdb_bin, f"LOAD {EXTENSION_NAME};\n{SMOKE_QUERY}", home)
        if proc.returncode != 0:
            _fail("the smoke query failed", proc)
        if not (proc.stdout or "").strip():
            _fail("the smoke query returned no output", proc)
        print(f"      -> {proc.stdout.strip()}")

        # 4. Real inference through ONNX Runtime on the fixture model.
        repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        fixtures = os.path.join(repo_root, "test", "fixtures").replace("\\", "/")
        if not os.path.isdir(fixtures):
            _fail(f"fixture directory not found: {fixtures}")
        sql = INFERENCE_QUERY.replace(_FIXTURES, fixtures)
        print("[4/6] Local inference on the committed fixture model")
        proc = _run_sql(duckdb_bin, f"LOAD {EXTENSION_NAME};\n{sql}", home)
        if proc.returncode != 0:
            _fail("fixture-model inference failed", proc)
        lines = [ln.strip() for ln in (proc.stdout or "").splitlines() if ln.strip()]
        if lines[-2:] != INFERENCE_EXPECT:
            _fail(
                f"inference output mismatch: expected {INFERENCE_EXPECT}, got {lines}",
                proc,
            )
        print(f"      -> probability_in_range={lines[-2]} choice_valid={lines[-1]}")

        # 5. TLS from our own HTTPS client on this OS: an unauthenticated call to the real Cloudflare API must
        #    come back as an HTTP 401 envelope. A failed handshake or missing CA store reads differently
        #    ("TLS", "certificate"), so this proves handshake + CA roots on Windows and macOS too.
        print("[5/6] HTTPS handshake and CA roots (expecting an HTTP 401 from api.cloudflare.com)")
        proc = _run_sql(
            duckdb_bin,
            f"LOAD {EXTENSION_NAME};\n"
            "SET anofox_decide_allow_remote = true;\nSET anofox_decide_max_retries = 0;\n"
            "SELECT decide_register_model('smoke-clef', 'cloudflare', "
            "MAP {'account_id': '00000000000000000000000000000000'});\n"
            "SELECT decide_probability('x', 'A refund is requested.', model := 'smoke-clef');",
            home,
            extra_env={"CLOUDFLARE_API_TOKEN": "smoke-test-not-a-key"},
        )
        combined = (proc.stdout or "") + (proc.stderr or "")
        if proc.returncode == 0 or "HTTP 401" not in combined:
            _fail("expected an HTTP 401 answer from api.cloudflare.com (a TLS or CA problem looks different)", proc)
        print("      -> HTTP 401 envelope received")

        # 6. The first-run path for local models: download a real model (Hugging Face, over our HTTPS client and
        #    its signed CDN redirect), then score with it through the embedded graph and the injected weights.
        print("[6/6] decide_download('julia-1') and a real local inference")
        cache = os.path.join(tmpdir, "cache").replace("\\", "/")
        sql = (
            f"LOAD {EXTENSION_NAME};\n"
            f"SET anofox_decide_cache_dir = '{cache}';\n"
            "SELECT count(*) FROM decide_download('julia-1');\n"
            "SELECT round(decide_probability('I want my money back.', 'A refund is requested.', "
            "model := 'julia-1'), 3);"
        )
        proc = _run_sql(duckdb_bin, sql, home, timeout=1200)
        if proc.returncode != 0:
            _fail("decide_download / local inference failed", proc)
        lines = [ln.strip() for ln in (proc.stdout or "").splitlines() if ln.strip()]
        if lines[-2:] != ["2", "0.991"]:
            _fail(f"expected 2 downloaded files and probability 0.991 (reference 0.99083), got {lines}", proc)
        print(f"      -> files={lines[-2]} probability={lines[-1]}")

    print(f"\nSmoke test PASSED ({EXTENSION_NAME} on {arch})")


if __name__ == "__main__":
    if len(sys.argv) != 4:
        raise SystemExit(
            f"Usage: {sys.argv[0]} <extension_path> <duckdb_version> <arch>"
        )
    run_smoke_test(
        extension_path=sys.argv[1],
        duckdb_version=sys.argv[2],
        arch=sys.argv[3],
    )
