#!/usr/bin/env python3
"""Fetch, verify and stage the CUDA runtime exactly as decide_download_runtime('cuda') does.

The pinned wheel (url, size, sha256) and the entries taken from it are READ from src/decide_accelerate.cpp, so
this can never drift from what the extension ships: if the extension's pin changes, the next on-hardware run
verifies the new one. Stages, into <dest>:

    libonnxruntime.so.<ver>                the core, UNPATCHED (the CMake build renames it with soname_patch_tool,
                                           the routine the extension uses; this script does not rename anything)
    libonnxruntime_providers_cuda.so       the CUDA provider
    libonnxruntime_providers_shared.so

usage: pinned_runtime.py <src/decide_accelerate.cpp> <dest dir> [--parse-only]
"""
import hashlib
import re
import sys
import urllib.request
import zipfile
from pathlib import Path


def literal(source: str, name: str) -> str:
    m = re.search(r"\b" + name + r"\s*=\s*(.*?);", source, re.S)
    if not m:
        raise SystemExit(f"{name} not found in decide_accelerate.cpp: the pin moved, update this script")
    return "".join(re.findall(r'"([^"]*)"', m.group(1))) or m.group(1).strip()


def main() -> int:
    source = Path(sys.argv[1]).read_text()
    dest = Path(sys.argv[2])
    parse_only = len(sys.argv) > 3 and sys.argv[3] == "--parse-only"
    dest.mkdir(parents=True, exist_ok=True)
    url = literal(source, "kOrtCudaWheelUrl")
    size = int(literal(source, "kOrtCudaWheelBytes"))
    sha = literal(source, "kOrtCudaWheelSha256")
    entries = re.findall(r'"(onnxruntime/capi/[^"]+)"', source)
    assert len(entries) == 3, entries
    if parse_only:
        print(url, size, sha, entries, sep="\n")
        return 0
    wheel = dest / "runtime.whl"
    print(f"fetching the pinned runtime ({size} bytes)")
    with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "anofox-decide-gpu-test/1.0"}),
                                timeout=600) as resp, open(wheel, "wb") as out:
        h = hashlib.sha256()
        total = 0
        while chunk := resp.read(1 << 20):
            out.write(chunk)
            h.update(chunk)
            total += len(chunk)
    ok = True
    print(f"PIN_SIZE {'OK' if total == size else 'MISMATCH'}: {total} bytes (pinned {size})")
    ok = ok and total == size
    print(f"PIN_SHA256 {'OK' if h.hexdigest() == sha else 'MISMATCH'}: {h.hexdigest()}")
    ok = ok and h.hexdigest() == sha
    z = zipfile.ZipFile(wheel)
    names = set(z.namelist())
    for entry in entries:
        present = entry in names
        print(f"  {'OK  ' if present else 'MISS'} {entry}")
        ok = ok and present
        if present:
            (dest / Path(entry).name).write_bytes(z.read(entry))
    wheel.unlink()
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
