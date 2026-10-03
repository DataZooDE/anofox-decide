"""export_julia CLI: inspect | export (mirrors export_onnx/cli.py)."""
from __future__ import annotations

import argparse
import sys

from . import export as export_mod
from . import inspect as inspect_mod


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="export_julia")
    sub = parser.add_subparsers(dest="cmd", required=True)
    inspect_p = sub.add_parser("inspect", help="Pin the upstream model spec")
    inspect_p.add_argument("--repo", default=inspect_mod.REPO_ID)
    inspect_p.add_argument("--out", required=True)
    export_p = sub.add_parser("export", help="Export to ONNX")
    export_p.add_argument("--spec", required=True)
    export_p.add_argument("--weights", required=True)
    export_p.add_argument("--out", required=True)
    export_p.add_argument("--opset", type=int, default=17)
    export_p.add_argument("--weight-free", action="store_true")
    export_p.add_argument("--arch", default="julia-1")
    args = parser.parse_args(argv)
    if args.cmd == "inspect":
        return inspect_mod.main(["--repo", args.repo, "--out", args.out])
    extra = (["--weight-free"] if args.weight_free else []) + ["--arch", args.arch]
    return export_mod.main(
        ["--spec", args.spec, "--weights", args.weights, "--out", args.out, "--opset", str(args.opset)] + extra
    )


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
