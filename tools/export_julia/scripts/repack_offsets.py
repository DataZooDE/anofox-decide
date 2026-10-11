#!/usr/bin/env python3
"""Add the packed f32 layout (offset/length on every stub, `pack` in the tensor map) to
existing weight-free graphs, in place. Only external-data metadata and the pack section
change; CPU injection still resolves weights by initializer name.

  repack_offsets.py ../../resources/graph_*.onnx ../../test/fixtures/local_tiny/graph_tiny.onnx
"""
import sys
from pathlib import Path

from export_julia.weight_free import repack


def main(argv):
    if not argv:
        print("usage: repack_offsets.py GRAPH.onnx...", file=sys.stderr)
        return 2
    for g in map(Path, argv):
        arch = g.stem.removeprefix("graph_")
        pack = repack(g, g.with_name(f"tensor_map_{arch}.json"))
        print(f"{g.name}: {len(pack['initializers'])} initializers, {pack['total_bytes']} packed bytes")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
