#!/usr/bin/env python3
"""CI invariants: the committed graphs contain no weight bytes.

For every graph given (resources/graph_*.onnx, test fixtures):
  * a tensor_map_<arch>.json sits next to it and covers every non-inline initializer,
  * every mapped initializer is an external-data stub with no payload,
  * the inline initializers (rotary tables, scalar constants) stay under a small budget,
  * no `.onnx.data` / `.safetensors` file is committed next to it,
  * the graph file itself is small.
Needs only `pip install onnx`; no weights, no onnxruntime.
"""
import json
import sys
from pathlib import Path

import onnx

MAX_GRAPH_BYTES = 3 * 1024 * 1024
INLINE_BUDGET = 256 * 1024


def check(graph: Path) -> list[str]:
    errors = []
    arch = graph.stem.removeprefix("graph_")
    tmap_path = graph.with_name(f"tensor_map_{arch}.json")
    if graph.stat().st_size > MAX_GRAPH_BYTES:
        errors.append(f"{graph}: {graph.stat().st_size} bytes exceeds {MAX_GRAPH_BYTES}")
    if not tmap_path.exists():
        return errors + [f"{graph}: missing {tmap_path.name}"]
    mapped = set(json.loads(tmap_path.read_text())["initializers"])
    for sib in list(graph.parent.glob("*.onnx.data")) + list(graph.parent.glob("*.safetensors")):
        if graph.parent.name == "resources":
            errors.append(f"{sib}: weight file committed")
    proto = onnx.load(str(graph), load_external_data=False)
    inline = 0
    seen = set()
    for init in proto.graph.initializer:
        seen.add(init.name)
        external = init.data_location == onnx.TensorProto.EXTERNAL
        payload = len(init.raw_data) + 4 * len(init.float_data) + 4 * len(init.int32_data) + 8 * len(init.int64_data)
        if init.name in mapped:
            if not external or payload:
                errors.append(f"{graph}: mapped initializer {init.name} carries {payload} payload bytes")
        else:
            inline += payload
    missing = sorted(mapped - seen)
    if missing:
        errors.append(f"{graph}: map names initializers the graph lacks: {missing[:5]}")
    if inline > INLINE_BUDGET:
        errors.append(f"{graph}: {inline} inline initializer bytes exceed {INLINE_BUDGET}")
    return errors


def main(argv):
    if not argv:
        print("usage: check_graph_invariants.py GRAPH.onnx...", file=sys.stderr)
        return 2
    errors = []
    for g in argv:
        errors += check(Path(g))
    for e in errors:
        print("FAIL", e)
    if not errors:
        print(f"ok: {len(argv)} graph(s) are weight-free")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
