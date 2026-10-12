#!/usr/bin/env python3
"""CI invariants: the committed graphs contain no weight bytes.

For every graph given (resources/graph_*.onnx, test fixtures):
  * a tensor_map_<arch>.json sits next to it and covers every non-inline initializer,
  * every mapped initializer is an external-data stub with no payload whose `offset`/`length`
    equal the tensor map's `pack` layout (64-byte aligned, sorted by name, f32, no overlap),
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
    tmap = json.loads(tmap_path.read_text())
    mapped = set(tmap["initializers"])
    pack = tmap.get("pack")
    if not pack:
        errors.append(f"{graph}: tensor map has no `pack` section (run scripts/repack_offsets.py)")
    else:
        if set(pack["initializers"]) != mapped:
            errors.append(f"{graph}: pack initializer names differ from the tensor map")
        end = 0
        for name in sorted(mapped):
            ent = pack["initializers"].get(name)
            if ent is None:
                errors.append(f"{graph}: pack lacks {name}")
                continue
            if ent["offset"] != end or ent["length"] < 0:
                errors.append(f"{graph}: pack offset/length of {name} is not canonical")
            end = (ent["offset"] + ent["length"] + 63) // 64 * 64
        if end != pack["total_bytes"] or pack["alignment"] != 64 or pack["itemsize"] != 4 \
                or pack.get("location") != "weights.bin":
            errors.append(f"{graph}: pack header inconsistent (location/total_bytes/alignment/itemsize)")
    weight_files = [p for pattern in ("*.onnx.data", "*.safetensors", "weights.bin")
                    for p in graph.parent.glob(pattern)]
    for sib in weight_files:
        if graph.parent.name == "resources":
            errors.append(f"{sib}: weight file committed")
    proto = onnx.load(str(graph), load_external_data=False)
    inline = 0
    seen = set()
    for init in proto.graph.initializer:
        seen.add(init.name)
        external = init.data_location == onnx.TensorProto.EXTERNAL
        payload = len(init.raw_data) + 4 * len(init.float_data) + 4 * len(init.int32_data) + 8 * len(init.int64_data) \
            + 8 * len(init.double_data) + 8 * len(init.uint64_data) + sum(map(len, init.string_data))
        if init.name in mapped:
            if not external or payload:
                errors.append(f"{graph}: mapped initializer {init.name} carries {payload} payload bytes")
            elif pack and init.name in pack["initializers"]:
                ext = {e.key: e.value for e in init.external_data}
                want = pack["initializers"][init.name]
                numel = 1
                for d in init.dims:
                    numel *= d
                if ext != {"location": pack.get("location"), "offset": str(want["offset"]),
                           "length": str(want["length"])} or len(init.external_data) != 3 \
                        or list(init.dims) != tmap["shapes"].get(init.name) \
                        or want["length"] != numel * 4 or init.data_type != onnx.TensorProto.FLOAT:
                    errors.append(f"{graph}: stub {init.name} {ext} does not match pack {want}")
        else:
            if external:
                errors.append(f"{graph}: unmapped external initializer {init.name}")
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
