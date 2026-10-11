#!/usr/bin/env python3
"""Rebuild the packed f32 blob from the safetensors using ONLY the tensor map's `pack` section
and check it against what the graph stubs describe.

  verify_pack.py GRAPH.onnx CHECKPOINT.safetensors

Checks, per mapped initializer: the stub's offset/length equal the pack entry; the entry is
64-byte aligned, inside total_bytes and overlap-free; and the blob bytes at the entry, viewed
as float32 with the stub's dims, equal the checkpoint tensor (upcast to f32, transposed if the
map says so), read through an independent loader (torch) than the one that built the blob.
"""
import json
import sys
from pathlib import Path

import onnx

from check_graph_invariants import check

from export_julia.weight_free import build_packed_blob, checkpoint_f32


def main(graph: Path, ckpt: Path) -> int:
    arch = graph.stem.removeprefix("graph_")
    tmap = json.loads(graph.with_name(f"tensor_map_{arch}.json").read_text())
    errors = check(graph)
    if errors:
        for error in errors:
            print("FAIL", error)
        return 1
    blob = build_packed_blob(tmap, ckpt)
    ref = checkpoint_f32(ckpt)
    pack = tmap["pack"]
    proto = onnx.load(str(graph), load_external_data=False)
    stubs = {i.name: i for i in proto.graph.initializer if i.name in tmap["initializers"]}
    for name in sorted(stubs):
        ent = pack["initializers"][name]
        a = ref[tmap["initializers"][name]]
        if tmap["transforms"].get(name) == "transpose":
            a = a.T
        got = blob[ent["offset"]:ent["offset"] + ent["length"]]
        if got.tobytes() != a.astype("<f4").tobytes():
            errors.append(f"{name}: blob bytes differ from the checkpoint")
    for e in errors[:20]:
        print("FAIL", e)
    if not errors:
        print(f"ok: {graph.name}: {len(stubs)} initializers, {blob.size} bytes rebuilt from the pack "
              "section equal the checkpoint")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main(Path(sys.argv[1]), Path(sys.argv[2])))
