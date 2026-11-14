#!/usr/bin/env python3
"""Half-ready graph passes (throwaway copies; never touches resources/).

Two CPU-neutral rewrites that make a graph safe to run with half-precision GEMMs:

  sanitize_masks                 every float Constant that is -inf or <= -65504 becomes -10000.0
                                 (the encoder's additive mask is finfo(float32).min, which is -inf in
                                 fp16/bf16 and turns a fully masked padded row into NaN).
  rotary_outer_product_to_mul    each MatMul named like `*rotary_emb*` (a K = 1 outer product
                                 [B,H,1] x [B,1,T]) becomes a broadcasting Mul with the same inputs and
                                 output, so every remaining MatMul is a real GEMM.

Usage:
  half_ready.py --out DIR graph_julia-1.onnx [graph_*.onnx ...] [--only masks|rotary]

Writes DIR/<graph name> (and copies the tensor map next to it); the inputs are never modified and
DIR may not be inside a `resources` directory.
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import numpy as np
import onnx
import onnx.numpy_helper

HALF_MASK_VALUE = -10000.0
HALF_MIN = -65504.0
_FLOAT_TYPES = (onnx.TensorProto.FLOAT, onnx.TensorProto.DOUBLE, onnx.TensorProto.FLOAT16)


def _walk_graphs(g):
    yield g
    for node in g.node:
        for attr in node.attribute:
            if attr.type == onnx.AttributeProto.GRAPH:
                yield from _walk_graphs(attr.g)
            elif attr.type == onnx.AttributeProto.GRAPHS:
                for sub in attr.graphs:
                    yield from _walk_graphs(sub)


def sanitize_masks(proto: onnx.ModelProto) -> int:
    """Rewrite -inf / <= -65504 float Constant values to -10000.0 in place; return the number of
    constants changed. +inf and NaN anywhere in a float Constant raise (never a mask)."""
    changed = 0
    for g in _walk_graphs(proto.graph):
        for node in g.node:
            if node.op_type != "Constant":
                continue
            for attr in node.attribute:
                if attr.name != "value" or attr.type != onnx.AttributeProto.TENSOR:
                    continue
                if attr.t.data_type not in _FLOAT_TYPES:
                    continue
                v = onnx.numpy_helper.to_array(attr.t)
                if np.isnan(v).any() or np.isposinf(v).any():
                    raise ValueError(f"{node.name}: +inf/NaN float constant, refusing to sanitize")
                bad = v <= HALF_MIN  # includes -inf
                if not bad.any():
                    continue
                new = np.where(bad, np.array(HALF_MASK_VALUE, dtype=v.dtype), v).astype(v.dtype)
                attr.t.CopyFrom(onnx.numpy_helper.from_array(new, attr.t.name))
                changed += 1
    for init in proto.graph.initializer:  # inline float initializers must not hide a -inf either
        if init.data_type in _FLOAT_TYPES and init.data_location != onnx.TensorProto.EXTERNAL:
            v = onnx.numpy_helper.to_array(init)
            if v.size and (not np.isfinite(v).all() or (v <= HALF_MIN).any()):
                raise ValueError(f"initializer {init.name} holds a non-finite or <= -65504 value")
    return changed


def rotary_outer_product_to_mul(proto: onnx.ModelProto) -> int:
    """MatMul nodes whose name contains `rotary_emb` become Mul (same inputs/outputs); return count."""
    changed = 0
    for g in _walk_graphs(proto.graph):
        for node in g.node:
            if node.op_type == "MatMul" and "rotary_emb" in node.name:
                node.op_type = "Mul"
                node.name += "_as_Mul"  # a plain `.../Mul` name already exists in the rotary block
                changed += 1
    return changed


def half_ready(proto: onnx.ModelProto, *, masks: bool = True, rotary: bool = True) -> dict:
    return {"masks": sanitize_masks(proto) if masks else 0,
            "rotary": rotary_outer_product_to_mul(proto) if rotary else 0}


def patch_file(src: Path, out_dir: Path, *, masks: bool = True, rotary: bool = True) -> dict:
    if "resources" in {p.name for p in out_dir.resolve().parents} | {out_dir.resolve().name}:
        raise SystemExit("refusing to write into a resources directory; use a scratch --out")
    out_dir.mkdir(parents=True, exist_ok=True)
    proto = onnx.load(str(src), load_external_data=False)  # keeps the weight-free stubs
    counts = half_ready(proto, masks=masks, rotary=rotary)
    dst = out_dir / src.name
    if dst.resolve() == src.resolve():
        raise SystemExit("--out would overwrite the input")
    onnx.save(proto, str(dst))
    tmap = src.with_name(src.name.replace("graph_", "tensor_map_").replace(".onnx", ".json"))
    if tmap.exists():
        shutil.copyfile(tmap, out_dir / tmap.name)
    return counts


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("graphs", nargs="+", type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--only", choices=["masks", "rotary"])
    a = ap.parse_args(argv)
    for g in a.graphs:
        c = patch_file(g, a.out, masks=a.only in (None, "masks"), rotary=a.only in (None, "rotary"))
        print(f"{g.name}: sanitized {c['masks']} constants, rotary MatMul->Mul {c['rotary']} -> {a.out / g.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
