#!/usr/bin/env python3
"""Gate G1: the fp16 NaN hazard is real on the original graph and gone on the half-ready graph.

For each model: convert (i) the original and (ii) the half-ready graph (masks + rotary Mul) with
onnxruntime.transformers.float16.convert_float_to_float16 (keep_io_types=True; RoPE chain, LayerNorm,
Softmax, Erf, Where and Cast kept fp32 via the block lists), run ORT CPU over padded batches (rows padded
> 64 tokens beyond their last token, so fully masked local-window rows exist) and count non-finite values
in `scores` on the VALID markers. Also runs the "half-GEMM only" policy (MatMul operands cast to fp16,
result cast back; everything else fp32) on the half-ready graph and reports max |delta p| versus fp32
(informational for G3).

  gate_g1_nan.py [--models laya-multilingual laya-typed-decisions julia-1 tiny] [--tickets tickets.csv]
                 [--ladder 128 256] [--threads 8] [--no-gemm-only]

PASS needs: both half-ready converted graphs (blocked, full, and full with non-saturating mask constants) have 0 non-finite valid-marker scores on every scenario. The
original graph's NaNs are reported (the hazard demonstration) but do not decide PASS/FAIL.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
import onnx  # noqa: E402
import onnxruntime as ort  # noqa: E402
from onnx import TensorProto, helper  # noqa: E402
from tokenizers import Tokenizer  # noqa: E402

import _gate_common as C  # noqa: E402
import half_ready as H  # noqa: E402

FP32_OPS = ["LayerNormalization", "Softmax", "Erf", "Where", "Cast"]


def convert_fp16(proto: onnx.ModelProto, policy: str = "blocked") -> onnx.ModelProto:
    """policy "blocked": FP32_OPS and the RoPE chain stay fp32 (the mask Constants feed a fp32 Where, so they
    are not narrowed); "full": only the RoPE chain and the graph's own Cast nodes stay fp32 (the converter
    produces a type-inconsistent graph if Cast is converted); Where, mask constants, LayerNorm, Softmax...
    are all converted."""
    from onnxruntime.transformers.float16 import DEFAULT_OP_BLOCK_LIST, convert_float_to_float16

    rope = [n.name for n in proto.graph.node if "rotary_emb" in n.name]
    ops = sorted(set(DEFAULT_OP_BLOCK_LIST) | set(FP32_OPS)) if policy == "blocked" else ["Cast"]
    out = convert_float_to_float16(proto, keep_io_types=True, op_block_list=ops, node_block_list=rope)
    # The converter emits the same boundary Cast (same input, same output) once per blocked consumer of a
    # tensor; keep the first and drop the exact duplicates.
    defined, keep, dropped = set(), [], 0
    for node in out.graph.node:
        if node.op_type == "Cast" and node.output[0] in defined:
            dropped += 1
            continue
        defined.update(node.output)
        keep.append(node)
    del out.graph.node[:]
    out.graph.node.extend(keep)
    return out


def unclamp_masks(proto: onnx.ModelProto) -> int:
    """What a plain (non-saturating) fp32 -> fp16 Cast does to the masks: float Constants <= -65504 become
    -inf. The ORT converter instead clamps them to -65504 (finite), which hides the hazard, so the
    `noclamp` variants model converters such as a MIGraphX/MLX/CUDA cast of the constant."""
    n = 0
    for node in proto.graph.node:
        if node.op_type == "Constant":
            for a in node.attribute:
                if a.name == "value" and a.t.data_type == TensorProto.FLOAT:
                    v = onnx.numpy_helper.to_array(a.t)
                    if (v <= H.HALF_MIN).any():
                        a.t.CopyFrom(onnx.numpy_helper.from_array(np.where(v <= H.HALF_MIN, -np.inf, v).astype(v.dtype), a.t.name))
                        n += 1
    return n


def half_gemm_only(proto: onnx.ModelProto) -> int:
    """Wrap every MatMul: Cast(to=fp16) on both operands, MatMul, Cast(to=fp32) on the result.

    ORT CPU has no fp16 MatMul kernel: it runs the MatMul in fp32 between inserted Casts and then deletes
    cast pairs, which silently removes the fp16 rounding (even at ORT_DISABLE_ALL). An Identity (which has an
    fp16 kernel) after each fp16 Cast and before the output Cast keeps the pairs apart, so the result is
    exactly "round operands to fp16, fp32 accumulate, round the result to fp16" (verified against numpy)."""
    g = proto.graph
    new_nodes, n = [], 0
    for node in g.node:
        if node.op_type != "MatMul":
            new_nodes.append(node)
            continue
        a, b = node.input
        out = node.output[0]
        ca, cb, mo = f"{a}__h{n}", f"{b}__h{n}", f"{out}__h{n}"
        new_nodes += [
            helper.make_node("Cast", [a], [ca + "c"], name=f"{node.name}/cast_a", to=TensorProto.FLOAT16),
            helper.make_node("Identity", [ca + "c"], [ca], name=f"{node.name}/keep_a"),
            helper.make_node("Cast", [b], [cb + "c"], name=f"{node.name}/cast_b", to=TensorProto.FLOAT16),
            helper.make_node("Identity", [cb + "c"], [cb], name=f"{node.name}/keep_b"),
            helper.make_node("MatMul", [ca, cb], [mo], name=node.name),
            helper.make_node("Identity", [mo], [mo + "k"], name=f"{node.name}/keep_out"),
            helper.make_node("Cast", [mo + "k"], [out], name=f"{node.name}/cast_out", to=TensorProto.FLOAT),
        ]
        n += 1
    del g.node[:]
    g.node.extend(new_nodes)
    return n


def plain_session(proto: onnx.ModelProto, threads: int):
    o = ort.SessionOptions()
    o.intra_op_num_threads = threads
    o.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL  # keep the fp16 Casts as written
    return ort.InferenceSession(proto.SerializeToString(), o, providers=["CPUExecutionProvider"])


def probs(scores, mm):
    s = np.where(mm, scores.astype(np.float64), -np.inf)
    e = np.exp(s - s.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


def nonfinite(scores, mm):
    return int((~np.isfinite(scores[mm])).sum())


def run_model(name, args) -> dict:
    files = C.tiny_files() if name == "tiny" else C.model_files(name)
    tmap = json.loads(Path(files["tmap"]).read_text())
    ckpt = C.load_ckpt(files["weights"])
    if name == "tiny":
        scen = C.random_scenarios()
    else:
        tok = Tokenizer.from_file(str(files["tokenizer"]))
        scen = C.scenarios(tok, C.specials(tok), C.read_tickets(args.tickets), tuple(args.ladder), max_b=8)
    res = {"model": name, "scenarios": [n for n, _ in scen]}

    orig = onnx.load(str(files["graph"]), load_external_data=False)
    ready = onnx.load(str(files["graph"]), load_external_data=False)
    res["patched"] = H.half_ready(ready)
    ref_sess = C.make_session(ready, tmap, ckpt, args.threads)
    ref = {n: ref_sess.run(["scores"], f)[0] for n, f in scen}

    for label, proto, policy in (("original", orig, "blocked"), ("half_ready", ready, "blocked"),
                                 ("original_full", orig, "full"), ("half_ready_full", ready, "full"),
                                 ("original_noclamp", orig, "full"), ("half_ready_noclamp", ready, "full")):
        proto = onnx.ModelProto.FromString(proto.SerializeToString())
        if label.endswith("noclamp"):
            res.setdefault("unclamped_constants", {})[label] = unclamp_masks(proto)
        conv = convert_fp16(C.fill_weights(proto, tmap, ckpt), policy)
        sess = plain_session(conv, args.threads)
        bad, dmax = {}, 0.0
        for n, f in scen:
            s = sess.run(["scores"], f)[0]
            nf = nonfinite(s, f["marker_mask"])
            if nf:
                bad[n] = nf
            else:
                dmax = max(dmax, float(np.abs(probs(s, f["marker_mask"]) - probs(ref[n], f["marker_mask"]))[f["marker_mask"]].max()))
        res[f"fp16_{label}"] = {"nonfinite_valid_by_scenario": bad, "nonfinite_total": sum(bad.values()),
                                "max_abs_dp_finite_scenarios": dmax}
        print(json.dumps({"model": name, label: res[f"fp16_{label}"]}), flush=True)
        del conv, sess

    if args.gemm_only:
        gp = onnx.load(str(files["graph"]), load_external_data=False)
        H.half_ready(gp)
        res["gemm_only_matmuls"] = half_gemm_only(gp)
        sess = C.make_session(gp, tmap, ckpt, args.threads, optimize=False)
        nf, ds, dp = 0, 0.0, 0.0
        for n, f in scen:
            s = sess.run(["scores"], f)[0]
            mm = f["marker_mask"]
            nf += nonfinite(s, mm)
            if np.isfinite(s[mm]).all():
                ds = max(ds, float(np.abs(s - ref[n])[mm].max()))
                dp = max(dp, float(np.abs(probs(s, mm) - probs(ref[n], mm))[mm].max()))
        res["half_gemm_only"] = {"nonfinite": nf, "max_abs_dscore": ds, "max_abs_dp": dp}
        print(json.dumps({"model": name, "half_gemm_only": res["half_gemm_only"]}), flush=True)
    return res


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", nargs="+", default=list(C.MODELS) + ["tiny"])
    ap.add_argument("--tickets", type=Path)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--ladder", nargs="+", type=int, default=[128, 256])
    ap.add_argument("--no-gemm-only", dest="gemm_only", action="store_false")
    a = ap.parse_args(argv)
    results = [run_model(m, a) for m in a.models]
    ok = all(r[f"fp16_half_ready{p}"]["nonfinite_total"] == 0 for r in results for p in ("", "_full", "_noclamp"))
    for p in ("", "_full", "_noclamp"):
        hazard = [r["model"] for r in results if r[f"fp16_original{p}"]["nonfinite_total"] > 0]
        print(f"hazard reproduced on the original graph ({p[1:] or 'blocked'} conversion) for: {hazard or 'none'}")
    print("G1 PASS: 0 non-finite valid-marker scores on the half-ready fp16 graph" if ok else
          "G1 FAIL: the half-ready fp16 graph still yields non-finite valid-marker scores")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
