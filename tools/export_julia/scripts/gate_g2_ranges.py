#!/usr/bin/env python3
"""Gate G2: activation ranges at every MatMul of the fp32 graph (is fp16 enough, or is bf16 needed?).

For every MatMul the script records max |x| and the smallest non-zero |x| of each non-weight input and of
the output over realistic padded batches, plus the number of non-zero values below the fp16 normal minimum
6.1e-5. The statistics are reduced INSIDE the graph (Abs/ReduceMax/ReduceMin nodes added to a scratch copy),
so even T = 8192 attention matrices are never held. Weight operands are measured directly from the checkpoint.

  gate_g2_ranges.py [--models laya-multilingual laya-typed-decisions julia-1 tiny] [--tickets tickets.csv]
                    [--ladder 128 256 1024] [--long 8192] [--threads 8] [--json out.json]

`--long N` adds one single-row batch of exactly N tokens for julia-1 (the longest bucket; memory heavy).
Per model: fp16 feasible when every activation |x| (MatMul inputs and outputs) stays below 3e4 (fp16 max is
65504, 3e4 leaves headroom), otherwise bf16 is needed. Prints a final G2 line; exit 0 always unless a run fails
(it is a measurement, the conclusion is in the output).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
import onnx  # noqa: E402
from onnx import TensorProto, helper  # noqa: E402
from tokenizers import Tokenizer  # noqa: E402

import _gate_common as C  # noqa: E402

FP16_HEADROOM = 3.0e4
FP16_NORMAL_MIN = 6.1e-5
BIG = float(np.finfo(np.float32).max)


def add_stats(proto: onnx.ModelProto, tensors: list[str]) -> dict[str, tuple[str, str, str]]:
    """Append max|x|, min non-zero |x| and #(0 < |x| < fp16 normal min) reductions for each tensor."""
    g = proto.graph
    consts = {
        "zero": helper.make_tensor("g2_zero", TensorProto.FLOAT, [], [0.0]),
        "big": helper.make_tensor("g2_big", TensorProto.FLOAT, [], [BIG]),
        "tiny": helper.make_tensor("g2_tiny", TensorProto.FLOAT, [], [FP16_NORMAL_MIN]),
    }
    for k, t in consts.items():
        g.initializer.append(t)
    out, after = {}, {}  # tensor -> stat nodes; they are placed right after the producer so the tensor can be freed
    for i, t in enumerate(tensors):
        p = f"g2_{i}"
        nodes = [
            helper.make_node("Abs", [t], [p + "_abs"]),
            helper.make_node("ReduceMax", [p + "_abs"], [p + "_max"], keepdims=0),
            helper.make_node("Equal", [p + "_abs", "g2_zero"], [p + "_isz"]),
            helper.make_node("Where", [p + "_isz", "g2_big", p + "_abs"], [p + "_nz"]),
            helper.make_node("ReduceMin", [p + "_nz"], [p + "_min"], keepdims=0),
            helper.make_node("Less", [p + "_abs", "g2_tiny"], [p + "_lt"]),
            helper.make_node("Not", [p + "_isz"], [p + "_nnz"]),
            helper.make_node("And", [p + "_lt", p + "_nnz"], [p + "_small"]),
            helper.make_node("Cast", [p + "_small"], [p + "_smallf"], to=TensorProto.FLOAT),
            helper.make_node("ReduceSum", [p + "_smallf"], [p + "_cnt"], keepdims=0),
        ]
        after[t] = nodes
        for o in (p + "_max", p + "_min", p + "_cnt"):
            g.output.append(helper.make_tensor_value_info(o, TensorProto.FLOAT, []))
        out[t] = (p + "_max", p + "_min", p + "_cnt")
    produced = {o for n in g.node for o in n.output}
    new = [n for t in tensors if t not in produced for n in after[t]]  # graph inputs: stats first
    for n in g.node:
        new.append(n)
        for o in n.output:
            if o in after:
                new.extend(after[o])
    del g.node[:]
    g.node.extend(new)
    return out


def weight_stats(proto, tmap, ckpt) -> dict[str, tuple[float, float]]:
    inline = {i.name: onnx.numpy_helper.to_array(i) for i in proto.graph.initializer
              if i.name not in tmap["initializers"] and i.data_type == TensorProto.FLOAT}
    st = {}
    for name in set(i.name for i in proto.graph.initializer):
        if name in tmap["initializers"]:
            a = ckpt[tmap["initializers"][name]]
        elif name in inline:
            a = inline[name]
        else:
            continue
        a = np.abs(a.astype(np.float32))
        nz = a[a > 0]
        st[name] = (float(a.max()) if a.size else 0.0, float(nz.min()) if nz.size else 0.0)
    # Linear weights reach the MatMul through Transpose/Cast of the initializer (no constant folding at
    # export): value-preserving ops carry the weight's statistics to their output.
    for n in proto.graph.node:
        if n.op_type in ("Transpose", "Cast", "Identity", "Reshape", "Unsqueeze", "Squeeze") and n.input[0] in st:
            if n.op_type == "Cast" and n.attribute[0].i != TensorProto.FLOAT:
                continue
            st[n.output[0]] = st[n.input[0]]
    return st


def run_model(name, args) -> dict:
    files = C.tiny_files() if name == "tiny" else C.model_files(name)
    tmap = json.loads(Path(files["tmap"]).read_text())
    ckpt = C.load_ckpt(files["weights"])
    proto = onnx.load(str(files["graph"]), load_external_data=False)
    wstats = weight_stats(proto, tmap, ckpt)
    mms = [n for n in proto.graph.node if n.op_type == "MatMul"]
    tensors = sorted({t for n in mms for t in (*n.input, n.output[0]) if t not in wstats})
    names = add_stats(proto, tensors)
    sess = C.make_session(proto, tmap, ckpt, args.threads)
    outs = [o for trio in names.values() for o in trio]

    if name == "tiny":
        scen = C.random_scenarios()
    else:
        tok = Tokenizer.from_file(str(files["tokenizer"]))
        sp = C.specials(tok)
        tickets = C.read_tickets(args.tickets)
        scen = C.scenarios(tok, sp, tickets, tuple(args.ladder))
        if args.long and name == "julia-1":
            row = C.collate(tok, sp, C.long_text(tickets, args.long * 6), *C.QUESTIONS[1], args.long)
            scen.append((f"B1_T{args.long}_single_full", C.pad_batch([row], args.long, sp[3])))

    mx, mn, cnt = {}, {}, {}
    for sname, feed in scen:
        vals = sess.run(outs, feed)
        for t, (a, b, c) in names.items():
            va, vb, vc = (float(np.asarray(v)) for v in vals[outs.index(a):outs.index(a) + 3])
            mx[t] = max(mx.get(t, 0.0), va)
            mn[t] = min(mn.get(t, BIG), vb) if vb < BIG else mn.get(t, BIG)
            cnt[t] = cnt.get(t, 0.0) + vc
        print(f"  {name}: {sname} done", file=sys.stderr, flush=True)

    per = []
    for n in mms:
        ins = [t for t in n.input if t not in wstats]
        wins = [t for t in n.input if t in wstats]
        ent = {"node": n.name,
               "in_max": max([mx[t] for t in ins] + [0.0]),
               "in_min_nonzero": min([mn.get(t, BIG) for t in ins] + [BIG]),
               "out_max": mx[n.output[0]], "out_min_nonzero": mn.get(n.output[0], BIG),
               "w_max": max([wstats[t][0] for t in wins] + [0.0]),
               "w_min_nonzero": min([wstats[t][1] or BIG for t in wins] + [BIG]),
               "n_below_fp16_min": sum(cnt[t] for t in ins) + cnt[n.output[0]]}
        ent["act_max"] = max(ent["in_max"], ent["out_max"])
        per.append(ent)
    per.sort(key=lambda e: -e["act_max"])
    act_max = per[0]["act_max"]
    act_min = min(min(e["in_min_nonzero"], e["out_min_nonzero"]) for e in per)
    res = {"model": name, "scenarios": [s for s, _ in scen], "matmuls": len(mms),
           "global_act_max": act_max, "global_act_min_nonzero": act_min,
           "global_weight_max": max(e["w_max"] for e in per),
           "global_weight_min_nonzero": min(e["w_min_nonzero"] for e in per),
           "matmuls_over_3e4": sum(e["act_max"] > FP16_HEADROOM for e in per),
           "matmuls_with_values_below_fp16_min": sum(e["n_below_fp16_min"] > 0 for e in per),
           "values_below_fp16_min_total": sum(e["n_below_fp16_min"] for e in per),
           "worst10": per[:10],
           "fp16_feasible": act_max < FP16_HEADROOM}
    print(json.dumps({k: v for k, v in res.items() if k != "worst10"}), flush=True)
    for e in per[:10]:
        print("   ", json.dumps(e), flush=True)
    res["per_matmul"] = per
    return res


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", nargs="+", default=list(C.MODELS) + ["tiny"])
    ap.add_argument("--tickets", type=Path)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--ladder", nargs="+", type=int, default=[128, 256, 1024])
    ap.add_argument("--long", type=int, default=0, help="julia-1: also one single row of this many tokens")
    ap.add_argument("--json", type=Path, help="write the full per-MatMul results here")
    a = ap.parse_args(argv)
    results = [run_model(m, a) for m in a.models]
    if a.json:
        a.json.write_text(json.dumps(results, indent=1))
    verdict = ", ".join(f"{r['model']}: {'fp16 feasible' if r['fp16_feasible'] else 'bf16 needed'} "
                        f"(max |x| {r['global_act_max']:.4g})" for r in results)
    print(f"G2 {'PASS' if all(r['fp16_feasible'] for r in results) else 'FINDING'}: {verdict}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
