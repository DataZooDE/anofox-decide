#!/usr/bin/env python3
"""Gate G0: the half-ready graph passes are CPU-identical.

Builds throwaway patched copies (masks only, rotary only, both) of the committed weight-free graphs, runs
original and patched on ORT CPU with the same injected weights and thread settings over padded batches
(different row lengths, a row padded > 64 tokens beyond its last token so a fully masked local-window row
exists, single-row batches, M = 20 markers) and requires max |delta| of `scores` on the VALID markers == 0.0.

  gate_g0_identity.py [--models laya-multilingual laya-typed-decisions julia-1 tiny] [--tickets tickets.csv]
                      [--threads 8] [--ladder 128 256 1024]

Needs the cached real weights (~/.cache/anofox-decide/<model>/<rev>/); `tiny` runs offline from test/fixtures.
Exit 0 and a final `G0 PASS` line only if every delta is exactly 0.0.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
import onnx  # noqa: E402
from tokenizers import Tokenizer  # noqa: E402

import _gate_common as C  # noqa: E402
import half_ready as H  # noqa: E402

VARIANTS = {"masks": dict(masks=True, rotary=False), "rotary": dict(masks=False, rotary=True),
            "both": dict(masks=True, rotary=True)}


# exact patch counts of the committed graphs (2 encoder finfo.min + 2 head -inf masks; rotary tables)
EXPECT = {"laya-multilingual": {"masks": 4, "rotary": 1}, "laya-typed-decisions": {"masks": 4, "rotary": 2},
          "julia-1": {"masks": 4, "rotary": 1}, "tiny": {"masks": 2, "rotary": 1}}


def run_model(name, args) -> list[dict]:
    files = C.tiny_files() if name == "tiny" else C.model_files(name)
    tmap = json.loads(Path(files["tmap"]).read_text())
    ckpt = C.load_ckpt(files["weights"])
    if name == "tiny":
        scen = C.random_scenarios()
    else:
        tok = Tokenizer.from_file(str(files["tokenizer"]))
        scen = C.scenarios(tok, C.specials(tok), C.read_tickets(args.tickets), tuple(args.ladder))
    base = C.make_session(files["graph"], tmap, ckpt, args.threads)
    ref = {n: base.run(["scores"], f)[0] for n, f in scen}
    rows = []
    for vname, kw in VARIANTS.items():
        proto = onnx.load(str(files["graph"]), load_external_data=False)
        counts = H.half_ready(proto, **kw)
        want = {k: v for k, v in EXPECT[name].items() if kw[k]}
        assert {k: counts[k] for k in want} == want, f"{name}/{vname}: patched {counts}, expected {want}"
        sess = C.make_session(proto, tmap, ckpt, args.threads)
        worst_valid = worst_pad = 0.0
        bad = []
        for n, f in scen:
            got = sess.run(["scores"], f)[0]
            mm = f["marker_mask"]
            dv = float(np.abs(got - ref[n])[mm].max())
            dp = float(np.abs(got - ref[n])[~mm].max()) if (~mm).any() else 0.0
            worst_valid, worst_pad = max(worst_valid, dv), max(worst_pad, dp)
            if dv != 0.0:
                bad.append((n, dv))
        rows.append({"model": name, "variant": vname, "patched": counts, "scenarios": len(scen),
                     "max_abs_delta_valid": worst_valid, "max_abs_delta_masked_slots": worst_pad,
                     "ok": worst_valid == 0.0, "failing": bad[:5]})
        print(json.dumps(rows[-1]), flush=True)
    return rows


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", nargs="+", default=list(C.MODELS) + ["tiny"])
    ap.add_argument("--tickets", type=Path)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--ladder", nargs="+", type=int, default=[128, 256, 1024])
    a = ap.parse_args(argv)
    allrows = [r for m in a.models for r in run_model(m, a)]
    ok = all(r["ok"] for r in allrows)
    print("G0 PASS: max |delta| on valid markers == 0.0 for every model and variant" if ok else
          "G0 FAIL: a patched graph differs from the original on valid markers")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
