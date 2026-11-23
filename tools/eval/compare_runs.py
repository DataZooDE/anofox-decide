#!/usr/bin/env python3
"""Compare two run_eval.py result directories ticket by ticket: label flips and probability differences.

The parity check for an engine switch (docs/GGML_DECIDE.md): the same tickets, the same models, two engines. A flip is a ticket whose
refund decision (p >= 0.5) or routing choice differs; the probabilities are compared too, and the report.json metrics side by side.

  python3 tools/eval/compare_runs.py <reference dir> <candidate dir> [--models julia-1,laya-multilingual,laya-typed-decisions]

Exits 1 when any flip is found or a metric differs at 3 decimals.
"""
import argparse
import csv
import json
import sys
from pathlib import Path


def load(path):
    rows = {}
    with open(path, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rows[(int(r["id"]), r["qid"])] = r
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("reference")
    ap.add_argument("candidate")
    ap.add_argument("--models", default="julia-1,laya-multilingual,laya-typed-decisions")
    a = ap.parse_args()
    ref_dir, cand_dir = Path(a.reference), Path(a.candidate)
    ref_report = json.loads((ref_dir / "report.json").read_text())
    cand_report = json.loads((cand_dir / "report.json").read_text())
    bad = False
    for m in a.models.split(","):
        ref, cand = load(ref_dir / f"results_{m}.csv"), load(cand_dir / f"results_{m}.csv")
        if ref.keys() != cand.keys():
            print(f"{m}: different tickets or questions ({len(ref)} vs {len(cand)} rows)")
            bad = True
            continue
        flips, max_diff = [], 0.0
        for k, r in ref.items():
            c = cand[k]
            pr, pc = float(r["probability"]), float(c["probability"])
            max_diff = max(max_diff, abs(pr - pc))
            if k[1] == "refund" and (pr >= 0.5) != (pc >= 0.5):
                flips.append(f"{k[0]}/refund {pr:.4f}->{pc:.4f}")
            if k[1] == "routing" and r["choice"] != c["choice"]:
                flips.append(f"{k[0]}/routing {r['choice']}->{c['choice']}")
        metrics = []
        rm, cm = ref_report.get(m, {}), cand_report.get(m, {})
        for name in sorted(k for k in set(rm) | set(cm) if k.startswith(("refund_", "routing_"))):
            rv, cv = rm.get(name), cm.get(name)
            if isinstance(rv, dict) and isinstance(cv, dict):  # per-class routing recalls
                pairs = [(f"{name}.{c}", rv.get(c), cv.get(c)) for c in sorted(set(rv) | set(cv))]
            elif isinstance(rv, list) and isinstance(cv, list):  # confidence intervals
                pairs = [(f"{name}[{i}]", x, y) for i, (x, y) in enumerate(zip(rv, cv))]
            else:
                pairs = [(name, rv, cv)]
            for label, x, y in pairs:
                if not (isinstance(x, (int, float)) and isinstance(y, (int, float))) or round(x, 3) != round(y, 3):
                    metrics.append(f"{label} {x} -> {y}")
        res_r, res_c = ref_report.get(m, {}).get("resources", {}), cand_report.get(m, {}).get("resources", {})
        print(f"{m}: {len(ref)} answers, {len(flips)} flips, max |dp| {max_diff:.2e}, metric changes {len(metrics)}; "
              f"wall {res_r.get('wall_s', 0):.0f}s -> {res_c.get('wall_s', 0):.0f}s, "
              f"peak RSS {res_r.get('peak_rss_gb') or 0:.2f} -> {res_c.get('peak_rss_gb') or 0:.2f} GB, "
              f"engine {res_r.get('local_engine')} -> {res_c.get('local_engine')}")
        for line in flips + metrics:
            print("   ", line)
        bad = bad or bool(flips) or bool(metrics)
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
