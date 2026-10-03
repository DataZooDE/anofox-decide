#!/usr/bin/env python3
"""Run the labelled ticket evaluation through anofox_decide and report metrics.

For every model one DuckDB session scores all tickets with ONE query over decide_answers
(two questions per ticket, one provider request per distinct ticket for hosted models; the requests
run concurrently, see --concurrency):
  refund   binary  "A refund is requested."
  routing  choice  billing | orders | account | other
Results are written to <out>/results_<model>.csv; metrics to <out>/report.md/json.

Models (all optional, pick with --models):
  d1, jev, clef, clef-flash    hosted APIs (LIQUID_API_KEY / TYPESAFE_API_KEY / CLOUDFLARE_API_TOKEN + CLOUDFLARE_ACCOUNT_ID;
                               ticket text leaves the machine)
  laya-ml, laya-td, julia      local ONNX graphs (env: LAYA_ML_DIR/LAYA_ML_ONNX, LAYA_TD_DIR/LAYA_TD_ONNX,
                               JULIA_WEIGHTS_DIR/JULIA_ONNX)
  kev                          a running Kev server (--kev-url, default http://127.0.0.1:8009)
  strands                      a running `strands-decider serve` (--strands-url, default http://127.0.0.1:8000)

--concurrency N sets anofox_decide_max_concurrency (0 = automatic: 8 for hosted providers, 1 for local).
Needs a built extension (default build/release). Telemetry is always disabled here.
"""
import argparse
import csv
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
QUESTIONS = json.dumps([
    {"id": "refund", "kind": "binary", "instruction": "A refund is requested."},
    {"id": "routing", "kind": "choice", "instruction": "Which team should handle this request?",
     "options": ["billing", "orders", "account", "other"]},
])
CLASSES = ["billing", "orders", "account", "other"]


def esc(s):
    return s.replace("'", "''")


def models(env, kev_url, strands_url):
    """name -> (display, registration SQL, model id, needs-key env var or None)."""
    def need(*names):
        missing = [n for n in names if not env.get(n)]
        return missing

    return {
        "d1": ("Liquid D1", "SELECT decide_register_model('d1:free', 'liquid');", "d1:free", ["LIQUID_API_KEY"]),
        "clef": ("Cloudflare Clef", "SELECT decide_register_model('clef', 'cloudflare');", "clef",
                 ["CLOUDFLARE_API_TOKEN", "CLOUDFLARE_ACCOUNT_ID"]),
        "clef-flash": ("Cloudflare Clef-flash", "SELECT decide_register_model('clef-flash', 'cloudflare');", "clef-flash",
                       ["CLOUDFLARE_API_TOKEN", "CLOUDFLARE_ACCOUNT_ID"]),
        "jev": ("TypeSafe Jev", "SELECT decide_register_model('jev-latest', 'typesafe');", "jev-latest",
                ["TYPESAFE_API_KEY"]),
        "laya-ml": ("Laya multilingual",
                    f"SELECT decide_register_model('laya-ml','local','{esc(env.get('LAYA_ML_ONNX',''))}',"
                    f"'{esc(env.get('LAYA_ML_DIR',''))}/tokenizer/tokenizer.json','laya');", "laya-ml",
                    ["LAYA_ML_ONNX", "LAYA_ML_DIR"]),
        "laya-td": ("Laya typed-decisions",
                    f"SELECT decide_register_model('laya-td','local','{esc(env.get('LAYA_TD_ONNX',''))}',"
                    f"'{esc(env.get('LAYA_TD_DIR',''))}/tokenizer/tokenizer.json','laya');", "laya-td",
                    ["LAYA_TD_ONNX", "LAYA_TD_DIR"]),
        "julia": ("Julia-1",
                  f"SELECT decide_register_model('julia','local','{esc(env.get('JULIA_ONNX',''))}',"
                  f"'{esc(env.get('JULIA_WEIGHTS_DIR',''))}/tokenizer/tokenizer.json');", "julia",
                  ["JULIA_ONNX", "JULIA_WEIGHTS_DIR"]),
        "kev": ("Kev-0.8B (server)",
                "CREATE SECRET (TYPE anofox_decide, API_KEY 'local', SCOPE '127.0.0.1');"
                f"SELECT decide_register_model('kev-latest','systemone',MAP {{'endpoint':'{kev_url}'}});",
                "kev-latest", []),
        "strands": ("strands-decider 2B (server)",
                    f"SELECT decide_register_model('strands-decider','strands',MAP {{'endpoint':'{strands_url}'}});",
                    "strands-decider", []),
    }


def run_model(name, spec, tickets, out, duckdb, ext, env, concurrency=None):
    display, register, model_id, keys = spec
    missing = [k for k in keys if not env.get(k)]
    if missing:
        print(f"SKIP {name}: missing {', '.join(missing)}", file=sys.stderr)
        return None
    res = out / f"results_{name}.csv"
    setting = f"SET anofox_decide_max_concurrency = {int(concurrency)};\n" if concurrency is not None else ""
    sql = f"""
LOAD '{esc(str(ext))}';
SET anofox_decide_allow_remote = true;
{setting}{register}
CREATE TABLE t AS SELECT * FROM read_csv('{esc(str(tickets))}', all_varchar=true);
COPY (
  SELECT id::INTEGER AS id, question_id AS qid, probability, choice
  FROM (SELECT id, unnest(decide_answers(text, '{esc(QUESTIONS)}', model := '{model_id}'), recursive := true) FROM t)
  ORDER BY 1, 2
) TO '{esc(str(res))}' (HEADER);
"""
    e = dict(env, DATAZOO_DISABLE_TELEMETRY="1")
    t0 = time.time()
    p = subprocess.run([str(duckdb), "-unsigned", "-c", sql], capture_output=True, text=True, env=e)
    dt = time.time() - t0
    if p.returncode != 0 or not res.exists():
        err = "\n".join(l for l in p.stderr.splitlines() if "Schema error" not in l and l.strip())[:600]
        print(f"FAIL {name}: {err}", file=sys.stderr)
        return None
    print(f"ok   {name}: {dt:.0f}s", file=sys.stderr)
    return dt


# --- metrics ---------------------------------------------------------------
def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((c - h) / d, (c + h) / d)


def auroc(scores, labels):
    pos = [s for s, y in zip(scores, labels) if y]
    neg = [s for s, y in zip(scores, labels) if not y]
    if not pos or not neg:
        return float("nan")
    wins = sum((p > n) + 0.5 * (p == n) for p in pos for n in neg)
    return wins / (len(pos) * len(neg))


def ece(probs, labels, bins=10):
    n = len(probs)
    tot = 0.0
    for b in range(bins):
        lo, hi = b / bins, (b + 1) / bins
        idx = [i for i, p in enumerate(probs) if (p > lo or b == 0) and p <= hi]
        if idx:
            conf = sum(probs[i] for i in idx) / len(idx)
            acc = sum(labels[i] for i in idx) / len(idx)
            tot += len(idx) / n * abs(conf - acc)
    return tot


def mcnemar_exact(a_correct, b_correct):
    """Two-sided exact McNemar p on paired correctness vectors."""
    b = sum(1 for x, y in zip(a_correct, b_correct) if x and not y)
    c = sum(1 for x, y in zip(a_correct, b_correct) if y and not x)
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    p = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * p)


def analyse(tickets, out, names, displays):
    truth = {int(r["id"]): r for r in csv.DictReader(open(tickets, encoding="utf-8"))}
    ids = sorted(truth)
    report = {}
    for name in names:
        f = out / f"results_{name}.csv"
        if not f.exists():
            continue
        p, c = {}, {}
        for r in csv.DictReader(open(f)):
            if r["qid"] == "refund":
                p[int(r["id"])] = float(r["probability"])
            else:
                c[int(r["id"])] = r["choice"]
        if set(p) != set(ids) or set(c) != set(ids):
            print(f"WARN {name}: incomplete results ({len(p)}/{len(c)} of {len(ids)})", file=sys.stderr)
            continue
        y = [truth[i]["refund"] == "true" for i in ids]
        probs = [p[i] for i in ids]
        pred = [x >= 0.5 for x in probs]
        ok_ref = [a == b for a, b in zip(pred, y)]
        tp = sum(1 for a, b in zip(pred, y) if a and b)
        fp = sum(1 for a, b in zip(pred, y) if a and not b)
        pos, neg = sum(y), len(y) - sum(y)
        recall = tp / pos if pos else float("nan")
        spec = (neg - fp) / neg if neg else float("nan")
        route_true = [truth[i]["routing"] for i in ids]
        ok_route = [c[i] == truth[i]["routing"] for i in ids]
        recalls = {}
        for cl in CLASSES:
            idx = [j for j, t in enumerate(route_true) if t == cl]
            recalls[cl] = sum(ok_route[j] for j in idx) / len(idx) if idx else float("nan")
        report[name] = {
            "display": displays[name], "n": len(ids),
            "refund_acc": sum(ok_ref) / len(ids), "refund_ci": wilson(sum(ok_ref), len(ids)),
            "refund_bal_acc": (recall + spec) / 2, "refund_recall": recall, "refund_specificity": spec,
            "refund_auroc": auroc(probs, y),
            "refund_brier": sum((a - (1 if b else 0)) ** 2 for a, b in zip(probs, y)) / len(ids),
            "refund_ece": ece(probs, [1 if b else 0 for b in y]),
            "routing_acc": sum(ok_route) / len(ids), "routing_ci": wilson(sum(ok_route), len(ids)),
            "routing_macro": sum(recalls.values()) / len(recalls), "routing_recalls": recalls,
            "_ok_ref": ok_ref, "_ok_route": ok_route,
        }
    return report, truth


def render(report, truth, times):
    ids = sorted(truth)
    pos = sum(1 for i in ids if truth[i]["refund"] == "true")
    routing = {cl: sum(1 for i in ids if truth[i]["routing"] == cl) for cl in CLASSES}
    maj = max(routing.values()) / len(ids)
    L = [f"# Evaluation report ({len(ids)} Bitext tickets)", "",
         f"Refund: {pos} positive / {len(ids) - pos} negative (always-no baseline {100 * (len(ids) - pos) / len(ids):.0f}%). "
         f"Routing mix: " + ", ".join(f"{k} {v}" for k, v in routing.items()) +
         f" (always-{max(routing, key=routing.get)} baseline {100 * maj:.0f}%).", "",
         "## Refund (yes/no, threshold 0.5)", "",
         "| Model | Accuracy [95% CI] | Balanced acc | Recall | Specificity | AUROC | Brier | ECE |",
         "|---|---|---|---|---|---|---|---|"]
    order = sorted(report, key=lambda k: -report[k]["refund_bal_acc"])
    for k in order:
        r = report[k]
        L.append(f"| {r['display']} | {100 * r['refund_acc']:.1f}% [{100 * r['refund_ci'][0]:.0f}-{100 * r['refund_ci'][1]:.0f}] "
                 f"| {100 * r['refund_bal_acc']:.1f}% | {100 * r['refund_recall']:.0f}% | {100 * r['refund_specificity']:.0f}% "
                 f"| {r['refund_auroc']:.3f} | {r['refund_brier']:.3f} | {r['refund_ece']:.3f} |")
    L += ["", "## Routing (billing / orders / account / other)", "",
          "| Model | Accuracy [95% CI] | Macro recall | " + " | ".join(f"{c} recall" for c in CLASSES) + " |",
          "|---|---|---|" + "---|" * len(CLASSES)]
    for k in sorted(report, key=lambda k: -report[k]["routing_macro"]):
        r = report[k]
        L.append(f"| {r['display']} | {100 * r['routing_acc']:.1f}% [{100 * r['routing_ci'][0]:.0f}-{100 * r['routing_ci'][1]:.0f}] "
                 f"| {100 * r['routing_macro']:.1f}% | " + " | ".join(f"{100 * r['routing_recalls'][c]:.0f}%" for c in CLASSES) + " |")
    if len(report) > 1:
        best_ref = order[0]
        best_route = max(report, key=lambda k: report[k]["routing_macro"])
        L += ["", f"## Significance (exact McNemar, paired on the same tickets)", "",
              f"Refund vs {report[best_ref]['display']}, routing vs {report[best_route]['display']}; p < 0.05 means the "
              "difference is unlikely to be chance on this sample.", "",
              "| Model | Refund p | Routing p |", "|---|---|---|"]
        for k in order:
            if k == best_ref and k == best_route:
                continue
            pr = "best" if k == best_ref else f"{mcnemar_exact(report[best_ref]['_ok_ref'], report[k]['_ok_ref']):.3f}"
            pt = "best" if k == best_route else f"{mcnemar_exact(report[best_route]['_ok_route'], report[k]['_ok_route']):.3f}"
            L.append(f"| {report[k]['display']} | {pr} | {pt} |")
    if times:
        L += ["", "## Wall time for the whole sample", "", "| Model | Seconds |", "|---|---|"]
        L += [f"| {report[k]['display']} | {times[k]:.0f} |" for k in order if k in times]
    return "\n".join(L) + "\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tickets", required=True, help="tickets.csv from make_sample.py")
    ap.add_argument("--out", required=True, help="output directory")
    ap.add_argument("--models", default="d1,jev,clef,clef-flash,laya-ml,laya-td,julia,kev,strands")
    ap.add_argument("--duckdb", default=str(ROOT / "build/release/duckdb"))
    ap.add_argument("--extension", default=str(ROOT / "build/release/extension/anofox_decide/anofox_decide.duckdb_extension"))
    ap.add_argument("--kev-url", default="http://127.0.0.1:8009")
    ap.add_argument("--strands-url", default="http://127.0.0.1:8000")
    ap.add_argument("--concurrency", type=int, default=None,
                    help="anofox_decide_max_concurrency for hosted models (0 = automatic, the extension default)")
    ap.add_argument("--analyse-only", action="store_true", help="skip scoring; reuse results_*.csv in --out")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    specs = models(env, a.kev_url, a.strands_url)
    names = [m for m in a.models.split(",") if m]
    unknown = [m for m in names if m not in specs]
    if unknown:
        sys.exit(f"unknown model(s): {unknown}; choose from {sorted(specs)}")
    times = {}
    if not a.analyse_only:
        for n in names:
            t = run_model(n, specs[n], a.tickets, out, a.duckdb, a.extension, env, a.concurrency)
            if t is not None:
                times[n] = t
    report, truth = analyse(a.tickets, out, names, {k: v[0] for k, v in specs.items()})
    if not report:
        sys.exit("no model produced complete results")
    md = render(report, truth, times)
    (out / "report.md").write_text(md)
    clean = {k: {kk: vv for kk, vv in v.items() if not kk.startswith("_")} for k, v in report.items()}
    (out / "report.json").write_text(json.dumps(clean, indent=2))
    print(md)


if __name__ == "__main__":
    main()
