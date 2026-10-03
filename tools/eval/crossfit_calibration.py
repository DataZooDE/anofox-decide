#!/usr/bin/env python3
"""Does Platt calibration help? Honest estimate by repeated 2-fold cross-fitting.

For every model's refund probabilities (results_<model>.csv written by run_eval.py) the tickets are
split in two stratified halves; Platt scaling is fitted on one half and applied to the other, and
both halves are scored this way, so no ticket is ever judged by a calibration fitted on itself.
Averaged over --repeats random splits. Same maths as decide_fit_calibration / the registration
option MAP {'calibration': 'platt:a,b'} (a pure-Python re-implementation, so it also cross-checks it).

  python3 tools/eval/crossfit_calibration.py --tickets tickets.csv --results out_dir [--models d1,jev]
"""
import argparse
import csv
import math
import random
from pathlib import Path

NAMES = {"d1": "Liquid D1", "jev": "TypeSafe Jev", "laya-ml": "Laya multilingual",
         "laya-td": "Laya typed-decisions", "kev": "Kev-0.8B", "strands": "strands-decider", "julia": "Julia-1"}


def logit(p):
    p = min(max(p, 1e-6), 1 - 1e-6)
    return math.log(p / (1 - p))


def sig(z):
    return 1 / (1 + math.exp(-z)) if z >= 0 else math.exp(z) / (1 + math.exp(z))


def fit(P, Y):
    """Platt's smoothed targets + damped Newton; returns (a, b) or None when not fittable."""
    pos = sum(Y)
    neg = len(Y) - pos
    f = [logit(p) for p in P]
    if pos == 0 or neg == 0 or max(f) - min(f) < 1e-9:
        return None
    hi, lo = (pos + 1) / (pos + 2), 1 / (neg + 2)
    t = [hi if y else lo for y in Y]

    def loss(A, B):
        s = 0.0
        for fi, ti in zip(f, t):
            z = A * fi + B
            s += (1 - ti) * z + math.log1p(math.exp(-z)) if z >= 0 else -ti * z + math.log1p(math.exp(z))
        return s

    A, B = 1.0, 0.0
    fv = loss(A, B)
    for _ in range(200):
        g1 = g2 = 0.0
        h11 = h22 = 1e-12
        h12 = 0.0
        for fi, ti in zip(f, t):
            q = sig(A * fi + B)
            d, w = q - ti, q * (1 - q)
            g1 += d * fi
            g2 += d
            h11 += w * fi * fi
            h12 += w * fi
            h22 += w
        if abs(g1) < 1e-9 and abs(g2) < 1e-9:
            break
        det = h11 * h22 - h12 * h12
        dA, dB = -(h22 * g1 - h12 * g2) / det, -(-h12 * g1 + h11 * g2) / det
        gd = g1 * dA + g2 * dB
        step, moved = 1.0, False
        while step >= 1e-12:
            nf = loss(A + step * dA, B + step * dB)
            if nf < fv + 1e-4 * step * gd:
                A, B, fv, moved = A + step * dA, B + step * dB, nf, True
                break
            step /= 2
        if not moved:
            break
    return (A, B) if A > 0 else None


def ece(P, Y, bins=10):
    n, tot = len(P), 0.0
    for b in range(bins):
        lo, hi = b / bins, (b + 1) / bins
        idx = [i for i, p in enumerate(P) if (p > lo or b == 0) and p <= hi]
        if idx:
            tot += len(idx) / n * abs(sum(P[i] for i in idx) / len(idx) - sum(Y[i] for i in idx) / len(idx))
    return tot


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tickets", required=True)
    ap.add_argument("--results", required=True, help="directory with results_<model>.csv")
    ap.add_argument("--models", default="d1,jev,laya-ml,laya-td,kev")
    ap.add_argument("--repeats", type=int, default=20)
    a = ap.parse_args()
    truth = {int(r["id"]): r["refund"] == "true" for r in csv.DictReader(open(a.tickets, encoding="utf-8"))}
    ids = sorted(truth)
    print("| Model | Accuracy @0.5 raw -> calibrated | Brier raw -> calibrated | ECE raw -> calibrated |")
    print("|---|---|---|---|")
    for m in [x for x in a.models.split(",") if x]:
        f = Path(a.results) / f"results_{m}.csv"
        if not f.exists():
            continue
        prob = {int(r["id"]): float(r["probability"]) for r in csv.DictReader(open(f)) if r["qid"] == "refund"}
        runs = []
        for seed in range(a.repeats):
            rng = random.Random(seed)
            pos = [i for i in ids if truth[i]]
            neg = [i for i in ids if not truth[i]]
            rng.shuffle(pos)
            rng.shuffle(neg)
            halves = [pos[:len(pos) // 2] + neg[:len(neg) // 2], pos[len(pos) // 2:] + neg[len(neg) // 2:]]
            raw, cal, y = [], [], []
            for k in (0, 1):
                fitted = fit([prob[i] for i in halves[1 - k]], [truth[i] for i in halves[1 - k]])
                if fitted is None:
                    break
                for i in halves[k]:
                    raw.append(prob[i])
                    y.append(1 if truth[i] else 0)
                    cal.append(sig(fitted[0] * logit(prob[i]) + fitted[1]))
            else:
                acc = lambda Q: sum((q >= 0.5) == bool(t) for q, t in zip(Q, y)) / len(y)
                brier = lambda Q: sum((q - t) ** 2 for q, t in zip(Q, y)) / len(y)
                runs.append((acc(raw), acc(cal), brier(raw), brier(cal), ece(raw, y), ece(cal, y)))
        if not runs:
            print(f"| {NAMES.get(m, m)} | not fittable (no positive slope or one outcome only) | | |")
            continue
        r = [sum(x[i] for x in runs) / len(runs) for i in range(6)]
        print(f"| {NAMES.get(m, m)} | {100 * r[0]:.1f}% -> {100 * r[1]:.1f}% | {r[2]:.3f} -> {r[3]:.3f} | {r[4]:.3f} -> {r[5]:.3f} |")


if __name__ == "__main__":
    main()
