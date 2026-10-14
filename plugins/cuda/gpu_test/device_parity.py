#!/usr/bin/env python3
"""Real-weights CPU-versus-CUDA parity through the extension, on a machine with an NVIDIA GPU.

Runs the SAME queries in two DuckDB sessions, `SET anofox_decide_device = 'cpu'` and `= 'cuda'` (an explicit
device never falls back: if CUDA cannot serve, the query fails and so does this check), and compares:

  1. the two recorded reference probabilities of Laya multilingual, within 2e-4 of 0.98209638 and 0.0012388999
     ('I want my money back.' / 'Thanks, everything works now!', question 'A refund is requested.');
  2. over the 200-ticket evaluation (two questions per ticket, as tools/eval/run_eval.py asks them), CPU versus
     CUDA: the REFUND probability of every ticket, maximum absolute difference <= 1e-4, and the refund decision
     (at 0.5) and the routing CHOICE (the argmax of the four options; the routing probabilities themselves are
     not compared) identical on every ticket;
  3. accuracy against the labels, identical between the devices;
  4. that the CUDA session really was served by cuda: decide_models().device says so after scoring.

It prints the numbers it checked (max probability delta, accuracies, wall time per device) so they can be
recorded, and ends with PARITY_RESULT: PASSED or FAILED. The recorded values only apply to laya-multilingual;
for another model only the CPU-versus-CUDA comparisons run.

usage: device_parity.py --duckdb <duckdb> --extension <anofox_decide.duckdb_extension> --tickets <tickets.csv>
                        [--model laya-multilingual] [--plugin-dir <dir with libanofox_decide_cuda_plugin.so>]
"""
import argparse
import csv
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

QUESTIONS = json.dumps([
    {"id": "refund", "kind": "binary", "instruction": "A refund is requested."},
    {"id": "routing", "kind": "choice", "instruction": "Which team should handle this request?",
     "options": ["billing", "orders", "account", "other"]},
])
RECORDED = {"laya-multilingual": (("I want my money back.", 0.98209638), ("Thanks, everything works now!", 0.0012388999))}
REFUND_Q = "A refund is requested."
PROB_TOL = 1e-4
RECORDED_TOL = 2e-4


def esc(s):
    return str(s).replace("'", "''")


def run_session(args, device, workdir, scored_pairs):
    """One DuckDB session on `device`; returns (probe rows, ticket results csv path, seconds, serving device)."""
    probe_csv = workdir / f"probe_{device}.csv"
    result_csv = workdir / f"results_{device}.csv"
    device_csv = workdir / f"serving_{device}.csv"
    probes = " UNION ALL ".join(
        f"SELECT '{esc(t)}' AS text, decide_probability('{esc(t)}', '{REFUND_Q}', model := '{args.model}')::DOUBLE AS p"
        for t, _ in scored_pairs)
    plugin = f"SET anofox_decide_plugin_dir = '{esc(args.plugin_dir)}';\n" if args.plugin_dir else ""
    sql = f"""
LOAD '{esc(args.extension)}';
{plugin}SET anofox_decide_device = '{device}';
COPY ({probes}) TO '{esc(probe_csv)}' (HEADER);
CREATE TABLE t AS SELECT * FROM read_csv('{esc(args.tickets)}', all_varchar=true);
COPY (
  SELECT id::INTEGER AS id, question_id AS qid, probability::DOUBLE AS probability, choice
  FROM (SELECT id, unnest(decide_answers(text, '{esc(QUESTIONS)}', model := '{args.model}'), recursive := true) FROM t)
  ORDER BY 1, 2
) TO '{esc(result_csv)}' (HEADER);
COPY (SELECT device FROM decide_models() WHERE model = '{args.model}') TO '{esc(device_csv)}' (HEADER);
"""
    env = dict(os.environ, DATAZOO_DISABLE_TELEMETRY="1")
    t0 = time.time()
    p = subprocess.run([args.duckdb, "-unsigned", "-c", sql], capture_output=True, text=True, env=env)
    seconds = time.time() - t0
    if p.returncode != 0:
        print(f"FAIL: the {device} session failed:\n{p.stderr[-1500:]}")
        sys.exit(1)
    probe = [(r["text"], float(r["p"])) for r in csv.DictReader(open(probe_csv))]
    serving = [r["device"] for r in csv.DictReader(open(device_csv))]
    return probe, result_csv, seconds, (serving[0] if serving else "")


def load_results(path):
    refund, routing = {}, {}
    for r in csv.DictReader(open(path)):
        if r["qid"] == "refund":
            refund[int(r["id"])] = float(r["probability"])
        else:
            routing[int(r["id"])] = r["choice"]
    return refund, routing


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--duckdb", required=True)
    ap.add_argument("--extension", required=True)
    ap.add_argument("--tickets", required=True, help="tickets.csv from tools/eval/make_sample.py")
    ap.add_argument("--model", default="laya-multilingual")
    ap.add_argument("--plugin-dir", default="")
    args = ap.parse_args()

    ok = True

    def check(cond, text):
        nonlocal ok
        print(("PASS  " if cond else "FAIL  ") + text)
        ok = ok and cond

    pairs = list(RECORDED.get(args.model, ())) or [("I want my money back.", None), ("Thanks, everything works now!", None)]
    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp)
        # The weights are fetched once (cached under the extension's cache directory) before either session.
        env = dict(os.environ, DATAZOO_DISABLE_TELEMETRY="1")
        subprocess.run([args.duckdb, "-unsigned", "-c",
                        f"LOAD '{esc(args.extension)}'; SET anofox_decide_allow_remote = true; "
                        f"CALL decide_download('{args.model}');"], check=True, env=env)
        cpu_probe, cpu_csv, cpu_s, cpu_dev = run_session(args, "cpu", work, pairs)
        gpu_probe, gpu_csv, gpu_s, gpu_dev = run_session(args, "cuda", work, pairs)
        # The result files live in the temporary directory: read them before it is removed.
        cpu_refund, cpu_route = load_results(cpu_csv)
        gpu_refund, gpu_route = load_results(gpu_csv)

    check(cpu_dev == "cpu", f"the cpu session was served by '{cpu_dev}'")
    check(gpu_dev.startswith("cuda"), f"the cuda session was served by '{gpu_dev}' (decide_models().device)")

    print("\nprobabilities of the two reference sentences")
    for (text, recorded), (_, cpu_p), (_, gpu_p) in zip(pairs, cpu_probe, gpu_probe):
        print(f"  {text!r}: cpu {cpu_p:.10g}  cuda {gpu_p:.10g}  |delta| {abs(cpu_p - gpu_p):.3g}")
        check(abs(cpu_p - gpu_p) <= PROB_TOL, f"cpu vs cuda within {PROB_TOL:g} for {text!r}")
        if recorded is not None:
            check(abs(gpu_p - recorded) <= RECORDED_TOL, f"cuda within {RECORDED_TOL:g} of the recorded {recorded} for {text!r}")

    ids = sorted(cpu_refund)
    check(len(ids) > 0 and set(ids) == set(gpu_refund) == set(cpu_route) == set(gpu_route),
          f"both devices answered every ticket ({len(ids)})")
    delta = max(abs(cpu_refund[i] - gpu_refund[i]) for i in ids)
    check(delta <= PROB_TOL, f"refund probabilities: max |cpu - cuda| = {delta:.3g} (limit {PROB_TOL:g})")
    flips_refund = [i for i in ids if (cpu_refund[i] >= 0.5) != (gpu_refund[i] >= 0.5)]
    flips_route = [i for i in ids if cpu_route[i] != gpu_route[i]]
    check(not flips_refund, f"refund decisions identical on every ticket (flips: {flips_refund[:10]})")
    check(not flips_route, f"routing argmax identical on every ticket (flips: {flips_route[:10]})")

    truth = {int(r["id"]): r for r in csv.DictReader(open(args.tickets, encoding="utf-8"))}

    def accuracy(refund, route):
        r = sum((refund[i] >= 0.5) == (truth[i]["refund"] == "true") for i in ids) / len(ids)
        t = sum(route[i] == truth[i]["routing"] for i in ids) / len(ids)
        return r, t

    cpu_acc, gpu_acc = accuracy(cpu_refund, cpu_route), accuracy(gpu_refund, gpu_route)
    print(f"\naccuracy  cpu: refund {100 * cpu_acc[0]:.1f}% routing {100 * cpu_acc[1]:.1f}%   "
          f"cuda: refund {100 * gpu_acc[0]:.1f}% routing {100 * gpu_acc[1]:.1f}%")
    check(cpu_acc == gpu_acc, "accuracy identical on both devices")
    print(f"wall time for the {len(ids)}-ticket evaluation incl. session start and model load: "
          f"cpu {cpu_s:.1f}s, cuda {gpu_s:.1f}s")
    print("PARITY_RESULT: " + ("PASSED" if ok else "FAILED"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
