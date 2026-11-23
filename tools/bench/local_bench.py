#!/usr/bin/env python3
"""Throughput and memory of the in-process models, one fresh DuckDB process per cell.

For each (model, text length, batch_tokens) cell it scores ROWS synthetic tickets of roughly that many tokens
with two questions (a refund probability and a 4-way routing choice) and reports wall time, CPU time, peak RSS
and rows per second. Model load is included, as a user sees it on a first query.

  python3 tools/bench/local_bench.py --models julia-1,laya-multilingual,laya-typed-decisions --tokens 128,512,2048

Needs a built extension and the models downloaded (CALL decide_download(...)).
"""
import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SENTENCES = [
    "I was charged twice for my March invoice and would like the extra amount refunded.",
    "The export dialog crashes on version 3.2 whenever the report contains more than one sheet.",
    "Our team moved to a new office and the delivery address on the open order is now wrong.",
    "Thanks for the quick help yesterday, the dashboard works again after the update.",
    "The login page returns error 500 on Safari although it works in Firefox.",
]
QUESTIONS = ('[{"id":"refund","kind":"binary","instruction":"A refund is requested."},'
             '{"id":"team","kind":"choice","instruction":"Which team owns this?","options":["billing","orders","account","other"]}]')


def text_of(tokens, i):
    # about 18 tokens per sentence. Every row is distinct (identical rows are scored once per chunk, which would
    # measure the cache, not the model): the ticket number and the rotation differ per row.
    n = max(1, tokens // 18)
    return f"Ticket {1000 + i}: " + " ".join(SENTENCES[(i + k) % len(SENTENCES)] for k in range(n))


WRAPPER = ("import resource, subprocess, sys\n"
           "r = subprocess.run(sys.argv[1:], stdout=subprocess.PIPE, text=True)\n"
           "print(r.stdout.strip().splitlines()[-1] if r.stdout.strip() else '', file=sys.stderr)\n"
           "u = resource.getrusage(resource.RUSAGE_CHILDREN)\n"
           "print(f'@@usage {u.ru_maxrss} {u.ru_utime + u.ru_stime}', file=sys.stderr)\n"
           "sys.exit(r.returncode)\n")


def cell(duckdb, ext, model, tokens, rows, batch, extra):
    texts = ",".join("('" + text_of(tokens, i).replace("'", "''") + "')" for i in range(rows))
    query = (f"SELECT count(*) FROM (SELECT unnest(decide_answers(text, '{QUESTIONS}', model := '{model}'), recursive := true) FROM t);\n"
             f"SELECT 'engine=' || current_setting('anofox_decide_local_engine') || ' tokens=' || round(avg(decide_token_count(text, '{model}'))) FROM t;")
    sql = (f"LOAD '{ext}';\n{extra}\nSET anofox_decide_batch_tokens = {batch};\n"
           f"CREATE TABLE t AS SELECT * FROM (VALUES {texts}) v(text);\n"
           + query + "\n")
    t0 = time.time()
    p = subprocess.run([sys.executable, "-I", "-c", WRAPPER, duckdb, "-unsigned", "-list", "-noheader"], input=sql,
                       capture_output=True, text=True, env={**os.environ, "DATAZOO_DISABLE_TELEMETRY": "1"})
    wall = time.time() - t0
    usage = [l.split() for l in p.stderr.splitlines() if l.startswith("@@usage ")]
    if p.returncode != 0 or not usage:
        return f"{model}\t{tokens}\t{batch}\tFAILED {p.stderr.strip()[:200]}"
    # ru_maxrss is KiB on Linux, bytes on macOS
    rss = int(usage[-1][1]) / (1024 ** 3 if sys.platform == "darwin" else 1024 ** 2)
    info = [l for l in p.stderr.splitlines() if l.startswith("engine=")]
    engine, tok = (info[-1].split()[0][7:], info[-1].split()[1][7:]) if info else ("?", "?")
    cpu = float(usage[-1][2])
    return f"{model}\t{engine}\t{tok}\t{batch}\t{rows}\t{wall:.1f}\t{cpu:.0f}\t{rss:.2f}\t{rows / wall:.2f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="julia-1,laya-multilingual,laya-typed-decisions")
    ap.add_argument("--tokens", default="128,512,2048")
    ap.add_argument("--rows", type=int, default=64)
    ap.add_argument("--batch-tokens", default="0")
    ap.add_argument("--set", default="", help="extra SET statements, e.g. \"SET anofox_decide_local_engine='ggml';\"")
    ap.add_argument("--duckdb", default=str(ROOT / "build/release/duckdb"))
    ap.add_argument("--extension", default=str(ROOT / "build/release/extension/anofox_decide/anofox_decide.duckdb_extension"))
    a = ap.parse_args()
    print("model\tengine\ttokens\tbatch_tokens\trows\twall_s\tcpu_s\tpeak_rss_gb\trows_per_s")
    for m in a.models.split(","):
        for t in map(int, a.tokens.split(",")):
            for b in map(int, a.batch_tokens.split(",")):
                print(cell(a.duckdb, a.extension, m, t, a.rows, b, a.set), flush=True)


if __name__ == "__main__":
    main()
