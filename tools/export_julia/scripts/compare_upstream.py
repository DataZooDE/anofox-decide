"""Compare the DuckDB local path with upstream Julia-1 Python (TransformerEngine).

Needs: JULIA_WEIGHTS_DIR (snapshot), JULIA_UPSTREAM_DIR (dir containing the
upstream `julia/` package, e.g. a Hugging Face snapshot), and a CSV of DuckDB
output (id,p,c) from:
  decide_probability(text,'A refund is requested.',model:='julia') AS p,
  decide_choice(text,'Which team owns this?',['billing','defect','other'],model:='julia') AS c
over test/fixtures/support_tickets.csv (pass its path as argv[1]).
Run from the repo root with the export_julia venv.
"""
import csv
import os
import sys

sys.path.insert(0, os.environ["JULIA_UPSTREAM_DIR"])
from julia.inference import TransformerEngine  # noqa: E402

eng = TransformerEngine(os.environ["JULIA_WEIGHTS_DIR"], device="cpu", max_length=8192, head_length=512)
tickets = list(csv.DictReader(open("test/fixtures/support_tickets.csv")))
duck = {int(r["id"]): r for r in csv.DictReader(open(sys.argv[1]))}
worst, agree = 0.0, 0
for t in tickets:
    r = eng.predict(state=t["text"], questions={
        "refund": {"type": "noul", "instructions": "A refund is requested."},
        "team": {"type": "choice", "instructions": "Which team owns this?",
                 "criteria": {"billing": "billing", "defect": "defect", "other": "other"}}})["answers"]
    d = duck[int(t["id"])]
    worst = max(worst, abs(r["refund"]["noul"] - float(d["p"])))
    agree += r["team"]["choice"] == d["c"]
    print(t["id"], f"upstream={r['refund']['noul']:.6f} duckdb={float(d['p']):.6f} team {r['team']['choice']}/{d['c']}")
print(f"max |dP|={worst:.2e} choice agreement {agree}/{len(tickets)}")
sys.exit(0 if worst < 1e-3 and agree == len(tickets) else 1)
