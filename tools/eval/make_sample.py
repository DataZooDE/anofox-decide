#!/usr/bin/env python3
"""Build the labelled evaluation sample from the public Bitext customer-support dataset.

Source: bitext/Bitext-customer-support-llm-chatbot-training-dataset (CDLA-Sharing-1.0),
27 intents in 11 categories, ~1000 utterances per intent. The data is downloaded to
--out and NOT committed; only this script, its seed and the aggregate results are.

Questions evaluated per ticket:
  refund (yes/no)  ground truth: intent == get_refund
  routing (choice) ground truth: category -> {billing, orders, account, other}
                   billing: REFUND, INVOICE, PAYMENT; orders: ORDER, SHIPPING, DELIVERY;
                   account: ACCOUNT, SUBSCRIPTION, CANCEL; other: FEEDBACK, CONTACT

Sample (default 200, seed 20260930, deterministic):
  60 positives: intent get_refund (all route to billing)
  140 negatives, balanced so routing is not dominated by billing: billing 30,
  orders 40, account 40, other 30, each spread evenly over that class's intents
  (billing negatives: check_refund_policy, check_invoice, get_invoice,
  check_payment_methods, payment_issue, i.e. hard ones that mention refunds or
  payments without requesting a refund).
  Resulting routing mix: billing 90, orders 40, account 40, other 30.
track_refund is excluded entirely: "I expect a refund of X" is label-ambiguous.
Placeholders like {{Order Number}} are replaced with realistic values (seeded).
"""
import argparse
import csv
import random
import re
import sys
import urllib.request
from pathlib import Path

URL = ("https://huggingface.co/datasets/bitext/Bitext-customer-support-llm-chatbot-training-dataset/"
       "resolve/main/Bitext_Sample_Customer_Support_Training_Dataset_27K_responses-v11.csv")
SEED = 20260930
ROUTING = {
    "REFUND": "billing", "INVOICE": "billing", "PAYMENT": "billing",
    "ORDER": "orders", "SHIPPING": "orders", "DELIVERY": "orders",
    "ACCOUNT": "account", "SUBSCRIPTION": "account", "CANCEL": "account",
    "FEEDBACK": "other", "CONTACT": "other",
}
FILL = {
    "Order Number": lambda r: f"#{r.randint(10000, 99999)}",
    "Invoice Number": lambda r: f"INV-{r.randint(1000, 9999)}",
    "Account Type": lambda r: r.choice(["premium", "basic", "business", "free"]),
    "Account Category": lambda r: r.choice(["personal", "family", "business", "student"]),
    "Person Name": lambda r: r.choice(["Maria Lopez", "John Smith", "Aisha Khan", "Tom Becker", "Li Wei"]),
    "Refund Amount": lambda r: str(r.choice([12, 25, 49, 80, 120, 249])),
    "Currency Symbol": lambda r: r.choice(["$", "EUR", "GBP"]),
    "Delivery City": lambda r: r.choice(["Berlin", "Lyon", "Austin", "Leeds", "Porto"]),
    "Delivery Country": lambda r: r.choice(["Germany", "France", "the US", "the UK", "Portugal"]),
}


def fill(text, rng):
    return re.sub(r"\{\{([^}]*)\}\}", lambda m: FILL.get(m.group(1), lambda r: m.group(1))(rng), text)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True, help="output directory (created; keep it out of git)")
    ap.add_argument("--positives", type=int, default=60, help="refund tickets (default 60)")
    ap.add_argument("--negatives", type=int, default=140, help="non-refund tickets (default 140)")
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    src = out / "bitext.csv"
    if not src.exists():
        print(f"downloading {URL}", file=sys.stderr)
        urllib.request.urlretrieve(URL, src)
    rows = list(csv.DictReader(open(src, encoding="utf-8")))
    rng = random.Random(args.seed)
    by_intent = {}
    for r in rows:
        by_intent.setdefault(r["intent"], []).append(r)
    chosen = [("pos", r) for r in rng.sample(by_intent["get_refund"], args.positives)]
    # Negative quota per routing class (30/140, 40/140, 40/140, 30/140 of the negatives).
    shares = {"billing": 30, "orders": 40, "account": 40, "other": 30}
    total_share = sum(shares.values())
    for cls, share in shares.items():
        quota = round(args.negatives * share / total_share)
        intents = sorted(i for i, rs in by_intent.items()
                         if ROUTING[rs[0]["category"]] == cls and i not in ("get_refund", "track_refund"))
        per, extra = divmod(quota, len(intents))
        for j, intent in enumerate(intents):
            chosen += [("neg", r) for r in rng.sample(by_intent[intent], per + (1 if j < extra else 0))]
    rng.shuffle(chosen)
    with open(out / "tickets.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["id", "text", "refund", "routing", "intent", "category"])
        for i, (kind, r) in enumerate(chosen, 1):
            w.writerow([i, " ".join(fill(r["instruction"], rng).split()), str(kind == "pos").lower(),
                        ROUTING[r["category"]], r["intent"], r["category"]])
    pos = sum(1 for k, _ in chosen if k == "pos")
    print(f"wrote {out / 'tickets.csv'}: {len(chosen)} tickets ({pos} refund, {len(chosen) - pos} not)")


if __name__ == "__main__":
    main()
