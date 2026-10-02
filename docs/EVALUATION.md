# Model evaluation on 200 labelled tickets

Run on 1 Oct 2026 with `tools/eval` (commit that added this file). Every model scores the
same 200 tickets through the extension, one `decide_table` call per ticket with two questions:

- **refund** (yes/no): "A refund is requested." Ground truth: the ticket's intent is `get_refund`.
- **routing** (choice): "Which team should handle this request?" with options
  `billing` / `orders` / `account` / `other`. Ground truth: the ticket's category mapped to a team.

## Data

Public [Bitext customer-support dataset](https://huggingface.co/datasets/bitext/Bitext-customer-support-llm-chatbot-training-dataset)
(CDLA-Sharing-1.0; 27 intents, 11 categories). It is downloaded by `tools/eval/make_sample.py`, not
committed. The sample is deterministic (seed 20260930): 60 `get_refund` positives and 140 negatives
spread over routing classes so that routing is not dominated by `billing`. Negatives include
deliberately hard ones (`check_refund_policy`, `payment_issue`, `cancel_order`, complaints);
`track_refund` is excluded because "I expect a refund of X" has no clear label. Routing mix: billing 90
(60 refund requests + 30 other billing), orders 40, account 40, other 30.

## Results

Refund: 60 positive / 140 negative (always-no baseline 70%). Routing mix: billing 90, orders 40, account 40, other 30 (always-billing baseline 45%).

## Refund (yes/no, threshold 0.5)

| Model | Accuracy [95% CI] | Balanced acc | Recall | Specificity | AUROC | Brier | ECE |
|---|---|---|---|---|---|---|---|
| TypeSafe Jev | 93.0% [89-96] | 94.5% | 98% | 91% | 0.984 | 0.066 | 0.125 |
| Laya multilingual | 92.0% [87-95] | 93.8% | 98% | 89% | 0.970 | 0.066 | 0.074 |
| Liquid D1 | 77.0% [71-82] | 83.6% | 100% | 67% | 0.970 | 0.157 | 0.225 |
| Laya typed-decisions | 88.5% [83-92] | 82.3% | 67% | 98% | 0.978 | 0.090 | 0.169 |
| Kev-0.8B (server) | 70.5% [64-76] | 76.1% | 90% | 62% | 0.850 | 0.206 | 0.245 |
| Julia-1 | 29.5% [24-36] | 45.8% | 87% | 5% | 0.407 | 0.633 | 0.637 |

## Routing (billing / orders / account / other)

| Model | Accuracy [95% CI] | Macro recall | billing recall | orders recall | account recall | other recall |
|---|---|---|---|---|---|---|
| TypeSafe Jev | 92.0% [87-95] | 90.7% | 98% | 95% | 70% | 100% |
| Liquid D1 | 91.0% [86-94] | 89.5% | 96% | 85% | 88% | 90% |
| Laya typed-decisions | 76.0% [70-81] | 77.4% | 71% | 88% | 78% | 73% |
| Laya multilingual | 62.0% [55-68] | 64.6% | 57% | 55% | 70% | 77% |
| Kev-0.8B (server) | 44.0% [37-51] | 55.5% | 18% | 48% | 60% | 97% |
| Julia-1 | 37.5% [31-44] | 42.5% | 20% | 70% | 50% | 30% |

## Significance (exact McNemar, paired on the same tickets)

Refund vs TypeSafe Jev, routing vs TypeSafe Jev; p < 0.05 means the difference is unlikely to be chance on this sample.

| Model | Refund p | Routing p |
|---|---|---|
| Laya multilingual | 0.832 | 0.000 |
| Liquid D1 | 0.000 | 0.832 |
| Laya typed-decisions | 0.163 | 0.000 |
| Kev-0.8B (server) | 0.000 | 0.000 |
| Julia-1 | 0.000 | 0.000 |

Wall time for the 200 tickets (not comparable across hardware and networks): Julia-1 21 s, Laya multilingual
48 s, Laya typed-decisions 87 s, Jev 72 s, Kev-0.8B 296 s (CPU), Liquid D1 1807 s. D1's free tier showed highly
variable latency (1 to 34 s per call in plain `curl` tests on the same payloads), so this is the API, not
the extension.

## What it shows

- **Jev is the most accurate overall** (93% refund, 92% routing) and the best calibrated on refund (Brier 0.066).
- **Laya multilingual matches Jev on refund** (92%, p = 0.83) and is the best-calibrated model (ECE 0.074), but
  routes much worse (62% vs 92%, p < 0.001). Among models that run in-process it is the best refund detector.
- **Laya typed-decisions is the best in-process router** (76%), but at the 0.5 cut-off it misses a third of
  refund requests (recall 67%, specificity 98%).
- **Liquid D1 matches Jev on routing** (91% vs 92%, p = 0.83) and ranks refunds as well as Laya multilingual
  (AUROC 0.970), but is over-confident on "refund": at 0.5 it flags every complaint, human-agent request and
  refund-policy question (specificity 67%). A cut-off near 0.9 would give about 90% on this sample. That
  number is chosen after the fact on the same data, so it is an indication of a calibration issue, not a
  fair accuracy estimate.
- **Kev-0.8B and Julia-1 are not competitive** on this task. Julia-1 answers "refund" for almost everything
  (AUROC below 0.5); its local port reproduces the upstream Python implementation exactly (see
  REVIEW_FOLLOWUP.md), so this is the model, not the port.
- The earlier 8-ticket comparison in the README ranked D1 first (8/8, 8/8). That sample was too small and
  too easy; on 200 tickets D1's refund result is clearly worse than Jev's and Laya multilingual's.

## Caveats

- **English only.** Bitext is English, so the multilingual claims of Laya multilingual and Julia-1 are not
  tested here.
- **Synthetic, template-generated text.** Bitext utterances are shorter and cleaner than real tickets, and
  intents map neatly to categories. Expect lower absolute numbers on your own data.
- **One sample, one seed.** Confidence intervals are Wilson 95% intervals on 200 tickets (about +/-5 points);
  the McNemar p-values compare models paired on the same tickets.
- **Label definitions are ours.** "Refund requested" is the `get_refund` intent and the routing classes are
  our grouping of Bitext categories; a different definition would move the numbers.
- **Hosted models change over time** (`jev-latest`, `d1:free`); the run is a dated snapshot. Ticket text was
  sent to Liquid AI and TypeSafe for those runs.
- **Local models use the extension's own conversion.** Each local model was verified against its upstream
  Python reference on the original 8 tickets; the 200-ticket run did not re-verify that.

## Reproduce

```bash
python3 tools/eval/make_sample.py --out /tmp/eval            # downloads Bitext, writes /tmp/eval/tickets.csv
# local models need exported graphs (see README); hosted need LIQUID_API_KEY / TYPESAFE_API_KEY
export JULIA_ONNX=... JULIA_WEIGHTS_DIR=... LAYA_ML_ONNX=... LAYA_ML_DIR=... LAYA_TD_ONNX=... LAYA_TD_DIR=...
python3 tools/eval/run_eval.py --tickets /tmp/eval/tickets.csv --out /tmp/eval/out --models d1,jev,laya-ml,laya-td,julia
python3 tools/eval/run_eval.py --tickets /tmp/eval/tickets.csv --out /tmp/eval/out --models kev   # needs a running Kev server
```

`run_eval.py --analyse-only` recomputes the report from existing `results_*.csv` files (copy results from
several runs into one directory first).
