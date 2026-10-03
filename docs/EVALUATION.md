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
| strands-decider 2B (server) | 81.0% [75-86] | 71.7% | 48% | 95% | 0.918 | 0.129 | 0.152 |
| Julia-1 | 29.5% [24-36] | 45.8% | 87% | 5% | 0.407 | 0.633 | 0.637 |

## Routing (billing / orders / account / other)

| Model | Accuracy [95% CI] | Macro recall | billing recall | orders recall | account recall | other recall |
|---|---|---|---|---|---|---|
| TypeSafe Jev | 92.0% [87-95] | 90.7% | 98% | 95% | 70% | 100% |
| Liquid D1 | 91.0% [86-94] | 89.5% | 96% | 85% | 88% | 90% |
| Laya typed-decisions | 76.0% [70-81] | 77.4% | 71% | 88% | 78% | 73% |
| strands-decider 2B (server) | 65.0% [58-71] | 70.7% | 51% | 82% | 62% | 87% |
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
| strands-decider 2B (server) | 0.000 | 0.000 |
| Julia-1 | 0.000 | 0.000 |

Wall time for the 200 tickets (not comparable across hardware and networks): Julia-1 21 s, Laya multilingual
48 s, Laya typed-decisions 87 s, Jev 72 s, Kev-0.8B 296 s (CPU), strands-decider 2B 396 s (CPU, no GPU on the
test machine), Liquid D1 1807 s. D1's free tier showed highly
variable latency (1 to 34 s per call in plain `curl` tests on the same payloads), so this is the API, not
the extension.

## Calibration

Raw cut-offs differ per model (D1 says "refund" for every complaint at 0.5), so each model's refund
probabilities were re-scaled with Platt scaling, the same maths as `decide_fit_calibration` and the
registration option `MAP {'calibration': 'platt:a,b'}`. To keep the estimate honest the tickets are split
into two stratified halves, the scaling is fitted on one half and applied to the other (both halves are
scored this way), averaged over 20 random splits. Each fit therefore uses 100 tickets (30 refund) and is
judged on tickets it never saw.

| Model | Accuracy @0.5 raw -> calibrated | Brier raw -> calibrated | ECE raw -> calibrated |
|---|---|---|---|
| Liquid D1 | 77.0% -> 91.0% | 0.157 -> 0.065 | 0.225 -> 0.049 |
| TypeSafe Jev | 93.0% -> 93.2% | 0.066 -> 0.050 | 0.125 -> 0.046 |
| Laya multilingual | 92.0% -> 93.7% | 0.066 -> 0.055 | 0.074 -> 0.052 |
| Laya typed-decisions | 88.5% -> 93.5% | 0.090 -> 0.054 | 0.169 -> 0.048 |
| Kev-0.8B | 70.5% -> 80.7% | 0.206 -> 0.143 | 0.245 -> 0.069 |

- **D1's weakness was calibration, not discrimination.** Scaling lifts it from 77% to 91% and its Brier score to
  Jev's level, so the model ranks refund requests well and only its cut-off was off.
- **Every model's probabilities get more honest** (Brier and ECE fall for all five), and the spread between
  the top four on refund shrinks from 16 points to under 3 (D1 91.0%, Jev 93.2%, Laya typed-decisions 93.5%,
  Laya multilingual 93.7%), which is inside the sampling interval of 200 tickets (about +/-5 points).
  Laya typed-decisions gains most among them (88.5% -> 93.5%), because its raw cut-off was too conservative.
- **Kev-0.8B improves but stays behind** (80.7%).
- **Julia-1 cannot be calibrated**: its probabilities rank refund requests worse than chance (AUROC 0.407), and
  the fit correctly refuses it.
- Routing is a choice question and is not calibrated; the routing table above is unchanged.
- This is in-distribution: the calibration is fitted and judged on the same kind of (template-generated,
  English) tickets. Expect a calibration fitted on one ticket stream to need a refit on another.

Reproduce from the results of `run_eval.py`: `python3 tools/eval/crossfit_calibration.py --tickets
/tmp/eval/tickets.csv --results /tmp/eval/out`.

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
- **strands-decider 2B** (added 2 Oct 2026; a Qwen3.5 torso with a pointer head, served locally) is mid-table:
  81% refund, 65% routing. It is conservative on refund (recall 48%, specificity 95%, AUROC 0.918): a cut-off
  near 0.27 would give about 86% on this sample, chosen after the fact on the same data. It routes better than
  Laya multilingual and Kev-0.8B but clearly worse than Jev and D1 (p < 0.001). It is the only model here that
  ran with a different choice-criteria mode (`criteria: 'name'`, required by its server schema); that is a wire
  detail, not a quality difference.
- **Kev-0.8B and Julia-1 are not competitive** on this task. Julia-1 answers "refund" for almost everything
  (AUROC below 0.5); its local port reproduces the upstream Python implementation exactly (see
  REVIEW_FOLLOWUP.md), so this is the model, not the port.
- An earlier 8-ticket check ranked D1 first (8/8, 8/8). That sample was too small and
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
# local models need exported graphs (see tools/export_julia/README.md); hosted need LIQUID_API_KEY / TYPESAFE_API_KEY
export JULIA_ONNX=... JULIA_WEIGHTS_DIR=... LAYA_ML_ONNX=... LAYA_ML_DIR=... LAYA_TD_ONNX=... LAYA_TD_DIR=...
python3 tools/eval/run_eval.py --tickets /tmp/eval/tickets.csv --out /tmp/eval/out --models d1,jev,laya-ml,laya-td,julia
python3 tools/eval/run_eval.py --tickets /tmp/eval/tickets.csv --out /tmp/eval/out --models kev       # needs a running Kev server
python3 tools/eval/run_eval.py --tickets /tmp/eval/tickets.csv --out /tmp/eval/out --models strands   # needs a running strands-decider server
```

`run_eval.py --analyse-only` recomputes the report from existing `results_*.csv` files (copy results from
several runs into one directory first).

## Run time and concurrency (2 Oct 2026)

The remote latency noted above (1 to 34 s per call) made sequential runs slow. Scalar `decide_*` calls now send a
chunk's requests concurrently (`anofox_decide_max_concurrency`, default 8 for hosted providers): 8 real D1 calls took
176.7 s one after another and 23.1 s together. `tools/eval/run_eval.py` does not benefit yet: it scores through
`LATERAL decide_table`, which DuckDB runs one row at a time (see "Scoring many rows" in the README). Moving it to the
scalar functions (or to a future scalar that returns the question rows) would cut the remote runs by about the
concurrency factor.

