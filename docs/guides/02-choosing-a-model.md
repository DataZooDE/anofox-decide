# Choosing a model

Hosted and local models are profiles over one wire format (System One: `binary`, `choice` and `score`
questions, probabilities back), so a model is just a registered id, and several can be used side by side
in one session with no `SET` in between. All of them answer the same questions through the same
functions, so you can swap the model without touching the query.

## You have, do this

`SELECT * FROM decide_doctor();` tells you what is ready and how to fix what is not. Then choose one:

| You have | Do this |
|---|---|
| Nothing yet | Use the built-in `stub` to try the SQL: `model := 'stub'`. It returns constants (0.5, the first option, the middle level) so it can never pass for a real answer. |
| A Liquid AI key ([D1](https://docs.liquid.ai/lfm/models/decision-models)) | `export LIQUID_API_KEY=...`, then register `d1:free` ([Getting Started](01-getting-started.md#2-check-the-setup-and-pick-a-model)). |
| A Cloudflare account ([Clef](https://blog.cloudflare.com/clef-decision-models/) on Workers AI) | `export CLOUDFLARE_API_TOKEN=...` and `CLOUDFLARE_ACCOUNT_ID=...`, then `decide_register_model('clef', 'cloudflare')` ([Cloudflare Clef](../api/02-providers.md#cloudflare-clef)). |
| A TypeSafe key (Jev) | `export TYPESAFE_API_KEY=...`, then `decide_register_model('jev-latest', 'typesafe')`. |
| A machine that can run a 2B model | Run [strands-decider](https://github.com/strands-labs/strands-decider) locally, no key: [Local servers](../api/02-providers.md#local-servers). |
| Nothing but a laptop and a download (about 0.7 GB) | `CALL decide_download('laya-multilingual');`, then `model := 'laya-multilingual'`. It runs inside DuckDB, offline after the download: [Local models](../api/03-local-models.md). |
| Your own exported ONNX model | Register the files yourself: [Your own exported model](../api/03-local-models.md#your-own-exported-model). |

## Hosted, local server or in-process

| | Hosted API | Local server | In-process |
|---|---|---|---|
| Models | TypeSafe Jev, Liquid AI D1, Cloudflare Clef / Clef-flash | strands-decider, Kev, any System One server | Julia-1, Laya multilingual, Laya typed-decisions |
| Your text goes to | the provider | your own server (`https://`, or `http://` on loopback) | nowhere: DuckDB scores it |
| Setup | an API key and `SET anofox_decide_allow_remote = true` | start the server, register it, the same opt-in | `CALL decide_download('<model>')` once |
| Speed | up to 8 requests at a time | 1 at a time for strands by default | CPU, or a GPU with a plugin ([GPU](../api/04-gpu.md)) |
| Reference | [Providers](../api/02-providers.md) | [Local servers](../api/02-providers.md#local-servers) | [Local models](../api/03-local-models.md) |

## Status and scope

What works today:

| Provider | Models | Questions | Runs where |
|---|---|---|---|
| `stub` | deterministic placeholder (0.5 / first option / middle level) | `binary`, `choice`, `score` | in-process, for tests and demos |
| `typesafe` | TypeSafe Jev (`jev-latest`) | `binary`, `choice`, `score` | hosted API; opt-in via `anofox_decide_allow_remote` |
| `liquid` | Liquid AI D1 (`d1:free`) | `binary`, `choice`, `score` | hosted API; same opt-in |
| `cloudflare` | Cloudflare Clef (`clef`), Clef-flash (`clef-flash`) | `binary`, `choice`, `score` | hosted API (Workers AI); same opt-in |
| `systemone` | any System One compatible server, e.g. Kev | `binary`, `choice`, `score` | your endpoint (`https://`, or `http://` on loopback) |
| `strands` | strands-decider 2B | `binary`, `choice`, `score` | local server on loopback, no key |
| `local` | Julia-1, Laya multilingual, Laya typed-decisions | `binary`, `choice`, `score` | in-process on CPU, offline after setup |

`score` is verified for parity with each model's upstream implementation and against the live services;
unlike refund and routing it has not been scored for accuracy on labelled data.

**Not yet:** images (Cloudflare Clef accepts them, our functions take text only); half-precision GPU
execution (the GPU backends are fp32; see [GPU acceleration](../api/04-gpu.md)); the Von model (it needs
order-invariant attention in the export); on macOS only Apple silicon is built (ONNX Runtime ships no Intel
macOS archive after v1.23.2), and WebAssembly, musl and MinGW are not built.

## How the models compare

Measured on 200 labelled tickets from the public Bitext customer-support dataset (60 refund requests,
140 not; routing into billing / orders / account / other). Full method, caveats and per-class results:
[docs/EVALUATION.md](../EVALUATION.md).

| Model | Runs | Refund accuracy | Refund AUROC | Routing accuracy |
|---|---|---|---|---|
| TypeSafe Jev (`typesafe`) | hosted API | 93.0% | 0.984 | 92.0% |
| Laya multilingual | local, in-process | 92.0% | 0.970 | 62.0% |
| Laya typed-decisions | local, in-process | 88.5% | 0.978 | 76.0% |
| Cloudflare Clef (`cloudflare`) | hosted API | 85.0% | 0.951 | 88.0% |
| strands-decider 2B (`strands`) | local server | 81.0% | 0.918 | 65.0% |
| Liquid D1 (`liquid`) | hosted API | 77.0% | 0.970 | 91.0% |
| Kev-0.8B (`systemone`) | local server | 70.5% | 0.850 | 44.0% |
| Cloudflare Clef-flash (`cloudflare`) | hosted API | 70.0% | 0.769 | 78.5% |
| Julia-1 | local, in-process | 29.5% | 0.407 | 37.5% |

Always-no scores 70% on refund and always-billing 45% on routing. Jev is the most accurate overall;
Clef is the third-best router (not significantly different from Jev) and a solid, conservative refund detector
(recall 62% at the 0.5 cut-off), while the smaller Clef-flash is no better than always-no on refund;
strands-decider is conservative on refund (recall 48% at the 0.5 cut-off) and a mid-table router; D1
routes as well as Jev, but at the 0.5 cut-off over-predicts refunds (its ranking is fine, AUROC 0.970);
Laya multilingual matches Jev on refund and is the best in-process model, with typed-decisions the best
in-process router. This is English, template-generated data: a smoke test of relative strength, not a
benchmark of your tickets, so evaluate on your own (how to reproduce this run:
[docs/EVALUATION.md](../EVALUATION.md#reproduce)). Each local ONNX model reproduces its upstream Python
reference to within 5e-5 in probability on a small check set; the hosted models are called as-is.

**Which model?** Jev was the most accurate in the 200-ticket evaluation, with Clef and D1 close behind on
routing. Among models that run in-process, Laya multilingual matched it on refund detection and Laya
typed-decisions routes best; there is no single best local model yet. Start with whichever you have a key
or the hardware for, then compare on your own labelled data.

## Comparing models on your own data

Several models can be registered side by side and named per call, and the metric aggregates work on any
`(probability, label)` pairs, so comparing models is one query over a table of runs:

```sql
SET anofox_decide_allow_remote = true;
SELECT decide_register_model('jev-latest', 'typesafe');
SELECT decide_register_model('d1:free', 'liquid');

CREATE TABLE runs AS
SELECT 'jev-latest' AS model, id, refund,
       decide_probability(text, 'A refund is requested.', model := 'jev-latest') AS p FROM tickets
UNION ALL
SELECT 'd1:free', id, refund,
       decide_probability(text, 'A refund is requested.', model := 'd1:free') FROM tickets;

SELECT model,
       decide_accuracy(p, refund)    AS accuracy,
       decide_brier_score(p, refund) AS brier,
       decide_ece(p, refund)         AS ece
FROM runs GROUP BY model ORDER BY brier;     -- see examples/04_compare_models.sql
```

The complete version is [`examples/04_compare_models.sql`](../../examples/04_compare_models.sql). How to read
the metrics, plot ROC and calibration curves per model, and fix a model's cut-off:
[Evaluate and calibrate](03-evaluate-and-calibrate.md).
