# Functions

Every function is available as `anofox_decide_<name>` and as the short alias `decide_<name>`.
`duckdb_functions()` carries a description, parameter names and a runnable example for each.

- [What you can ask](#what-you-can-ask)
- [Function reference](#function-reference)
- [Several questions at once](#several-questions-at-once)
- [NULLs](#nulls)
- [Scoring many rows](#scoring-many-rows)

## What you can ask

Every question has a **kind**. You write the question and, for the kinds that need them, the answers —
at query time, per row if you like.

| Kind | You provide | You get | Function |
|---|---|---|---|
| `binary` | a statement | the probability (0 to 1) that it holds for the text | `decide_probability`, and `decide_decision` with a threshold |
| `choice` | a question and a list of options | the best option, plus the probability of each | `decide_choice` |
| `score` | a question and an ordered rubric of 2 to 10 levels, lowest first | the expected level (0-based, e.g. `1.4` between `frustrated` and `angry`), plus the probability of each level | `decide_score` |

## Function reference

### Decisions

| Function | Returns | Purpose |
|---|---|---|
| `decide_probability(state, question[, model])` | `DOUBLE` | P(`question` holds for `state`) |
| `decide_decision(state, question, threshold[, model])` | `BOOLEAN` | `true` when that probability reaches an explicit `threshold` (0 to 1); never guessed |
| `decide_choice(state, question, options[, model])` | `VARCHAR` | the best of a list of options |
| `decide_choice_distribution(state, question, options[, model])` | `MAP(VARCHAR, DOUBLE)` | the probability of every option (sums to 1), so you can threshold or abstain in SQL ([Abstaining](../guides/03-evaluate-and-calibrate.md#abstaining-on-choice-questions)) |
| `decide_score(state, question, levels[, model])` | `DOUBLE` | the expected 0-based level on an ordered rubric |
| `decide_many(state, questions[, model])` | `VARCHAR` (JSON) | several questions about one text in one request |
| `decide_table(state, questions[, model])` | table | the same, one row per question |
| `decide_answers(state, questions[, model])` | `LIST(STRUCT(...))` | the rows of `decide_table` as a value: `unnest(decide_answers(...), recursive := true)` gives its columns for every row of a table, scored concurrently |

### Evaluation and calibration

See [Evaluate and calibrate](../guides/03-evaluate-and-calibrate.md) for how to use them.

| Function | Returns | Purpose |
|---|---|---|
| `decide_accuracy(p, outcome[, threshold])` | `DOUBLE` | share of rows where thresholding `p` reproduces the label (aggregate) |
| `decide_brier_score(p, outcome)` | `DOUBLE` | mean squared error of the probabilities; 0 is perfect, 0.25 a constant 0.5 (aggregate) |
| `decide_ece(p, outcome)` | `DOUBLE` | expected calibration error over ten equal-width bins (aggregate) |
| `decide_log_loss(p, outcome)` | `DOUBLE` | mean binary cross-entropy, `p` clipped to [1e-15, 1 - 1e-15] (aggregate) |
| `decide_auc(p, outcome)` | `DOUBLE` | area under the ROC curve, ties count one half; NULL with one outcome only (aggregate) |
| `decide_roc_curve_by(source, group, p, outcome)` | table | the ROC curve per group, ready for `anofox_plot` ([Plot](../guides/03-evaluate-and-calibrate.md#plot-with-anofox-visualization)) |
| `decide_calibration_curve_by(source, group, p, outcome[, n_bins := 10])` | table | the calibration curve per group, with Wilson intervals ([Plot](../guides/03-evaluate-and-calibrate.md#plot-with-anofox-visualization)) |
| `decide_metrics_by(source, group, p, outcome)` | table | accuracy, auc, brier, ece and log loss per group, one row each ([Plot](../guides/03-evaluate-and-calibrate.md#plot-with-anofox-visualization)) |
| `decide_fit_calibration(p, outcome)` | `VARCHAR` | fits Platt scaling to a model's raw probabilities and returns the `'platt:a,b'` spec for `decide_register_model` ([Calibration](../guides/03-evaluate-and-calibrate.md#calibration)) (aggregate) |

### Models, local runtime and setup

| Function | Returns | Purpose |
|---|---|---|
| `decide_register_model(id[, provider[, ...]])` | `BOOLEAN` | register a model for this database instance ([Providers](02-providers.md)) |
| `decide_unregister_model(id)` | `BOOLEAN` | remove a registered model so it can be registered again |
| `decide_models()` | table | the registered models, whether each can be called now, what to do if not, and the `device` a local model is scored on |
| `decide_doctor()` | table | `item`, `status` (`ok` / `warn` / `fail`), `detail`, `fix` for the whole setup |
| `decide_download(model)` | table | download a local model's weights from Hugging Face into the cache directory, verified and resumable ([Local models](03-local-models.md)) |
| `decide_token_count(text[, model])` | `BIGINT` | tokens `text` has for a local model, to find rows that do not fit its window ([Long text](03-local-models.md#long-text)) |
| `decide_devices()` | table | the devices this machine can score local models on (cpu, and any NVIDIA, AMD or Apple GPU found), with `usable` and why not ([GPU](04-gpu.md)) |
| `decide_backends()` | table | which device can serve which local model, and why not: the same check `anofox_decide_device = 'auto'` uses ([GPU](04-gpu.md)) |
| `decide_accelerate()` | table | finds the GPU, downloads and verifies its plugin, activates it for this session ([GPU](04-gpu.md)) |
| `decide_download_runtime(backend)` | table | download and verify the plugin of `'cuda'`, `'rocm'` or `'mlx'` ([GPU](04-gpu.md)) |
| `decide_gpu_precompile(model[, seq := N])` | table | compile a local model's shape buckets ahead of time on the ROCm backend, with the seconds each took ([ROCm](../GPU_ROCM.md)) |

## Several questions at once

Several questions about the same text go in one call, as a JSON array of
`{id, kind, instruction[, options | levels]}`:

```sql
SELECT question_id, kind, probability, choice, score, distribution
FROM decide_table('The bill is wrong and I want my money back.',
  '[{"id": "refund", "kind": "binary", "instruction": "A refund is requested."},
    {"id": "team",   "kind": "choice", "instruction": "Which team owns this?", "options": ["billing", "defect", "other"]},
    {"id": "mood",   "kind": "score",  "instruction": "How frustrated is the writer?", "levels": ["calm", "frustrated", "angry"]}]',
  model := 'd1:free');
```

At most `anofox_decide_max_questions` questions (default 100) go in one `decide_many` / `decide_table` call.

### Output columns of `decide_table`

| Column | Type | Meaning |
|---|---|---|
| `question_id` | `VARCHAR` | the `id` you gave the question |
| `kind` | `VARCHAR` | `binary`, `choice` or `score` |
| `probability` | `DOUBLE` | `binary`: P(yes). `choice`: probability of the chosen option. `score`: probability of the most likely level |
| `choice` | `VARCHAR` | the chosen option (`choice` only) |
| `confidence` | `DOUBLE` | the model's own confidence when the service reports one, otherwise `NULL` |
| `model` | `VARCHAR` | the model that answered |
| `score` | `DOUBLE` | the expected 0-based level (`score` only) |
| `distribution` | `MAP(VARCHAR, DOUBLE)` | probability per option or level (`choice`, `score`) |

`decide_many` returns the same answers as one JSON document per text:
`{"model": "...", "results": [{"id", "kind", "probability", ...}]}`.

`decide_answers` returns the same rows as a list of structs, one list per text. To score every row of a
table and get `decide_table`'s columns, unnest it (this runs concurrently, unlike a lateral join):

```sql
SELECT id, question_id, probability, choice
FROM (SELECT id, unnest(decide_answers(body, '[{"id": "refund", "kind": "binary", "instruction": "A refund is requested."}]',
                                        model := 'd1:free'), recursive := true)
      FROM tickets);
```

## NULLs

A NULL text, question, option list or threshold gives a NULL result: no model is resolved and nothing is
sent. A NULL `model` argument falls back to the session default.

## Scoring many rows

A remote call takes one text plus any number of questions, so the cost is **one request per row**
whatever the SQL looks like. What you control is how many run at the same time.

| You write | Requests | Notes |
|---|---|---|
| `decide_probability` / `decide_choice` / `decide_score` / `decide_decision` / `decide_many` over a table | one per distinct row, **up to 8 at a time** | The fast path. A chunk of rows is evaluated together: identical rows are sent once, connections are reused, and a `429` makes the query back off |
| `decide_table(text, questions)` with constant arguments | one request for all questions | One document, many questions |
| `unnest(decide_answers(t.text, questions), recursive := true)` over a table | one per distinct row, **up to 8 at a time** | The fast way to get `decide_table`'s rows (one per question) for every row of a table: same columns, same answers, scalar speed |
| `LATERAL decide_table(t.text, ...)` over a table | one per row, **one after another** | Gives the same rows, but DuckDB gives the function one row per call in a lateral join, so it cannot be concurrent (measured: with 8 DuckDB threads the peak is still 1 request in flight). Fine for a few rows; use `decide_answers` for many |

Measured on the real Liquid D1 service (8 tickets, three questions each): 176.7 s with
`SET anofox_decide_max_concurrency = 1` against 23.1 s with the default, same answers (D1 itself varies
slightly from call to call). The [triage example](../guides/01-getting-started.md#3-decide) takes about 11 s.

```sql
SET anofox_decide_max_concurrency = 4;   -- 0 = automatic (default): 8 hosted, 1 for the local strands server; 1 = one at a time
SET anofox_decide_timeout_ms = 60000;    -- slow models: more time per request
SET anofox_decide_max_retries = 5;       -- retries after a 429, a 5xx or a connection failure
```

Lower the limit if your provider rate limits you. The query already halves its window after a `429`,
honours `Retry-After`, and stops starting new requests after the first error (the error of the first
failing row is the one reported). Two identical `(text, question)` pairs in one chunk cost one request;
nothing is shared across chunks or queries, because a service need not answer the same question the same
way twice. Local models run one after another (or in batches with
[`anofox_decide_batch_tokens`](03-local-models.md#batching-rows)). See
[`examples/03_scoring_many_rows.sql`](../../examples/03_scoring_many_rows.sql).
