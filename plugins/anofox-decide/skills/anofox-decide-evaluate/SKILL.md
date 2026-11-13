---
name: anofox-decide-evaluate
description: >
  Measuring, comparing and calibrating decision-model probabilities with the anofox_decide DuckDB
  extension. Covers the metric aggregates decide_accuracy (with an optional threshold),
  decide_brier_score, decide_ece, decide_log_loss and decide_auc. Covers the long-table producers
  decide_roc_curve_by, decide_calibration_curve_by (n_bins, Wilson intervals) and decide_metrics_by,
  which plot directly with anofox-visualization's anofox_plot. Covers Platt-scaling calibration
  with decide_fit_calibration, giving a 'platt:a,b' spec that is registered through
  decide_register_model's calibration option; choosing a threshold for decide_decision; abstaining
  on choice questions with decide_choice_distribution; and comparing several models on one labelled
  table. Use when evaluating accuracy on labelled data, comparing models or providers, picking a
  cut-off, fixing a model that says yes too often or too rarely, or plotting ROC / reliability
  curves.
version: 2026.11.13
user-invocable: false
---

# Anofox Decide — Evaluate, Compare & Calibrate

**Extension:** `anofox_decide` v2026.11.13 | **DuckDB:** v1.5.6 / v2.0 | **Dual naming:** `decide_*` and
`anofox_decide_*`

The models answer with probabilities, so their quality is measurable on labelled data. The workflow:
score a labelled sample with each candidate model into one `runs(model, id, label, p)` table, compare the
models, calibrate the chosen one, then pick the threshold.

## Critical gotchas

1. **Metrics take `(probability, outcome)`.** The probability must be 0 to 1, not percents or logits. The
   outcome is BOOLEAN or 0/1. Anything else is an error that says what to fix. NaN is an error too
   (`NULLIF(p, 'NaN'::DOUBLE)`).
2. **NULL rows are skipped**, so refusals silently drop out, and two models end up compared on different
   rows (the one that refuses the hard cases looks better). Always report `count(*) FILTER (WHERE p IS NULL)`
   and score a refusal as wrong: `coalesce(p, CASE WHEN label THEN 0.0 ELSE 1.0 END)`. Empty input gives
   NULL. `decide_auc` is NULL when only one outcome is present.
3. **`decide_accuracy` thresholds at 0.5** unless you pass a third argument.
4. **Calibration only changes yes/no probabilities** (`decide_probability`, `decide_decision`,
   `decide_many` / `decide_table` binary answers). Choice and score are untouched. Platt scaling has
   `a > 0`, so the ranking (AUC) never changes. It fixes the cut-off and the sharpness.
5. **`decide_fit_calibration` refuses** a single outcome, constant probabilities, and models that rank no
   better than chance. Fit on about 50 or more labelled rows. **Fit on one sample and check on another**,
   which costs no requests: the check runs on the stored probabilities (see Calibration). Refit for each
   data distribution.
6. **The `_by` producers take a table or view name as a string** plus unquoted columns:
   `decide_metrics_by('runs', model, p, label)`. A CTE is not visible to them: create a (temp) view or table
   first. They return **long** tables (one row per point or metric), not one column per metric.
7. **Don't evaluate on `stub`.** It returns 0.5 for every yes/no question, 1.0 on the first option of a
   choice and a uniform distribution for a score.

## Metric aggregates

| Function | Meaning | Best |
|---|---|---|
| `decide_accuracy(p, outcome[, threshold])` | share where `p >= threshold` reproduces the label | 1 |
| `decide_brier_score(p, outcome)` | mean squared error of `p`; a constant 0.5 scores 0.25 | 0 |
| `decide_ece(p, outcome)` | expected calibration error, 10 equal-width bins on the decided answer's confidence | 0 |
| `decide_log_loss(p, outcome)` | mean binary cross-entropy, `p` clipped to [1e-15, 1-1e-15] | 0 |
| `decide_auc(p, outcome)` | area under ROC, ties count 1/2 | 1 |

```sql
-- DEMO DATA: 16 made-up rows to show the calls; evaluate and calibrate on 50+ real labelled rows.
-- One row per (model, item); in practice p comes from decide_probability (see the next section).
CREATE OR REPLACE TABLE runs AS SELECT * FROM (VALUES
  ('model-a', 1, true, 0.92), ('model-a', 2, false, 0.20), ('model-a', 3, true, 0.71), ('model-a', 4, false, 0.75),
  ('model-a', 5, true, 0.64), ('model-a', 6, false, 0.08), ('model-a', 7, true, 0.88), ('model-a', 8, false, 0.35),
  ('model-b', 1, true, 0.99), ('model-b', 2, false, 0.81), ('model-b', 3, true, 0.97), ('model-b', 4, false, 0.76),
  ('model-b', 5, true, 0.93), ('model-b', 6, false, 0.62), ('model-b', 7, true, 0.98), ('model-b', 8, false, 0.70)
) t(model, id, refund, p);

SELECT model,
       count(*) FILTER (WHERE p IS NULL)         AS refusals,   -- report them, and count them as wrong:
       round(decide_accuracy(coalesce(p, CASE WHEN refund THEN 0.0 ELSE 1.0 END), refund), 3) AS acc_refusal_wrong,
       round(decide_accuracy(p, refund), 3)      AS acc_at_50,
       round(decide_accuracy(p, refund, 0.8), 3) AS acc_at_80,
       round(decide_auc(p, refund), 3)           AS auc,
       round(decide_brier_score(p, refund), 3)   AS brier,
       round(decide_ece(p, refund), 3)           AS ece,
       round(decide_log_loss(p, refund), 3)      AS log_loss
FROM runs GROUP BY model ORDER BY brier;
```

Read it like this: AUC says whether a model *ranks* well. Brier, ECE and log loss say whether its
probabilities can be *taken at face value*. In the demo, `model-b` ranks perfectly (AUC 1) but is right only
half the time at 0.5: its cut-off is wrong, so calibrate it or pick another threshold. `model-a` ranks two of
its 16 (true, false) pairs wrongly (AUC 0.875) but is right 7 of 8 times at 0.5. Rank candidates by AUC
first: calibration fixes the cut-off and the sharpness, never the ranking.

## Building `runs` from real models

```sql
-- needs: the two models registered (see anofox-decide-models) and a labelled tickets(id, text, refund) table
CREATE OR REPLACE TABLE runs AS
SELECT 'jev-latest' AS model, id, refund,
       decide_probability(text, 'A refund is requested.', model := 'jev-latest') AS p FROM tickets
UNION ALL
SELECT 'd1', id, refund,
       decide_probability(text, 'A refund is requested.', model := 'd1') FROM tickets;
```

`jev-latest` (TypeSafe) and `d1` (Liquid) are **hosted** models: each row is a paid request, the text
leaves the machine, and they need a key, `SET anofox_decide_allow_remote = true` and registration (see
`anofox-decide-models`). Store `runs` once and compute every metric from the table. Local catalog models
(`laya-multilingual`, `laya-typed-decisions`, `julia-1`) cost nothing per row (see `anofox-decide-local`).

For choice questions, compare `decide_choice(...) = label` with `avg(...::INT)` per model. For score
questions, compare `round(decide_score(...))` with the label's level index.

## Curves and summary tables (anofox-visualization ready)

```text
decide_roc_curve_by(source VARCHAR, group_col, p_col, outcome_col)                    -> TABLE (curve schema)
decide_calibration_curve_by(source, group_col, p_col, outcome_col[, n_bins := 10])    -> TABLE (curve schema, Wilson intervals)
decide_metrics_by(source, group_col, p_col, outcome_col)                              -> TABLE(group_col, model_id, metric, value, conf_low,
                                                                                         conf_high, row_order)  -- long: one row per (group, metric);
                                                                                         accuracy at 0.5; no auc row when one outcome is present;
                                                                                         conf_low / conf_high are always NULL (no intervals)
```

```sql
SELECT * FROM decide_metrics_by('runs', model, p, refund);
SELECT * FROM decide_roc_curve_by('runs', model, p, refund);
SELECT * FROM decide_calibration_curve_by('runs', model, p, refund, n_bins := 5);

-- one row per model, one column per metric
PIVOT (SELECT model_id, metric, value FROM decide_metrics_by('runs', model, p, refund)) ON metric USING first(value);
```

```sql
-- needs: the anofox_visualization extension
LOAD json;                     -- anofox_plot needs it (INSTALL json; once, if it is not installed)
LOAD anofox_visualization;
SELECT anofox_plot(t) FROM decide_roc_curve_by('runs', model, p, refund) t;          -- one ROC curve per model
SELECT anofox_plot(t) FROM decide_calibration_curve_by('runs', model, p, refund) t;  -- reliability diagram
SELECT anofox_plot(t) FROM decide_metrics_by('runs', model, p, refund) t;            -- model comparison
```

These follow the anofox integration contract (`curve` / `summary` schemas, a trailing `row_order`). They
`UNION ALL BY NAME` with anofox-statistics' `roc_curve_by` / `calibration_curve_by`.

## Calibration (Platt scaling)

Fit on one half, check on the other, then register. The check costs no requests: a calibration is a pure
function of the raw probability, `sigmoid(a * logit(p) + b)` with `p` clamped to [1e-6, 1 - 1e-6], exactly
what a registered `calibration` applies, so it runs on the stored `p`.

```sql
-- the demo has 8 rows per model: only to show the calls (split on ids here; with real data hash(id) % 2 = 0)
SET VARIABLE spec = (SELECT decide_fit_calibration(p, refund) FROM runs WHERE model = 'model-b' AND id <= 4);
SELECT getvariable('spec') AS spec;                                                  -- 'platt:a,b'

-- held-out check: raw vs calibrated on the other half; Brier and ECE should drop, AUC stays
WITH ab AS (SELECT string_split(replace(getvariable('spec'), 'platt:', ''), ',') AS v),
held AS (SELECT refund, p, greatest(least(p, 1 - 1e-6), 1e-6) AS q
         FROM runs WHERE model = 'model-b' AND id > 4)
SELECT round(decide_brier_score(p, refund), 3) AS brier_raw,
       round(decide_brier_score(1 / (1 + exp(-(v[1]::DOUBLE * ln(q / (1 - q)) + v[2]::DOUBLE))), refund), 3) AS brier_cal,
       round(decide_ece(p, refund), 3) AS ece_raw,
       round(decide_ece(1 / (1 + exp(-(v[1]::DOUBLE * ln(q / (1 - q)) + v[2]::DOUBLE))), refund), 3) AS ece_cal
FROM held, ab;
```

When the check passes, refit on all rows of **that** model and register the spec on a variant of **the
same** model. A calibration never carries over to another model or another kind of data:

```sql
-- needs: a Liquid key and runs holding d1's probabilities on 50+ labelled rows (built as in the section above)
SET anofox_decide_allow_remote = true;
SET VARIABLE d1_spec = (SELECT decide_fit_calibration(p, refund) FROM runs WHERE model = 'd1');   -- 1. fit
SELECT decide_register_model('d1-cal', 'liquid',
                             MAP {'model': 'd1', 'calibration': getvariable('d1_spec')});       -- 2. register
SELECT calibration FROM decide_models() WHERE model = 'd1-cal';                                 -- 3. check
-- every yes/no probability of 'd1-cal' is now calibrated. Hosted models take the MAP as the 3rd argument,
-- a model registered from your own ONNX files as the 6th (see anofox-decide-local)
```

In the project's evaluation, calibration cut D1's refund error from 23% to 9%.

## Picking a threshold for `decide_decision`

```sql
-- accuracy at each cut-off, per model; take the best (or the one that meets a precision / recall target)
SELECT model, t AS threshold, round(decide_accuracy(p, refund, t), 3) AS accuracy
FROM runs, (SELECT unnest([0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]) AS t)
GROUP BY model, t ORDER BY model, accuracy DESC, t;

-- precision and recall at one cut-off
SELECT model,
       count(*) FILTER (WHERE p >= 0.8 AND refund) / nullif(count(*) FILTER (WHERE p >= 0.8), 0) AS precision,
       count(*) FILTER (WHERE p >= 0.8 AND refund) / nullif(count(*) FILTER (WHERE refund), 0)   AS recall
FROM runs GROUP BY model;
```

Then use it explicitly: `decide_decision(text, 'A refund is requested.', 0.8, model := '...')`.

## Abstaining on choice questions

```sql
-- route only when the winning option has at least 60% probability; NULL = send to a human
-- (on stub the first option always wins with 1.0: a real model is needed to see a NULL)
SELECT id,
       CASE WHEN list_max(map_values(d)) >= 0.6
            THEN list_extract(map_keys(d), list_position(map_values(d), list_max(map_values(d)))) END AS team
FROM (SELECT id, decide_choice_distribution(text, 'Which team owns this?', ['billing', 'defect', 'other'],
                                            model := 'stub') AS d
      FROM (VALUES (1, 'I was charged twice.')) t(id, text));
```

Reference pages: [Evaluate and calibrate](https://github.com/DataZooDE/anofox-decide/blob/main/docs/guides/03-evaluate-and-calibrate.md),
[Choosing a model](https://github.com/DataZooDE/anofox-decide/blob/main/docs/guides/02-choosing-a-model.md),
[Evaluation](https://github.com/DataZooDE/anofox-decide/blob/main/docs/EVALUATION.md).
