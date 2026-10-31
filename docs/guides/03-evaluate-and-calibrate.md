# Evaluate and calibrate

A decision model answers with probabilities, so you can measure how good they are on labelled data, plot
them per model, and fix a model's cut-off before you act on it.

## Metrics

The metric aggregates work on any `(probability, label)` pairs, so comparing models is one query:

```sql
SELECT model,
       decide_accuracy(p, refund)    AS accuracy,
       decide_brier_score(p, refund) AS brier,
       decide_ece(p, refund)         AS ece
FROM runs GROUP BY model ORDER BY brier;     -- see examples/04_compare_models.sql
```

| Aggregate | Measures |
|---|---|
| `decide_accuracy(p, outcome[, threshold])` | share of rows where thresholding `p` reproduces the label |
| `decide_brier_score(p, outcome)` | mean squared error of the probabilities; 0 is perfect, 0.25 a constant 0.5 |
| `decide_ece(p, outcome)` | expected calibration error over ten equal-width bins |
| `decide_log_loss(p, outcome)` | mean binary cross-entropy, `p` clipped to [1e-15, 1 - 1e-15] |
| `decide_auc(p, outcome)` | area under the ROC curve, ties count one half; NULL with one outcome only |

`decide_accuracy` thresholds at 0.5 unless you pass a threshold. NULL rows are skipped and empty input
returns NULL; probabilities outside 0 to 1, NaN, or labels that are not 0/1 raise an error that says what
to fix. `decide_auc` and `decide_log_loss` complete the set. How to build the `runs` table from several
models: [Comparing models on your own data](02-choosing-a-model.md#comparing-models-on-your-own-data).

## Plot with anofox-visualization

`decide_roc_curve_by`, `decide_calibration_curve_by` and `decide_metrics_by` return the long tables of the
anofox integration contract (the `curve` and `summary` schemas), so
[anofox-visualization](https://github.com/DataZooDE/anofox-visualization) plots them with no reshaping. Each
takes a table name, the column that tells the models (or groups) apart, the probability and the outcome:

```sql
LOAD anofox_visualization;
SELECT anofox_plot(t) FROM decide_roc_curve_by('runs', model, p, refund) t;          -- one ROC curve per model
SELECT anofox_plot(t) FROM decide_calibration_curve_by('runs', model, p, refund) t;  -- reliability diagram
SELECT anofox_plot(t) FROM decide_metrics_by('runs', model, p, refund) t;            -- model comparison
```

`decide_calibration_curve_by` takes an optional `n_bins := 10` and returns Wilson intervals per bin;
`decide_metrics_by` returns accuracy, auc, brier, ece and log loss per group, one row each.

The curves have the columns of anofox-statistics' `roc_curve_by` / `calibration_curve_by`, so they
`UNION ALL BY NAME` with them into one plot. Every producer ends with `row_order`, the contract's optional
1-based ordinal: the order the rows come out in (along the curve, by bin, metrics alphabetically), so a
consumer that aggregates the rows can keep it. The calibration curve bins on P(yes) (`floor(p * n_bins)`, the edges
`decide_ece` uses); `decide_ece` itself bins on the confidence of the decided answer, so the two are related but
not the same number. The producers carry `duckdb_functions()` tags (`anofox.contract = '1'`,
`anofox.output = 'curve'` / `'summary'`), every decide function `anofox.family = 'decide'`, and
`anofox_contract_version()` returns `'1'`.

## Calibration

A model's yes/no probability is only as good as its cut-off. Some models say "yes" too often at 0.5 (Liquid
D1 flagged every complaint as a refund request) and others too rarely. Per-model **Platt scaling**
(`p' = sigmoid(a * logit(p) + b)`, `a > 0`, so the ranking never changes) fixes the cut-off and the
sharpness for yes/no questions; choice and score answers are untouched. Fit it from labelled data, register
the model with it:

```sql
-- 1. fit on a labelled sample of that model's raw probabilities
SELECT decide_fit_calibration(p, y) FROM labelled;                      -- 'platt:1.92,-0.72'
-- 2. register the model with the fitted spec (any provider; local models take the MAP as 6th argument)
SELECT decide_register_model('d1-cal', 'liquid', MAP {'model': 'd1:free', 'calibration': 'platt:1.92,-0.72'});
SELECT decide_register_model('laya-cal', 'local', '/m/laya.onnx', '/m/tokenizer.json', 'laya', MAP {'calibration': 'platt:1.1,-0.3'});
-- 3. every yes/no probability from that model (decide_probability, decide_decision, decide_many, decide_table) is calibrated
SELECT calibration FROM decide_models();                                   -- the spec in use, NULL when none
```

`decide_fit_calibration` needs both outcomes and non-constant probabilities, ignores NULL rows, and refuses a
model whose probabilities rank outcomes no better than chance (a negative slope would invert it). Fit on one
sample and check on another with `decide_brier_score` and `decide_ece`; about 50 or more labelled rows give a
stable fit. On the 200-ticket evaluation, fitting on one half and scoring the other (20 random splits)
cut D1's refund error from 23% to 9% and its Brier score from 0.157 to 0.065; see
[docs/EVALUATION.md](../EVALUATION.md#calibration) (`tools/eval/crossfit_calibration.py` reproduces it).
A calibration is fitted to one data distribution: refit it for your own tickets.

## Abstaining on choice questions

`decide_choice_distribution(state, question, options[, model])` returns the probability of every option as a
`MAP(VARCHAR, DOUBLE)` (it sums to 1), so you can threshold the winner's probability and abstain in plain SQL:

```sql
SELECT body, decide_choice_distribution(body, 'Which team owns this?', ['billing','defect','other'], model := 'jev-latest') AS dist
FROM tickets;
-- abstain below 60% confidence
SELECT CASE WHEN list_max(map_values(d)) >= 0.6 THEN list_extract(map_keys(d), list_position(map_values(d), list_max(map_values(d)))) END AS team
FROM (SELECT decide_choice_distribution(body, 'Which team owns this?', ['billing','defect','other'], model := 'jev-latest') AS d FROM tickets);
```

More on the calibration and performance measurements: [docs/CALIBRATION_PERF.md](../CALIBRATION_PERF.md).
