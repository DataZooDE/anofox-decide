# Examples

Runnable SQL for `anofox_decide`. Run one from the repository root (the data paths are relative to it):

```bash
duckdb -unsigned < examples/01_quickstart.sql
```

(`-unsigned` loads the extension from a local build; with the extension installed from the
community repository, plain `duckdb` works.)

| File | What it shows | Runs offline |
|---|---|---|
| [`01_quickstart.sql`](01_quickstart.sql) | The whole SQL surface (binary, choice, score, decision, table, many) on the built-in test model | yes |
| [`02_support_triage.sql`](02_support_triage.sql) | Refund, routing and frustration for tickets, then accuracy / Brier / calibration against labels | yes (stub) |
| [`03_scoring_many_rows.sql`](03_scoring_many_rows.sql) | Scoring a table: concurrency, retries, timeouts, and which SQL shape is fast | yes (stub) |
| [`04_compare_models.sql`](04_compare_models.sql) | Several hosted models on the same labelled tickets, side by side | no (keys) |
| [`05_local_model.sql`](05_local_model.sql) | A local ONNX decision model, fully offline after setup | no (your graph) |

The offline examples start on the built-in `stub` model, which returns constants: they check that your
SQL is right, not that an answer is. Swap in a real model with the lines marked `REAL MODEL` at the top of
each file. `test/cpp/test_decide_docs.cpp` runs the offline examples on every build and parses the others,
so they cannot drift from the extension.

`data/support_tickets.csv` is a small hand-written set of eight labelled tickets (refund yes/no, owning
team). For a real comparison use a few hundred of your own; the 200-ticket evaluation in
[`docs/EVALUATION.md`](../docs/EVALUATION.md) shows how.
