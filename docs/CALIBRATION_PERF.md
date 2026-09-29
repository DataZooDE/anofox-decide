# AnoFox Decide — calibration and performance report (BRD §6, Next scope)

Date: 29 Sept 2026. Machine: 32× AMD Ryzen 9 3950X, 125 GB RAM.
Build: release, DuckDB v1.5.5, `CMAKE_PREFIX_PATH` at tabfm's ORT tree.
Method: single-user `EXPLAIN ANALYZE`, median of 5 runs unless noted.
Telemetry disabled (`DATAZOO_DISABLE_TELEMETRY=1`).

> Remote (TypeSafe) numbers are intentionally absent: this sandbox has no
> outbound HTTPS, so no latency/accuracy claim is made for `jev-latest`.
> Use `make test-live` (needs `TYPESAFE_API_KEY`) plus the SQL recipes below
> on a networked machine to fill that column in.

## 1. Calibration: SQL recipe + measured values

Recipe (held-out labeled table with a boolean outcome column):

```sql
SELECT decide_brier_score(decide_probability(s, '<proposition>', model := '<m>'), y),
       decide_ece(decide_probability(s, '<proposition>', model := '<m>'), y),
       decide_accuracy(decide_probability(s, '<proposition>', model := '<m>'), y, <threshold>)
FROM labeled;
```

`decide_accuracy` without a threshold decides at a documented 0.5 boundary
(same `p >= t` rule as `decide_decision`); pass an explicit threshold to
validate operating points on held-out data.

Measured on an 8-row balanced toy set (4 refund / 4 not-refund,
proposition `'A refund is requested.'`):

| model | Brier ↓ | ECE ↓ | accuracy @0.5 ↑ |
|---|---|---|---|
| `stub` | 0.25 | 0.0 | 0.5 |
| `tiny-local` (random-init fixture) | 0.2500 | 0.3750 | 0.125 |

Reading: the stub scores a constant 0.5, so Brier is exactly 0.25 and
accuracy is chance — the aggregates behave per definition. `tiny-local` uses
the committed **random-weight** fixture (`test/fixtures/julia1_tiny.onnx`),
so chance-level-or-worse numbers are expected: they prove the pipeline runs
end to end, not that Julia-1 is a good model. Real calibration claims
(Brier/ECE by model and task, reliability plots, thresholds validated on
held-out data) require the gated real-weight runs (`JULIA_ONNX` +
`JULIA_WEIGHTS_DIR`) plus agreed labeled benchmarks — explicitly out of
scope for this report.

## 2. Performance (local providers, CPU)

State ≈ 100 chars; questions `{"id","kind":"binary","instruction"}` ≈ 60 chars.

| workload | median total | per call | per question |
|---|---|---|---|
| `decide_probability`, stub, 200k rows | 0.454 s | 2.3 µs/row | — (≈440k rows/s) |
| `decide_many`, stub, 1 q × 200 calls | 0.0095 s | 48 µs | 48 µs |
| `decide_many`, stub, 10 q × 200 calls | 0.0076 s | 38 µs | 3.8 µs |
| `decide_many`, stub, 100 q × 50 calls | 0.0016 s | 32 µs | 0.32 µs |
| `decide_probability`, tiny-local, 50 rows | 0.0011 s | 22 µs/row | — |
| `decide_many`, tiny-local, 10 q × 10 calls | 0.0010 s | 100 µs | 10 µs |
| `decide_table`, tiny-local, 10 q (1 scan) | 0.0009 s | 900 µs/scan | 90 µs |

Notes:

- Stub cost is per-row fixed overhead (`GetValue` × N, registry lookup,
  JSON parse/serialize of the batch envelope) — per-question marginal cost
  is sub-microsecond. The scalar row path (`decide_probability`,
  2.3 µs/row) is ~20× cheaper per row than the batch-JSON path, so
  single-question enrichment over large tables should use the scalars;
  `decide_many`/`decide_table` pay off at ≥10 questions/row by sharing one
  parse and (for remote) one request.
- `tiny-local` is ~10× the stub per row (BPE tokenize + micro-graph ORT
  run). Real Julia-1 weights (144M params) will be orders of magnitude
  slower than this fixture — re-run this table after export before quoting
  local-model SLAs.
- `decide_table` evaluates once per scan (bind parses, init-global scores,
  scan emits rows): batching N questions over one state costs one provider
  round trip for remote, one ORT session run per row for local.

## 3. Reliability (no partial success as complete)

Covered by contract tests, not numbers: `test/sql/decide_table.test`
(zero rows on NULL, actionable errors naming `decide_table`),
`test/sql/decide_decision.test` (NULL → NULL, explicit threshold),
`test/sql/decide_calibration.test` (NULL skip, NULL-on-empty, finite
probabilities only). Remote retry policy (429/529/5xx, bounded backoff,
never 401/422) is pinned by the hermetic Catch2 `[remote]` suite.
