# anofox-decide

DuckDB extension evaluating natural-language predicates and runtime-defined answer sets against text or structured state. Modeled after `../anofox-tabfm` (same DuckDB pin, extension-ci-tools harness, red/green TDD, sqllogictest + Catch2 DoD).

```sql
LOAD anofox_decide;
SELECT decide_probability('The customer requests a refund.', 'A refund is requested.', model := 'stub');
SELECT decide_choice('Primary issue?', ['billing','defect','other'], model := 'stub');
SELECT * FROM decide_models();
```

## Status

Scaffold stage (stub provider, deterministic 0.5). See [docs/SPIKE_RESULTS.md](docs/SPIKE_RESULTS.md) for the pinned provider contract: TypeSafe `POST /v1/systemone` remote + open local decision models (Julia-1 first).

## Build & test

```bash
make init      # duckdb v1.5.5 + extension-ci-tools v1.5-variegata submodules
make release
make test_debug
./build/debug/test/unittest test/sql/decide_contract.test   # single file
```

Tests always run with `DATAZOO_DISABLE_TELEMETRY=1` (Makefile does this).
