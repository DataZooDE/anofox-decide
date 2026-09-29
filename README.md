# anofox-decide

DuckDB extension evaluating natural-language predicates and runtime-defined answer sets against text or structured state. Modeled after `../anofox-tabfm` (same DuckDB pin, extension-ci-tools harness, red/green TDD, sqllogictest + Catch2 DoD).

```sql
LOAD anofox_decide;
SELECT decide_probability('The customer requests a refund.', 'A refund is requested.', model := 'stub');
SELECT decide_choice('Primary issue?', ['billing','defect','other'], model := 'stub');
SELECT decide_decision('The customer requests a refund.', 'A refund is requested.', 0.7, model := 'stub');
SELECT * FROM decide_table('The bill is wrong.',
  '[{"id":"refund","kind":"binary","instruction":"A refund is requested."}]', 'stub');
SELECT * FROM tickets, LATERAL (SELECT * FROM decide_table(tickets.body, '[...]', 'stub')) dt;
SELECT decide_brier_score(p, y) FROM labeled;  -- + decide_ece / decide_accuracy(p, y[, threshold])
SELECT * FROM decide_models();
```

Remote (TypeSafe) usage:

```sql
SET anofox_decide_allow_remote=true;  -- explicit opt-in, default off
SELECT decide_register_model('jev-latest', 'typesafe');
SELECT decide_probability('...', '...', model := 'jev-latest');  -- key from TYPESAFE_API_KEY
```

Key management (preferred first): a stored secret (write-only, redacted in
`duckdb_secrets()`, optionally scoped to an endpoint host) beats the legacy
setting, which stays readable via `current_setting`:

```sql
CREATE SECRET (TYPE anofox_decide, API_KEY 'sk-...');                     -- preferred
CREATE SECRET (TYPE anofox_decide, API_KEY 'sk-...', SCOPE 'custom.host'); -- per-endpoint
SET anofox_decide_api_key='sk-...';  -- legacy override, visible in settings
```

Trust note: `TYPESAFE_API_KEY` attaches only to the default endpoint
(`https://api.typesafe.ai`); a custom `anofox_decide_endpoint` needs a
secret or the explicit setting, and cleartext `http://` works for
loopback hosts only.

Local NLI (Julia-1, fully offline after setup):

```sql
SELECT decide_register_model('julia-1', 'local', '<path>/julia1.onnx', '<path>/tokenizer/tokenizer.json');
SELECT decide_probability('...', '...', model := 'julia-1');
```

## Status

Providers: `stub` (deterministic) + `typesafe` remote (DuckDB-bundled httplib/yyjson, no vcpkg)
+ `local` Julia-1 (ORT + hand-ported BPE tokenizer, collation mirrors `julia/data.py`).
See [docs/SPIKE_RESULTS.md](docs/SPIKE_RESULTS.md) for the pinned contracts,
[docs/julia-1-spec.json](docs/julia-1-spec.json) for the pinned model spec, and
[docs/CALIBRATION_PERF.md](docs/CALIBRATION_PERF.md) for the calibration/perf
report (BRD §6: recipes, measured stub + fixture numbers, reliability notes).

## Build & test

```bash
make release VCPKG_TOOLCHAIN_PATH= CMAKE_PREFIX_PATH=<ort-install>
# e.g. CMAKE_PREFIX_PATH=../anofox-tabfm/build/release/vcpkg_installed/x64-linux
# (vcpkg disabled: this sandbox has no vcpkg write; ORT comes from a package
# or the cmake/ort.cmake prebuilt download)
make test_debug                      # offline suite: contract + registry + local + decision + table + calibration + remote-config + Catch2
make test-live                       # live TypeSafe E2E; explicit SKIP without TYPESAFE_API_KEY
./build/release/test/unittest test/sql/decide_contract.test   # single file
```

Gated (need weights/network, WARN-and-pass without): Catch2 `[tokenizer]` goldens
(`JULIA_WEIGHTS_DIR`), `[local]` real graph (`JULIA_ONNX`), export pytest
(`tools/export_julia`: `JULIA_WEIGHTS_DIR`/`JULIA_SRC_MODEL`/`JULIA_ONNX`).

Tests always run with `DATAZOO_DISABLE_TELEMETRY=1` (Makefile does this).
