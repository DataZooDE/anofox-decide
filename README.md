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

## Local models

Julia-1, fully offline after setup:

```sql
SELECT decide_register_model('julia-1', 'local', '<path>/julia1.onnx', '<path>/tokenizer/tokenizer.json');
SELECT decide_probability('...', '...', model := 'julia-1');
```

Local models take an optional 5th argument, the **profile**: `julia-1`
(default) or `laya`. The profile fixes how questions are rendered for that
model family, the sequence limits, and calibration. Laya
([convaiinnovations/laya](https://huggingface.co/convaiinnovations/laya),
Apache 2.0, ModernBERT / mmBERT encoders) needs `rl_agent_config.json` next
to the graph (limits + calibration temperatures):

```sql
SELECT decide_register_model('laya', 'local', '<dir>/julia1.onnx',
                             '<snapshot>/multilingual/tokenizer/tokenizer.json', 'laya');
```

Export any of the marker-head checkpoints with `tools/export_julia`
(`python -m export_julia.export --spec <spec.json> --weights <snapshot subdir> --out <dir>`;
the weights dir holds `encoder/config.json` + `model.safetensors`; the
spec needs `{"head": {"head_layers": 2, "dropout": 0.1}}`) and copy the
checkpoint's `rl_agent_config.json` beside the graph. Tokenizers: the
SentencePiece-style one (Julia-1, Laya multilingual) and ByteLevel BPE (Laya
English / typed-decisions) are both supported.

Measured on 8 labelled support tickets (`test/fixtures/support_tickets.csv`),
correct refund / team:

| Model | Refund | Team |
|---|---|---|
| TypeSafe Jev (remote) | 8/8 | 7/8 |
| Laya multilingual (local) | 8/8 | 7/8 |
| Laya typed-decisions (local) | 7/8 | 6/8 |
| Kev-0.8B (local server, remote provider) | 6/8 | 6/8 |
| Julia-1 (local) | 4/8 | 4/8 |

[Kev](https://github.com/jaredpalmer/kev) (Qwen3.5 + LoRA, Apache 2.0) serves
TypeSafe's `/v1/systemone` contract, so it needs no extra code: run
`python -m kev.serve --run jaredpalmer/kev-0.8b --port 8009`, then
`SET anofox_decide_endpoint='http://127.0.0.1:8009'; SET anofox_decide_api_key='local';`
and register `decide_register_model('kev-latest', 'typesafe')`. The endpoint
setting is per session, so use one DuckDB connection per endpoint if you
mix Jev and Kev. Kev-4B/9B/27B are more accurate than 0.8B and need a GPU
or a large Mac.

Each local ONNX model reproduces its upstream Python reference to within 5e-5
in probability on these tickets. Eight tickets is a smoke test, not a
benchmark: evaluate on your own data.

## Status

Early (0.1): the SQL surface, remote and local providers, calibration
metrics and CI are working. What you can use today:

| Provider | Models | Questions | Runs where |
|---|---|---|---|
| `stub` | deterministic placeholder (0.5 / first option) | `binary`, `choice` | in-process, for tests and demos |
| `typesafe` (remote) | Jev (`jev-latest`), or any server speaking TypeSafe's `/v1/systemone`, e.g. [Kev](https://github.com/jaredpalmer/kev) | `binary`, `choice` | API call; opt-in via `anofox_decide_allow_remote` |
| `local` (ONNX Runtime) | Julia-1, Laya multilingual, Laya typed-decisions (profiles `julia-1`, `laya`) | `binary`, `choice` | in-process on CPU, fully offline after setup |

Not supported yet: `score` (ordinal) questions, GPU execution, and the
Von model (needs order-invariant attention in the export).

Which local model to pick: Laya multilingual matched Jev on our 8-ticket
smoke test and Julia-1 did not (see the table above), so it is the
recommended local model. Model weights are not shipped in the repo; see
[Local models](#local-models) for setup.

CI builds and tests Linux (amd64, arm64), macOS (arm64) and Windows (amd64)
on every push, then smoke-tests the shipped `.duckdb_extension` in the stock
DuckDB CLI. Releases (`v*` tags) upload through the deploy workflow once the
org's AWS role trusts this repository.

Further reading: [docs/SPIKE_RESULTS.md](docs/SPIKE_RESULTS.md) (provider
contracts), [docs/julia-1-spec.json](docs/julia-1-spec.json) (pinned Julia-1
spec), [docs/CALIBRATION_PERF.md](docs/CALIBRATION_PERF.md) (calibration and
performance), [docs/REVIEW_FOLLOWUP.md](docs/REVIEW_FOLLOWUP.md) (review
decisions and the model investigations).

## Build & test

Clone with submodules (`git clone --recurse-submodules`), then:

```bash
make release                          # release build; ONNX Runtime built via vcpkg (first build is slow)
make release DECIDE_ORT_VCPKG=0 CMAKE_PREFIX_PATH=<ort-install>   # faster local build against an existing ONNX Runtime
make test_release                     # offline suite: SQL tests + Catch2 (no network, no weights)
make test-live                        # live TypeSafe E2E; explicit SKIP without TYPESAFE_API_KEY
./build/release/test/unittest test/sql/decide_contract.test       # a single file
```

Some tests need model weights or network and warn-and-pass without them:
Catch2 `[tokenizer]` goldens (`JULIA_WEIGHTS_DIR`, `LAYA_TYPED_DIR`), `[local]`
real graphs (`JULIA_ONNX`, `LAYA_MULTILINGUAL_DIR` + `LAYA_ONNX`,
`LAYA_TYPED_DIR` + `LAYA_TYPED_ONNX`), and the export pytest in
`tools/export_julia` (`JULIA_WEIGHTS_DIR`, `JULIA_SRC_MODEL`, `JULIA_ONNX`).
Run the suite with `TYPESAFE_API_KEY` unset, as CI does, so tests do not
depend on your own key.

Tests always run with `DATAZOO_DISABLE_TELEMETRY=1` (Makefile does this).
