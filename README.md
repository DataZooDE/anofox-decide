# anofox-decide

DuckDB extension evaluating natural-language predicates and runtime-defined answer sets against text or structured state. Modeled after `../anofox-tabfm` (same DuckDB pin, extension-ci-tools harness, red/green TDD, sqllogictest + Catch2 DoD).

```sql
LOAD anofox_decide;
SELECT decide_probability('The customer requests a refund.', 'A refund is requested.', model := 'stub');
SELECT decide_choice('Primary issue?', ['billing','defect','other'], model := 'stub');
SELECT decide_score('Help! My payouts failed for 3 days!', 'How frustrated is the writer?', ['calm','frustrated','angry'], model := 'stub');  -- expected level, 0-based
SELECT decide_decision('The customer requests a refund.', 'A refund is requested.', 0.7, model := 'stub');
SELECT * FROM decide_table('The bill is wrong.',
  '[{"id":"refund","kind":"binary","instruction":"A refund is requested."},
    {"id":"mood","kind":"score","instruction":"How frustrated?","levels":["calm","frustrated","angry"]}]', 'stub');
-- columns: question_id, kind, probability, choice, confidence, model, score, distribution
SELECT * FROM tickets, LATERAL (SELECT * FROM decide_table(tickets.body, '[...]', 'stub')) dt;
SELECT decide_brier_score(p, y) FROM labeled;  -- + decide_ece / decide_accuracy(p, y[, threshold])
SELECT * FROM decide_models();
```

## First run

There is no implicit model: a call that names no model (and with no `SET anofox_decide_model`) fails with
instructions instead of quietly returning a constant, so a test model can never pass for a real answer. The
fastest way to see where you stand:

```sql
SELECT * FROM decide_doctor();   -- ok / warn / fail per check, with the fix for each problem
SELECT model, ready, hint FROM decide_models();   -- what is registered, whether it can be called, and why not
```

Typical first steps: register a model (see [Remote providers](#remote-providers) or
[Local models](#local-models)), then name it per call (`model := 'jev-latest'`) or set it once
(`SET anofox_decide_model = 'jev-latest'`). `model := 'stub'` is a built-in test model that returns constants
(probability 0.5, the first option) for trying the SQL surface. `decide_unregister_model('id')` removes a model
so it can be registered again with different settings.

## Remote providers

Remote models send `state` to the provider's API, so they sit behind an
explicit opt-in. Providers are profiles over the same System One wire format
(`noul` / `choice` / `score`); each has its own endpoint, path and key variable, so
several can be used side by side in one session with no `SET` in between:

| Provider | Service | Default endpoint | Key env var |
|---|---|---|---|
| `typesafe` | TypeSafe Jev | `https://api.typesafe.ai` | `TYPESAFE_API_KEY` |
| `liquid` | Liquid AI D1 | `https://api.liquid.ai` (`/decisions/v1/systemone`) | `LIQUID_API_KEY` |
| `systemone` | any compatible server, e.g. [Kev](https://github.com/jaredpalmer/kev) | you set it | none (see `key_env`) |
| `strands` | [strands-decider](https://github.com/strands-labs/strands-decider) local server | `http://127.0.0.1:8000` | none (keyless) |

```sql
SET anofox_decide_allow_remote=true;  -- explicit opt-in, default off
SELECT decide_register_model('jev-latest', 'typesafe');
SELECT decide_register_model('d1:free', 'liquid');            -- the id is the model name on the wire
SELECT decide_probability('...', '...', model := 'jev-latest');
SELECT decide_probability('...', '...', model := 'd1:free');
```

Per-model options go in a `MAP` (keys `endpoint`, `path`, `model`, `key_env`,
`criteria`), e.g. a self-hosted server, a different wire model name, an explicit
key variable for a custom endpoint, or `criteria: 'name'` for servers whose
schema needs a string description per choice option instead of `null`:

```sql
SELECT decide_register_model('kev-latest', 'systemone', MAP {'endpoint': 'http://127.0.0.1:8009'});
SELECT decide_register_model('d1', 'liquid', MAP {'model': 'd1:free'});
SELECT decide_register_model('mine', 'systemone', MAP {'endpoint': 'https://llm.internal', 'key_env': 'MY_KEY_VAR'});
```

**API keys.** Environment variables work out of the box (`export
LIQUID_API_KEY=...`), and a DuckDB secret always overrides them. Precedence,
highest first:

1. A stored secret (write-only, redacted in `duckdb_secrets()`), matched by the endpoint host:
   ```sql
   CREATE SECRET (TYPE anofox_decide, API_KEY 'sk-...', SCOPE 'api.liquid.ai');
   CREATE SECRET (TYPE anofox_decide, API_KEY getenv('LIQUID_API_KEY'), SCOPE 'api.liquid.ai');  -- copy from the env
   ```
2. `anofox_decide_api_key` (legacy, `typesafe` provider only; visible via `current_setting`).
3. The model's explicit `key_env` variable.
4. The provider's own env var (table above), only on that provider's default host.

A key is never sent to a host it was not configured for: a redirected
endpoint gets nothing from the environment, and a secret scoped to one host
is not used for another. Cleartext `http://` works for loopback hosts only.
`anofox_decide_endpoint` remains as a legacy setting for the `typesafe`
provider.

Kev (Qwen3.5 + LoRA, Apache 2.0) is a local server, not an in-process model:
run `python -m kev.serve --run jaredpalmer/kev-0.8b --port 8009`, then register
it as above and give it any key (it ignores the value but the provider requires
one), e.g. `CREATE SECRET (TYPE anofox_decide, API_KEY 'local', SCOPE '127.0.0.1')`.
Kev-4B/9B/27B are more accurate than 0.8B and need a GPU or a large Mac.

[strands-decider](https://github.com/strands-labs/strands-decider) (Apache 2.0, a 1.9B
Qwen3.5 torso with a pointer head) is also a local server. It needs no key and its
schema requires a string per choice option, which the `strands` profile handles
(a choice question may have at most 24 options on this model). It listens on
loopback without authentication, so keep it local:

```bash
python3.12 -m venv .venv && . .venv/bin/activate
pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cpu   # or a CUDA/MPS build
pip install strands-decider
strands-decider serve StrandsAgents/strands-decider-2B-hobson-v19 --device cpu --port 8000   # first run downloads ~4.5 GB
```
```sql
SET anofox_decide_allow_remote=true;
SELECT decide_register_model('strands-decider', 'strands');   -- default endpoint http://127.0.0.1:8000
SELECT decide_choice('Help! My payouts have been failing.', 'Which team?', ['billing','sales','retail'], model := 'strands-decider');
```

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

## Model comparison

Measured on 200 labelled tickets from the public Bitext customer-support dataset (60 refund requests,
140 not; routing into billing / orders / account / other). Full method, caveats and per-class results:
[docs/EVALUATION.md](docs/EVALUATION.md).

| Model | Runs | Refund accuracy | Refund AUROC | Routing accuracy |
|---|---|---|---|---|
| TypeSafe Jev (`typesafe`) | hosted API | 93.0% | 0.984 | 92.0% |
| Laya multilingual | local, in-process | 92.0% | 0.970 | 62.0% |
| Liquid D1 (`liquid`) | hosted API | 77.0% | 0.970 | 91.0% |
| Laya typed-decisions | local, in-process | 88.5% | 0.978 | 76.0% |
| strands-decider 2B (`strands`) | local server | 81.0% | 0.918 | 65.0% |
| Kev-0.8B (`systemone`) | local server | 70.5% | 0.850 | 44.0% |
| Julia-1 | local, in-process | 29.5% | 0.407 | 37.5% |

Always-no scores 70% on refund and always-billing 45% on routing. Jev is the most accurate overall; strands-decider is
conservative on refund (recall 48% at the 0.5 cut-off) and a mid-table router; D1
routes as well as Jev, but at the 0.5 cut-off over-predicts refunds (its ranking is fine, AUROC 0.970);
Laya multilingual matches Jev on refund and is the best in-process model, with typed-decisions the best
in-process router. English-only, template-generated data: a smoke test of relative strength, not a
benchmark of your tickets, so evaluate on your own data (`tools/eval` reproduces this run).

Each local ONNX model reproduces its upstream Python reference to within 5e-5 in probability on a small
check set; the hosted models are called as-is.

## Status

Early (0.1): the SQL surface, remote and local providers, calibration
metrics and CI are working. What you can use today:

| Provider | Models | Questions | Runs where |
|---|---|---|---|
| `stub` | deterministic placeholder (0.5 / first option) | `binary`, `choice`, `score` | in-process, for tests and demos |
| `typesafe` (remote) | TypeSafe Jev (`jev-latest`) | `binary`, `choice`, `score` | hosted API; opt-in via `anofox_decide_allow_remote` |
| `liquid` (remote) | Liquid AI D1 (`d1:free`) | `binary`, `choice`, `score` | hosted API; same opt-in |
| `systemone` (remote) | any TypeSafe-compatible server, e.g. [Kev](https://github.com/jaredpalmer/kev) | `binary`, `choice`, `score` | your endpoint (loopback `http://` or `https://`) |
| `strands` (remote) | [strands-decider](https://github.com/strands-labs/strands-decider) 2B | `binary`, `choice`, `score` | local server on loopback, no key |
| `local` (ONNX Runtime) | Julia-1, Laya multilingual, Laya typed-decisions (profiles `julia-1`, `laya`) | `binary`, `choice`, `score` | in-process on CPU, fully offline after setup |

Question kinds: `binary` (yes/no, `decide_probability` / `decide_decision`), `choice` (one of N,
`decide_choice`) and `score` (ordinal: an ordered rubric of 2 to 10 level descriptions, lowest first;
`decide_score` returns the expected 0-based level, and `decide_many` / `decide_table` also return the
per-level probabilities). `score` is verified for parity with each model's upstream implementation and
against the live services; unlike refund and routing it has not been scored for accuracy on labelled data.
Not supported yet: GPU execution and the Von model (needs order-invariant attention in the export).

Which model to pick: Jev was the most accurate in our 200-ticket evaluation. Among models that run
in-process, Laya multilingual matched it on refund detection and Laya typed-decisions routes best
(see the comparison above); there is no single best local model yet. Model weights are not shipped in the repo; see
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

## Telemetry

Sends **anonymous** usage telemetry (extension load and per-function call
counts; no state, questions, answers, model ids, endpoints, keys or SQL) through
the shared [`posthog-telemetry`](https://github.com/DataZooDE/posthog-telemetry)
library, exactly as [anofox-tabfm](https://github.com/DataZooDE/anofox-tabfm)
does. Turn it off with any of:

```bash
export DATAZOO_DISABLE_TELEMETRY=1        # environment
```
```sql
SET anofox_telemetry_enabled = false;      -- SQL
```

CI environments are auto-detected and telemetry is disabled there. The full
list of what is collected is in [TELEMETRY.md](TELEMETRY.md). The load banner
(once a day, terminal only, never in CI or pipes) is silenced with
`SET datazoo_banner = false` or `DATAZOO_NO_BANNER=1`.

## Troubleshooting

Every error follows one shape, `<function>: <what went wrong>. Fix: <what to run>`, and echoes the value that
caused it. Run `SELECT * FROM decide_doctor();` first when something does not work.

| You see | It means | Do this |
|---|---|---|
| `no model selected` | the call names no model and no default is set | `decide_probability(..., model := '<id>')` or `SET anofox_decide_model = '<id>'`; register one first with `decide_register_model` |
| `model '<id>' is not registered (given by ...)` | the id is a typo, or never registered, or removed | the message lists the registered ids and a close match; `SELECT * FROM decide_models()` |
| `model '<id>' is a ... model: calling it sends your text to ..., and remote calls are off` | remote models send your text to an endpoint and are off by default | `SET anofox_decide_allow_remote = true;` |
| `no API key for ... at <host>` | no secret, setting or env var supplies a key | export the provider's variable (`LIQUID_API_KEY`, `TYPESAFE_API_KEY`) or `CREATE SECRET (TYPE anofox_decide, API_KEY '...', SCOPE '<host>')`; a keyless local server uses provider `strands` |
| `... rejected the API key (HTTP 401)` | the service refused the key; the message says where the key came from | replace it: `CREATE OR REPLACE SECRET (TYPE anofox_decide, API_KEY '<key>', SCOPE '<host>')` (a secret overrides env vars) |
| `... does not know the model '<x>' ... (HTTP 404)` | the model name sent to the service is wrong (it is the registered id unless you set one) | `decide_register_model('<id>', '<provider>', MAP {'model': '<provider model name>'})` |
| `... rejected the request (HTTP 422)` | the service refused the question or options; its own reason is quoted | for a `criteria` complaint register the model with `MAP {'criteria': 'name'}` |
| `... is rate limiting requests (HTTP 429)` | too many calls; retried with the service's Retry-After | `SET anofox_decide_max_retries = 8;` or slow down |
| `... had a server error (HTTP 5xx)` | the service failed; not caused by your query | retry later or use another model |
| `... is not reachable: nothing is listening on 127.0.0.1:<port>` | a local server (strands-decider, Kev) is not running | start it; connection refused on loopback is not retried |
| `... did not answer within N ms` | a slow model or server | `SET anofox_decide_timeout_ms = 60000;` (or `SET anofox_decide_max_retries = 0;` to fail fast) |
| `... answered HTTP 200 but not with JSON (an HTML page)` | the endpoint or path is not a System One server (a login or error page) | check the endpoint and path in `decide_models()` |
| `endpoint '<x>' contains a path` / `has an invalid port` | an endpoint is only `scheme://host[:port]` | `MAP {'endpoint': 'https://host', 'path': '/v1/...'}` |
| `the graph file '...' does not exist` | the path is wrong; relative paths resolve against the DuckDB working directory | pass the full path of the exported `.onnx`; see [Local models](#local-models) |
| `provider '<x>' takes no file paths` | a remote provider was given a path argument | remote providers take an options `MAP`: `decide_register_model('m', 'typesafe', MAP {'endpoint': 'https://host'})` |
| `model '<id>' is already registered` | ids are unique per database instance | `decide_unregister_model('<id>')`, then register again |
| `<setting>: must be between A and B, got X` | a setting is out of range | the message shows the default and an example `SET` |
| `threshold must be between 0 and 1, got 70` | thresholds are fractions, not percents | use `0.7` for 70% |
| `probability must be between 0 and 1` (metrics) | the first argument is not a probability | pass probabilities, not percents or logits (`p / 100.0`); NaN: `NULLIF(p, 'NaN'::DOUBLE)` |
| `No function matches ... explicit type casts` | DuckDB's own message for argument types | cast the arguments (`x::VARCHAR`, `y = 1`); `decide_choice` options must be a list of strings `['a','b']` |

NULL rules: a NULL state, question, option list or threshold gives a NULL result (no model is resolved and
nothing is sent); a NULL `model` argument falls back to the session default. Local models print many
`Schema error: ... already registered` lines on stderr the first time one loads; they come from the bundled
ONNX Runtime and are harmless.

## Build & test

Clone with submodules (`git clone --recurse-submodules`; they include the shared `posthog-telemetry` and `datazoo-banner` libraries), then:

```bash
make release                          # release build; ONNX Runtime built via vcpkg (first build is slow)
make release DECIDE_ORT_VCPKG=0 CMAKE_PREFIX_PATH=<ort-install>   # faster local build against an existing ONNX Runtime
make test_release                     # offline suite: SQL tests + Catch2 (no network, no weights)
make test-live                        # live TypeSafe + Liquid D1 E2E; explicit SKIP without TYPESAFE_API_KEY / LIQUID_API_KEY
./build/release/test/unittest test/sql/decide_contract.test       # a single file
```

Some tests need model weights or network and warn-and-pass without them:
Catch2 `[tokenizer]` goldens (`JULIA_WEIGHTS_DIR`, `LAYA_TYPED_DIR`), `[local]`
real graphs (`JULIA_ONNX`, `LAYA_MULTILINGUAL_DIR` + `LAYA_ONNX`,
`LAYA_TYPED_DIR` + `LAYA_TYPED_ONNX`), and the export pytest in
`tools/export_julia` (`JULIA_WEIGHTS_DIR`, `JULIA_SRC_MODEL`, `JULIA_ONNX`).
Run the suite with `TYPESAFE_API_KEY` and `LIQUID_API_KEY` unset, as CI does, so tests do not
depend on your own key.

Tests always run with `DATAZOO_DISABLE_TELEMETRY=1` (Makefile does this).
