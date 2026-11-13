---
name: anofox-decide-models
description: >
  Choosing, registering and configuring decision models for the anofox_decide DuckDB extension,
  and fixing setup errors. Covers decide_register_model / decide_unregister_model and their
  per-model options MAP (endpoint, path, model, key_env, criteria, calibration, account_id, input,
  prompt_cache_key). Covers every provider: stub, typesafe (TypeSafe Jev), liquid (Liquid AI D1),
  cloudflare (Clef, Clef-flash), openai (OpenAI Decisions gpt-6-luna), gemini (Gemini Decisions
  gemini-decision-1), systemone (Kev or any System One server), strands (strands-decider), and
  local (Julia-1, Laya). Covers the anofox_decide_allow_remote opt-in, the anofox_decide_model
  default, API keys as DuckDB secrets (CREATE SECRET TYPE anofox_decide, SERVICE ...) or env vars
  and their precedence, decide_services, decide_models, decide_doctor, every anofox_decide_*
  setting, the error-message shape and the common errors with their fixes, and installing / loading
  the extension (INSTALL anofox_decide FROM community). Use when setting up a
  model or a key, picking a provider, comparing hosted vs local, or when a decide_* call fails with
  "no model selected", "not registered", "remote calls are off", "no API key" or an HTTP error.
version: 2026.11.13
user-invocable: false
---

# Anofox Decide — Models, Providers, Keys & Setup

**Extension:** `anofox_decide` v2026.11.13 | **DuckDB:** v1.5.6 / v2.0 | **Dual naming:** `decide_*` and
`anofox_decide_*` (identical)

A model is a **registered id**. Every provider answers the same three question kinds through the same
functions (see `anofox-decide-questions`), so swapping a model never changes the query. Local models
(Julia-1, Laya) and GPUs: `anofox-decide-local`.

## Critical gotchas

1. **Start with `SELECT * FROM decide_doctor();`.** It returns `item`, `status` (`ok`/`warn`/`fail`),
   `detail` and `fix` for the whole setup: models, keys, settings and GPU.
2. **Remote is opt-in.** Hosted APIs and local servers (strands, Kev) send the text you score to an
   endpoint. They need `SET anofox_decide_allow_remote = true;` first, otherwise the call fails with
   `... remote calls are off`.
3. **Registrations and settings are per process.** They are not stored in the database file. Repeat
   `decide_register_model` and the `SET`s in every new DuckDB process or script. Settings apply to the
   whole database instance.
4. **Registering is not calling.** `decide_register_model` checks the arguments, and for a `local` model
   the files too. `decide_models()` / `decide_doctor()` show a missing key or a missing opt-in. A wrong
   remote model name or a rejected key shows only on the first call.
5. **The id is the model name on the wire** unless you set `MAP {'model': '...'}`. Liquid renamed `d1:free`
   to `d1` (Oct 2026): `d1:free` now answers HTTP 404.
6. **Ids are unique.** Registering an existing id fails. Call `decide_unregister_model('<id>')` first.
7. **Keys: a secret beats an env var.** A key is never sent to a host it was not configured for. Do not
   put keys in SQL that you share. Prefer env vars or `CREATE SECRET` with a value read outside the script.
8. **The endpoint is only `scheme://host[:port]`.** A path goes in `MAP {'path': ...}`. Cleartext
   `http://` works for loopback only.
9. **`stub` and the `local` catalog models need no registration.** `model := 'stub'` works at once.
   `model := 'laya-multilingual'` works once `CALL decide_download('laya-multilingual')` has run.

## Providers

| Provider | Model ids | Runs | Key (env var) |
|---|---|---|---|
| `stub` | `stub` | in-process, constants (0.5 / first option / middle level) | none |
| `typesafe` | `jev-latest` | hosted | `TYPESAFE_API_KEY` |
| `liquid` | `d1`, `d1.1`, `d1-ultra` | hosted | `LIQUID_API_KEY` |
| `cloudflare` | `clef` (27B), `clef-flash` (9B) | hosted, Workers AI; max 64 questions per request | `CLOUDFLARE_API_TOKEN` + `CLOUDFLARE_ACCOUNT_ID` |
| `openai` | `gpt-6-luna` | hosted, OpenAI Decisions; up to 400 questions (raise `anofox_decide_max_questions`, default 100); may refuse | `OPENAI_API_KEY` |
| `gemini` | `gemini-decision-1` | hosted, Gemini Decisions; up to 1000 questions (same setting); safety blocks are refusals | `GEMINI_API_KEY` |
| `strands` | your id | local server, default `http://127.0.0.1:8000`; max 24 options | none |
| `systemone` | your id | any System One server (e.g. Kev) at your `endpoint` | required (any value for keyless servers) |
| `local` | `julia-1`, `laya-multilingual`, `laya-typed-decisions`, or your own ONNX export | in-process, offline | none |

**Which model?** On the project's 200-ticket evaluation, TypeSafe Jev was the most accurate overall.
gemini-decision-1 was level with it in a separate run, and Clef, D1 and gpt-6-luna route well. At the 0.5
cut-off, D1 and Gemini over-predict "yes" and Clef and gpt-6-luna err towards "no": calibrate them, or pick
the threshold from labelled data (see `anofox-decide-evaluate`). Laya multilingual is the best in-process model (refund
detection on par with Jev), and Laya typed-decisions is the best in-process router. Julia-1 is weak.
That data is English and template-generated, so **compare models on the user's own labelled data**.

## Registering

```text
decide_register_model(id VARCHAR[, provider VARCHAR[, options MAP(VARCHAR, VARCHAR)]]) -> BOOLEAN
decide_register_model(id, 'local', graph_path, tokenizer_path[, profile ('julia-1' | 'laya')[, options MAP]])
decide_unregister_model(id) -> BOOLEAN
```

```sql
SET anofox_decide_allow_remote = true;                       -- explicit opt-in: the text goes to the provider
SELECT decide_register_model('jev-latest', 'typesafe');
SELECT decide_register_model('d1', 'liquid');
SELECT decide_register_model('clef', 'cloudflare');          -- or 'clef-flash'
SELECT decide_register_model('gpt-6-luna', 'openai');
SELECT decide_register_model('gemini-decision-1', 'gemini');
SELECT decide_register_model('strands-decider', 'strands');  -- default endpoint http://127.0.0.1:8000

-- per-model options
SELECT decide_register_model('liquid-d1', 'liquid', MAP {'model': 'd1.1'});                  -- other wire name
SELECT decide_register_model('kev', 'systemone', MAP {'endpoint': 'http://127.0.0.1:8009'});
SELECT decide_register_model('mine', 'systemone', MAP {'endpoint': 'https://llm.internal', 'key_env': 'MY_KEY_VAR'});
SELECT decide_register_model('clef-acct', 'cloudflare', MAP {'model': 'clef', 'account_id': '0123456789abcdef0123456789abcdef'});
SELECT decide_register_model('luna-json', 'openai', MAP {'model': 'gpt-6-luna', 'input': 'json', 'prompt_cache_key': 'tickets-v1'});

SET anofox_decide_model = 'd1';                              -- the default for calls that name no model
SELECT model, provider, ready, hint FROM decide_models();   -- what is registered and whether it can be called now

SELECT decide_unregister_model('liquid-d1');
```

| Option | For | Meaning |
|---|---|---|
| `endpoint` | all remote | `scheme://host[:port]` only |
| `path` | all remote | the request path, if not the provider default |
| `model` | all remote | the model name sent to the service (default: the id) |
| `key_env` | all remote | name of an env var holding the key (any host) |
| `criteria` | System One | `'name'` when a server wants a string per choice option instead of `null` (fixes HTTP 422 criteria complaints) |
| `calibration` | all, incl. local | `'platt:a,b'` from `decide_fit_calibration` |
| `account_id` | cloudflare | the 32-character account id |
| `input` | openai | `'auto'` (default: JSON state sent as JSON), `'text'`, `'json'` |
| `prompt_cache_key` | openai | default: the model id. `''` sends none |

## API keys and secrets

One secret type, `anofox_decide`, holds the key for any service. `SERVICE` names which one:

```sql
-- one secret per service: give each a name (an unnamed secret takes the one default name)
CREATE SECRET gemini_key (TYPE anofox_decide, SERVICE 'gemini', API_KEY 'AIza-example');
CREATE SECRET cf_key (TYPE anofox_decide, SERVICE 'cloudflare', API_KEY 'cf-example', ACCOUNT_ID '0123456789abcdef0123456789abcdef');
CREATE SECRET kev_key (TYPE anofox_decide, SERVICE 'systemone', API_KEY 'local', SCOPE '127.0.0.1');   -- Kev ignores the value
CREATE SECRET hf_key (TYPE anofox_decide, SERVICE 'huggingface', API_KEY 'hf_example');               -- gated downloads only
CREATE OR REPLACE SECRET gemini_key (TYPE anofox_decide, SERVICE 'gemini', API_KEY 'AIza-rotated');    -- replace a key
SELECT service, env_var, key_found, key_source FROM decide_services();   -- never shows the key
```

Or export the env var before starting DuckDB: `export LIQUID_API_KEY=...`. The services are `typesafe`,
`liquid`, `cloudflare`, `openai`, `gemini`, `systemone` (needs `SCOPE` = your host; no env var, use
`key_env`), `huggingface` (`HF_TOKEN`) and `strands` (keyless).

**Precedence, highest first:**
1. a stored secret for the endpoint host. A `SERVICE` secret applies only to that service and its default
   host, or exactly its `SCOPE`.
2. the model's `key_env` variable.
3. the service's env var, on its default host only.
4. (`typesafe` only, deprecated) `anofox_decide_api_key`.

Ties: longest scope, then `SERVICE` over older-form secrets, then temporary over persistent, then the
lower name. `decide_models()` has a `key_source` column. Older secrets without `SERVICE` or `SCOPE` reach
every remote host, and `decide_doctor()` warns about them.

## Settings

| Setting | Default | Meaning |
|---|---|---|
| `anofox_decide_model` | unset | default model id |
| `anofox_decide_allow_remote` | `false` | opt-in for hosted and server models |
| `anofox_decide_max_concurrency` | `0` (auto: 8 hosted, 1 strands) | remote requests in flight per endpoint and key, across all threads and queries of the process, 0 to 64 |
| `anofox_decide_timeout_ms` | `30000` | per remote request |
| `anofox_decide_max_retries` | `3` | after 429 / 5xx / connection failure, 0 to 10 |
| `anofox_decide_max_questions` | `100` | per `decide_table` / `decide_many` call, up to 1000 |
| `anofox_decide_api_key`, `anofox_decide_endpoint` | unset / TypeSafe URL | legacy, `typesafe` only. Use a secret instead |

Local-model and GPU settings: `anofox-decide-local`. Proxies: `HTTPS_PROXY`, `HTTP_PROXY`, `ALL_PROXY`
and `NO_PROXY` are read (loopback is never proxied). Telemetry: `SET anofox_telemetry_enabled = false` or
`DATAZOO_DISABLE_TELEMETRY=1`.

## Errors and fixes

Every error has the shape `<function>: <what went wrong>. Fix: <what to run>`. Run `decide_doctor()` first.

| Message contains | Fix |
|---|---|
| `no model selected` | `model := '<id>'` or `SET anofox_decide_model = '<id>'` |
| `model '<id>' is not registered` | the message lists registered ids and a close match. Register it, or check `decide_models()` |
| `... remote calls are off` | `SET anofox_decide_allow_remote = true;` |
| `no API key for <service> at <host>` | `CREATE SECRET <name> (TYPE anofox_decide, SERVICE '<service>', API_KEY '...')`, plus `SCOPE '<host>'` (host only, no scheme or path) when the model's endpoint is not the provider's default host, and always for `systemone`. Or export the env var (default host only) or the model's `key_env`. A keyless local server uses provider `strands` |
| `rejected the API key (HTTP 401)` (Gemini: 400) | replace it: `CREATE OR REPLACE SECRET ...` |
| `Authentication error (code 10000)` (Cloudflare) | wrong token, or a token of another account: check `CLOUDFLARE_ACCOUNT_ID` and the token's Workers AI Read+Edit |
| `does not know the model ... (HTTP 404)` | register with `MAP {'model': '<provider model name>'}` |
| `HTTP 422` | for a `criteria` complaint: `MAP {'criteria': 'name'}` |
| `HTTP 429` | `SET anofox_decide_max_concurrency = 2;` / `SET anofox_decide_max_retries = 8;` |
| `nothing is listening on 127.0.0.1:<port>` | start the local server |
| `did not answer within N ms` | `SET anofox_decide_timeout_ms = 60000;` |
| `accepts at most 64 questions` | split the questions (Clef) |
| `endpoint ... contains a path` | `MAP {'endpoint': 'https://host', 'path': '/v1/...'}` |
| `provider '<x>' takes no file paths` | remote providers take an options MAP, not paths |
| `already registered` | `decide_unregister_model('<id>')`, then register again |
| `threshold must be between 0 and 1, got 70` | use `0.7` |

## Recipe: from nothing to a real answer

```sql
-- needs: LIQUID_API_KEY exported before starting DuckDB, and network access
INSTALL anofox_decide FROM community;   -- once per machine
LOAD anofox_decide;                     -- every session
SELECT * FROM decide_doctor();
SET anofox_decide_allow_remote = true;
SELECT decide_register_model('d1', 'liquid');
SET anofox_decide_model = 'd1';
SELECT decide_probability('I want my money back.', 'A refund is requested.') AS p;
```

No key at all? Use the in-process Laya model: `CALL decide_download('laya-multilingual');`, then
`model := 'laya-multilingual'` (see `anofox-decide-local`).

Reference pages: [Providers](https://github.com/DataZooDE/anofox-decide/blob/main/docs/api/02-providers.md),
[Settings](https://github.com/DataZooDE/anofox-decide/blob/main/docs/api/05-settings.md),
[Troubleshooting](https://github.com/DataZooDE/anofox-decide/blob/main/docs/api/06-troubleshooting.md),
[Choosing a model](https://github.com/DataZooDE/anofox-decide/blob/main/docs/guides/02-choosing-a-model.md).
