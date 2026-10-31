# Providers

Hosted and local models are profiles over one wire format (System One: `binary`, `choice` and `score`
questions, probabilities back), so a model is just a registered id, and several can be used side by side
in one session with no `SET` in between. Which one to pick: [Choosing a model](../guides/02-choosing-a-model.md).

- [Overview](#overview)
- [Registering a model](#registering-a-model)
- [Hosted providers](#hosted-providers): [TypeSafe Jev](#typesafe-jev), [Liquid AI D1](#liquid-ai-d1), [Cloudflare Clef](#cloudflare-clef)
- [API keys and secrets](#api-keys-and-secrets)
- [Local servers](#local-servers)

## Overview

| Provider | Service | Runs | Default endpoint | Key |
|---|---|---|---|---|
| `typesafe` | TypeSafe Jev | hosted API | `https://api.typesafe.ai` | `TYPESAFE_API_KEY` |
| `liquid` | Liquid AI D1 | hosted API | `https://api.liquid.ai` | `LIQUID_API_KEY` |
| `cloudflare` | Cloudflare Clef, Clef-flash | hosted API (Workers AI) | `https://api.cloudflare.com` | `CLOUDFLARE_API_TOKEN` |
| `strands` | [strands-decider](https://github.com/strands-labs/strands-decider) 2B | local server | `http://127.0.0.1:8000` | none |
| `systemone` | any compatible server, e.g. [Kev](https://github.com/jaredpalmer/kev) | your server | you set it | optional (`key_env`) |
| `local` | Julia-1, Laya | in-process, offline | — | none ([Local models](03-local-models.md)) |
| `stub` | built-in test model | in-process | — | none |

## Registering a model

A hosted or server model sends the text you score to its endpoint, so it sits behind an explicit opt-in that
is off by default. Register it once per database instance (registrations and settings are not stored: repeat
them in each new process), then name it per call or make it the default with `SET anofox_decide_model`:

```sql
SET anofox_decide_allow_remote = true;                       -- hosted and server models need the opt-in
SELECT decide_register_model('jev-latest', 'typesafe');
SELECT decide_register_model('d1:free', 'liquid');
SELECT decide_register_model('clef', 'cloudflare');             -- Cloudflare Clef; 'clef-flash' is the smaller one
SELECT decide_probability('...', '...', model := 'jev-latest');
SELECT decide_probability('...', '...', model := 'd1:free');
```

Per-model options go in a `MAP` (keys `endpoint`, `path`, `model`, `key_env`, `criteria`, `calibration`, `account_id`): a self-hosted
server, a different model name on the wire, an explicit key variable for a custom endpoint, or
`criteria: 'name'` for servers whose schema wants a string description per choice option instead of `null`, or a fitted `calibration` ([Calibration](../guides/03-evaluate-and-calibrate.md#calibration)):

```sql
SELECT decide_register_model('kev-latest', 'systemone', MAP {'endpoint': 'http://127.0.0.1:8009'});
SELECT decide_register_model('d1', 'liquid', MAP {'model': 'd1:free'});
SELECT decide_register_model('mine', 'systemone', MAP {'endpoint': 'https://llm.internal', 'key_env': 'MY_KEY_VAR'});
```

An endpoint is only `scheme://host[:port]`; a path goes in `path`. Cleartext `http://` works for loopback
hosts only. `decide_unregister_model('<id>')` removes a model so the id can be registered again, and
`SELECT * FROM decide_models();` lists what is registered and whether each can be called now.

## Hosted providers

Each hosted provider receives the text you score. The quality numbers below come from the 200-ticket
evaluation in [Choosing a model](../guides/02-choosing-a-model.md#how-the-models-compare).

### TypeSafe Jev

| | |
|---|---|
| Provider | `typesafe` |
| Model id | `jev-latest` |
| Default endpoint | `https://api.typesafe.ai` |
| Key | `TYPESAFE_API_KEY`, or a secret scoped to `api.typesafe.ai` |

```bash
export TYPESAFE_API_KEY=...          # before starting DuckDB
```
```sql
SET anofox_decide_allow_remote = true;
SELECT decide_register_model('jev-latest', 'typesafe');
SELECT decide_probability('I want my money back.', 'A refund is requested.', model := 'jev-latest');
```

Jev was the most accurate model in the 200-ticket evaluation. Two legacy settings apply to this provider
only: `anofox_decide_api_key` (kept in plain text and visible through `current_setting`, so prefer a secret
or the env var) and `anofox_decide_endpoint`; other providers take a per-model `endpoint`.

### Liquid AI D1

| | |
|---|---|
| Provider | `liquid` |
| Model id | `d1:free` ([D1](https://docs.liquid.ai/lfm/models/decision-models)) |
| Default endpoint | `https://api.liquid.ai` |
| Key | `LIQUID_API_KEY`, or a secret scoped to `api.liquid.ai` |

```bash
export LIQUID_API_KEY=...            # before starting DuckDB
```
```sql
SET anofox_decide_allow_remote = true;
SELECT decide_register_model('d1:free', 'liquid');               -- the id is the model name on the wire
SELECT decide_register_model('d1', 'liquid', MAP {'model': 'd1:free'});   -- or a shorter id
```

D1 routes as well as Jev, but at the 0.5 cut-off over-predicts refunds (its ranking is fine, AUROC 0.970);
a fitted [calibration](../guides/03-evaluate-and-calibrate.md#calibration) cut its refund error from 23% to 9% on that evaluation.

### Cloudflare Clef

[Clef](https://blog.cloudflare.com/clef-decision-models/) is Cloudflare's open-weight family of decision models
(Apache 2.0 weights on Hugging Face), served on Workers AI: `clef` (27B) and the smaller `clef-flash` (9B). It
speaks the same questions as the other hosted models, so the call is the same:

```bash
export CLOUDFLARE_API_TOKEN=...      # before starting DuckDB: a token with "Workers AI - Read" and "Workers AI - Edit"
export CLOUDFLARE_ACCOUNT_ID=...     # the 32-character account id from your Cloudflare dashboard
```
```sql
SET anofox_decide_allow_remote = true;
SELECT decide_register_model('clef', 'cloudflare');          -- or 'clef-flash'; the id is the model name in the URL
SELECT decide_choice('Checkout has been failing for every customer.', 'Which team should handle this?',
                     ['billing', 'technical', 'sales'], model := 'clef');
-- the account id can also be given per model: decide_register_model('clef', 'cloudflare', MAP {'account_id': '<id>'})
```

Things worth knowing:

- **Cost and speed:** $0.24 per million input tokens for `clef` and $0.09 for `clef-flash`; one call took about
  0.3 to 1 s in our runs. A request takes at most **64 questions** (a call with more is refused before anything is
  sent) and a 64k-token text.
- **Quality:** on the 200-ticket evaluation `clef` is the third-best router (88%) and a solid refund detector (85%);
  `clef-flash` is much weaker on refund (70%). `clef` errs towards "no" at the 0.5 cut-off (recall 62%, specificity
  95%); [Calibration](../guides/03-evaluate-and-calibrate.md#calibration) shows how to fit the cut-off to your own
  data. See [How the models compare](../guides/02-choosing-a-model.md#how-the-models-compare).
- **Errors:** Cloudflare answers a wrong token and a valid token of a different account with the same
  "Authentication error", so the message names both causes. A wrong model name is an HTTP 400 "No route for
  that URI"; the message says the models are `clef` and `clef-flash`.
- **Not supported:** Clef also accepts images (up to four per request); the functions here take text only. The
  27B and 9B models are too large to run in-process like Julia-1 or Laya; to run them yourself, serve them behind a
  System One compatible endpoint (see the model card for serving options) and register that endpoint with
  provider `systemone`.

<!--
  ADD MORE HOSTED PROVIDERS HERE.
  One "### <Service name>" section per provider, in the same shape as the ones above:
  a short table (provider, model ids, default endpoint, key), the export + register + call snippet,
  and "things worth knowing". Also add a row to the Overview table at the top of this page, to the
  "Hosted providers" link list, to the model table in README.md, and to docs/guides/02-choosing-a-model.md.
-->

## API keys and secrets

Environment variables work out of the box (`export LIQUID_API_KEY=...`), and a DuckDB secret always
overrides them. Precedence, highest first:

1. A stored secret (write-only, redacted in `duckdb_secrets()`), matched by the endpoint host:
   ```sql
   CREATE SECRET (TYPE anofox_decide, API_KEY 'sk-...', SCOPE 'api.liquid.ai');
   CREATE SECRET (TYPE anofox_decide, API_KEY getenv('LIQUID_API_KEY'), SCOPE 'api.liquid.ai');  -- copy from the env
   ```
2. `anofox_decide_api_key` (legacy, `typesafe` only; visible through `current_setting`, so prefer a secret).
3. The model's explicit `key_env` variable.
4. The provider's own env var ([Overview](#overview)), only on that provider's default host.

A key is never sent to a host it was not configured for: a redirected endpoint gets nothing from the
environment, and a secret scoped to one host is not used for another. Cleartext `http://` works for
loopback hosts only. Behind a proxy, set `HTTPS_PROXY`.

## Local servers

[strands-decider](https://github.com/strands-labs/strands-decider) (Apache 2.0; a 1.9B Qwen3.5 torso with
a pointer head) needs no key. A choice question may have at most 24 options on this model. It listens on
loopback without authentication, so keep it local:

```bash
python3.12 -m venv .venv && . .venv/bin/activate
pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cpu   # or a CUDA/MPS build
pip install strands-decider
strands-decider serve StrandsAgents/strands-decider-2B-hobson-v19 --device cpu --port 8000   # first run downloads ~4.5 GB
```
```sql
SET anofox_decide_allow_remote = true;
SELECT decide_register_model('strands-decider', 'strands');   -- default endpoint http://127.0.0.1:8000
SELECT decide_choice('Help! My payouts have been failing.', 'Which team?', ['billing', 'sales', 'retail'],
                     model := 'strands-decider');
```

Kev (Qwen3.5 + LoRA, Apache 2.0) is run the same way: `python -m kev.serve --run jaredpalmer/kev-0.8b
--port 8009`, register it as `systemone` ([Registering a model](#registering-a-model)) and give it any key,
since it ignores the value but the provider requires one:
`CREATE SECRET (TYPE anofox_decide, API_KEY 'local', SCOPE '127.0.0.1')`. Kev-4B, 9B and 27B are more
accurate than 0.8B and need a GPU or a large Mac.

By default the strands server gets one request at a time (`anofox_decide_max_concurrency = 0` means 8 for
hosted models and 1 for the local strands server); see [Scoring many rows](01-functions.md#scoring-many-rows).
