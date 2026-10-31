# Settings

Settings apply to the whole database instance (`SET name = value;` affects every connection to it, and is
gone when the process ends) and are validated: a bad value fails with the allowed range, the default and an
example.

| Setting | Default | Meaning |
|---|---|---|
| `anofox_decide_model` | *(unset)* | Model id used when a call names none. Unset: name one per call or set it |
| `anofox_decide_allow_remote` | `false` | Opt-in for hosted and server models; the text you score goes to their endpoint ([Providers](02-providers.md)) |
| `anofox_decide_max_concurrency` | `0` | Remote requests in flight at once, 0 to 64. `0` = automatic: 8 hosted, 1 for the local strands server ([Scoring many rows](01-functions.md#scoring-many-rows)) |
| `anofox_decide_timeout_ms` | `30000` | Per-request timeout for remote calls |
| `anofox_decide_max_retries` | `3` | Retries after a `429`, a `5xx` or a connection failure, 0 to 10 |
| `anofox_decide_max_questions` | `100` | Most questions in one `decide_many` / `decide_table` call, up to 1000 |
| `anofox_decide_cache_dir` | *(unset: `~/.cache/anofox-decide`)* | Where `decide_download` keeps the local models' weights |
| `anofox_decide_max_length` | `8192` | Tokens a `julia-1` local model reads per question (Laya uses its own config) ([Long text](03-local-models.md#long-text)) |
| `anofox_decide_head_length` | `512` | Tokens reserved for the question and its options in local models |
| `anofox_decide_batch_tokens` | `0` | Local models: score the rows of a whole chunk together, in batches of at most this many padded tokens (batch size times padded length; `1024` to `131072`). `0` scores one text at a time. Answers are the same either way (identical to the last bit on the 200-ticket evaluation); on CPU the effect on speed is small and inconsistent, it exists to feed GPUs ([Batching rows](03-local-models.md#batching-rows)) |
| `anofox_decide_device` | `auto` | Where local models run: `auto` (a GPU only when its plugin is installed, the device is usable and the plugin serves the model; else the CPU), `cpu`, or `cuda` / `rocm` / `mlx`, which fail with the fix when that device cannot serve the model ([GPU](04-gpu.md#choosing-the-device)) |
| `anofox_decide_gpu_precision` | `fp32` | Numeric precision of the GPU backends. Only `fp32`; other values are rejected, never silently run in fp32 |
| `anofox_decide_plugin_dir` | *(unset: `plugins` inside the cache directory)* | Where the GPU plugins are installed and loaded from; `decide_accelerate()` fills it |
| `anofox_decide_on_truncate` | `error` | `error`: a local model refuses text, question or options that do not fit its window; `ignore`: score the shortened input |
| `anofox_decide_api_key` | *(unset)* | Legacy key for the `typesafe` provider, kept in plain text; prefer `CREATE SECRET` or the env var ([API keys](02-providers.md#api-keys-and-secrets)) |
| `anofox_decide_endpoint` | `https://api.typesafe.ai` | Legacy endpoint for `typesafe`; other providers take a per-model `endpoint` |
| `anofox_telemetry_enabled` | `true` | Anonymous usage telemetry, see [TELEMETRY.md](../../TELEMETRY.md) |
| `anofox_telemetry_key` | *(the project's PostHog key)* | PostHog API key the telemetry is sent with; there is no need to change it |
| `datazoo_banner` | `true` | The once-a-day feedback banner in interactive terminals |

## Environment variables

| Variable | Used for |
|---|---|
| `TYPESAFE_API_KEY`, `LIQUID_API_KEY`, `CLOUDFLARE_API_TOKEN` | the hosted providers' keys, on their default hosts only ([API keys](02-providers.md#api-keys-and-secrets)) |
| `CLOUDFLARE_ACCOUNT_ID` | the Cloudflare account in the Workers AI URL, unless a model sets `account_id` |
| `HTTPS_PROXY`, `HTTP_PROXY`, `ALL_PROXY`, `NO_PROXY` | a proxy for remote models (`HTTPS_PROXY` for `https://` endpoints, `HTTP_PROXY` for `http://`, `ALL_PROXY` as the fallback) and for `decide_download`; `NO_PROXY` lists hosts to reach directly. The lowercase forms are also read; `localhost`, `127.0.0.1` and `::1` are never proxied |
| `DATAZOO_DISABLE_TELEMETRY=1` | turns telemetry off, like `SET anofox_telemetry_enabled = false` |
| `DATAZOO_NO_BANNER=1` | silences the feedback banner, like `SET datazoo_banner = false` |
