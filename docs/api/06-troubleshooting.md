# Troubleshooting

Every error follows one shape, `<function>: <what went wrong>. Fix: <what to run>`, and echoes the value
that caused it. Run `SELECT * FROM decide_doctor();` first when something does not work. If an error does
not tell you what to do, please [open an issue](https://github.com/DataZooDE/anofox-decide/issues) with the
output of `decide_doctor()` and the failing call (without your key).

- [Choosing and calling a model](#choosing-and-calling-a-model), [Providers, keys and endpoints](#providers-keys-and-endpoints), [Local models](#local-models), [GPU](#gpu), [Values and types](#values-and-types)

## Choosing and calling a model

| You see | It means | Do this |
|---|---|---|
| `no model selected` | the call names no model and no default is set | `decide_probability(..., model := '<id>')` or `SET anofox_decide_model = '<id>'`; register one first with `decide_register_model` |
| `model '<id>' is not registered (given by ...)` | the id is a typo, or never registered, or removed | the message lists the registered ids and a close match; `SELECT * FROM decide_models()` |
| `model '<id>' is a ... model: calling it sends your text to ..., and remote calls are off` | remote models send your text to an endpoint and are off by default | `SET anofox_decide_allow_remote = true;` |

## Providers, keys and endpoints

| You see | It means | Do this |
|---|---|---|
| `no API key for ... at <host>` | no secret, setting or env var supplies a key | export the provider's variable (`LIQUID_API_KEY`, `TYPESAFE_API_KEY`) or `CREATE SECRET (TYPE anofox_decide, API_KEY '...', SCOPE '<host>')`; a keyless local server uses provider `strands` |
| `... rejected the request (HTTP 401): "Authentication error (code 10000)"` (Cloudflare) | Cloudflare gives this answer for a wrong token and for a token that belongs to another account | check the token has Workers AI - Read and Edit and that `CLOUDFLARE_ACCOUNT_ID` is the account the token belongs to |
| `no Cloudflare account id for model '...'` | a `cloudflare` model needs the account id for its URL | `export CLOUDFLARE_ACCOUNT_ID=<id>` or `MAP {'account_id': '<id>'}` |
| `... accepts at most 64 questions per request` | Cloudflare Clef takes at most 64 questions in one request | split the questions across several calls |
| `... rejected the API key (HTTP 401)` | the service refused the key; the message says where the key came from | replace it: `CREATE OR REPLACE SECRET (TYPE anofox_decide, API_KEY '<key>', SCOPE '<host>')` (a secret overrides env vars) |
| `... does not know the model '<x>' ... (HTTP 404)` | the model name sent to the service is wrong (it is the registered id unless you set one) | `decide_register_model('<id>', '<provider>', MAP {'model': '<provider model name>'})` |
| `... rejected the request (HTTP 422)` | the service refused the question or options; its own reason is quoted | for a `criteria` complaint register the model with `MAP {'criteria': 'name'}` |
| `... is rate limiting requests (HTTP 429)` | too many calls; retried with the service's Retry-After | `SET anofox_decide_max_retries = 8;` or `SET anofox_decide_max_concurrency = 2;` |
| `... had a server error (HTTP 5xx)` | the service failed; not caused by your query | retry later or use another model |
| `... is not reachable: nothing is listening on 127.0.0.1:<port>` | a local server (strands-decider, Kev) is not running | start it; connection refused on loopback is not retried |
| `... did not answer within N ms` | a slow model or server | `SET anofox_decide_timeout_ms = 60000;` (or `SET anofox_decide_max_retries = 0;` to fail fast) |
| `... answered HTTP 200 but not with JSON (an HTML page)` | the endpoint or path is not a System One server (a login or error page) | check the endpoint and path in `decide_models()` |
| `endpoint '<x>' contains a path` / `has an invalid port` | an endpoint is only `scheme://host[:port]` | `MAP {'endpoint': 'https://host', 'path': '/v1/...'}` |

## Local models

| You see | It means | Do this |
|---|---|---|
| `the graph file '...' does not exist` | the path is wrong; relative paths resolve against the DuckDB working directory | pass the full path of the exported `.onnx`; see [Local models](03-local-models.md) |
| `the text is N tokens but the local model reads at most M ...` | the text does not fit the local model's window, and the end would be ignored | shorten the text, raise `anofox_decide_max_length` (`julia-1` only), or `SET anofox_decide_on_truncate = 'ignore';`; `decide_token_count(text)` measures it |
| `... the question is N tokens but only M fit in the head budget` / `option N (...) of question '...' is N tokens but ...` | the question or an option is too long for the part of the window they share | shorten it, raise `anofox_decide_head_length`, or `SET anofox_decide_on_truncate = 'ignore';` |
| `the graph file '...' is a safetensors weights file, not an ONNX graph` (also: text, JSON, HTML, zip, pickle) | the graph argument is not the exported `.onnx` | pass the `.onnx` file that `tools/export_julia` wrote; see [Local models](03-local-models.md) |
| `ONNX Runtime could not load the graph '...'` / `the graph '...' is not a readable ONNX model` | the file passed the registration checks but ONNX Runtime cannot use it (damaged, incomplete, or an unsupported ONNX version); the reason from ONNX Runtime is in parentheses | export the model again with `tools/export_julia` and register that file |
| `the graph '...' has the inputs ..., but a local model is fed ...` | the graph is not a scores graph from `tools/export_julia`; the message names the missing and unused inputs | export it again with `tools/export_julia` |
| `the tokenizer file '...' is a Unigram tokenizer, but local models need a BPE tokenizer` | the tokenizer belongs to another model family | pass the `tokenizer.json` of a Julia-1 or Laya checkpoint |
| `the Laya config file '...' lacks the keys ...` | `rl_agent_config.json` is not the one of a Laya checkpoint | copy it from the checkpoint folder next to the graph |
| `'.../rl_agent_config.json' sits next to the graph ..., so it comes from a Laya checkpoint` | a Laya graph registered without the profile would be scored as `julia-1` | add the 5th argument: `decide_register_model('<id>', 'local', '<graph>', '<tokenizer>', 'laya')` |
| `provider '<x>' takes no file paths` | a remote provider was given a path argument | remote providers take an options `MAP`: `decide_register_model('m', 'typesafe', MAP {'endpoint': 'https://host'})` |
| `model '<id>' is already registered` | ids are unique per database instance | `decide_unregister_model('<id>')`, then register again |

## GPU

| You see | It means | Do this |
|---|---|---|
| `anofox_decide_device is 'cuda' but no 'cuda' device was found on this machine` (also `rocm`, `mlx`) | an explicit GPU device on a machine where none was discovered | `CALL decide_accelerate();` shows what this machine can use, or `SET anofox_decide_device = 'auto';` |
| `anofox_decide_device is 'cuda' but model '<id>' cannot run on cuda:0: ... its plugin is not installed in <dir>` (also `rocm`, `mlx`) | an explicit GPU device whose plugin is missing, unusable, or cannot serve that model; the reason follows the colon | `CALL decide_accelerate();` (see `decide_backends()` for the reason), or `SET anofox_decide_device = 'auto';` |
| `anofox_decide_device is 'cuda' but loading a GPU plugin is not allowed because enable_external_access is off` | DuckDB's own setting forbids loading plugins | `SET enable_external_access = true;` or `SET anofox_decide_device = 'cpu';` |
| `decide_download_runtime: the '<backend>' plugin does not match its published checksum (expected ..., got ...)` and `... was not loaded: it does not match its published checksum` | a damaged download, or the file was replaced after it was installed; the plugin **and** its `.sha256` file have been deleted | retry `CALL decide_download_runtime('<backend>');` (or `CALL decide_accelerate();`), which fetches a verified copy; report it if it recurs |
| `could not fetch the published checksum for the '<backend>' plugin` / `the published checksum file ... does not mention '<file>'` | the release has no usable `.sha256` asset, or the network failed; the plugin was deleted rather than used unverified | retry the call; if it persists the release is incomplete: update the extension (`FORCE INSTALL anofox_decide FROM community;`) |
| `the 'cuda' backend ... a runtime library it needs is missing` / `ran out of device memory` | CUDA 12.8+ with cuDNN 9, the ROCm MIGraphX runtime or Apple MLX is not installed; or the batch does not fit the card | install it and `CALL decide_download_runtime('cuda');`; for memory `SET anofox_decide_batch_tokens = 4096;` or `SET anofox_decide_device = 'cpu';`. More in [docs/GPU_BACKENDS.md](../GPU_BACKENDS.md#troubleshooting) |

## Values and types

| You see | It means | Do this |
|---|---|---|
| `<setting>: must be between A and B, got X` | a setting is out of range | the message shows the default and an example `SET` |
| `threshold must be between 0 and 1, got 70` | thresholds are fractions, not percents | use `0.7` for 70% |
| `probability must be between 0 and 1` (metrics) | the first argument is not a probability | pass probabilities, not percents or logits (`p / 100.0`); NaN: `NULLIF(p, 'NaN'::DOUBLE)` |
| `No function matches ... explicit type casts` | DuckDB's own message for argument types | cast the arguments (`x::VARCHAR`, `y = 1`); `decide_choice` options must be a list of strings `['a','b']` |

Related: [Providers](02-providers.md), [Local models](03-local-models.md), [GPU acceleration](04-gpu.md), [Settings](05-settings.md).
