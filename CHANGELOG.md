# Changelog

All notable changes to `anofox_decide` are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/); this project uses calendar versioning
(`YYYY.MM.DD`, tagged `vYYYY.MM.DD`, like the other anofox extensions).

## [Unreleased]

### Added
- `anofox_decide_ep_path`: the directory the GPU plugins are installed in and loaded from (default: `plugins` inside
  the cache directory), the same name as in anofox-tabfm.

### Changed
- **The in-process models run on the ggml engine** (vendor/anofox-ggml-adapter, shared with anofox-tabfm) instead of ONNX Runtime.
  Same answers (0 label flips on the 200-ticket evaluation against ONNX Runtime, on every platform and GPU path), faster and with
  less memory: on a Linux x64 CPU 4.7 / 10.7 / 25.9 s instead of 5.4 / 13.0 / 30.8 s at 0.5 / 1.9 / 2.4 GB instead of 1.5 / 3.1 /
  4.9 GB (Julia-1 / Laya multilingual / Laya typed-decisions); on an Apple M3 CPU 5 / 13 / 35 s instead of 11 / 34 / 88 s; on the
  GPU plugins 1.2x to 3.9x faster than the previous ones at up to 4.7x less host memory. A model loads in 0.3 to 0.7 s
  (was 1.4 to 5.5 s). Details: docs/GGML_DECIDE.md.
- `SET threads` now limits the local models: the CPU engine runs `min(threads, physical cores)` threads, at most one per physical
  core (a second thread per core adds no arithmetic and slows every op's barrier; ONNX Runtime ignored the setting), so
  `SET threads = 1` is slower than before.
- A graph the engine cannot run says so with the op it lacks: `the graph '...' cannot run on the ggml engine (...)`.
- `decide_gpu_precompile` is deprecated: the plugins build a shape's graph on its first use, so it compiles nothing (on ROCm
  it still loads the model onto the GPU and returns one row per bucket).
- A session served by a GPU plugin no longer keeps the model's checkpoint resident in the process once the plugin has copied or
  uploaded the weights (the mapping stays valid; pages come back from the file only if read again). Up to ~1.3 GB lower peak RSS for
  Laya on MLX.
- Local model sessions: one per model, built outside any shared lock (a first load no longer stalls other models' scoring); a
  `SET threads` or device switch replaces the model's session instead of keeping a second copy of its weights; the
  held memory is reported to DuckDB's object cache. Ctrl-C stops a chunk of texts between model runs.
- Fewer compiled shapes: text lengths are bucketed (steps of 32 up to 512 tokens, then 64/128/256, also with batching, whose
  doubling ladder wasted up to half a batch), markers rounded to multiples of 4, GPU batches to powers of two. A text asked
  several questions is tokenised once. The GELU, the embedding lookup and the attention run as native ggml ops on GPUs too.
- GPU sessions hand the plugin the checkpoint's own 16-bit weights instead of a float32 copy (1.3-1.7 GB less host memory per
  Laya GPU session); registered models' weights are uploaded to the device once instead of on every run.
- Only a graph refusal sends a registered model to the CPU for good under `anofox_decide_device = 'auto'`; an out-of-memory or
  driver failure is raised and not remembered, and a refusal ends when the plugin or the graph file changes.
- Long texts: the ggml CPU engine fuses each encoder layer's attention into one flash-attention op (the adapter's
  `ggml_flash_attn_ext` lowering). julia-1 at about 2,000 tokens is about 40% faster than ONNX Runtime (it was 12% slower),
  and every model is faster at about 500 tokens; answers unchanged (0 flips on the evaluation).
- **GPU plugins come from anofox-ggml-adapter.** The local models (julia-1, laya-multilingual, laya-typed-decisions)
  run on the GPU through the adapter's plugins (plugin ABI v3), shared with anofox-tabfm, instead of decide's own
  ONNX Runtime CUDA, MIGraphX and MLX plugins: `libanofox_ggml-cuda_plugin.so` (CUDA runtime and cuBLAS linked
  statically, needs only the NVIDIA driver's `libcuda.so.1`; sm_75 and newer), `libanofox_ggml-rocm_plugin.so`
  (links the system's HIP, hipBLAS and rocBLAS; gfx90a, gfx942, gfx1030, gfx1100, gfx1101, gfx1102, gfx1150, gfx1151,
  gfx1200, gfx1201) and `libanofox_mlx_plugin.dylib` (Apple silicon, Homebrew `mlx` and `mlx-c`). On Windows the
  names drop `lib` and end in `.dll`. They are fetched from the adapter's release
  ([v2026.11.25](https://github.com/DataZooDE/anofox-ggml-adapter/releases)) with their sha256 sidecars
  (`ggml-cuda_plugin.sha256`, `ggml-rocm_plugin.sha256`, `mlx_plugin.sha256`) and verified before they load.
  **Upgrading:** run `CALL decide_accelerate();` once. The old plugin files (`libanofox_decide_*_plugin.*` and the ONNX
  Runtime libraries next to them) are no longer used and can be deleted; until the new plugin is installed, `'auto'`
  scores on the CPU and an explicit device names the fix. decide no longer publishes plugins on its own releases.
- `decide_download_runtime('cuda')` and `decide_accelerate()` fetch the plugin alone: no ONNX Runtime wheel, and no
  CUDA toolkit or cuDNN to install.
- ROCm no longer compiles per shape: no shape buckets, no `.mxr` cache and no bucket limit on text length.
  `decide_gpu_precompile` is still accepted on ROCm but compiles nothing (it only loads the model onto the GPU).
- **A model registered with its own `.onnx` runs on a GPU too** (it used to be refused there): the plugin gets the
  whole graph, which carries its weights. A graph the plugin cannot run (an op it lacks, weights in an external file)
  is remembered as not servable on that device: `'auto'` then scores it on the CPU and `decide_backends()` says why.
- `SET threads` takes effect on the next call of a local model on the ggml engine (its session used to keep the thread
  count it was built with).
- `__anofox_decide_local_sessions()` reports the engine `plugin:cuda`, `plugin:rocm` or `plugin:mlx` for a GPU session.
- Verified with the local end-to-end suite on the GPU (`test/sql/decide_local_e2e_gpu.test`: probabilities within
  1e-3 of the CPU reference, labels exact, every block served by the plugin) on AMD gfx1150 and gfx1201 (ROCm) and an
  Apple M3 (MLX).

### Deprecated
- `anofox_decide_plugin_dir`, the old name of `anofox_decide_ep_path`: read when `anofox_decide_ep_path` is empty,
  removed in the next release.

### Removed
- ONNX Runtime: it is no longer built, linked or downloaded (no vcpkg `onnxruntime` port, no `ort-vcpkg` feature, no
  `DECIDE_ORT_VCPKG`, no ONNX Runtime CUDA runtime download). The extension links only OpenSSL and the vendored engine.
- decide's own GPU plugins (`plugins/cuda`, `plugins/migraphx`, `plugins/mlx`, plugin ABI v1): the CUDA, ROCm and MLX plugins are
  anofox-ggml-adapter's.

### Fixed
- A GPU plugin that runs out of device memory is reported as such, with the batch-size remedy (the adapter's "out of device memory"
  was reported as a generic plugin failure). A missing runtime library names what that backend's plugin links: the NVIDIA driver,
  ROCm's HIP, hipBLAS and rocBLAS, or `brew install mlx mlx-c`.

## [2026.11.24] - 2026-11-23

### Fixed
- DuckDB v1.5.6 binaries carry the release tag as their extension version (they were stamped with the short
  commit SHA: DuckDB 1.5.6 passes the tag pattern to `git describe --match` with literal single quotes, so no tag
  matched). `extension_config.cmake` now passes `EXTENSION_VERSION` itself when HEAD is on a `vX.Y.Z` tag.

### Known issues
- The CUDA plugin is still not attached (see 2026.11.22).

## [2026.11.23] - 2026-11-22

### Fixed
- Release deploys to get.erpl.io upload with a repo-local copy of the aws-cli upload script
  (`scripts/extension-upload-single.sh`). DuckDB 2.0.1's own copy needs rclone and `AWS_ENDPOINT_URL`, so every
  DuckDB 2.0 deploy of v2026.11.22 failed and that release has no 2.0.1 binaries on get.erpl.io; this release does.

### Known issues
- The CUDA plugin is still not attached (see 2026.11.22).

## [2026.11.22] - 2026-11-22

### Changed
- **Builds on DuckDB v2.0.1** (was v2.0.0), the version community-extensions builds against. The 2.0 line in CI
  now builds every platform on pull requests too, runs the smoke test against the released v2.0.1 CLI, and
  deploys. The v1.5.6 line is unchanged. Local builds default to v2.0.1 (the `duckdb` submodule).
- Release deploys write the versioned path (`anofox_decide/vYYYY.MM.DD/...`) only from the tag run, so a
  `main` push of the same commit can no longer overwrite a release with its pre-tag build.
- The ONNX Runtime source archive (v1.29.0) falls back to a DataZoo mirror
  (`DataZooDE/onnxruntime-mirror`, the unmodified upstream archive, same SHA512) when github.com/microsoft/onnxruntime
  is unreachable.

### Known issues
- The CUDA plugin is not attached to this release: its build downloads the prebuilt ONNX Runtime GPU archive from
  github.com/microsoft/onnxruntime, which is unavailable. The MIGraphX and MLX plugins are attached as before.

## [2026.11.13] - 2026-11-13

### Added
- **Claude Code plugin.** `/plugin marketplace add DataZooDE/anofox-decide`, then
  `/plugin install anofox-decide@anofox-decide`, installs four skills (`anofox-decide-questions`,
  `anofox-decide-models`, `anofox-decide-local` and `anofox-decide-evaluate`). They teach Claude Code every function,
  question format, provider, key and setting, the error messages and their fixes, local models and GPUs, and
  evaluation and calibration. The docs tests check them like the documentation: their SQL runs on the stub model,
  every name they use exists, every function has a skill, and their versions follow the release. The README,
  Getting Started and `docs/BUILDING.md` (which now also lists every version string to bump at a release) describe it.
- **Gemini Decisions** (`gemini-decision-1`, `POST https://generativelanguage.googleapis.com/v1beta/decisions`) as
  the provider `gemini`: `export GEMINI_API_KEY=...`, then `SELECT decide_register_model('gemini-decision-1',
  'gemini');`. The same functions and questions as the other models; up to 1000 questions per request. Question
  ids travel as `q0`, `q1`, ... (Gemini only takes `[A-Za-z0-9_-]{1,64}`), so any id works and every message names
  yours. The key goes in the `x-goog-api-key` header. A request Gemini blocks for safety reasons (HTTP 400
  `safety`, `prohibited_content`, `blocklist`, `image_safety`, `content_blocked`) is a refusal of every question of
  that row (NULL, `kind = 'refusal'`), not a failed query. A bad key (Gemini answers HTTP 400) is reported as a key
  error. Text only; a JSON state is sent as text.
- **One credential scheme for every online service.** `CREATE SECRET (TYPE anofox_decide, SERVICE '<id>',
  API_KEY '...')` with `SERVICE` one of `typesafe`, `liquid`, `cloudflare`, `openai`, `gemini`, `systemone`,
  `huggingface`. Without `SCOPE` the secret belongs to the service's default host and to that service only: it
  never follows a redirected endpoint and never reaches another provider at the same host. `SCOPE` (a host) serves
  a self-hosted endpoint; `SERVICE 'systemone'` requires it. `ACCOUNT_ID` (Cloudflare only) is redacted like the
  key, and the account id order is the `account_id` option, then the secret, then `CLOUDFLARE_ACCOUNT_ID`.
- **Hugging Face tokens** for `decide_download` from gated repositories: a `SERVICE 'huggingface'` secret, else
  `HF_TOKEN`. The token is sent only to the catalog host and dropped when a download is redirected to another
  host (the CDN). A 401/403 names `HF_TOKEN` and the secret.
- `decide_services()` lists every online service: default host, env var, secret fields, whether a key is found
  and where it comes from (never the key). `decide_models()` has a `key_source` column, and `decide_doctor()` shows
  where each remote model's key comes from (also while remote calls are off) and an `api keys` row.
- Every service reports a missing key with the same message: `no API key for <Service> at <host>. Fix: provide
  one: (1) CREATE SECRET (TYPE anofox_decide, SERVICE '<id>', API_KEY '<key>'); (2) export <ENV>=<key> before
  starting DuckDB`.

### Changed
- **Key precedence is the same for every service:** a stored secret, then the model's `key_env`, then the
  service's env var (default host only), then, for `typesafe` only, the `anofox_decide_api_key` setting. The
  setting used to come second (before `key_env` and `TYPESAFE_API_KEY`); it now comes last and is
  **deprecated**: use a secret. A session that sets both the setting and `TYPESAFE_API_KEY` (or a `key_env`) now
  sends the env key. The setting alone still works as before, and existing secrets without `SERVICE` keep their
  meaning.
  `decide_doctor()` shows a `warn` row while the setting is set, saying whether something else wins, and one per
  older-form secret without `SERVICE` and `SCOPE` (it reaches every remote host, also `gemini`). When several secrets
  match, DuckDB's own order applies (longest scope, then a `SERVICE` secret, temporary before persistent, then the
  name); `key_source` and key errors name the secret used.

### Fixed
- **A release no longer leaves two empty draft releases behind.** The three GPU plugin workflows each created a
  draft for the tag (a draft is not bound to its tag, so none of them saw the others'). They now share one
  (`tools/ci/release_draft.sh`: the first draft created for the tag; empty duplicates are deleted), and upload and
  publish address that release by id, so the assets can never split across drafts. The eight empty drafts of the
  2026.10.14 to 2026.10.31 releases were removed.
- **`anofox_decide_max_concurrency` now bounds what the provider sees.** The limit was kept per chunk of rows,
  so DuckDB threads scoring several chunks in parallel, or two queries at once, multiplied the requests in
  flight (a rate limit of 2 could see 4). It is now shared by every request to the same endpoint with the same
  key in the process, and a `429` halves the window for all of them. Each query still starts with the
  configured limit.

## [2026.10.31] - 2026-10-31

### Added
- **OpenAI Decisions** (`gpt-6-luna`, `POST https://api.openai.com/v1/decisions`) as the provider `openai`:
  `export OPENAI_API_KEY=...`, then `SELECT decide_register_model('gpt-6-luna', 'openai');`. The same functions
  and the same questions as for the other models; up to 400 questions per request (raise
  `anofox_decide_max_questions`, default 100, to send more than 100). A state that is a JSON object
  or array (`to_json(row)`, a JSON column) is sent as structured data (`input_json`); the model option
  `input` ('auto', 'text', 'json') overrides that. `prompt_cache_key` defaults to the model id (so the rows of one
  query share OpenAI's prompt cache) and can be set, or set to `''` to send none.
- **Refusals**: when a model declines to answer (OpenAI's `refusal` answer type) the scalars return NULL, and
  `decide_table` / `decide_answers` return the row with `kind = 'refusal'` and NULL answer fields, instead of
  failing the query. `decide_accuracy` and the other metrics skip NULLs; to score a refusal as wrong, coalesce it.
- `decide_models()` has two more columns, `input` and `prompt_cache_key` (OpenAI models only).
- Usage tokens (cached and uncached input) are not recorded yet.

### Fixed
- **Liquid D1**: Liquid renamed `d1:free` to `d1` (and added `d1.1` and `d1-ultra`); `d1:free` now answers HTTP 404.
  Every example, page and test uses `d1`. A Liquid 404 names the current models and the registration that fixes it.
- A server message that contains the API key is no longer replaced as a whole: the key is cut out of every
  response body before anything parses, quotes or truncates it, so neither the key nor a prefix of it can reach an
  error message.

### Changed
- Documentation restructured: short README, guides and API pages under docs/.
- Error messages and `decide_doctor()` hints link to the new pages under docs/ instead of README anchors
  (the old `#local-models` and `#models-and-providers` anchors still lead there).

## [2026.10.26] - 2026-10-26

### Added
- **`row_order` on the contract producers** (`decide_roc_curve_by`, `decide_calibration_curve_by`,
  `decide_metrics_by`): a last column, the contract's optional 1-based ordinal over the whole table in the
  producer's own order (DataZooDE/anofox-statistics#242, DataZooDE/anofox-visualization#445). The curves still
  combine with anofox-statistics' with `UNION ALL BY NAME`; a positional `UNION ALL` needs the same column on
  both sides.

## [2026.10.24] - 2026-10-24

### Added
- **anofox integration contract** (#34): decide's evaluation results plot in
  [anofox-visualization](https://github.com/DataZooDE/anofox-visualization) with no hand-written SQL.
  `decide_roc_curve_by(source, group, p, outcome[, drop_intermediate := true])` (scikit-learn `roc_curve`) and
  `decide_calibration_curve_by(source, group, p, outcome[, n_bins := 10])` (equal-width bins on P(yes), 95% Wilson
  intervals) return the contract's `curve` schema, column for column like anofox-statistics' `roc_curve_by`;
  `decide_metrics_by(source, group, p, outcome)` returns the `summary` schema (accuracy, auc, brier, ece,
  log_loss per group). They are tagged in `duckdb_functions()` (`anofox.contract = '1'`, `anofox.output`), every
  decide function carries `anofox.family = 'decide'`, and `anofox_contract_version()` returns `'1'` (registered
  only if no other anofox extension did). New aggregates `decide_auc` and `decide_log_loss`.
- **Visualization pickup check** (`scripts/visualization_pickup.py`, `make test_visualization`, a CI job that deploy
  waits for): the built extension next to a released anofox_visualization in the official DuckDB CLI; both load
  orders, the tags, and every producer's output through `anofox_plot`, which must auto-detect its schema.

### Changed
- **Builds on DuckDB 2.0** (#35), the line community-extensions builds against, as well as 1.5.6: one source
  tree, with the API differences behind the [duckdb-compat](https://github.com/DataZooDE/duckdb-compat) shims.
  On 2.0, `model := 'x'` and the other named arguments bind by name again. CI has a DuckDB 2.0 build and test job.
- **DuckDB 1.4.x LTS is no longer supported** (#36): the supported lines are 1.5.6 and 2.0. Published 1.4.5
  binaries stay where they are.

## [2026.10.14] - 2026-10-14

GPU acceleration for the local models through plugins (NVIDIA CUDA, AMD ROCm/MIGraphX, Apple MLX), cross-row
batching, and fail-closed plugin verification. The extension itself stays one CPU-only file.

### Security
- **Plugin verification no longer fails open.** A plugin is downloaded under a temporary name and moved into place
  only after its sha256 matches the checksum file published with the release; every failure (a mismatch, a
  checksum file that is missing, unreadable or does not name the plugin) deletes the plugin, and `decide_accelerate()`
  does not probe a plugin whose download failed. The extension also checks a plugin against the `.sha256` file beside
  it once per process before it first loads it (cached by path, size and modification time; a plugin placed by hand
  with no `.sha256` file is trusted, and the docs say so). The extracted CUDA runtime files have their sha256 pinned
  and are checked on every install call. Downloads refuse redirects off https (and, for GitHub release assets, off
  GitHub's hosts), scratch files are private to the process, and installing into a plugin directory that other users
  can write to is refused (POSIX). The checksum is integrity from the same release, not authenticity: see
  `docs/GPU_BACKENDS.md` ("What is verified, and what is trusted").

### Fixed
- The plugin probe and the GPU session cache are keyed by the plugin file's identity and directory, and sessions are
  dropped when a plugin is installed, so a plugin replaced by `decide_accelerate()` or by hand is used at once.
- MIGraphX compiles in a temporary directory unique per model and shape (two models compiling the same bucket in one
  process could overwrite each other's inputs), and the plugin refuses null input buffers.
- `decide_gpu_precompile` is refused on backends that do not compile per shape instead of listing buckets it did not
  compile, and a `seq` longer than every bucket is an error naming the largest bucket.

### Added
- **GPU release assets, evaluation by device, GPU documentation.** A `v*` tag now attaches all three plugins
  (`libanofox_decide_cuda_plugin.so`, `libanofox_decide_migraphx_plugin.so`, `libanofox_decide_mlx_plugin.dylib`)
  and their `cuda_plugin.sha256` / `migraphx_plugin.sha256` / `mlx_plugin.sha256` sidecars to the release, under the
  names `decide_accelerate()` and `decide_download_runtime()` ask for. The three plugin workflows share one reusable
  publish workflow (`_gpu_release_assets.yml`) that checks the file names, that the sidecar lists the plugin and
  its digest, and that `DECIDE_PLUGIN_RELEASE_TAG` equals the tag, uploads to a draft release and publishes it only
  when all six assets are attached (`tools/ci/release_assets_complete.sh`); a manual run verifies without
  releasing; the new `GPU release gate` workflow
  (`tools/ci/check_plugin_pin.sh`) fails a tag build at once when it does not. `tools/eval/run_eval.py` gained
  `--device auto|cpu|cuda|rocm|mlx`: it sets `anofox_decide_device`, refuses to report (exit status 2, no report,
  results deleted) any model that `decide_models().device` shows was not served on the requested device, and records
  the serving device and wall time; catalog models (`laya-multilingual`, `laya-typed-decisions`, `julia-1`) can be
  evaluated by it, which is what the GPU backends serve. Tests: `tools/eval/test_run_eval.py` (no GPU needed).
  Documentation: the README section "GPU acceleration" and `docs/GPU_BACKENDS.md` with the measured numbers
  (200-ticket evaluation, fp32: Laya multilingual CPU 97.0 s against 11.8 s on CUDA RTX 4090, 51-81 s against 9 s on
  ROCm RX 9070 XT, 80.8 s against 9.5 s on MLX Apple M3, each CPU from the same box; largest probability difference
  to the CPU 4e-5, no routing or refund decision changed, accuracy identical), requirements, limits (fp32 only, no
  Windows or CUDA 13, ROCm compile 2 to 5 minutes and 0.6 to 1.75 GB per shape) and troubleshooting.
- **ROCm backend (MIGraphX plugin).** `plugins/migraphx/decide_migraphx_plugin.cpp` builds
  `libanofox_decide_migraphx_plugin.so`, which drives MIGraphX directly through `libmigraphx_c` (no ONNX Runtime) and
  scores the catalog models on AMD GPUs. Fixed shapes through buckets (20 markers; Laya 256 or 1024 tokens, Julia-1
  256 to 8192; one row or a full batch of at most 8192 padded tokens), a compiled-program cache
  (`<cache dir>/gpu-cache/migraphx/*.mxr`, keyed by graph, tensor map, every byte of the checkpoint, GPU
  architecture and shape; a damaged file is rebuilt), weights from the packed blob it builds from the safetensors
  and the tensor map while compiling, fp32 only, compile progress on stderr. New `decide_gpu_precompile(model[,
  seq := N])` compiles the buckets ahead of time and reports the seconds each took. A separate workflow
  (`gpu_plugin_migraphx.yml`) builds the plugin and checks the exported ABI symbol, its dynamic dependencies and a
  real dlopen with refusal cases. Measurements and limits: `docs/GPU_ROCM.md`.
- **MLX plugin source (`plugins/mlx`, macOS arm64 only).** The Apple GPU backend of the plugin ABI: an
  interpreter that runs the shipped ONNX graphs over Apple's `libmlx`/`libmlxc` (no ONNX Runtime, no protobuf),
  weights uploaded once from the safetensors file through the tensor map (F16 and BF16 checkpoints cast to fp32 on
  the device), last-use eviction of intermediates, and a token budget derived from the machine's RAM that splits
  a batch into chunks and always scores at least one row (Julia-1 at 8192 tokens runs as one row on a 16 GB Mac).
  Built by its own CMake target and the `MLX plugin` workflow (build, exported-symbol and dependency checks,
  dlopen and refusal smoke test; CI has no Apple GPU); Linux and Windows configures skip it. Its release asset is
  `libanofox_decide_mlx_plugin.dylib` with `mlx_plugin.sha256`, fetched by `decide_download_runtime('mlx')`.
- **GPU infrastructure.** The extension stays one CPU-only file; a GPU backend is a plugin it loads behind one C
  interface (`decide_plugin_abi.h`, ABI version 1). Everything around the plugins lives in the extension, so each
  backend is a library that needs no change to it: the loader (ABI checked before the
  library is used, `RTLD_DEEPBIND` on Linux, never unloaded), device discovery (NVML, the KFD topology with the
  MIGraphX architecture list, the Apple SoC), `decide_devices()`, `decide_backends()` (one servability predicate
  shared with device selection), `decide_accelerate()` and `decide_download_runtime('cuda' | 'rocm' | 'mlx')`
  (platform refusals before any download, sha256 sidecar verification of every plugin before it is moved into place
  and again on every later install call, the pinned CUDA runtime with its SONAME rename), and the settings `anofox_decide_device` (`auto` by default, which
  uses a GPU only when its plugin is installed, the device is usable and the plugin serves the model; explicit
  devices that cannot serve a model are hard errors naming `CALL decide_accelerate()`),
  `anofox_decide_gpu_precision` (`fp32` only; other values are rejected) and `anofox_decide_plugin_dir`.
  `decide_models()` gained a `device` column and `decide_doctor()` rows for the GPU and its plugin. The session
  cache is keyed by device and precision. Without a plugin installed nothing changes: local models score on the
  CPU exactly as before.
- **`anofox_decide_batch_tokens`** (default `0` = off): local models score the rows of a whole chunk together,
  in batches of at most this many padded tokens (batch size times padded length, `1024` to `131072`; one row
  always fits). Rows are collated exactly as before (so truncation behaves the same), sorted by length, padded
  to 128/256/.../8192 (never above the model window) and run in as few ONNX Runtime calls as the budget allows;
  answers come back in the original order. Padding is masked exactly, so answers match one-text-at-a-time scoring
  exactly: on the 200-ticket evaluation with Laya multilingual (400 questions) the probabilities were identical
  to the last bit, with no routing or refund-decision differences, at 16384 and at 4096. Off by default because
  on CPU the effect is small and not consistent: wall time for the 200 tickets was 112 s and 122 s with it off,
  98 s and 103 s at 16384 and 146 s at 4096 (one machine under heavy load, load average 58 to 69 on 32 cores,
  so indicative only). It is the prerequisite for GPU execution, where a batch of eight rows measured about the
  same time as one. `tools/eval/run_eval.py` gained `--batch-tokens`.
- **CUDA plugin (`plugins/cuda`).** The NVIDIA backend behind the plugin interface:
  `libanofox_decide_cuda_plugin.so`, built as its own CMake project with its own workflow
  (`gpu_plugin_cuda.yml`). It links a private ONNX Runtime core with the SONAME renamed to `libanofoxort_gpu.so`
  (the core and the CUDA provider come from the pinned Microsoft wheel that `decide_download_runtime('cuda')`
  fetches; CUDA 12.8+ and cuDNN 9 stay the user's), opens the staged graph with the weights injected from its own
  mapping of the safetensors file (float32 tensors in place, F16/BF16 upcast), and runs the CUDA provider with TF32
  switched off so `fp32` means strict fp32. Shapes are dynamic, so `precompile` does nothing. Tested without a GPU:
  the artifact's symbols and `DT_NEEDED`, a real `dlopen` with the loader's flags, clean refusals, the weight
  feeding on ORT's CPU provider against reference scores from Python onnxruntime, and a host that already holds an
  upstream-named ORT core. Hardware validation (`plugins/cuda/gpu_test`) is a separate, manual step.

### Changed
- **The committed weight-free graphs carry a packed weight layout** (groundwork for the GPU backends). Every
  external-data stub now has an `offset` and a `length`, and each `tensor_map_*.json` has a `pack` section: the
  injected float32 weights in one blob, initializers in sorted name order, 64-byte aligned (0.58 / 1.29 / 1.68 GB
  for Julia-1 / Laya multilingual / Laya typed-decisions). A GPU plugin builds its blob from the downloaded
  safetensors and the map alone. Nothing else in the graphs or maps changed and CPU scores are bit-identical;
  all stubs use the shared `weights.bin` location. `tools/export_julia/scripts/repack_offsets.py` upgrades older graphs,
  `verify_pack.py` checks a blob against a checkpoint, and `check_graph_invariants.py` (CI) now requires and
  verifies the layout.

## [2026.10.04] - 2026-10-04

Local models that work after `INSTALL`, Cloudflare Clef, relational scoring at scalar speed, and local models
that no longer cut text silently.

### Added
- **`decide_answers(state, questions[, model])`**: the rows of `decide_table` as a `LIST(STRUCT(question_id, kind,
  probability, choice, confidence, model, score, distribution))`, one list per text. `unnest(decide_answers(...),
  recursive := true)` over a table gives `decide_table`'s columns at scalar speed: a chunk's requests run
  concurrently (`anofox_decide_max_concurrency`), identical texts are sent once and calibration applies, which a
  `LATERAL decide_table` join cannot do. NULL state or questions give a NULL list (no rows after unnest). Both
  functions build their rows with one shared builder, so they always agree. `tools/eval/run_eval.py` now scores
  through it and has a `--concurrency` option.
- **Local models work out of the box.** `CALL decide_download('laya-multilingual')` (also `laya-typed-decisions` and
  `julia-1`) fetches the upstream weights from Hugging Face into `~/.cache/anofox-decide`
  (`anofox_decide_cache_dir`), pinned to a fixed revision, with size and SHA-256 verification, HTTP Range resume
  and redirect following over the bundled HTTPS client (no `INSTALL httpfs`, no Python). The extension embeds the
  weight-free ONNX graphs; the weights are injected into ONNX Runtime at load (float16 checkpoints are upcast).
  `model := 'laya-multilingual'` then works with no registration; before the download it fails with
  `model '...' is not downloaded. Fix: CALL decide_download('...');`. `decide_models()` lists the catalog models
  with `ready = false` until downloaded and `decide_doctor()` reports them. Registering your own exported graph
  with `decide_register_model(..., 'local', ...)` still works. `tools/export_julia --weight-free` produces the
  embedded graphs and tensor maps; CI checks that the committed graphs carry no weight bytes.
- **Cloudflare Clef** as a hosted provider: `decide_register_model('clef', 'cloudflare')` (or `'clef-flash'`) with
  `CLOUDFLARE_API_TOKEN` and `CLOUDFLARE_ACCOUNT_ID` (or `MAP {'account_id': '...'}`). The Workers AI response
  envelope (`{"result": ...}`) and its `errors[]` are understood for every provider; the account id and model go
  into the URL; a request takes at most 64 questions (the provider's own limit, checked before anything is sent);
  a 401 names both causes Cloudflare cannot tell apart (token and account id). Measured on the 200-ticket
  evaluation: `clef` 85.0% refund and 88.0% routing, `clef-flash` 70.0% and 78.5% (see `docs/EVALUATION.md`).
  Images are not supported. `make test-live` runs a live Cloudflare test when both variables are set.
- `decide_token_count(text[, model])` returns how many tokens a text has for a local model, so you can find the
  rows that do not fit its window. Hosted and stub models raise an error that says so.
- `anofox_decide_on_truncate` (`'error'` by default, or `'ignore'`) for local models.

### Changed
- **Local models no longer cut text silently.** Text, question or options that do not fit the model's window
  now raise an error that says how many tokens would be ignored and for which question; set
  `anofox_decide_on_truncate = 'ignore'` for the old behaviour. Laya models take their limits from
  `rl_agent_config.json` and the message says so. (Before, the cut was silent, and an over-long option was never
  reported at all.)
- Local scoring errors use the public vocabulary (`binary`, not `noul`), name the function you called and the
  question, and no longer mention internal rules.
- **Registration checks local files deeper than "can be opened".** A weights file, text, JSON, HTML, zip or pickle
  as the graph, a tokenizer that is not BPE (for example Unigram), and a Laya config without its keys are rejected
  at `decide_register_model` with a message that names the problem; a `rl_agent_config.json` next to a graph
  registered without a profile is reported as a probable Laya checkpoint. `decide_models()` and
  `decide_doctor()` use the same checks, so they no longer say `ready` for files registration would reject.
- ONNX Runtime failures (an unreadable or damaged graph, an unsupported ONNX version, running out of memory) are
  reported as `<function>: <what>. Fix: <what to run>` instead of the raw ONNX Runtime text; a graph with other
  inputs than a local model is fed names the missing and the unused inputs.
- The bundled ONNX is now built with `ONNX_DISABLE_STATIC_REGISTRATION=ON`, which stops the 634
  `Schema error: ... already registered` lines that the first local model load used to print on stderr.
  Checked on the release-build artifacts of the CI run: the build without the flag printed 634 such lines
  (1268 stderr lines in all), the build with it printed none, and the model answered the same.

## [2026.10.03] - 2026-10-03

First public release.

### Added
- **Natural-language decisions as SQL functions.** Three kinds of question, all written at query time:
  `binary` (`decide_probability`, and `decide_decision` with an explicit threshold), `choice` (one of an
  option list you define, `decide_choice`) and `score` (an ordered rubric of 2 to 10 levels, lowest first;
  `decide_score` returns the expected 0-based level). `decide_many` answers several questions about one
  text in a single request as JSON, and `decide_table` returns the same answers as rows
  (`question_id`, `kind`, `probability`, `choice`, `confidence`, `model`, `score`, `distribution`).
  Every function is available as `anofox_decide_<name>` and as the short alias `decide_<name>`, and
  documents itself in `duckdb_functions()` (description, parameter names, a runnable example, categories).
- **Hosted, local-server and in-process models behind one interface.** Providers: `typesafe` (TypeSafe
  Jev), `liquid` (Liquid AI D1), `strands` (a local, keyless
  [strands-decider](https://github.com/strands-labs/strands-decider) server), `systemone` (any compatible
  server, e.g. Kev), `local` (Julia-1 and Laya on ONNX Runtime, CPU, offline after setup) and the built-in
  `stub` test model. Per-model options (`endpoint`, `path`, `model`, `key_env`, `criteria`) go in a `MAP`
  on `decide_register_model`; `decide_unregister_model` removes a model so it can be registered again.
- **Keys that stay where they belong.** Environment variables work out of the box and a DuckDB secret
  (`CREATE SECRET (TYPE anofox_decide, API_KEY '...', SCOPE '<host>')`, write-only, redacted in
  `duckdb_secrets()`) always overrides them. A key is only ever sent to the host it was configured for;
  cleartext `http://` is limited to loopback. Hosted models need the explicit opt-in
  `SET anofox_decide_allow_remote = true`.
- **Model quality metrics.** `decide_accuracy`, `decide_brier_score` and `decide_ece` aggregate over
  `(probability, label)` pairs, so comparing models is one query. A 200-ticket evaluation of every model
  on public data (Bitext customer support) is in `docs/EVALUATION.md`; `tools/eval` reproduces it.
- **Per-model calibration of yes/no probabilities.** Some models rank well but are wrong at the 0.5 cut-off.
  `decide_fit_calibration(p, outcome)` fits Platt scaling to a labelled sample and returns a `'platt:a,b'` spec;
  register a model with `MAP {'calibration': 'platt:a,b'}` (every provider; local models take the MAP as the sixth
  argument) and every yes/no probability from it is calibrated. The ranking never changes, and choice and score
  answers are untouched. `decide_models()` gains a `calibration` column.
- **`decide_choice_distribution(state, question, options[, model])`** returns the probability of every option as a
  `MAP(VARCHAR, DOUBLE)`, so the winner's probability can be thresholded and a low-confidence row left undecided in plain SQL.
- **Diagnostics.** `decide_doctor()` checks the whole setup (default model, remote opt-in, files, keys and
  where they came from, endpoints, telemetry) and gives the fix for each problem; `decide_models()` lists
  every registered model with `ready`, a `hint` for what to do when it is not, the profile, endpoint and
  wire model.
- **Concurrent scoring of many rows.** The scalar functions evaluate a chunk of rows together: identical
  `(model, text, questions)` requests are sent once, up to `anofox_decide_max_concurrency` remote requests
  run at the same time (default automatic: 8 for hosted providers, 1 for the local strands server),
  connections are reused, a `429` halves the window and `Retry-After` is honoured. On the real Liquid D1
  service, 8 tickets with three questions each took 176.7 s one request at a time and 23.1 s concurrently,
  with the same answers. `LATERAL decide_table` stays one row at a time: DuckDB hands an in-out function a
  single row per call there (documented in the README).
- **Settings:** `anofox_decide_model`, `anofox_decide_allow_remote`, `anofox_decide_max_concurrency`,
  `anofox_decide_timeout_ms`, `anofox_decide_max_retries`, `anofox_decide_max_questions`,
  `anofox_decide_max_length`, `anofox_decide_head_length`, and the legacy `anofox_decide_api_key` and
  `anofox_decide_endpoint` for the `typesafe` provider. Every setting is validated and its error shows the
  range, the default and an example `SET`.
- **Examples** (`examples/`): the whole SQL surface, support-ticket triage with accuracy and calibration,
  scoring many rows, comparing models, and a local model. The offline ones run in the test suite on every
  build and the others are parsed, so they cannot drift from the extension.
- **Anonymous usage telemetry and the feedback banner**, shared with the other anofox extensions (opt out
  with `DATAZOO_DISABLE_TELEMETRY=1` or `SET anofox_telemetry_enabled = false`; see `TELEMETRY.md`).
- Releases are published to the anofox extension repository (`get.anofox.com`) on every `v*` tag and every push to `main`, in addition to the DuckDB community repository; the README documents installing from either.

### Changed
- **There is no implicit model.** A call that names no model and has no `anofox_decide_model` fails with
  instructions for choosing and registering one, instead of quietly returning the `stub`'s constant. The
  `stub` is for trying the SQL and says so everywhere it appears.
- **Errors say what to do.** Every error has the shape `<function>: <what went wrong>. Fix: <what to run>`
  and echoes the value that caused it. Remote failures quote the service's own message and give the next
  step per status (401 names where the key came from, 404 points at the model option, 422 explains the
  `criteria` option, 429 and 5xx say how many attempts were made), a local server that is not running says
  to start it and is not retried, and the opt-in check comes before the key check. Plain user mistakes no
  longer carry the "Unexpected? Please report it" footer, which is kept for genuinely unexpected errors.
- **The documentation matches the extension.** `decide_many` is documented as returning
  `{"model", "results": [...]}`; `decide_table` documents the `score` kind; function examples run as
  written. A test checks that every function and setting the README names exists.

### Known limitations
- Local models are not downloaded for you: export a checkpoint to ONNX once with `tools/export_julia`.
  GPU execution for local models is not supported, and neither is the Von model.
- `score` is verified for parity with each model's upstream implementation and against the live services,
  but has not been scored for accuracy on labelled data.
- Text longer than a local model's limit is cut off without a warning.
- Platforms: Linux (amd64, arm64), macOS (arm64) and Windows (amd64) against DuckDB 1.5.6. Not built:
  macOS on Intel (ONNX Runtime ships no archive after v1.23.2), WebAssembly, musl and MinGW.

[Unreleased]: https://github.com/DataZooDE/anofox-decide/compare/v2026.11.24...HEAD
[2026.11.24]: https://github.com/DataZooDE/anofox-decide/compare/v2026.11.23...v2026.11.24
[2026.11.23]: https://github.com/DataZooDE/anofox-decide/compare/v2026.11.22...v2026.11.23
[2026.11.22]: https://github.com/DataZooDE/anofox-decide/compare/v2026.11.13...v2026.11.22
[2026.11.13]: https://github.com/DataZooDE/anofox-decide/compare/v2026.10.31...v2026.11.13
[2026.10.31]: https://github.com/DataZooDE/anofox-decide/compare/v2026.10.26...v2026.10.31
[2026.10.26]: https://github.com/DataZooDE/anofox-decide/compare/v2026.10.24...v2026.10.26
[2026.10.24]: https://github.com/DataZooDE/anofox-decide/compare/v2026.10.14...v2026.10.24
[2026.10.14]: https://github.com/DataZooDE/anofox-decide/compare/v2026.10.04...v2026.10.14
[2026.10.04]: https://github.com/DataZooDE/anofox-decide/compare/v2026.10.03...v2026.10.04
[2026.10.03]: https://github.com/DataZooDE/anofox-decide/releases/tag/v2026.10.03
