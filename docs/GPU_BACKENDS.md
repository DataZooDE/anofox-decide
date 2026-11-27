# GPU backends (CUDA, ROCm, Apple MLX)

The extension is one CPU-only file. A GPU backend is a separate plugin it loads when asked. The plugins are
[anofox-ggml-adapter](https://github.com/DataZooDE/anofox-ggml-adapter)'s, shared with anofox-tabfm, and speak its
generic plugin interface (plugin ABI v3, `vendor/anofox-ggml-adapter/include/anofox_ggml_adapter/plugin_abi.h`).
Three exist (on Windows the names drop `lib` and end in `.dll`):

| Backend | Device | Plugin | Source | Verified on |
|---|---|---|---|---|
| CUDA | NVIDIA, sm_75 or newer | `libanofox_ggml-cuda_plugin.so` | adapter `src/plugin_ggml.cpp` | (in progress) |
| ROCm | AMD | `libanofox_ggml-rocm_plugin.so` | adapter `src/plugin_ggml.cpp`, [docs/GPU_ROCM.md](GPU_ROCM.md) | gfx1150, gfx1201 |
| MLX | Apple silicon | `libanofox_mlx_plugin.dylib` | adapter `src/plugin_mlx.cpp` | Apple M3 |

All three run the same ONNX graphs as the CPU, in fp32: the CUDA and ROCm plugins on the same ggml engine as the
CPU, the MLX plugin on Apple's MLX. Each gets the extension's embedded weight-free graph and the weights the
extension reads from the downloaded safetensors file; nothing is staged to disk. They are an option for speed, not a
different model. The local end-to-end suite (`test/sql/decide_local_e2e_gpu.test`: 40 tickets, all three catalog
models) passes on every verified device with every probability within 1e-3 of the CPU reference, the labels exact
(where the reference is not a near tie), and every block served by the plugin: `__anofox_decide_local_sessions()`
reports the engine `plugin:cuda`, `plugin:rocm` or `plugin:mlx` for a GPU session.

## Turning it on

```sql
SELECT * FROM decide_devices();     -- cpu, plus any NVIDIA / AMD / Apple GPU found; usable, and why not
CALL decide_accelerate();           -- find the GPU, download and verify its plugin, use it from now on
SELECT * FROM decide_backends();    -- per local model and device: supported, or the reason and the fix
SELECT model, device FROM decide_models();   -- where each local model is scored right now
SET anofox_decide_batch_tokens = 8192;       -- advisable on a GPU: score a whole chunk of rows per call
```

`decide_accelerate()` is `decide_download_runtime(<backend>)` for the device it finds. To fetch one explicitly:
`CALL decide_download_runtime('cuda');` (or `'rocm'`, `'mlx'`). Both return one row per file, and the plugin is
used from then on in the same session (and by every new session) under `auto`.
`anofox_decide_ep_path` is where plugins are installed and loaded from (default: `plugins` inside the cache
directory; the deprecated `anofox_decide_plugin_dir` is read when it is empty), and a plugin built against another
interface version or struct layout is refused with the fix.

### What is verified, and what is trusted

Plugins are native code the extension loads, so the checks are spelled out here exactly:

- **At download.** The plugin is fetched under a temporary name and moved into place only after its sha256 equals
  the line for it in the `.sha256` file published next to it in the pinned release of anofox-ggml-adapter
  (`ggml-cuda_plugin.sha256`, `ggml-rocm_plugin.sha256`, `mlx_plugin.sha256`). If the
  checksum file cannot be fetched, does not name the plugin, or disagrees, the plugin is deleted (and the
  checksum file with it); nothing unverified stays where it could be loaded. Redirects must stay on https and, for
  GitHub release assets, on GitHub's own hosts.
- **On every later `decide_accelerate()` / `decide_download_runtime()` call.** The installed plugin is hashed
  and compared with its checksum file again (and the checksum file is re-fetched when it is stale). A mismatch
  deletes the plugin.
- **Before the extension first loads a plugin in a process** (the probe behind `decide_backends()` and `auto`, or
  scoring). If a `.sha256` file sits beside the plugin it is checked once, keyed by the file's path, size and
  modification time, so scoring does not rehash and a replaced file is checked again. The check hashes the file,
  confirms it did not change meanwhile, and then loads it by path: a process that can write to the plugin
  directory could still swap the file between the check and the load. The directory is part of the trust boundary,
  and `decide_download_runtime` refuses to install into one that other users can write to.
- **A plugin you placed by hand, with no `.sha256` file beside it, is trusted** and loaded as it is.
- **The checksum file comes from the same release as the plugin**, so all of this is integrity (damage, a swapped
  asset), not authenticity: it does not defend against a compromised release or a local attacker who can write to
  the plugin directory.

The models from `decide_download` and models registered with their own `.onnx` graph both run on a GPU: a catalog model's
plugin gets the weight-free graph and the safetensors weights, a registered model's plugin the whole graph with its weights.

| Setting | Meaning |
|---|---|
| `anofox_decide_device` | `auto` (default): a GPU only when its plugin is installed, the device is usable and the plugin serves the model, otherwise the CPU. `cpu`: always the CPU. `cuda` / `rocm` / `mlx`: that device, or an error naming `CALL decide_accelerate()`; never a silent CPU run |
| `anofox_decide_batch_tokens` | Score the rows of a chunk together, in batches of at most this many padded tokens. `0` (default) scores one text at a time. A GPU waits on one row per call, so set it on a GPU (`8192` to `16384` in the runs below). It is **not** a CPU speed-up: on CPU its effect is small and inconsistent |
| `anofox_decide_gpu_precision` | `fp32` only. Other values are rejected, never silently run in fp32 |
| `anofox_decide_ep_path` | Where plugins are installed and loaded from |
| `anofox_decide_plugin_dir` | Deprecated: the old name of `anofox_decide_ep_path`, read when it is empty; removed in the next release |

`decide_gpu_precompile('<model>')` compiled a model's shape buckets ahead of time for the former MIGraphX plugin.
The ggml and MLX plugins build a shape's graph on its first use, so it is no longer needed: it is still accepted on
ROCm, where it compiles nothing (it only loads the model onto the GPU and returns one row per former bucket), and refused on CUDA, MLX and the CPU.

## Requirements

| Backend | You need | The extension fetches |
|---|---|---|
| CUDA | Linux x86-64, an NVIDIA GPU of compute capability **sm_75 or newer** and the NVIDIA driver. The plugin links the CUDA runtime (12.8) and cuBLAS statically and needs only the driver's `libcuda.so.1`: no CUDA toolkit, no cuDNN. An older card is shown by `decide_devices()` as not usable, with its compute capability. Windows is not published yet | the plugin only |
| ROCm | Linux x86-64, a ROCm install (the plugin links the system's HIP, hipBLAS and rocBLAS; it is built against ROCm 7.2) and a GPU whose architecture the plugin carries code objects for: gfx90a, gfx942, gfx1030, gfx1100, gfx1101, gfx1102, gfx1150, gfx1151, gfx1200, gfx1201. Any other architecture is shown by `decide_devices()` as not usable, with its name | the plugin only |
| MLX | macOS on Apple silicon (arm64), `brew install mlx mlx-c` | the plugin only; it links Homebrew's `libmlx` and `libmlxc` |

Disk: the cached models are the ones `decide_download` already put in the cache directory; the plugins write no
compiled programs. Host memory on the 200-ticket evaluation: 1.0 to 1.4 GB per process on CUDA and ROCm, about 1.3 GB on
MLX ([Measured](#measured)). Device memory was measured only with the previous plugins (CUDA 1.6 to 2.6 GB of VRAM up to
4096 tokens), not yet with these.

## Measured

The current plugins against the previous ones (decide v2026.11.24: ONNX Runtime CUDA, MIGraphX, decide's own MLX plugin), each
pair on the same machine, engines alternating, warm. All numbers are fp32 and come from the 200-ticket evaluation of
[EVALUATION.md](EVALUATION.md): two questions per ticket (400 answers), the wall time includes session start and model load,
produced by `tools/eval/run_eval.py --device <device>`, which refuses to report a model that the requested device did not serve.
The full gate, the CPU rows included, is in [GGML_DECIDE.md](GGML_DECIDE.md).

**Wall time, 200 tickets** (previous plugin in parentheses)

| Model | CUDA, RTX 4090 | ROCm, RX 9070 (gfx1201) | MLX, Apple M3 |
|---|---|---|---|
| Julia-1 | 4 s (11 s) | 6 s (16 s) | 6 s (8 s) |
| Laya multilingual | 6 s (15 s) | 7 s (26 s) | 8 s (11 s) |
| Laya typed-decisions | 7 s (18 s) | 11 s (43 s) | 14 s (17 s) |

**Peak host memory (RSS)**

| Model | CUDA | ROCm | MLX |
|---|---|---|---|
| Julia-1 | 1.04 GB (1.78 GB) | 0.98 GB (2.59 GB) | 1.29 GB (1.30 GB) |
| Laya multilingual | 1.21 GB (4.02 GB) | 1.09 GB (4.61 GB) | 1.27 GB (1.27 GB) |
| Laya typed-decisions | 1.35 GB (6.05 GB) | 1.21 GB (5.74 GB) | 1.27 GB (1.28 GB) |

**Agreement:** no answer changed on any device (0 label flips over the 400 answers of every model, metrics identical), and no
probability moved by more than 4e-5 (MLX: bit-identical to the previous MLX plugin). The ggml plugins use more host CPU time than the
previous ones (on the RTX 4090 16 / 23 / 37 s against 12 / 16 / 20 s): the per-op launch work runs on the host.

The MLX row is measured with the MLX fixes of anofox-ggml-adapter v2026.11.26 (shape arithmetic on MLX's CPU stream, 16-bit weights
widened on the GPU, the buffer cache released); with the plugin of v2026.11.25 the same run took 13 / 15 / 23 s at 1.3 / 2.4 / 2.6 GB.

Test conditions: CUDA on an RTX 4090 (RunPod), the ggml-CUDA plugin of anofox-ggml-adapter v2026.11.25 against the previous ONNX Runtime 1.29
CUDA plugin of decide v2026.11.13 (the last release that shipped one). ROCm on gfx1201 with ROCm 7.2, against the previous MIGraphX 2.15 plugin. MLX 0.32
on an M3 with 16 GB.

## Limits

- **fp32 only.** Half precision needs a re-export of the graphs (the attention mask constants overflow in fp16
  and bf16) and is the next milestone; `anofox_decide_gpu_precision` rejects anything else.
- **CUDA:** Linux x86-64 and sm_75 or newer only. No Windows yet.
- **ROCm:** Linux x86-64 and the architectures listed under Requirements only.
- **MLX:** Apple silicon only.
- **Speed depends on the shape.** The GPU gains are largest for Laya typed-decisions and long texts; for short
  texts the CPU is already fast. Batching helps Laya multilingual and Julia-1 on a GPU, not Laya typed-decisions on
  MLX. Measure on your own data with `tools/eval/run_eval.py`.
- A registered `.onnx` that uses an ONNX op the plugin lacks is refused with the op names (score it on the CPU).
- The GPU backends are not run in CI (the hosted runners have none). The plugins are built, checked and released
  by anofox-ggml-adapter; scoring is validated by hand on the hardware above with
  `test/sql/decide_local_e2e_gpu.test`.

## Reproduce

`run_eval.py` sets the device but does not install a plugin: a machine without the plugin exits with status 2 on
`--device cuda`. Install it first, in any DuckDB session with the extension loaded:

```sql
CALL decide_accelerate();            -- or: CALL decide_download_runtime('cuda');   ('rocm', 'mlx')
```

```bash
python3 tools/eval/make_sample.py --out sample                       # the 200 tickets
python3 tools/eval/run_eval.py --tickets sample/tickets.csv --out out-cpu \
    --models laya-multilingual,laya-typed-decisions,julia-1 --device cpu
python3 tools/eval/run_eval.py --tickets sample/tickets.csv --out out-gpu \
    --models laya-multilingual,laya-typed-decisions,julia-1 --device cuda   # or rocm, mlx; add --batch-tokens 8192
```

`--device` sets `anofox_decide_device`, then reads `decide_models().device`. When a model was not served on the
requested device (a GPU request that fell to the CPU, a missing plugin, a hosted model) the run exits with status 2
and writes no report. With the device, the report records the serving device and the wall time. The first
run downloads the models (`decide_download`, about 2 GB). The hardware check that also compares against the CPU is
`test/sql/decide_local_e2e_gpu.test` (generated from the CPU suite by `tools/e2e/make_local_e2e.py`): it runs the
40-ticket local suite on the device named by `DECIDE_E2E_DEVICE`, requires every probability within 1e-3 of the CPU
reference and the labels to match where the reference is not a near tie, and checks that each block was served by `plugin:<device>`.

## Troubleshooting

| You see | It means | Do this |
|---|---|---|
| `anofox_decide_device is 'cuda' but no 'cuda' device was found on this machine` (or `rocm`, `mlx`) | an explicit device on a machine where none was discovered | `CALL decide_accelerate();` shows what the machine can use, or `SET anofox_decide_device = 'auto';` |
| `anofox_decide_device is 'cuda' but model '<id>' cannot run on cuda:0: ... its plugin is not installed in <dir>` (or `rocm`, `mlx`) | an explicit device whose plugin is missing, cannot load (the loader's text follows), or cannot serve that model | `CALL decide_accelerate();`, or `SET anofox_decide_device = 'auto';` / `'cpu'` |
| `... loading a GPU plugin is not allowed because enable_external_access is off` | DuckDB's own setting forbids loading plugins | `SET enable_external_access = true;` or `SET anofox_decide_device = 'cpu';` |
| `decide_devices()` lists the GPU with `usable = false` | the driver or runtime is missing, the card is older than sm_75 (CUDA), the ROCm plugin carries no code objects for its architecture, or (MLX) the process runs under Rosetta; the reason names it | install the NVIDIA driver, ROCm (HIP, hipBLAS, rocBLAS), or `brew install mlx mlx-c`; `SELECT * FROM decide_doctor();` |
| `no CUDA plugin is published for <os>/<arch>` (and the ROCm and MLX equivalents) | a platform with no published plugin (CUDA and ROCm are linux/amd64 only, MLX is macOS/arm64 only) | `SET anofox_decide_device = 'cpu';` (on Apple silicon use `'mlx'`, on Linux `'cuda'` or `'rocm'`) |
| `the '<backend>' backend ... a runtime library it needs is missing` | `libcuda.so.1` (the NVIDIA driver), the ROCm HIP / hipBLAS / rocBLAS libraries or `libmlx` was not found | install the vendor runtime; the plugin itself comes from `CALL decide_download_runtime('<backend>');` |
| `the '<backend>' backend ran out of device memory` | the batch or the text does not fit the card | `SET anofox_decide_batch_tokens = 4096;`, shorten the text (`decide_token_count`), or `SET anofox_decide_device = 'cpu';` |
| `the '<backend>' backend ... the execution provider failed` | the driver or runtime rejected the work | check `decide_doctor()`, use the CPU until fixed, and report it at the issue tracker |
| `the GPU plugin '<path>' was ...` (another plugin ABI version or struct layout), or `'<path>' is not an anofox GPU plugin` | an old plugin next to a newer extension (for example one of decide's former plugins), or a file with the right name but the wrong contents | `CALL decide_accelerate();` installs the matching one |
| `the '<backend>' plugin does not match its published checksum (expected ..., got ...)`, or `the GPU plugin '<path>' was not loaded: it does not match its published checksum` | a damaged download, or the file was replaced after it was installed while a `.sha256` file sits beside it (a plugin copied in by hand with no `.sha256` file is not checked). **The plugin and its `.sha256` file have been deleted** | retry `CALL decide_download_runtime('<backend>');` (or `CALL decide_accelerate();`): the failed attempt deleted the file, the next one fetches a verified copy. Report it if it recurs |
| `could not fetch the published checksum for the '<backend>' plugin (...)`, or `the published checksum file ... does not mention '<file>', so it vouches for nothing` | the release has no usable `.sha256` asset, or the network failed; the plugin was deleted rather than used unverified | retry the call; if it persists the release is incomplete: `FORCE INSTALL anofox_decide FROM community;` |
| `the release <tag> has no '<plugin file>' asset: this release was published without GPU plugins` | the extension points at a release that carries no plugin assets | update the extension (`FORCE INSTALL anofox_decide FROM community;`) or `SET anofox_decide_device = 'cpu';` |
| `the plugin directory '<dir>' is writable by other users` | `decide_download_runtime` refuses to install native code into a directory others can write to | `chmod go-w '<dir>'`, or `SET anofox_decide_ep_path` to a directory only you can write |
| `refusing to follow a redirect from <host> to <host>` | a download was redirected off https, or (for a GitHub release asset) off GitHub's own hosts | check the network or proxy; retry |
| `fetching a GPU plugin is not allowed because enable_external_access is off` | DuckDB's own setting forbids the download | `SET enable_external_access = true;` or copy the plugin into `anofox_decide_ep_path` |
| `decide_models().device` says `cpu` after `decide_accelerate()` | the plugin does not serve this model (a hand-registered `.onnx`) | `SELECT * FROM decide_backends();` gives the reason per model and device |

## Releasing

The plugins are released by [anofox-ggml-adapter](https://github.com/DataZooDE/anofox-ggml-adapter/releases), not
by this repository: `decide_download_runtime` fetches from the adapter release that `DECIDE_PLUGIN_RELEASE_TAG`
(`src/include/decide_plugin_artifacts.hpp`) names, which is the adapter's `ANOFOX_GGML_PLUGIN_RELEASE_TAG`
(currently `v2026.11.27`). A stale pin is safe: the loader refuses a plugin of another ABI version or struct layout.
Checklist for a release:

1. **Bump the adapter submodule** (`vendor/anofox-ggml-adapter`) when a new plugin release is wanted; the pin moves
   with it.
2. **Run the hardware end-to-end suite** on the release candidate and keep its output with the release notes:
   `test/sql/decide_local_e2e_gpu.test` with `DECIDE_REAL_WEIGHTS=1`, `DECIDE_E2E_DEVICE` set to `cuda`, `rocm` or
   `mlx` and `DECIDE_E2E_EP_PATH` the plugin's directory. CI has no GPU, so this is the only scoring evidence.
3. **After publishing**, on a clean Linux and a clean macOS machine run `CALL decide_accelerate();` against the real
   assets and score the two recorded sentences.

The former plugin sources (`plugins/cuda`, `plugins/migraphx`, `plugins/mlx`) and their release workflows are gone;
ONNX Runtime was removed with them.
