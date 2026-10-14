# GPU backends (CUDA, ROCm, Apple MLX)

The extension is one CPU-only file. A GPU backend is a separate plugin it loads when asked, behind one C
interface (`src/include/decide_plugin_abi.h`). Three exist:

| Backend | Device | Plugin | Source | Hardware it was validated on |
|---|---|---|---|---|
| CUDA | NVIDIA | `libanofox_decide_cuda_plugin.so` | [`plugins/cuda`](../plugins/cuda/README.md) | RTX 4090 (RunPod) |
| ROCm | AMD | `libanofox_decide_migraphx_plugin.so` | `plugins/migraphx`, [docs/GPU_ROCM.md](GPU_ROCM.md) | RX 9070 XT (gfx1201) |
| MLX | Apple silicon | `libanofox_decide_mlx_plugin.dylib` | `plugins/mlx` | Apple M3, 16 GB |

All three run the same ONNX graphs as the CPU, in fp32, and the answers agree with the CPU to within 4e-5 in
probability (table below). They are an option for speed, not a different model: accuracy on the 200-ticket
evaluation is identical on every device.

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
`anofox_decide_plugin_dir` is where plugins are installed and loaded from (default: `plugins` inside the cache
directory), and a plugin built against another interface version is refused with the fix.

### What is verified, and what is trusted

Plugins are native code the extension loads, so the checks are spelled out here exactly:

- **At download.** The plugin is fetched under a temporary name and moved into place only after its sha256 equals
  the line for it in the `<backend>_plugin.sha256` file published next to it in the pinned release. If the
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
- **CUDA runtime files** (the renamed ONNX Runtime core and the two provider libraries extracted from Microsoft's
  wheel) have their sha256 pinned in the extension and are compared on every install call; a mismatch or a
  half-finished extraction deletes them, and they are extracted again.

Only the models from `decide_download` run on a GPU (the plugin reads their safetensors file itself); a model
registered with its own `.onnx` graph stays on the CPU.

| Setting | Meaning |
|---|---|
| `anofox_decide_device` | `auto` (default): a GPU only when its plugin is installed, the device is usable and the plugin serves the model, otherwise the CPU. `cpu`: always the CPU. `cuda` / `rocm` / `mlx`: that device, or an error naming `CALL decide_accelerate()`; never a silent CPU run |
| `anofox_decide_batch_tokens` | Score the rows of a chunk together, in batches of at most this many padded tokens. `0` (default) scores one text at a time. A GPU waits on one row per call, so set it on a GPU (`8192` to `16384` in the runs below). It is **not** a CPU speed-up: on CPU its effect is small and inconsistent |
| `anofox_decide_gpu_precision` | `fp32` only. Other values are rejected, never silently run in fp32 |
| `anofox_decide_plugin_dir` | Where plugins are installed and loaded from |

`decide_gpu_precompile('<model>')` compiles a model's shape buckets ahead of time. It matters on ROCm only (see
below); on CUDA and MLX shapes are dynamic and it is not needed.

## Requirements

| Backend | You need | The extension fetches |
|---|---|---|
| CUDA | Linux x86-64, an NVIDIA GPU, a driver with **CUDA 12.8 or newer** and **cuDNN 9** installed on the machine. CUDA 13 and Windows are not published yet: the pinned ONNX Runtime provider is the CUDA 12 build. `decide_devices()` does not check the CUDA version (NVML reports the driver, and every NVIDIA card it finds is listed `usable = true`), so on a machine with only CUDA 13 runtime libraries the plugin loads and then fails when it initialises with `a runtime library it needs is missing` (naming the CUDA 12 library) | the plugin, and the pinned ONNX Runtime GPU core and CUDA provider (the wheel is size- and sha256-checked, and so is each extracted file on every install call; taken from Microsoft's wheel and renamed to `libanofoxort_gpu.so` so it cannot clash with the statically linked CPU runtime) |
| ROCm | Linux x86-64, ROCm **7.x** with MIGraphX (`apt install migraphx`), a GPU whose architecture MIGraphX supports: gfx900, gfx906, gfx908, gfx90a, gfx940, gfx941, gfx942, gfx1030, gfx1100, gfx1101, gfx1102, gfx1200, gfx1201. Any other architecture is shown by `decide_devices()` as not usable, with its name | the plugin only; it drives `libmigraphx_c` from your ROCm install, ONNX Runtime is not involved |
| MLX | macOS on Apple silicon (arm64), `brew install mlx mlx-c` | the plugin only; it links Apple's `libmlx` and `libmlxc` |

Disk: the cached models are the ones `decide_download` already put in the cache directory; ROCm adds the compiled
programs (below). Memory: CUDA 1.6 to 2.6 GB of VRAM up to 4096 tokens; MLX uses about 1.2 to 1.3 GB for the
smaller shapes and Julia-1 at 8192 tokens peaks at 7.4 GB (on a 16 GB Mac it runs as one row at a time).

## Measured

All numbers are fp32 and come from the 200-ticket evaluation of [EVALUATION.md](EVALUATION.md): two questions per
ticket (400 answers), the wall time includes session start and model load, and it is produced by
`tools/eval/run_eval.py --device <device>`, which refuses to report a model that the requested device did not
serve. **CPU numbers come from each box's own CPU and are not comparable across boxes**: the CUDA pod had a slow
shared CPU, the ROCm machine was heavily loaded. Read each row as "the same machine, CPU against GPU", not as a
ranking of GPUs, and read every ratio as an upper bound: the CPU baselines came from slow or loaded machines, so a
quiet, fast CPU narrows the gap.

**Wall time, 200 tickets**

| Model | CUDA, RTX 4090 | ROCm, RX 9070 XT | MLX, Apple M3 16 GB |
|---|---|---|---|
| Laya multilingual | CPU 97.0 s, GPU 11.8 s | CPU 51-81 s; 18 s one text at a time, 9 s with `batch_tokens = 8192` | CPU 80.8 s; 17.4 s (9.5 s at `batch_tokens = 16384`) |
| Laya typed-decisions | not measured | CPU 186 s, GPU 29 s | CPU 167.5 s; 20.0 s, 22.0 s with batching (batching does not help it) |
| Julia-1 | not measured | CPU 21 s, GPU 10 s | CPU 24.1 s; 14.2 s, 5.1 s with batching |

**Agreement with the CPU** (largest absolute probability difference over the 400 answers; no routing or refund
decision changed on any device, and accuracy is identical: refund 92.0%, routing 62.0% for Laya multilingual)

| Model | CUDA | ROCm | MLX |
|---|---|---|---|
| Laya multilingual | 3.0e-6 | 9.9e-6 | 6.6e-6 |
| Laya typed-decisions | not measured | 3.9e-5 | 1.4e-5 |
| Julia-1 | not measured | 1.7e-5 | 2.8e-5 |

The recorded values of Laya multilingual (`I want my money back.` 0.98209638, `Thanks, everything works now!`
0.0012388999) are reproduced within 1e-7 on CUDA and within 6e-8 between the CPU and MLX; the ROCm values and
their differences from the reference are in [GPU_ROCM.md](GPU_ROCM.md). The accuracy figures equal EVALUATION.md
on every device; Julia-1's 29.5% refund and 37.5% routing accuracy is its documented weakness, not a GPU effect.

Test conditions: CUDA on an RTX 4090 (RunPod) with the ONNX Runtime 1.29 GPU core and TF32 switched off
(`use_tf32=0`), so `fp32` means strict fp32. ROCm on gfx1201 with ROCm 7.2.4 and MIGraphX 2.15. MLX 0.32.3.

## ROCm: compile time and disk

MIGraphX compiles one program per tensor shape. The plugin serves a few fixed shapes (20 markers; Laya 256 or
1024 tokens, Julia-1 256 to 8192 tokens; one row or a full batch) and caches each compiled program under
`<cache dir>/gpu-cache/migraphx/`.

- **Compiling a bucket takes 2 to 5 minutes, once**, and its `.mxr` file is 0.6 to 1.75 GB. The first query that
  meets a new shape compiles it while you wait (a line on stderr says so, every 30 seconds, which a notebook may
  not show); run `CALL decide_gpu_precompile('laya-multilingual');` once before a batch job to do it up front.
  It is refused on CUDA and MLX (nothing to compile there) and errors, naming the largest bucket, for a `seq`
  longer than any bucket.
- A cached program takes 4 to 10 s to load into the GPU on its first use in a process; after that a call is fast
  (Laya multilingual 1 x 256 tokens 26 ms, 1 x 1024 tokens 53 ms; Julia-1 1 x 8192 tokens 451 ms).
- All buckets of one model: about 20 to 26 minutes and 5.7 to 10.3 GB. A compile needs about twice the model's
  packed weights free while it runs.

The full table of buckets is in [GPU_ROCM.md](GPU_ROCM.md).

## Limits

- **fp32 only.** Half precision needs a re-export of the graphs (the attention mask constants overflow in fp16
  and bf16) and is the next milestone; `anofox_decide_gpu_precision` rejects anything else.
- **CUDA:** Linux x86-64 and CUDA 12.8+ with cuDNN 9 only. No Windows and no CUDA 13 yet. The CUDA plugin keeps a
  host float32 copy of the weights.
- **ROCm:** the compile cost above, and a text longer than the largest bucket (1024 tokens for Laya, 8192 for
  Julia-1) is refused with the limit; `SET anofox_decide_device = 'cpu'` scores it on the CPU.
- **MLX:** Apple silicon only. A 16 GB Mac is a budget: a batch is split by the machine's RAM and at least one row
  is always scored.
- **Speed depends on the shape.** The GPU gains are largest for Laya typed-decisions and long texts; for short
  texts the CPU is already fast. Batching helps Laya multilingual and Julia-1 on a GPU, not Laya typed-decisions on
  MLX. Measure on your own data with `tools/eval/run_eval.py`.
- Only models from `decide_download` run on a GPU; a model registered with its own `.onnx` stays on the CPU.
- The GPU backends are not run in CI (the hosted runners have none): CI builds each plugin and checks its exported
  symbol, its dependencies and that it loads and refuses cleanly. Scoring is validated by hand on the hardware
  above before a release.

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
run downloads the models (`decide_download`, about 2 GB). The per-backend hardware checks that also compare against the CPU are `plugins/cuda/gpu_test/device_parity.py`,
`plugins/migraphx/gpu_test/device_parity.py` and `plugins/mlx/gpu_test/device_parity.py`. Each compares, on the
200 tickets, the refund probability of every ticket (largest difference 1e-4) and the refund decision and routing
choice (the argmax of the options), plus the two recorded reference sentences; the routing probabilities
themselves are not compared.

## Troubleshooting

| You see | It means | Do this |
|---|---|---|
| `anofox_decide_device is 'cuda' but no 'cuda' device was found on this machine` (or `rocm`, `mlx`) | an explicit device on a machine where none was discovered | `CALL decide_accelerate();` shows what the machine can use, or `SET anofox_decide_device = 'auto';` |
| `anofox_decide_device is 'cuda' but model '<id>' cannot run on cuda:0: ... its plugin is not installed in <dir>` (or `rocm`, `mlx`) | an explicit device whose plugin is missing, cannot load (the loader's text follows), or cannot serve that model | `CALL decide_accelerate();`, or `SET anofox_decide_device = 'auto';` / `'cpu'` |
| `... loading a GPU plugin is not allowed because enable_external_access is off` | DuckDB's own setting forbids loading plugins | `SET enable_external_access = true;` or `SET anofox_decide_device = 'cpu';` |
| `decide_devices()` lists the GPU with `usable = false` | the driver or runtime is missing, or (ROCm) the architecture is not supported by MIGraphX; the reason names it | install CUDA 12.8+ with cuDNN 9, ROCm 7.x with MIGraphX, or `brew install mlx mlx-c`; `SELECT * FROM decide_doctor();` |
| `no CUDA plugin is published for <os>/<arch>` (and the ROCm and MLX equivalents) | a platform with no published plugin (CUDA and ROCm are linux/amd64 only, MLX is macOS/arm64 only) | `SET anofox_decide_device = 'cpu';` (on Apple silicon use `'mlx'`, on Linux `'cuda'` or `'rocm'`) |
| `the '<backend>' backend ... a runtime library it needs is missing` | `libcudnn`, `libmigraphx_c`, `libmlx` or the ONNX Runtime core was not found | install the vendor runtime, or `CALL decide_download_runtime('<backend>');` for the parts that are downloadable |
| `the '<backend>' backend ran out of device memory` | the batch or the text does not fit the card | `SET anofox_decide_batch_tokens = 4096;`, shorten the text (`decide_token_count`), or `SET anofox_decide_device = 'cpu';` |
| `the '<backend>' backend ... the execution provider failed` | the driver or runtime rejected the work | check `decide_doctor()`, use the CPU until fixed, and report it at the issue tracker |
| `was built against plugin ABI version N, but this build speaks version M` | an old plugin next to a newer extension | `CALL decide_accelerate();` installs the matching one |
| `the '<backend>' plugin does not match its published checksum (expected ..., got ...)`, or `the GPU plugin '<path>' was not loaded: it does not match its published checksum` | a damaged download, or the file was replaced after it was installed while a `.sha256` file sits beside it (a plugin copied in by hand with no `.sha256` file is not checked). **The plugin and its `.sha256` file have been deleted** | retry `CALL decide_download_runtime('<backend>');` (or `CALL decide_accelerate();`): the failed attempt deleted the file, the next one fetches a verified copy. Report it if it recurs |
| `could not fetch the published checksum for the '<backend>' plugin (...)`, or `the published checksum file ... does not mention '<file>', so it vouches for nothing` | the release has no usable `.sha256` asset, or the network failed; the plugin was deleted rather than used unverified | retry the call; if it persists the release is incomplete: `FORCE INSTALL anofox_decide FROM community;` |
| `the release <tag> has no '<plugin file>' asset: this release was published without GPU plugins` | the extension points at a release that carries no plugin assets | update the extension (`FORCE INSTALL anofox_decide FROM community;`) or `SET anofox_decide_device = 'cpu';` |
| `the plugin directory '<dir>' is writable by other users` | `decide_download_runtime` refuses to install native code into a directory others can write to | `chmod go-w '<dir>'`, or `SET anofox_decide_plugin_dir` to a directory only you can write |
| `refusing to follow a redirect from <host> to <host>` | a download was redirected off https, or (for a GitHub release asset) off GitHub's own hosts | check the network or proxy; retry |
| the first ROCm query takes minutes | MIGraphX is compiling a new shape (once, then cached) | `CALL decide_gpu_precompile('<model>');` ahead of time; the progress line on stderr shows it is alive |
| `fetching a GPU plugin is not allowed because enable_external_access is off` | DuckDB's own setting forbids the download | `SET enable_external_access = true;` or copy the plugin into `anofox_decide_plugin_dir` |
| `decide_models().device` says `cpu` after `decide_accelerate()` | the plugin does not serve this model (a hand-registered `.onnx`, or a shape it has no bucket for) | `SELECT * FROM decide_backends();` gives the reason per model and device |

## Releasing

The plugins are attached to the GitHub release of the tag, and `decide_download_runtime` fetches from the release
that `DECIDE_PLUGIN_RELEASE_TAG` (`src/include/decide_plugin_artifacts.hpp`) names. Checklist for a release:

1. **Bump the pin and the version together** in the release commit: `DECIDE_PLUGIN_RELEASE_TAG`, the version
   strings, the CHANGELOG heading. `tools/ci/check_plugin_pin.sh <tag>` (and `gpu_release_gate.yml`) fail a tag
   that does not equal the pin; nothing else stops a stale pin, because the plugin ABI check only refuses a
   *different* ABI version, not an older plugin of the same one.
2. **Run the hardware parity suites** on the release candidate and keep their output with the release notes:
   `plugins/cuda/gpu_test/device_parity.py` (NVIDIA), `plugins/migraphx/gpu_test/device_parity.py` (AMD, with
   `--precompile`) and `plugins/mlx/gpu_test/device_parity.py` (Apple silicon). CI has no GPU, so these are the only
   scoring evidence. `DECIDE_TEST_MIGRAPHX=1` also runs the MIGraphX tiny-fixture test.
3. **Push the tag.** The three plugin workflows build, verify and upload to a *draft* release, which is published
   only when all six assets (three plugins, three `.sha256` files) are attached. A manual (workflow_dispatch) run of
   a plugin workflow does the verification without touching any release.
4. **After publishing**, on a clean Linux and a clean macOS machine run `CALL decide_accelerate();` against the real
   assets and score the two recorded sentences; confirm that a wrong tag's 404 names the tag and the asset.
