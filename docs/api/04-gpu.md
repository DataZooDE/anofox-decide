# GPU acceleration

The extension stays one CPU-only file. A GPU backend (NVIDIA CUDA, AMD ROCm/MIGraphX, Apple MLX) is a separate
plugin the extension loads when asked, fetched from the release and verified before it is used. Nothing changes
until you ask: without a plugin every [local model](03-local-models.md) scores on the CPU exactly as before.

The full reference (all measurements, requirements, limits and troubleshooting) is
[docs/GPU_BACKENDS.md](../GPU_BACKENDS.md); the ROCm shape-bucket table is [docs/GPU_ROCM.md](../GPU_ROCM.md).
This page is the short version.

## Quick start

```sql
SELECT * FROM decide_devices();     -- cpu, plus any NVIDIA / AMD / Apple GPU found, usable or not and why
CALL decide_accelerate();           -- find the GPU, download and verify its plugin, use it from now on
SELECT * FROM decide_backends();    -- per local model and device: supported, or the reason and the fix
SELECT model, device FROM decide_models();   -- where each local model is scored right now
SET anofox_decide_batch_tokens = 8192;       -- advisable on a GPU: score the rows of a chunk together
```

`decide_download_runtime('cuda')` (or `'rocm'`, `'mlx'`) fetches one backend's plugin explicitly. Requirements, all
fp32: CUDA 12.8 or newer with cuDNN 9 on Linux x86-64; ROCm 7.x with MIGraphX on Linux x86-64 for a supported
architecture; macOS on Apple silicon with `brew install mlx mlx-c`.

## Choosing the device

`anofox_decide_device` is `auto` by default: a GPU is used only when its plugin is installed, the device is usable
and the plugin serves the model, and `decide_backends()` is the same check, so `auto` never names a device that
table calls unsupported. `cpu` always scores on the CPU. `cuda`, `rocm` or `mlx` are hard errors naming
`CALL decide_accelerate()` when that device is missing, unusable or cannot serve the model: never a silent CPU run.
Only the models from `decide_download` can run on a GPU (the plugin loads their safetensors file itself); a model
registered with its own `.onnx` graph stays on the CPU. `anofox_decide_gpu_precision` accepts only `fp32`;
`anofox_decide_plugin_dir` is where plugins are installed and loaded from.

## What is verified

Plugins are native code. A download lands under a temporary name and moves into place only
after its sha256 matches the `.sha256` file published next to it in the release; any failure on the way (a
mismatch, a missing or unreadable checksum file) deletes the plugin. `decide_accelerate()` and
`decide_download_runtime()` check the installed plugin against that file again on every call, and the extension
checks a plugin against the `.sha256` file beside it once per process (by path, size and modification time) before
it first loads it, which scoring and `decide_backends()` do not repeat. The checksum comes from the same release as
the plugin, so this catches damage and a swapped asset, not a compromised release. **A plugin you copy into
`anofox_decide_plugin_dir` yourself, with no `.sha256` file beside it, is trusted and loaded as it is.** Treat the
directory like any directory you load code from: the extension refuses to install into one that other users can
write to. A plugin built against another interface version is refused with the fix.

## ROCm needs a compile

MIGraphX compiles one program per tensor shape: 2 to 5 minutes once per shape, and
0.6 to 1.75 GB of cache each, under `<cache dir>/gpu-cache/migraphx/`. Run `CALL decide_gpu_precompile('laya-multilingual');`
once before a batch job: it compiles every shape up front with progress (on CUDA and MLX it is refused, there is
nothing to compile); otherwise the first query that meets a shape pays for it, minutes at a time. Details:
[docs/GPU_ROCM.md](../GPU_ROCM.md).

## Measured

On the 200-ticket evaluation Laya multilingual ran several times faster on an RTX 4090 (CUDA), an RX 9070 XT
(ROCm) and an Apple M3 (MLX) than on the CPU of the same machine, with probabilities within 4e-5 of the CPU
and no decision changed. The gains depend on the model and the machine; every number, and why the ratios are
upper bounds: [docs/GPU_BACKENDS.md](../GPU_BACKENDS.md).

**Not yet:** half precision, CUDA on Windows, CUDA 13.
Source: [`plugins/cuda`](../../plugins/cuda/README.md), `plugins/migraphx`, `plugins/mlx`.

GPU errors and their fixes are in [Troubleshooting](06-troubleshooting.md#gpu) and
[docs/GPU_BACKENDS.md](../GPU_BACKENDS.md#troubleshooting).
