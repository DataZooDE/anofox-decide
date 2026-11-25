# GPU acceleration

The extension stays one CPU-only file. A GPU backend (NVIDIA CUDA, AMD ROCm, Apple MLX) is a separate
plugin the extension loads when asked, fetched from the release and verified before it is used. The plugins are
[anofox-ggml-adapter](https://github.com/DataZooDE/anofox-ggml-adapter)'s, shared with anofox-tabfm: the ggml
CUDA and ROCm plugins run the same engine as the CPU, the MLX plugin runs the same ONNX graphs on Apple's MLX. Nothing changes
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

`decide_download_runtime('cuda')` (or `'rocm'`, `'mlx'`) fetches one backend's plugin explicitly: the plugin alone,
no CUDA toolkit or cuDNN. Requirements, all fp32: on Linux x86-64 an NVIDIA GPU of compute
capability sm_75 or newer and its driver (the CUDA plugin links the CUDA runtime and cuBLAS statically and needs
only the driver's `libcuda.so.1`); on Linux x86-64 a ROCm install (HIP, hipBLAS, rocBLAS) and an AMD GPU of
architecture gfx90a, gfx942, gfx1030, gfx1100, gfx1101, gfx1102, gfx1150, gfx1151, gfx1200 or gfx1201; macOS on
Apple silicon with `brew install mlx mlx-c`.

## Choosing the device

`anofox_decide_device` is `auto` by default: a GPU is used only when its plugin is installed, the device is usable
and the plugin serves the model, and `decide_backends()` is the same check, so `auto` never names a device that
table calls unsupported. `cpu` always scores on the CPU. `cuda`, `rocm` or `mlx` are hard errors naming
`CALL decide_accelerate()` when that device is missing, unusable or cannot serve the model: never a silent CPU run.
Both kinds of local model run on a GPU: the models from `decide_download` (the plugin gets their weight-free graph and
the weights from the safetensors file) and a model registered with its own `.onnx` (the plugin gets the whole graph,
which carries its weights). A registered graph that needs an ONNX op the plugin lacks is refused with the op names;
`SET anofox_decide_device = 'cpu'` scores it on the CPU. `anofox_decide_gpu_precision` accepts only `fp32`;
`anofox_decide_ep_path` is where plugins are installed and loaded from (`anofox_decide_plugin_dir`, its old name,
is still read when `anofox_decide_ep_path` is empty, and is removed in the next release).

## What is verified

Plugins are native code. A download lands under a temporary name and moves into place only
after its sha256 matches the `.sha256` file published next to it in the release; any failure on the way (a
mismatch, a missing or unreadable checksum file) deletes the plugin. `decide_accelerate()` and
`decide_download_runtime()` check the installed plugin against that file again on every call, and the extension
checks a plugin against the `.sha256` file beside it once per process (by path, size and modification time) before
it first loads it, which scoring and `decide_backends()` do not repeat. The checksum comes from the same release as
the plugin, so this catches damage and a swapped asset, not a compromised release. **A plugin you copy into
`anofox_decide_ep_path` yourself, with no `.sha256` file beside it, is trusted and loaded as it is.** Treat the
directory like any directory you load code from: the extension refuses to install into one that other users can
write to. A plugin built against another interface version (plugin ABI v3) or struct layout is refused with the fix.

## No ahead-of-time compile

The ggml and MLX plugins build their compute graph for a shape on its first use, as part of that call: no
minutes-long compile and no compiled programs on disk, so there is nothing to compile ahead. `decide_gpu_precompile` is still accepted on ROCm (and refused on CUDA, MLX
and the CPU), but with the ggml ROCm plugin it compiles nothing (it only loads the model onto the GPU and returns one row per former bucket). Details: [docs/GPU_ROCM.md](../GPU_ROCM.md).

## Verified

The local end-to-end suite (40 tickets, all three catalog models) passes on AMD gfx1150 and gfx1201 (ROCm) and on
an Apple M3 (MLX): every probability within 1e-3 of the CPU reference, the labels exact (where the reference is not a
near tie), and every block served by the plugin. Against the previous plugins on the same machines, the 200-ticket evaluation
is 1.2x to 3.9x faster and needs up to 4.7x less host memory, with no answer changed ([docs/GPU_BACKENDS.md](../GPU_BACKENDS.md#measured)).

**Not yet:** half precision, CUDA on Windows.
Source: [anofox-ggml-adapter](https://github.com/DataZooDE/anofox-ggml-adapter) (`src/plugin_ggml.cpp`,
`src/plugin_mlx.cpp`); the plugins are released there.

GPU errors and their fixes are in [Troubleshooting](06-troubleshooting.md#gpu) and
[docs/GPU_BACKENDS.md](../GPU_BACKENDS.md#troubleshooting).
