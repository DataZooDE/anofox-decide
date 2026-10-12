# CUDA plugin

`libanofox_decide_cuda_plugin.so`: the NVIDIA backend behind the plugin interface in
[`src/include/decide_plugin_abi.h`](../../src/include/decide_plugin_abi.h) (ABI version 1, unchanged). The
extension loads it when `anofox_decide_device` is `cuda` (or `auto` finds an NVIDIA GPU and the plugin installed).
Linux x86_64 only; CUDA 12.8+ with cuDNN 9 are the user's.

## How it works

- **Its own ONNX Runtime.** The extension links ORT statically, which cannot host ORT's provider libraries (the
  executable's copies of the core's classes win symbol resolution and corrupt the heap). The plugin therefore links
  a private shared core, taken from the same wheel as the CUDA provider
  (`decide_download_runtime('cuda')` extracts it from the pinned, size- and sha256-checked Microsoft wheel), with its
  SONAME rewritten from `libonnxruntime.so.1` to `libanofoxort_gpu.so` (same length, so a byte patch;
  [`decide_soname_patch.hpp`](../../src/include/decide_soname_patch.hpp) is the one routine, used by the download
  and by this build). The loader adds `RTLD_DEEPBIND`; the rename and the flag are independent layers.
- **Weights.** The staged weight-free graph is opened with the tensors injected through `AddExternalInitializers`
  from the plugin's own `mmap` of the safetensors file, using the tensor map ([`decide_cuda_weights.hpp`](decide_cuda_weights.hpp)).
  Float32 tensors are referenced in place, F16/BF16 are upcast into one host arena, `transpose` entries are
  transposed. This is the one backend that keeps a host float32 copy (ORT owns the upload). The graph's packed
  offsets are ignored.
- **Precision.** `use_tf32=0`: ORT's CUDA provider rounds fp32 matmuls to TF32 on Ampere and newer by default,
  which would make a device switch move the answers. Only `fp32` is accepted.
- **Shapes.** Dynamic; `precompile` is a no-op.

## Build and test (no GPU needed)

```bash
cmake -S plugins/cuda -B build-cuda-plugin           # fetches the ORT CPU archive for headers and a core to rename
cmake --build build-cuda-plugin -j
ctest --test-dir build-cuda-plugin --output-on-failure
```

`-DDECIDE_CUDA_ORT_DIR=<dir>` uses an extracted ORT distribution (`include/`, `lib/libonnxruntime.so.1.29.0`)
instead; any `libonnxruntime_providers_*.so` in its `lib/` is copied next to the renamed core. The tests:

| test | proves |
|---|---|
| `cuda_plugin_artifact` | the entry symbol is exported, `DT_NEEDED` names `libanofoxort_gpu.so` and never `libonnxruntime`, no CUDA library is linked directly, `$ORIGIN` rpath |
| `cuda_plugin_load_check` | a real `dlopen` (loader flags) resolves, the ABI matches, `create()` refuses with a message |
| `cuda_plugin_no_cuda_refusal` | against a core without the CUDA provider, `create()` names it after the checkpoint passed |
| `cuda_plugin_cpu_ep_weights` | the identical code on ORT's CPU provider (a test-only build, `DECIDE_CUDA_PLUGIN_CPU_TEST_EP`) reproduces reference scores from Python onnxruntime exactly: F16 / F32 aligned / F32 unaligned / transposed checkpoints, batch vs single row, 4 threads on one handle, every refusal |
| `cuda_plugin_survives_shadow_core` | the same, in a process that already loaded an upstream-named core globally |

## On hardware

`gpu_test/` holds the checks that need an NVIDIA GPU; they are run by hand (they cost money on a rented pod):
`gpu_test/runpod_run.py` rents a RunPod pod, ships the sources and runs `gpu_test/cuda_pod_verify.sh`, which
fetches the pinned runtime, builds against it, repeats the tests on the real core, runs the tiny fixture on the
CUDA provider (plus a no-device negative control) and, when given a DuckDB CLI and the extension,
`gpu_test/device_parity.py`: the recorded Laya multilingual values within 2e-4, CPU against CUDA within 1e-4 on
the 200-ticket evaluation with identical argmax and accuracy.
