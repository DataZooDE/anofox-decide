# ROCm backend (AMD GPUs)

The ROCm backend is a plugin, `libanofox_ggml-rocm_plugin.so`, from
[anofox-ggml-adapter](https://github.com/DataZooDE/anofox-ggml-adapter) (shared with anofox-tabfm). It runs the
same ggml engine as the CPU on HIP, and speaks the adapter's plugin interface (plugin ABI v3). It replaces the former
MIGraphX plugin (`libanofox_decide_migraphx_plugin.so`), removed with ONNX Runtime.

Requirements: Linux x86-64, a ROCm install (the plugin links the system's HIP, hipBLAS and rocBLAS; it is built
against ROCm 7.2), and a GPU architecture the plugin carries code objects for: gfx90a, gfx942, gfx1030, gfx1100,
gfx1101, gfx1102, gfx1150, gfx1151, gfx1200, gfx1201 (`decide_devices()` says which). Verified with the local
end-to-end suite on gfx1150 and gfx1201.

## How it works

The extension hands the plugin its embedded weight-free graph and the weights it reads from the downloaded
safetensors file (`decide_download`), renamed and transposed through the tensor map exactly as for the CPU; nothing
is staged to disk and no compiled program is cached. ggml builds the compute graph for a shape on its first use, as
part of that call, so there are no shape buckets and no minutes-long compile: a text is limited only by the model's
window, as on the CPU.

## Using it

```sql
CALL decide_accelerate();                    -- installs the plugin (or copy it into anofox_decide_ep_path)
SET anofox_decide_device = 'rocm';
SET anofox_decide_batch_tokens = 8192;       -- feed the GPU batches; the default (0) scores one text at a time
```

`decide_gpu_precompile` is deprecated: still accepted on ROCm, but with this plugin it compiles nothing (it only loads the model onto the GPU and returns one row per former bucket); there is nothing to
compile ahead.

Agreement with the CPU: the local end-to-end suite (`test/sql/decide_local_e2e_gpu.test`, 40 tickets, all three
catalog models) gives every probability within 1e-3 of the CPU reference and the labels exact where the reference
is not a near tie, with every block served by `plugin:rocm`. On gfx1201 the 200-ticket evaluation takes 6 / 7 / 11 s
(Julia-1, Laya multilingual, Laya typed-decisions) at 1.0 to 1.2 GB host memory, against 16 / 26 / 43 s and 2.6 to 5.7 GB
with the former MIGraphX plugin ([GPU_BACKENDS.md](GPU_BACKENDS.md#measured)).

## Limits

- fp32 only. Half precision needs a re-export of the graphs (the mask constants overflow in fp16 and bf16) and is
  planned.
- Linux x86-64 and the architectures above only; any other card is listed by `decide_devices()` as not usable,
  with its architecture.
