# ROCm backend (AMD GPUs, MIGraphX)

The ROCm backend is a plugin, `libanofox_decide_migraphx_plugin.so`, that drives MIGraphX directly through
`libmigraphx_c`. ONNX Runtime is not involved: its ROCm execution provider no longer exists and no prebuilt
build carries the MIGraphX one. The plugin is built from `plugins/migraphx/` and speaks the plugin interface of
`src/include/decide_plugin_abi.h`.

Requirements: Linux x86-64, a ROCm install with MIGraphX (`migraphx` and its runtime; on Debian and Ubuntu
`apt install migraphx`), and a GPU architecture MIGraphX supports (`decide_devices()` says which). Validated on
a Radeon RX 9070 XT (gfx1201) with ROCm 7.2.4 and MIGraphX 2.15.

## How it works

MIGraphX compiles a program for one concrete tensor shape, and a compile takes minutes. The plugin therefore
serves a small grid of shapes and caches what it compiles:

| | |
|---|---|
| markers per row | 20 (the most options a question can have; unused slots are masked) |
| tokens per row | Laya models: 256 or 1024. Julia-1: 256, 1024, 4096 or 8192. A batch pads up to the next bucket |
| rows | 1, 4 (one text with a few questions), or as many as fit 8192 padded tokens (32 at 256 tokens, 8 at 1024, 2 at 4096, 1 at 8192). A request pads up to the next of these; a bigger batch is cut into chunks of the largest. A bigger program that is already cached is used rather than compiling the exact one in the middle of a query |

The weights come from the downloaded safetensors file (`decide_download`) through the tensor map: while it
compiles, the plugin writes the packed float32 blob the graph's external-data stubs describe next to a copy of
the graph, lets MIGraphX read it, and deletes it again. The compiled program (`.mxr`) embeds the weights, so a
cached program loads without touching the checkpoint.

The cache lives in `<cache dir>/gpu-cache/migraphx/`. A program is keyed by the model, a hash of the graph, the
tensor map and every byte of the checkpoint, the GPU architecture and the shape, so a different checkpoint or
card never reuses it. A truncated or unreadable file is dropped and compiled again. Each program holds its own
copy of the weights on the GPU, so the plugin keeps the two most recently used resident
(`DECIDE_MIGRAPHX_MAX_RESIDENT` changes that, 1 to 16).

## Using it

```sql
CALL decide_accelerate();                    -- installs the plugin (or copy it into anofox_decide_plugin_dir)
SET anofox_decide_device = 'rocm';
SET anofox_decide_batch_tokens = 8192;       -- feed the GPU batches; the default (0) scores one text at a time
CALL decide_gpu_precompile('laya-multilingual');   -- optional: compile every bucket now
```

Without `decide_gpu_precompile` the first query that meets a shape compiles it (a line on stderr says so and
repeats every 30 seconds). Compiling all buckets of a model is a coffee break and a few GB of disk:

Measured on a Radeon RX 9070 XT (gfx1201, 16 GB, ROCm 7.2.4, MIGraphX 2.15), fp32. Compile and size are per program;
the steady call is the median of repeated calls of the shape on the already loaded program (a cached program
takes 4 to 10 s to load into the GPU on its first use in a process).

| model | rows x tokens | compile | .mxr | steady call |
|---|---|---|---|---|
| laya-multilingual | 1 x 256 / 4 x 256 / 32 x 256 | 190 s / 163 s / 196 s | 1.30 GB | 26 ms / 46 ms / 175-230 ms |
| laya-multilingual | 1 x 1024 / 4 x 1024 / 8 x 1024 | 190 s / 210 s / 215 s | 1.34 GB | 53 ms / 165 ms / 215-280 ms |
| laya-typed-decisions | 1 x 256 / 4 x 256 / 32 x 256 | 211 s / 223 s / 291 s | 1.70 GB | 49 ms / 95 ms / 358 ms |
| laya-typed-decisions | 1 x 1024 / 4 x 1024 / 8 x 1024 | 226 s / 279 s / 328 s | 1.75 GB | 112 ms / 248 ms / 463 ms |
| julia-1 | 1 x 256 / 4 x 256 / 32 x 256 | 128 s / 123 s / 141 s | 0.59 GB | 18 ms / 26 ms / 106 ms |
| julia-1 | 1 x 1024 / 4 x 1024 / 8 x 1024 | 133 s / 136 s / 152 s | 0.60 GB | 32 ms / 83 ms / 136 ms |
| julia-1 | 1 x 4096 / 2 x 4096 | 169 s / 192 s | 0.69 GB | 189 ms / 268 ms |
| julia-1 | 1 x 8192 | 256 s | 0.83 GB | 451 ms |

All 9 julia-1 buckets take about 21 minutes and 5.7 GB, the 6 of laya-multilingual about 20 minutes and 7.9 GB,
the 6 of laya-typed-decisions about 26 minutes and 10.3 GB. Batching pays: 32 rows of 256 tokens cost 175-230 ms
on laya-multilingual against 26 ms for one row.

Agreement with the CPU (ONNX Runtime) on the 200-ticket evaluation sample (400 answers): largest probability
difference 9.9e-6 (laya-multilingual), 3.9e-5 (laya-typed-decisions), 1.7e-5 (julia-1); no routing choice or
refund decision differs. The recorded values for laya-multilingual reproduce: 0.98209641 (reference 0.98209638)
and 0.00123894 (reference 0.00123890). Wall time of the 200-ticket sample including the first load of the
programs: laya-multilingual 9 s batched (`anofox_decide_batch_tokens = 8192`) and 18 s one text at a time, against
51-81 s on the CPU (a busy 16-core machine); laya-typed-decisions 29 s against 186 s; julia-1 10 s against 21 s.

`decide_gpu_precompile(model, seq := N)` compiles only the token bucket that holds N tokens.

## Limits

- fp32 only. Half precision needs a re-export of the graphs (the mask constants overflow in fp16 and bf16) and is
  planned.
- A text that needs more tokens than the largest bucket of the model (1024 for Laya, 8192 for Julia-1) is
  refused with the limit; `SET anofox_decide_device = 'cpu'` scores it on the CPU.
- Disk: each compiled program takes 0.6 to 1.7 GB, and a compile needs about twice the model's packed weights
  free while it runs.
