---
name: anofox-decide-local
description: >
  Running decision models in-process and offline with the anofox_decide DuckDB extension, on CPU or
  GPU. Covers the built-in catalog (julia-1, laya-multilingual, laya-typed-decisions) and
  decide_download (verified, resumable Hugging Face download; gated repos via a huggingface secret or
  HF_TOKEN). Covers registering your own exported ONNX model with decide_register_model(id, 'local',
  graph, tokenizer[, profile[, options]]), long text and decide_token_count, and the window settings
  anofox_decide_max_length, anofox_decide_head_length and anofox_decide_on_truncate. Covers batching
  with anofox_decide_batch_tokens, anofox_decide_cache_dir, and GPU acceleration through verified
  CUDA / ROCm / Apple MLX plugins (anofox-ggml-adapter's): decide_devices, decide_backends,
  decide_accelerate, decide_download_runtime, anofox_decide_device, anofox_decide_ep_path (old name
  anofox_decide_plugin_dir, deprecated) and anofox_decide_gpu_precision. Use when the user wants no
  API key, no data leaving the machine, an offline or air-gapped setup, a Julia-1 / Laya model, a
  GPU, or hits "not downloaded", "tokens but the local model reads at most" or GPU plugin errors.
version: 2026.11.24
user-invocable: false
---

# Anofox Decide — Local Models & GPU

**Extension:** `anofox_decide` v2026.11.24 | **DuckDB:** v1.5.6 / v2.0 | **Runtime:** built-in ggml engine, linked
in | **Dual naming:** `decide_*` and `anofox_decide_*`

A local model runs inside DuckDB. After the one-time download nothing leaves the machine: no key, no
`anofox_decide_allow_remote`, no registration, no Python, no `httpfs`. The calls are the same as for
hosted models (see `anofox-decide-questions`).

## Critical gotchas

1. **Download once, then name it.** `CALL decide_download('<model>')` (or `SELECT * FROM decide_download(...)`)
   is needed before the first call. Otherwise the call fails with
   `model '<id>' is not downloaded. Fix: CALL decide_download('<id>');`. The catalog models need **no**
   `decide_register_model`.
2. **Long text is an error, not a silent cut.** When a text, question or option does not fit the window,
   the call fails and says how much would be lost. Find long rows with `decide_token_count`. To accept the
   cut, use `SET anofox_decide_on_truncate = 'ignore'`.
3. **The first call in a session loads the model:** about 0.3 s (julia-1) to 0.7 s (Laya).
   Later calls reuse it. Peak memory is 0.4 to 2.4 GB. The CPU engine runs `min(threads, physical cores)` threads,
   so DuckDB's `SET threads = 1` makes local models slower; the default needs no change.
4. **Local models score one text at a time** unless `anofox_decide_batch_tokens` is set. Batching gives
   identical answers. On CPU it barely changes speed. On a GPU, set it (e.g. `8192`).
5. **An explicit GPU device never falls back to the CPU silently.** `anofox_decide_device = 'auto'` (the default) uses a GPU only when its
   plugin is installed, the device is usable and the plugin serves the model. `'cuda'`, `'rocm'` or `'mlx'`
   is a hard error with the fix when that device cannot serve the model. A model registered from your own
   `.onnx` graph runs on a GPU too, unless it uses an ONNX op the plugin lacks (the error names the ops).
6. **GPU plugins are native code.** They are verified against the release's `.sha256` file. A plugin copied
   by hand without a `.sha256` file beside it is trusted as-is.
7. **No GPU needs a compile step.** The plugins build a shape's graph on first use. `decide_gpu_precompile`
   is deprecated: still accepted on ROCm, where it compiles nothing (it only loads the model onto the GPU), and refused on CUDA and MLX.

## The catalog

| Model | Download | Peak memory | Notes |
|---|---|---|---|
| `laya-multilingual` | 0.68 GB | 1.9 GB | multilingual (mmBERT-base). **The recommended start** |
| `laya-typed-decisions` | 0.85 GB | 2.4 GB | English (ModernBERT-large). The best in-process router |
| `julia-1` | 0.61 GB | 0.4 GB | smallest and fastest, the weakest |

```sql
-- needs: network (downloads about 0.7 GB from Hugging Face, once)
CALL decide_download('laya-multilingual');
SELECT decide_probability('I want my money back.', 'A refund is requested.', model := 'laya-multilingual');
SET anofox_decide_model = 'laya-multilingual';      -- or make it the default
```

```sql
SELECT model, ready, hint, device FROM decide_models();   -- the catalog is listed; ready = false until downloaded
```

The weights are pinned to a fixed upstream revision, checked by size and SHA-256, and resumed after an
interruption. They are kept in `~/.cache/anofox-decide` (`SET anofox_decide_cache_dir = '<dir>'`).
`HTTPS_PROXY` is honoured. Gated repositories need
`CREATE SECRET (TYPE anofox_decide, SERVICE 'huggingface', API_KEY 'hf_...')` or `HF_TOKEN`. The token
goes to `huggingface.co` only.

## Your own exported model

Export a checkpoint (a fine-tune, another Laya variant) to ONNX once with `tools/export_julia` from the
repository (Python + torch), then register the files:

```sql
-- needs: the exported files on disk
SELECT decide_register_model('my-julia', 'local', '/models/julia1.onnx', '/models/tokenizer/tokenizer.json');
-- 5th argument: profile 'julia-1' (default) or 'laya'; Laya needs rl_agent_config.json next to the graph
SELECT decide_register_model('my-laya', 'local', '/models/laya/julia1.onnx', '/models/laya/tokenizer.json', 'laya');
-- 6th argument: options MAP, e.g. a fitted calibration
SELECT decide_register_model('my-laya-cal', 'local', '/models/laya/julia1.onnx', '/models/laya/tokenizer.json', 'laya',
                             MAP {'calibration': 'platt:1.1,-0.3'});
```

Registration checks the files themselves. It rejects a `.safetensors` passed as the graph, a non-BPE
tokenizer, and a Laya config without its keys, and the message says what the file is. A Laya graph without
the `'laya'` profile is refused. Relative paths resolve against DuckDB's working directory.

## Long text

```sql
-- needs: the julia-1 weights (CALL decide_download('julia-1'))
SELECT id, decide_token_count(text, model := 'julia-1') AS tokens FROM tickets ORDER BY tokens DESC LIMIT 10;
```

```sql
SET anofox_decide_on_truncate = 'ignore';   -- score the shortened input instead of failing (default 'error')
SET anofox_decide_max_length = 4096;        -- julia-1 window, 32..8192 (8192 = default = maximum); lower saves memory. Laya ignores it
SET anofox_decide_head_length = 512;        -- tokens reserved for the question and options
SET anofox_decide_batch_tokens = 16384;     -- batch the rows of a chunk (0 = off; 1024..131072)
SET anofox_decide_max_length = 8192;
SET anofox_decide_on_truncate = 'error';
SET anofox_decide_batch_tokens = 0;
```

## GPU

```sql
SELECT * FROM decide_devices();     -- cpu plus any NVIDIA / AMD / Apple GPU, usable or not and why
SELECT * FROM decide_backends();    -- per local model and device: supported, or the reason and the fix
```

```sql
-- needs: a GPU and network (downloads and verifies the plugin from the release); works for CUDA, ROCm and MLX
CALL decide_accelerate();                     -- find the GPU, fetch + verify its plugin, use it from now on
SET anofox_decide_batch_tokens = 8192;        -- keep the GPU busy
SELECT model, device FROM decide_models();    -- where each local model is scored now
```

```sql
-- needs: network; fetch one backend's plugin explicitly instead of letting decide_accelerate choose
CALL decide_download_runtime('cuda');         -- or 'rocm', 'mlx'
```

No GPU needs a compile step: the plugins build a shape's graph on first use. `decide_gpu_precompile` is still
accepted on ROCm, where it compiles nothing and only loads the model onto the GPU (deprecated; refused on CUDA, MLX and the CPU).

| Setting | Default | Meaning |
|---|---|---|
| `anofox_decide_device` | `auto` | `auto`, `cpu`, `cuda`, `rocm`, `mlx` |
| `anofox_decide_gpu_precision` | `fp32` | only `fp32`. Other values are rejected |
| `anofox_decide_ep_path` | `<cache dir>/plugins` | where plugins are installed and loaded from. Must not be writable by other users. `anofox_decide_plugin_dir` is its deprecated old name, read when it is empty |

Requirements (the plugin is the only download; no CUDA toolkit or cuDNN): Linux x86-64 with an
NVIDIA GPU of sm_75 or newer and its driver; Linux x86-64 with ROCm (HIP, hipBLAS, rocBLAS) and an AMD GPU of
gfx90a, gfx942, gfx1030, gfx1100-gfx1102, gfx1150, gfx1151, gfx1200 or gfx1201; or macOS on Apple silicon with
`brew install mlx mlx-c`. Not supported yet: half precision, CUDA on Windows. GPU probabilities match the CPU to
within 1e-3 on the local end-to-end suite, labels exact.

| Error contains | Fix |
|---|---|
| `anofox_decide_device is 'cuda' but no 'cuda' device was found` | `CALL decide_accelerate();` or `SET anofox_decide_device = 'auto';` |
| `... cannot run on cuda:0: ... its plugin is not installed` | `CALL decide_accelerate();`. `decide_backends()` shows the reason |
| `... enable_external_access is off` | `SET enable_external_access = true;` or `SET anofox_decide_device = 'cpu';` |
| `does not match its published checksum` | the plugin was deleted. Retry `CALL decide_download_runtime('<backend>');` |
| `a runtime library it needs is missing` / `ran out of device memory` | install the NVIDIA driver, ROCm (HIP, hipBLAS, rocBLAS) or MLX. For memory, `SET anofox_decide_batch_tokens = 4096;` |
| `the text is N tokens but the local model reads at most M` | the window cannot grow (8192 tokens is the maximum). Shorten the text (strip quoted replies and signatures, or split it into parts and combine the scores), lower `anofox_decide_head_length` when the question is short (julia-1), or accept the cut: `SET anofox_decide_on_truncate = 'ignore';` |

Reference pages: [Local models](https://github.com/DataZooDE/anofox-decide/blob/main/docs/api/03-local-models.md),
[GPU acceleration](https://github.com/DataZooDE/anofox-decide/blob/main/docs/api/04-gpu.md),
[GPU backends](https://github.com/DataZooDE/anofox-decide/blob/main/docs/GPU_BACKENDS.md).
