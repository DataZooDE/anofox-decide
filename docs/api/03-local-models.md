# Local models

A local model runs inside DuckDB on ONNX Runtime, on the CPU and fully offline once its weights are on disk.
For a GPU, see [GPU acceleration](04-gpu.md).

- [The built-in catalog: Julia-1 and Laya](#the-built-in-catalog-julia-1-and-laya)
- [Your own exported model](#your-own-exported-model)
- [Long text](#long-text)
- [Batching rows](#batching-rows)

## The built-in catalog: Julia-1 and Laya

Three models work out of the box:

| Model | Download | Peak memory | Notes |
|---|---|---|---|
| `laya-multilingual` | 0.68 GB | 3.4 GB | multilingual (mmBERT-base encoder); the recommended start |
| `laya-typed-decisions` | 0.85 GB | 5.1 GB | English (ModernBERT-large encoder); the best router in our evaluation |
| `julia-1` | 0.61 GB | 1.5 GB | smallest and fastest, but the weakest in our evaluation |

```sql
CALL decide_download('laya-multilingual');           -- once: fetches and verifies the files (8 to 11 seconds on the test machine's connection)
SELECT decide_probability('I want my money back.', 'A refund is requested.', model := 'laya-multilingual');
```

That is all: no registration, no Python, no `INSTALL httpfs`. The extension embeds the model graphs (about
1 MB each, without weights); `decide_download` fetches the upstream weights from Hugging Face, pinned to a
fixed revision, checks size and SHA-256 of every file, resumes an interrupted download, and keeps them in
`~/.cache/anofox-decide` (change it with `SET anofox_decide_cache_dir = '<dir>'`). From then on nothing leaves
your machine. `SELECT model, ready, hint FROM decide_models();` lists the three with `ready = false` and the
call to run until they are downloaded, `decide_doctor()` reports them, and naming a model that is not
downloaded fails with `model 'laya-multilingual' is not downloaded. Fix: CALL decide_download('laya-multilingual');`.
Behind a proxy, set `HTTPS_PROXY` as for the hosted models ([Environment variables](05-settings.md#environment-variables)).

The first call in a session loads the model (4.0 s for `laya-multilingual`, 5.5 s for
`laya-typed-decisions`, 1.4 s for `julia-1` on the test machine, files in the page cache); later calls reuse it.
Peak resident memory is the table above (measured with `getrusage` over a whole DuckDB process): the Laya
checkpoints are float16 and ONNX Runtime computes in float32.

## Your own exported model

To run a checkpoint that is not in the catalog (a fine-tune, another Laya variant), export it to ONNX once with
[`tools/export_julia`](../../tools/export_julia/README.md) (Python, torch) and register the files:

```sql
SELECT decide_register_model('julia-1', 'local', '<path>/julia1.onnx', '<path>/tokenizer/tokenizer.json');
SELECT decide_probability('...', '...', model := 'julia-1');
```

The optional 5th argument is the **profile**, `julia-1` (default) or `laya`. It fixes how questions are
rendered for that model family, the sequence limits and the calibration. Laya
([convaiinnovations/laya](https://huggingface.co/convaiinnovations/laya), Apache 2.0, ModernBERT / mmBERT
encoders) needs `rl_agent_config.json` from its checkpoint next to the graph:

```sql
SELECT decide_register_model('laya', 'local', '<dir>/julia1.onnx',
                             '<snapshot>/multilingual/tokenizer/tokenizer.json', 'laya');
```

An options `MAP` as the 6th argument takes a fitted `calibration`
([Calibration](../guides/03-evaluate-and-calibrate.md#calibration)).

Both SentencePiece-style tokenizers (Julia-1, Laya multilingual) and ByteLevel BPE (Laya English and
typed-decisions) are supported. Registration checks the files, not only that they exist: a weights file
(`.safetensors`), a text file or a zip as the graph, a tokenizer that is not BPE, and a Laya config
without its keys are rejected with a message that says what the file is. `decide_models()` and
`decide_doctor()` run the same checks. A model registered with its own `.onnx` graph stays on the CPU
([GPU acceleration](04-gpu.md)).

## Long text

**Long text is an error, not a silent cut.** A local model reads a fixed window (8192 tokens for
`julia-1`, adjustable with `anofox_decide_max_length` and `anofox_decide_head_length`; Laya models use the
`max_len` and `head_max_len` of their `rl_agent_config.json` and ignore those two settings). When the text,
the question or an option does not fit, the call fails and says how much would be ignored, and for which
question. Find the long rows first with `decide_token_count`, or accept the cut with
`SET anofox_decide_on_truncate = 'ignore';`:

```sql
SELECT id, decide_token_count(text, model := 'julia-1') AS tokens FROM tickets ORDER BY tokens DESC LIMIT 10;
SET anofox_decide_on_truncate = 'ignore';  -- score the shortened text instead of failing
```

## Batching rows

By default a local model scores one text (with its questions) at a time.
`SET anofox_decide_batch_tokens = 16384;` scores the rows of a whole chunk together instead: rows are sorted by
length, padded to 128, 256, ... 8192 tokens (never above the model's window) and run in batches of at most that
many padded tokens. Padding is masked exactly, so the answers are the same: on the 200-ticket evaluation with
Laya multilingual the probabilities were identical to the last bit and no routing or refund decision changed.
On CPU the effect on speed is small and inconsistent (200 tickets: 112 s and 122 s off, 98 s and 103 s at
`16384`, 146 s at `4096`, on a machine under heavy load, so indicative only), which is why it is off. It exists
to keep a GPU busy: with one row per call a GPU waits, with a batch it does not.

See [`examples/05_local_model.sql`](../../examples/05_local_model.sql).
