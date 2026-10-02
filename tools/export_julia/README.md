# tools/export_julia — decision model → ONNX exporter

Exports a marker-head decision checkpoint to the ONNX graph that `anofox_decide` runs in-process
(`decide_register_model(id, 'local', <graph>, <tokenizer>[, profile])`). It handles
[SupersonicLabs/Julia-1](https://huggingface.co/SupersonicLabs/Julia-1) (profile `julia-1`, Apache 2.0) and
the [Laya](https://huggingface.co/convaiinnovations/laya) checkpoints (profile `laya`, Apache 2.0), which
share the architecture: an encoder (mmBERT or ModernBERT) plus a decision head that scores one marker per
question.

**No weights live in this repository** and none are redistributed: you download a checkpoint from Hugging
Face, export it once, and register the files. The extension needs only the exported `.onnx`, the model's
`tokenizer.json`, and, for Laya, its `rl_agent_config.json`.

## Export

```bash
cd tools/export_julia
uv sync                                          # torch comes from the PyTorch CPU index (no CUDA wheels)

# 1. Download a checkpoint (any marker-head one) from Hugging Face.
uv run python -c "from huggingface_hub import snapshot_download; snapshot_download('SupersonicLabs/Julia-1', local_dir='weights/julia-1')"

# 2. Julia-1 only: pin the upstream spec. docs/julia-1-spec.json is already pinned for you.
uv run export_julia inspect --out ./out

# 3. Export. --weights is the directory holding encoder/config.json and model.safetensors.
uv run export_julia export --spec ../../docs/julia-1-spec.json --weights ./weights/julia-1 --out ./out
```

This writes `julia1.onnx` and `julia1_tensor_map.json` to `--out` (opset 17 by default; `--opset` changes
it). For a Laya checkpoint, use the snapshot subfolder that holds its encoder and a minimal spec of your own:

```json
{"head": {"head_layers": 2, "dropout": 0.1}}
```

`repo_id` and `sha` are optional in the spec (they are only printed). Copy the checkpoint's
`rl_agent_config.json` next to the graph: the `laya` profile reads its sequence limits and calibration
temperatures from it.

## Use it

```sql
-- Julia-1
SELECT decide_register_model('julia-1', 'local', 'out/julia1.onnx', 'weights/julia-1/tokenizer/tokenizer.json');

-- Laya: profile 'laya', rl_agent_config.json beside the graph
SELECT decide_register_model('laya', 'local', 'out/julia1.onnx', 'weights/laya/tokenizer/tokenizer.json', 'laya');

SELECT decide_probability('I want my money back.', 'A refund is requested.', model := 'laya');
```

`SELECT * FROM decide_doctor();` opens the files and names the one that is missing. Both SentencePiece-style
tokenizers (Julia-1, Laya multilingual) and ByteLevel BPE (Laya English, typed-decisions) are supported.

## Check the export

```bash
uv run pytest                                    # a weight-free tiny export always runs
JULIA_WEIGHTS_DIR=./weights/julia-1 JULIA_SRC_MODEL=<upstream julia/model.py> uv run pytest   # matches the upstream model
JULIA_WEIGHTS_DIR=./weights/julia-1 JULIA_ONNX=./out/julia1.onnx uv run pytest                # torch against ONNX Runtime, fp32 1e-4
```

`scripts/compare_upstream.py` compares the DuckDB output with the upstream Python implementation on the
example tickets (`test/fixtures/support_tickets.csv`); each local ONNX model reproduces its upstream
reference to within 5e-5 in probability on a small check set.
