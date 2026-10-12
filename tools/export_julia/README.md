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

## Weight-free graphs (what the extension embeds)

`export_julia export ... --weight-free --arch <julia-1|laya-multilingual|laya-typed-decisions>` writes
`graph_<arch>.onnx` and `tensor_map_<arch>.json` instead of the full graph. The graph is exported without
constant folding (so parameters keep their names), every checkpoint tensor is replaced by an external-data
stub, and the map lists `initializer name -> safetensors key` (plus the shape the loader verifies). Every
mapping is value-checked against the checkpoint at export time. The three graphs in `resources/` were
produced this way (165 / 165 / 201 mapped initializers, all by exact name, no transposes, 0.6-1.0 MB each).
`scripts/gate_parity.py` runs the weight-free graph with the real safetensors injected next to the full-weights
graph on real tickets; `scripts/check_graph_invariants.py` is the CI check that committed graphs carry no
weight bytes; `scripts/make_tiny_fixture.py` regenerates the random-init test fixture.

### Packed layout for GPU backends

Every stub also carries an `offset` and a `length`, and the tensor map has a `pack` section with the same
numbers: the injected weights, as little-endian float32, laid out in one blob with the initializers in sorted name order,
each at a 64-byte aligned offset (`length = numel * 4`; `pack` also holds `alignment`, `itemsize` and
`total_bytes`, and `location = weights.bin`). The layout depends only on names and shapes, so a GPU plugin can build the blob from the
safetensors and the tensor map alone (`export_julia.weight_free.build_packed_blob` is the reference). Every stub names the same
`weights.bin` file, which a plugin stages beside its cached graph. Use a separate directory per model/checkpoint.
The CPU path injects by initializer name and never opens that file. The blob contains checkpoint tensors upcast
to float32 (and transposed where mapped), plus zero alignment padding; it is not a copy of the safetensors container.

`scripts/repack_offsets.py GRAPH...` adds the layout to an existing weight-free graph and map in place, touching
only external-data metadata and the map's `pack` section (this is how the committed graphs were upgraded).
Repacking is deterministic and idempotent; tiny-model tests compare it byte-for-byte with a fresh export.
`scripts/verify_pack.py GRAPH CHECKPOINT.safetensors` rebuilds the blob from the `pack` section only and checks
each tensor byte-for-byte against the stubs and the independently loaded, converted checkpoint; `check_graph_invariants.py` now also requires the layout and checks
every stub against it.

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
