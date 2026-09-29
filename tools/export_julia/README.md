# tools/export_julia — Julia-1 → ONNX exporter

Replicates `../anofox-tabfm/tools/export_onnx` for the local NLI provider: SupersonicLabs/Julia-1 (Apache 2.0) becomes the `julia-1` model behind `decide_register_model(id, 'local')` once the C++ ORT side lands.

## Status: scaffold only (offline env)

Authored, not run. `uv sync` needs PyPI + HuggingFace, both unreachable here. No weights anywhere in the repo (license wall, same rule as tabfm).

## Pipeline (when networked)

```bash
cd tools/export_julia
uv sync
uv run export_julia inspect --out ./out                          # pins julia-1-spec.json
uv run export_julia export --spec ./out/julia-1-spec.json \
  --weights $JULIA_WEIGHTS_DIR --out ./out                       # julia1.onnx + tensor map + tiny fixture
JULIA_WEIGHTS_DIR=... JULIA_SPEC_PATH=... uv run pytest          # parity gate
```

## C++ side (pending)

Mirror `cmake/ort.cmake` + `tabfm_ort_engine.cpp`: prebuilt ORT archive for debug, vcpkg static ORT for release single-file; `decide_local_nli.cpp` loads `julia1.onnx`, scores (state, question) pairs, and serves the same `DecideQuestion → DecideAnswer` contract as the remote module. Registration flips `'local'` from rejected to available in `decide_provider.cpp`.
