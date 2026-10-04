# MLX spike (2026-10-04): run the decision models on Apple GPU

Throwaway spike for the local-GPU plan. `tabfm_mlx_graph*` / `tabfm_onnx_reader*` are copies of anofox-tabfm
src/ (ONNX interpreter over mlx-c) with: `Split`, `Flatten`, `Relu`, `ConstantOfShape` added, typed int64/bool feeds,
and weights read from the upstream safetensors through the tensor map (F16 cast to F32 on device).

- `mkindex.py st.safetensors tensor_map.json out.idx` flattens tensor map + safetensors header into a TSV the harness reads.
- `inline.py` inlines the weights into the weight-free graph (for ORT CPU baselines); `cpubench.py` times ORT CPU.
- `main.cpp` + `build.sh` build `mlxrun graph.onnx weights.safetensors index.tsv B T M inputs.bin out.bin reps`.

Measured on an Apple M3 (16 GB), steady-state ms per call, versus ORT CPU 1.30 on the same machine:

| model, shape | ORT CPU | MLX | max raw-score difference |
|---|---|---|---|
| Laya multilingual B=1 T=1024 | 1.0 s | 0.36 s | 5.9e-6 |
| Laya multilingual B=8 T=256 | 1.8 s | 0.47 s | 2.2e-4 |
| Laya typed-decisions B=1 T=1024 | 3.1-3.4 s | 0.79 s | 2.1e-6 |
| Julia-1 B=1 T=2048 | 1.8 s | 0.53 s | 4.0e-5 |
| Julia-1 B=1 T=4096 | 6-8 s | 1.67 s | 6.6e-5 |
| Julia-1 B=1 T=8192 | not run | 6.0 s | n/a |

Peak RSS 1.2-1.3 GB throughout. Mac ORT versus Linux ORT differs by up to 1.9e-4 on the same inputs.
