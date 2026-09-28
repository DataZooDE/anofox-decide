# AnoFox Decide — spike results (plan step 1, 28 Sept 2026)

## Remote provider: TypeSafe System One API

- Endpoint: `POST https://api.typesafe.ai/v1/systemone`, `Authorization: Bearer <API_KEY>`.
- Request: `{state, model, questions}` where `questions` is a caller-keyed map; answers come back under the same keys.
- Question types: `noul` (yes/no → P(yes)), `choice` (one distribution over options, sums to 1), `score` (ordinal levels).
- Models (inspected [Models](https://docs.typesafe.ai/models)): `jev-1.13.0`, aliases `jev-latest` (= `jev-1.13.0`), `jev-preview`. Response echoes the versioned ID that answered — log it per row for auditability.
- Limits: 64k tokens/request; 32k for `state` + longest question; 429 on rate limit; client SDKs retry with backoff honoring `retry-after`.
- Mapping to BRD: `decide_probability` → one `noul`; `decide_choice` → one `choice`; `decide_many` → fan-out map of `noul`/`choice` in ONE request (speculative fan-out pattern). `model := '...'` passes through the `model` field untouched so new IDs need no code change.

## Local provider: open decision models (self-hosted)

User asked for Supersonic Labs Julia 1, Laya, Kev, NanoJev. Research 28 Sept 2026:

- **Julia-1** (SupersonicLabs, Apache 2.0, 144.3M params, mmBERT-small encoder + decision head, 2–20 answers, CPU-runnable, [HuggingFace](https://huggingface.co/SupersonicLabs/Julia-1)). First local target: open license, CPU story matches DuckDB embedding, small enough to ship like tabfm's weight-free fixture + download flow.
- **Laya** (ConvAI Innovations, open reply to Jev, typed-decision checkpoints). Second local target after Julia-1 packaging is proven.
- **Kev / NanoJev** (community Jev replicas, e.g. NanoJev serves `POST /api/evaluate`). Compatible via the same provider contract; pin exact checkpoint revisions at implementation time.

Decision: local provider loads an ONNX-exported open decision model via ONNX Runtime (tabfm `cmake/ort.cmake` pattern: prebuilt archive for debug, vcpkg static ORT for release single-file). Registry accepts model IDs `julia-1`, `laya`, `kev`, `nanojev` plus explicit revision pins.

## Consequences for the build

- Stub stage (this commit): deterministic `stub` provider, no network, no weights — SQL surface + harness go green first.
- Plan step 4: remote provider needs `httplib` (already in DuckDB tree) + DuckDB secrets-manager integration; `anofox_decide_allow_remote=false` default.
- Plan step 5: local NLI needs ORT wiring + `test/fixtures/` weight-free ONNX fixture + license-gated download (tabfm WS-D pattern).
