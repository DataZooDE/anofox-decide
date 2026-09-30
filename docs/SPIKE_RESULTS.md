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

## Verified wire facts (28 Sept 2026, from API docs + offline hermetic tests)

- Request: `{"state", "model", "questions": {id: {"type": "noul", "instructions", ...} | {"type": "choice", "instructions", "criteria": {opt: null, ...}}}}` — choice options travel as a **criteria map**, not an array (max 255).
- Response: `{"model": "<versioned-id>", "answers": {id: {"type": "noul", "noul": p} | {"type": "choice", "choice", "probabilities", "confidence"}}, "usage": {...}}`.
- Errors: 401 (key) / 422 (validation) never retried; 429 / 529 / 5xx retried with bounded backoff honoring Retry-After.
- **Offline env finding:** this machine has no outbound HTTPS (dead proxy, no DNS) — live E2E is committed as `test/sql/decide_remote_live.test` + `make test-live` but stays RED-blocked on network. Hermetic coverage (request/response mapping, retry counts, 401-no-retry, key redaction, auth header) is green in Catch2 `[remote]`.

## Consequences for the build

- Remote provider: DuckDB-bundled `httplib` + `yyjson` only, system OpenSSL linked by the extension (no vcpkg). Key from `TYPESAFE_API_KEY` env (or `anofox_decide_api_key` setting override); `anofox_decide_allow_remote=false` default; key never logged.
- Local NLI: `tools/export_julia` scaffold (mirrors tabfm `tools/export_onnx`); C++ ORT side pending network (inspect → export → parity → fixture). No weights in repo (license wall).

## Liquid AI D1 (30 Sept 2026)

Hosted System One-compatible API (`POST https://api.liquid.ai/decisions/v1/systemone`,
model `d1:free`, Bearer key from `LIQUID_API_KEY`, no open weights). Wire format
verified identical to TypeSafe's (see docs/REVIEW_FOLLOWUP.md); integrated as the
`liquid` remote provider profile.
