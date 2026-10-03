# AnoFox Decide — spike results (plan step 1, 28 Sept 2026)

> **Historical design notes** from the first spike (28 to 30 September 2026), kept for the reasoning behind
> the provider contracts. Status lines below describe the project *at the time*: the local ONNX provider,
> live tests against TypeSafe and Liquid D1, and the evaluation have since been built (see the
> [README](../README.md), [EVALUATION.md](EVALUATION.md) and [REVIEW_FOLLOWUP.md](REVIEW_FOLLOWUP.md)).

## Remote provider: TypeSafe System One API

- Endpoint: `POST https://api.typesafe.ai/v1/systemone`, `Authorization: Bearer <API_KEY>`.
- Request: `{state, model, questions}` where `questions` is a caller-keyed map; answers come back under the same keys.
- Question types: `noul` (yes/no → P(yes)), `choice` (one distribution over options, sums to 1), `score` (ordinal levels).
- Models (inspected [Models](https://docs.typesafe.ai/models)): `jev-1.13.0`, aliases `jev-latest` (= `jev-1.13.0`), `jev-preview`. Response echoes the versioned ID that answered — log it per row for auditability.
- Limits: 64k tokens/request; 32k for `state` + longest question; 429 on rate limit; client SDKs retry with backoff honoring `retry-after`.
- Mapping to the SQL surface: `decide_probability` → one `noul`; `decide_choice` → one `choice`; `decide_many` → fan-out map of `noul`/`choice` in ONE request (speculative fan-out pattern). `model := '...'` passes through the `model` field untouched so new IDs need no code change.

## Local provider: open decision models (self-hosted)

Candidates considered (research 28 Sept 2026): Supersonic Labs Julia 1, Laya, Kev, NanoJev.

- **Julia-1** (SupersonicLabs, Apache 2.0, 144.3M params, mmBERT-small encoder + decision head, 2–20 answers, CPU-runnable, [HuggingFace](https://huggingface.co/SupersonicLabs/Julia-1)). First local target: open license, CPU story matches DuckDB embedding, small enough to test with a weight-free fixture.
- **Laya** (ConvAI Innovations, open reply to Jev, typed-decision checkpoints). Second local target after Julia-1 packaging is proven.
- **Kev / NanoJev** (community Jev replicas, e.g. NanoJev serves `POST /api/evaluate`). Compatible via the same provider contract; pin exact checkpoint revisions at implementation time.

Decision: local provider loads an ONNX-exported open decision model via ONNX Runtime (prebuilt archive for debug, vcpkg static ORT for the release single-file build). Registry accepts model IDs `julia-1`, `laya`, `kev`, `nanojev` plus explicit revision pins.

## Verified wire facts (28 Sept 2026, from API docs + offline hermetic tests)

- Request: `{"state", "model", "questions": {id: {"type": "noul", "instructions", ...} | {"type": "choice", "instructions", "criteria": {opt: null, ...}}}}` — choice options travel as a **criteria map**, not an array (max 255).
- Response: `{"model": "<versioned-id>", "answers": {id: {"type": "noul", "noul": p} | {"type": "choice", "choice", "probabilities", "confidence"}}, "usage": {...}}`.
- Errors: 401 (key) / 422 (validation) never retried; 429 / 529 / 5xx retried with bounded backoff honoring Retry-After.
- **Offline env finding (at the time; live tests have since passed against both services):** this machine had no outbound HTTPS (dead proxy, no DNS) — live E2E is committed as `test/sql/decide_remote_live.test` + `make test-live` but stayed blocked on network at the time. Hermetic coverage (request/response mapping, retry counts, 401-no-retry, key redaction, auth header) is green in Catch2 `[remote]`.

## Consequences for the build

- Remote provider: DuckDB-bundled `httplib` + `yyjson` only, system OpenSSL linked by the extension (no vcpkg). Key from `TYPESAFE_API_KEY` env (or `anofox_decide_api_key` setting override); `anofox_decide_allow_remote=false` default; key never logged.
- Local NLI (done since): `tools/export_julia` exports the checkpoint to ONNX and the C++ ONNX Runtime side scores it in-process, with parity checks against the upstream Python. No weights in the repo (license wall).

## Liquid AI D1 (30 Sept 2026)

Hosted System One-compatible API (`POST https://api.liquid.ai/decisions/v1/systemone`,
model `d1:free`, Bearer key from `LIQUID_API_KEY`, no open weights). Wire format
verified identical to TypeSafe's (see docs/REVIEW_FOLLOWUP.md); integrated as the
`liquid` remote provider profile.

## Ordinal `score` wire format (2 Oct 2026)

Verified live (Jev `jev-1.13.0`, Liquid `d1:free`, strands-decider): request question
`{"type":"score","instructions":...,"criteria":["low", ..., "high"]}` (ascending array of 2 to 10 strings);
answer `{"type":"score","score":<expected index>,"legend":{"0":...},"probabilities":{"0":p0,...},"confidence":c}`.

## Cloudflare Clef (3 Oct 2026)

Hosted on Workers AI: `POST https://api.cloudflare.com/client/v4/accounts/{ACCOUNT_ID}/ai/run/@cf/cloudflare/{clef|clef-flash}`,
`Authorization: Bearer <token>` (token permissions: Workers AI - Read and Edit), body `{"model", "state", "questions"}`.
Recorded against the real API with a real token (responses sanitised in `test/fixtures/cloudflare/`):

- **Request:** the body our request builder already produces works as is (choice `criteria` map, score `criteria`
  array). A choice with **`null` criteria values is accepted** as well as one with descriptions. The URL carries the
  model name and the body's `model` must be exactly `clef` or `clef-flash` (anything else: HTTP 400, code 5006,
  "'/model' failed test ^\s*(clef|clef-flash)\s*$").
- **Success is wrapped in the Workers AI envelope:** `{"result": {"model", "answers", "usage"}, "success": true,
  "errors": [], "messages": []}`. Inside `result` the answers are the System One shape: `noul` (`noul`), `choice`
  (`choice`, `probabilities`, `confidence`), `score` (`score`, `legend`, `probabilities`, `confidence`). clef-flash answered the
  three-question example in about 0.3 s.
- **Errors use `errors[]`, not the System One error bodies:**
  - bad or missing token: HTTP 401, `{"result":null,"success":false,"errors":[{"code":10000,"message":"Authentication error"}]}`;
  - a **valid token with the wrong account id gives the same 401 and message**, so the two cannot be told apart and the guidance has to name both;
  - a wrong model name in the URL: HTTP **400** (not 404), code 7000, "No route for that URI";
  - validation (65 questions, limit 64): HTTP 422, code 5012, with the details as a JSON string nested inside `message`
    (`AiError: AiError: {"error":{"type":"invalid_request","message":"Request body failed validation","details":{"fieldErrors":{...}}}} (<request id>)`).
- **Limits** (documentation, confirmed for the question count): 64 questions per request, 64k-token context, up to 4 images
  (not supported by `anofox_decide`).
- Pricing: $0.24 per million input tokens (clef), $0.09 (clef-flash). No rate-limit headers were returned.
