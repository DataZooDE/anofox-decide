# Review follow-up decisions (DRAFT)

Records the owner's answers to the agent-crew review open questions
(`.crew/runs/20260929_060825_review_review-the-uncommitted-anofox-decide-roa/result.md`).

## Q1 — Deployment threat model (SETTLED)

**Decision:** trusted team / internal service. Colleagues or internal apps
share the database; no untrusted SQL authors.

**Consequence:** F4 (file-access gates) and F11 (registry hardening) stay
documented limitations for a future untrusted-deployment pass. F2
(endpoint-scoped key, loopback-only http) and F3 (secret type) were
implemented after all: F2 as cheap hardening, F3 per the owner's explicit
step-2 decision. Code-correctness items (F1, F5, F6, F8, F9, F12) all done.

## Q2 — decide_accuracy threshold form (SETTLED)

**Decision:** add a 3-arg form, keep the 2-arg form as documented @0.5.
`decide_accuracy(p, y, threshold)` shares `RequireThreshold` with
`decide_decision`; the 2-arg form stays as convenience with its 0.5
boundary stated in the function description and in CALIBRATION_PERF.md.

## Q3 — decide_table per-row state (SETTLED)

**Decision:** build the per-row version now. `decide_table` must work in
`LATERAL` over a table of states (one provider round trip per scan for
remote, per-row scoring otherwise), with its own red/green test cycle.
The `model :=` named parameter rides along for surface consistency.

## F4 decision (step 1, owner-selected)

Fail fast at registration, enforce at load: `decide_register_model` opens
both files through DuckDB's filesystem immediately (existence + policy),
and scoring reads bytes through the same gate.

## Implementation record (fix pass, 29 Sept 2026)

- F1: metrics read through `UnifiedVectorFormat::GetData`; filtered-scan
  probe observed `DICTIONARY_VECTOR` on the label input. Fixed + filtered,
  join, and BIGINT-label tests (`decide_calibration.test`).
- F9: integer label overload widened to `BIGINT`.
- F6: `DecideEvaluate` / `DecideDefaultModel` / `DecideMaxQuestions` /
  `RequireKnownProvider` live in `decide_provider.cpp`; scalars and table
  route through them.
- F8: `DecideParseManyQuestions` takes `func_name`; the table rewriter is
  gone; one regex pins the `decide_table:` colon.
- F5: `decide_accuracy(p, y, threshold)` shares `RequireThresholdDouble`
  with `decide_decision`; 2-arg stays, documented @0.5.
- F12: per-call `DecideModelCache` in the scalars; `decide_many` stub
  choice-probability pin in `decide_contract.test`.
- Q3: `in_out_function` per-row operator with resume state (constant path
  unchanged); `model :=` declared for constant calls, positional in
  lateral calls; spillover test past 2048 rows.
- Q1 (trusted): env key scoped to `api.typesafe.ai`, cleartext http
  loopback-only (`decide_remote_config.test` + Catch2 pure tests). F3, F4,
  F10, F11 remain documented limitations for a future untrusted-deployment
  pass.
- F4 (step 1, done): `decide_register_model` opens graph + tokenizer through
  DuckDB's filesystem (fail fast); scoring reads through the same gate.
  `decide_local_access.test` covers baseline, missing file, allowlist,
  denial, and load-time denial; wired into `make test_*_internal`. The
  Catch2 missing-graph assertion now expects "cannot open graph file".

## Julia-1 local scoring investigation (29 Sept 2026)

Realistic 8-ticket set (`test/fixtures/support_tickets.csv`): TypeSafe 8/8
refund, 7/8 team; local Julia-1 4/8 refund, 3/8 team. Real-text parity
(`test_real_ticket_parity`): torch vs ORT raw logits agree, single vs padded
batch agree, so the export/padding path is faithful. `[false,true]` option
order swings raw logits by ~4.8, so the 0.5 threshold is unreliable for the
local model; treat weak local discrimination as model behaviour (upstream
model.py parity still unverified). Fixed along the way: max_length/head_length
defaults (8192/512), head cap, `<0xXX>` byte fallback. The agent-crew run was
partial (claude OAuth expired, muse quota); codex members only.

## Laya local profile (29 Sept 2026)

Laya (convaiinnovations, Apache 2.0) shares Julia-1's marker-head
architecture, so `tools/export_julia` exports it unchanged. Added: registry
`profile` argument (`julia-1` | `laya`), laya rendering (`false: no, the
statement does not hold` / `true: ...`), limits + temperatures from
`rl_agent_config.json`, and a ByteLevel BPE tokenizer (GPT-2 pre-tokenizer,
NFC, added-token extraction) verified against HF goldens. Both the
multilingual and typed-decisions checkpoints reproduce upstream
`RLAgent.system_one` within 5e-5. 8-ticket smoke test (refund/team):
Jev 8/8, 7/8; Laya multilingual 8/8, 7/8; Laya typed-decisions 7/8, 6/8;
Julia-1 4/8, 4/8. Also fixed an out-of-bounds read for mixed-width
questions in one local batch. Not done: `score` questions locally, larger
labelled evaluation, Von.

## Kev via the remote provider (29 Sept 2026)

Kev-0.8B (jaredpalmer/kev, Apache 2.0) ran on this CPU-only machine through
`python -m kev.serve` and the existing `typesafe` provider with
`anofox_decide_endpoint=http://127.0.0.1:8009` and no code change. 8-ticket
smoke test: refund 6/8, team 6/8 (Jev 8/8, 7/8; Laya multilingual 8/8, 7/8).
Larger Kev sizes (4B/9B/27B) were not tried (no GPU). Von needs order-invariant
attention masking in the export and was not attempted.

## Liquid AI D1 and generalized remote providers (30 Sept 2026)

D1 (Liquid AI, hosted, API only: no downloadable weights) speaks TypeSafe's
System One wire format. Verified live against
`POST https://api.liquid.ai/decisions/v1/systemone` with model `d1:free`:
request/response identical (`answers.<id>.noul` / `.choice` / `.probabilities`
/ `.confidence`, `usage.output_tokens` 0); errors are JSON
`{"error": {"message", "type"}}` with 401 (bad key), 422 (no questions), 404
(unknown model). Docs name the key variable `LIQUID_API_KEY`.

Instead of a one-off provider, remote providers are now profiles
(`typesafe`, `liquid`, generic `systemone`) with per-model options in a `MAP`
(`endpoint`, `path`, `model`, `key_env`): Jev, D1 and Kev work side by side in
one session; `anofox_decide_endpoint` / `anofox_decide_api_key` stay as legacy
settings for `typesafe` only. Named arguments cannot carry the options
(DuckDB scalar functions treat `x := v` as positional), hence the MAP.
Key precedence: stored secret (host-scoped) > legacy setting (typesafe) >
model `key_env` > the provider's env var on its default host, so env vars work
and secrets always override them (asserted in Catch2 and against the live
service: a bogus stored secret makes a call fail while the valid env key
stays untouched). 8-ticket smoke test: D1 8/8 refund, 8/8 team (Jev 8/8, 7/8).
Kev now registers as `systemone` with a loopback endpoint and a host-scoped
secret, no session `SET`. Not done: `score` questions.

## 200-ticket evaluation (1 Oct 2026)

`tools/eval` scores six models on a deterministic 200-ticket sample of the public Bitext customer-support
dataset (refund yes/no + 4-way routing); results and caveats in docs/EVALUATION.md. It changed earlier
conclusions drawn from 8 tickets: Jev is the most accurate (93% refund / 92% routing), Laya multilingual
ties it on refund but routes at 62%, D1 matches Jev on routing but over-predicts refunds at the 0.5
cut-off (AUROC 0.970, spec 67%), typed-decisions is the best in-process router (76%), Kev-0.8B and
Julia-1 are not competitive. D1's free tier had very variable latency (1-34 s per call). English only and
template-generated, so absolute numbers will be lower on real tickets.

## strands-decider provider (2 Oct 2026)

strands-decider (strands-labs, Apache-2.0; Qwen3.5-2B torso + LoRA + pointer head, same family as Kev) serves
`POST /v1/systemone`, so it is a remote provider, not an in-process model (hybrid Gated DeltaNet layers, custom
masking and a 2B torso are the same blocker as Kev). Its schema validates choice `criteria` as `dict[str, str]`,
so the `null` descriptions we send everywhere else are rejected with HTTP 422 (confirmed with curl). Added a
`strands` provider profile: loopback default endpoint, no API key (no Authorization header unless a key is
configured), and choice options sent as their own descriptions; the options MAP gained `criteria: 'null' |
'name'` for any other server with the same schema. Server limit: 24 options per choice question
(`num_slots`). On the 200-ticket evaluation it scores 81% refund / 65% routing (6.6 min on CPU).

## `score` (ordinal) questions (2 Oct 2026)

Added end to end: `decide_score(state, question, levels[, model]) -> DOUBLE` (expected 0-based level), the
`score` kind in `decide_many` (`levels` array) and `decide_table` (new trailing columns `score` and
`distribution`, the latter also filled for choice rows), the stub provider, all remote providers and the local
Julia-1 / Laya models. Levels are an ordered rubric of 2 to 10 unique non-empty descriptions. Wire format
verified live against Jev, D1 and strands-decider: `criteria` is a JSON array, answers carry `score`,
`legend`, `probabilities` keyed `"0"..` and `confidence`; the parser derives the expected level from the
distribution when a server omits `score`. Local scoring: Julia-1 uses the level descriptions as given
(`julia/typed.py`), Laya renders `level <i>: <text>` (0-based) and applies its `score:*` temperatures; the
temperature index had no score case before. Parity with the upstream Python on 3 questions per model:
Julia-1 1.00148/0.99969/1.02204 vs 1.00148/0.99969/1.02204, Laya multilingual 1.0581/0.5510/1.8222, Laya
typed-decisions 1.4665/0.1438/1.7720 (all within 1e-4 / the upstream 4-decimal rounding).

Fixes found on the way: the remote response parser treated every non-noul answer as choice (a latent bug for
any new kind), `decide_many`/`decide_table` labelled every non-binary answer "choice", and the probability-sum
check (1e-3) rejected valid Jev answers because Jev rounds to 2 decimals (1 of 120 real score answers summed to
0.99); the tolerance is now 1e-3 + 0.005 per option for choice and score. Not done: an accuracy evaluation of
score on labelled data (the Bitext set has no ordinal labels).

## Error guidance, first-run experience (2 Oct 2026, PR C1)

Audit of every user-facing message (SQL surface, remote provider, local models) found: the report-a-bug footer on
every user mistake, a silent stub default, raw JSON and jargon in messages, thin diagnostics. PR C1 (foundation +
SQL surface): message convention `<function>: <what>. Fix: <sql>` with offending values echoed (new
`src/decide_errors.*`); `DECIDE_GUARD` adds the issue-link footer only to unexpected errors (user mistakes keep
their DuckDB exception class and no footer); **no implicit model** (`anofox_decide_model` is unset by default;
a call without a model explains how to choose and register one; resolution stays lazy so NULL inputs need none);
unknown-model errors name the source, the registered ids and a close match; `decide_models()` gains `is_default`,
`ready`, `hint`, `profile`, `endpoint`, `wire_model`; new `decide_doctor()` (item, status, detail, fix) and
`decide_unregister_model(id)`; registration rejects stray path/profile arguments for remote providers, explains
duplicate ids, and file-open errors use plain text, the right role and a Hugging Face id hint; settings validators
show range, default and an example, `head_length` and `max_length` are cross-checked; threshold and metric errors
echo the value; empty questions and empty `CREATE SECRET` are rejected. Planned next: PR C2 (remote HTTP/transport
errors) and PR C3 (local truncation, pairing, registration depth).

## Error guidance, remote provider (2 Oct 2026, PR C2)

HTTP and transport failures now say what happened, to whom, and what to do. Server error bodies are parsed
(`DecideExtractServerMessage`: `{"error":{"message","type"}}`, `{"error":"..."}`, `{"detail":"..."}`, FastAPI
`{"detail":[{"loc","msg"}]}` summarised to three entries, `{"detail":{"message"}}`; HTML detected and described;
capped at 240 chars; redacted if it echoes the key). Per-status text and next step: 401 (names the key's source,
`CREATE OR REPLACE SECRET` fix), 403, 404 and a 400 about the model (shows the wire model, the registered id and the
`MAP {'model': ...}` fix), 422 (with the `criteria` hint), 429 (real attempt count, `max_retries`), 5xx ("not your
query"), timeouts. Transport errors keep httplib's reason (`error_kind`: refused, timeout, read, tls, proxy,
invalid endpoint); connection refused on loopback is not retried and says to start the server; the attempt count is
the number actually sent. The `allow_remote` gate is checked before key resolution and names the model and host; the
missing-key message lists the options per provider and notices an unset `key_env`; endpoint validation fixes the raw
`std::stoi` crash and explains path/port/userinfo mistakes; one wording for the cleartext-http refusal; response
parse failures name the service and model, use public kind names, list what the server did send and add the
`criteria` hint for renamed options. Tested against a real HTTP server on loopback (12 scenarios: 401, 404, 400,
422, 429, 503, HTML 200, error body 200, timeout, refused, success, key reflection).
