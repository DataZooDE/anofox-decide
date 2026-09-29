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
