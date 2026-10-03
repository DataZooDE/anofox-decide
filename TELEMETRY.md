# anofox_decide Telemetry

`anofox_decide` collects **anonymous, privacy-preserving usage telemetry** so we
can see which capabilities are used, on which platforms, and where they fail —
and prioritise accordingly. It is **on by default** and **trivial to turn off**.

Telemetry is emitted through the shared
[`DataZooDE/posthog-telemetry`](https://github.com/DataZooDE/posthog-telemetry)
library and follows the cross-product **`telemetry_schema: 2`** envelope
(`posthog-telemetry/TELEMETRY-SCHEMA.md`). Ingestion is the EU PostHog cloud.

## How to turn it off

Any one of these fully short-circuits telemetry — when disabled, **nothing
leaves the machine** (the opt-out is enforced at the transport, not just at the
call sites):

```sql
SET anofox_telemetry_enabled = false;   -- DuckDB setting (per session)
```

```bash
export DATAZOO_DISABLE_TELEMETRY=1       # environment (1|true|yes)
```

Telemetry is also disabled automatically when a CI environment is detected. The
test suite always runs with `DATAZOO_DISABLE_TELEMETRY=1`.

## The guarantee: bounded, enumerated, non-PII

Every property we send is **either** a constant drawn from a small,
code-controlled enumeration **or** a pure number (durations, counts). The library
additionally clamps every outgoing string to 512 bytes as a backstop.

We **never** send: file paths, cache/model directories, table names, column
names, target/feature values, model weights or any row/result data, SQL text,
`WHERE`/`FILTER` clauses, device names, or error messages. Only the bounded
identifiers documented below leave the process.

The instrumentation is centralised: extension load is captured once in
`src/anofox_decide_extension.cpp`, and each user-facing function records exactly
one aggregated call at bind/registration time (never on a per-row path).

## What is collected

### Envelope (attached to every event)

`product` (`anofox_decide`), `product_version`, `product_edition` (`oss`),
`telemetry_schema` (`2`), `duckdb_version`, `os`, `arch`, `platform`, `is_ci`,
`is_container`, a per-process `$session_id`, and — once associated — the
`deployment` group. `distinct_id` is the SHA-256 of a machine id: a **stable,
pseudonymous** identifier, not tied to any personal data.

### Events

| Event | When | Properties (beyond the envelope) |
|---|---|---|
| `extension_loaded` | the `anofox_decide` extension loads | — |
| `function_executed` | a DuckDB function runs — **aggregated** per function per session (not per row) | `function_name`, `call_count`, `duration_ms_p50` |

`function_name` is always one of a fixed, code-controlled set of the extension's
own function names:

- `decide_probability` / `decide_choice` / `decide_choice_distribution` / `decide_score` /
  `decide_decision` / `decide_many` (scoring)
- `decide_table` / `decide_answers` (relational batch scoring)
- `decide_register_model` / `decide_unregister_model` / `decide_models` (model registry)
- `decide_doctor` (setup check)
- `decide_brier_score` / `decide_ece` / `decide_accuracy` / `decide_fit_calibration`
  (calibration metrics)

The short `decide_*` aliases and the primary `anofox_decide_*` names are recorded under the same
`decide_*` identifier.

No arguments, options, model ids, provider names, endpoints, API keys, state
text, questions or answers are ever attached. Remote providers receive the
`state` text you pass to them (that is the function's purpose); telemetry
never does.

## Function-call aggregation

DuckDB function calls are recorded via `RecordFunctionCall(function_name)`, which
aggregates in-process into a single `function_executed` event per function per
session (carrying `call_count` and `duration_ms_p50`), flushed at session end.
Because recording happens at bind time and never on a per-row path, a
million-row scoring query produces O(1) telemetry rows, not a firehose.

## Enterprise / account analytics

OSS `anofox_decide` associates only the `deployment` group. It has no license key,
so no `account` group is associated.
