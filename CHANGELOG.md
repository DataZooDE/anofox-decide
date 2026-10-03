# Changelog

All notable changes to `anofox_decide` are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/); this project uses calendar versioning
(`YYYY.MM.DD`, tagged `vYYYY.MM.DD`, like the other anofox extensions).

## [Unreleased]

## [2026.10.03] - 2026-10-03

First public release.

### Added
- **Natural-language decisions as SQL functions.** Three kinds of question, all written at query time:
  `binary` (`decide_probability`, and `decide_decision` with an explicit threshold), `choice` (one of an
  option list you define, `decide_choice`) and `score` (an ordered rubric of 2 to 10 levels, lowest first;
  `decide_score` returns the expected 0-based level). `decide_many` answers several questions about one
  text in a single request as JSON, and `decide_table` returns the same answers as rows
  (`question_id`, `kind`, `probability`, `choice`, `confidence`, `model`, `score`, `distribution`).
  Every function is available as `anofox_decide_<name>` and as the short alias `decide_<name>`, and
  documents itself in `duckdb_functions()` (description, parameter names, a runnable example, categories).
- **Hosted, local-server and in-process models behind one interface.** Providers: `typesafe` (TypeSafe
  Jev), `liquid` (Liquid AI D1), `strands` (a local, keyless
  [strands-decider](https://github.com/strands-labs/strands-decider) server), `systemone` (any compatible
  server, e.g. Kev), `local` (Julia-1 and Laya on ONNX Runtime, CPU, offline after setup) and the built-in
  `stub` test model. Per-model options (`endpoint`, `path`, `model`, `key_env`, `criteria`) go in a `MAP`
  on `decide_register_model`; `decide_unregister_model` removes a model so it can be registered again.
- **Keys that stay where they belong.** Environment variables work out of the box and a DuckDB secret
  (`CREATE SECRET (TYPE anofox_decide, API_KEY '...', SCOPE '<host>')`, write-only, redacted in
  `duckdb_secrets()`) always overrides them. A key is only ever sent to the host it was configured for;
  cleartext `http://` is limited to loopback. Hosted models need the explicit opt-in
  `SET anofox_decide_allow_remote = true`.
- **Model quality metrics.** `decide_accuracy`, `decide_brier_score` and `decide_ece` aggregate over
  `(probability, label)` pairs, so comparing models is one query. A 200-ticket evaluation of every model
  on public data (Bitext customer support) is in `docs/EVALUATION.md`; `tools/eval` reproduces it.
- **Diagnostics.** `decide_doctor()` checks the whole setup (default model, remote opt-in, files, keys and
  where they came from, endpoints, telemetry) and gives the fix for each problem; `decide_models()` lists
  every registered model with `ready`, a `hint` for what to do when it is not, the profile, endpoint and
  wire model.
- **Concurrent scoring of many rows.** The scalar functions evaluate a chunk of rows together: identical
  `(model, text, questions)` requests are sent once, up to `anofox_decide_max_concurrency` remote requests
  run at the same time (default automatic: 8 for hosted providers, 1 for the local strands server),
  connections are reused, a `429` halves the window and `Retry-After` is honoured. On the real Liquid D1
  service, 8 tickets with three questions each took 176.7 s one request at a time and 23.1 s concurrently,
  with the same answers. `LATERAL decide_table` stays one row at a time: DuckDB hands an in-out function a
  single row per call there (documented in the README).
- **Settings:** `anofox_decide_model`, `anofox_decide_allow_remote`, `anofox_decide_max_concurrency`,
  `anofox_decide_timeout_ms`, `anofox_decide_max_retries`, `anofox_decide_max_questions`,
  `anofox_decide_max_length`, `anofox_decide_head_length`, and the legacy `anofox_decide_api_key` and
  `anofox_decide_endpoint` for the `typesafe` provider. Every setting is validated and its error shows the
  range, the default and an example `SET`.
- **Examples** (`examples/`): the whole SQL surface, support-ticket triage with accuracy and calibration,
  scoring many rows, comparing models, and a local model. The offline ones run in the test suite on every
  build and the others are parsed, so they cannot drift from the extension.
- **Anonymous usage telemetry and the feedback banner**, shared with the other anofox extensions (opt out
  with `DATAZOO_DISABLE_TELEMETRY=1` or `SET anofox_telemetry_enabled = false`; see `TELEMETRY.md`).
- Releases are published to the anofox extension repository (`get.anofox.com`) on every `v*` tag and every push to `main`, in addition to the DuckDB community repository; the README documents installing from either.

### Changed
- **There is no implicit model.** A call that names no model and has no `anofox_decide_model` fails with
  instructions for choosing and registering one, instead of quietly returning the `stub`'s constant. The
  `stub` is for trying the SQL and says so everywhere it appears.
- **Errors say what to do.** Every error has the shape `<function>: <what went wrong>. Fix: <what to run>`
  and echoes the value that caused it. Remote failures quote the service's own message and give the next
  step per status (401 names where the key came from, 404 points at the model option, 422 explains the
  `criteria` option, 429 and 5xx say how many attempts were made), a local server that is not running says
  to start it and is not retried, and the opt-in check comes before the key check. Plain user mistakes no
  longer carry the "Unexpected? Please report it" footer, which is kept for genuinely unexpected errors.
- **The documentation matches the extension.** `decide_many` is documented as returning
  `{"model", "results": [...]}`; `decide_table` documents the `score` kind; function examples run as
  written. A test checks that every function and setting the README names exists.

### Known limitations
- Local models are not downloaded for you: export a checkpoint to ONNX once with `tools/export_julia`.
  GPU execution for local models is not supported, and neither is the Von model.
- `score` is verified for parity with each model's upstream implementation and against the live services,
  but has not been scored for accuracy on labelled data.
- Text longer than a local model's limit is cut off without a warning.
- Platforms: Linux (amd64, arm64), macOS (arm64) and Windows (amd64) against DuckDB 1.5.6. Not built:
  macOS on Intel (ONNX Runtime ships no archive after v1.23.2), WebAssembly, musl and MinGW.

[Unreleased]: https://github.com/DataZooDE/anofox-decide/compare/v2026.10.03...HEAD
[2026.10.03]: https://github.com/DataZooDE/anofox-decide/releases/tag/v2026.10.03
