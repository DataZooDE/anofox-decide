<h1 align="center">Anofox Decide</h1>
<p align="center"><strong>Natural-language decisions on your data, inside DuckDB</strong></p>

<p align="center">
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-blue.svg" alt="License: MIT"></a>
  <a href="https://duckdb.org"><img src="https://img.shields.io/badge/DuckDB-1.5.6%20%7C%202.0-green.svg" alt="DuckDB 1.5.6 | 2.0"></a>
  <a href="https://github.com/DataZooDE/anofox-decide/actions/workflows/MainDistributionPipeline.yml"><img src="https://github.com/DataZooDE/anofox-decide/actions/workflows/MainDistributionPipeline.yml/badge.svg?branch=main" alt="Build"></a>
</p>

Ask a yes/no question, pick one of the options *you* define, or rate a text on an ordered scale — as a
SQL function, over any text or structured state. Purpose-built **decision models** answer with calibrated
probabilities, not just labels, so you choose the threshold and can measure how good the answers are. No
training, no prompt engineering, no glue code: the questions and the answer sets are written in the query.

```sql
SELECT id,
       decide_probability(text, 'A refund is requested.')                            AS p_refund,
       decide_choice(text, 'Which team owns this?', ['billing', 'defect', 'other'])  AS team,
       decide_score(text, 'How frustrated is the writer?', ['calm', 'frustrated', 'angry']) AS frustration
FROM tickets;
```

## ✨ Key Features

### 🎯 Three kinds of question
- **binary**: the probability that a statement holds (`decide_probability`, `decide_decision`)
- **choice**: the best of a list of options, with the probability of each (`decide_choice`, `decide_choice_distribution`)
- **score**: the expected level on an ordered rubric (`decide_score`)
- **Many at once**: several questions about one text in one request (`decide_table`, `decide_answers`, `decide_many`)

### 🧠 Hosted or on your machine
- **Hosted**: TypeSafe **Jev**, Liquid AI **D1**, Cloudflare **Clef**
- **Local servers**: **strands-decider**, **Kev** (or any System One compatible server)
- **In-process, fully offline**: **Julia-1** and **Laya** on ONNX Runtime, optional GPU (CUDA, ROCm, Apple MLX)
- **One SQL surface**: swap the model without touching the query, compare models on your own data in one statement

### 📊 Measure and calibrate
- **Metrics**: accuracy, Brier score, calibration error, log loss, AUC as aggregates
- **Curves**: ROC, calibration and metrics tables ready for [anofox-visualization](https://github.com/DataZooDE/anofox-visualization)
- **Calibration**: per-model Platt scaling fitted in SQL

### 🔒 Safe by default
- **No weights shipped, nothing called** until you register a model
- **Explicit opt-in** before any text leaves your machine; keys as DuckDB secrets, never sent to another host
- **Errors that say what to run**, and `decide_doctor()` for the whole setup

## 📋 Table of Contents

- [Installation](#installation)
- [Quick Start](#-quick-start)
- [Models](#-models)
- [Documentation](#-documentation)
- [Status](#status)
- [Development](#-development)
- [License](#-license)
- [Telemetry](#telemetry)
- [Feedback](#feedback)

## Installation

```sql
INSTALL anofox_decide FROM community;
LOAD anofox_decide;
```

> **Latest builds from DataZoo.** The community build is signed but can trail a release by days. The
> DataZoo repository has a release the moment it is out; its builds are **unsigned**, so start DuckDB with
> `duckdb -unsigned`, and **install `httpfs` first** (otherwise DuckDB autoloads `httpfs` from
> `get.anofox.com`, which does not serve it):
>
> ```sql
> INSTALL httpfs; LOAD httpfs;
> SET custom_extension_repository = 'https://get.anofox.com';
> INSTALL anofox_decide;
> LOAD anofox_decide;
> ```

Check which build you got:

```sql
SELECT extension_version FROM duckdb_extensions() WHERE extension_name = 'anofox_decide';
```

Details and the errors these two gotchas produce: [Getting Started](docs/guides/01-getting-started.md).

## 🚀 Quick Start

```sql
-- 1. What is ready, and the fix for what is not
SELECT * FROM decide_doctor();

-- 2. Try the SQL on the built-in stub (it returns constants, so it can never pass for a real answer)
SELECT decide_probability('I want my money back.', 'A refund is requested.', model := 'stub');

-- 3. A real model: Liquid D1 (export LIQUID_API_KEY=... before starting DuckDB)
SET anofox_decide_allow_remote = true;                 -- explicit opt-in: the text goes to the provider
SELECT decide_register_model('d1:free', 'liquid');     -- registrations last for this process only
SET anofox_decide_model = 'd1:free';                   -- the default for calls that name no model

-- 4. Several questions over a small table, one row per question
CREATE TABLE tickets AS SELECT * FROM (VALUES
  (1, 'I was charged twice for my March invoice, please refund the extra 49 EUR.'),
  (2, 'The app crashes every time I open the export dialog on version 3.2.'),
  (3, 'Cancel my subscription and reimburse the unused months.')
) t(id, body);

SELECT t.id, d.question_id, round(d.probability, 3) AS probability, d.choice
FROM tickets t, LATERAL decide_table(t.body,
  '[{"id": "refund", "kind": "binary", "instruction": "A refund is requested."},
    {"id": "team",   "kind": "choice", "instruction": "Which team owns this?", "options": ["billing", "defect", "other"]}]') d
ORDER BY t.id, d.question_id;
```

No key? `CALL decide_download('laya-multilingual');` fetches a model that runs inside DuckDB, offline
after the 0.7 GB download, then use `model := 'laya-multilingual'`. For many rows use `decide_answers`
instead of the lateral join: it runs concurrently ([Scoring many rows](docs/api/01-functions.md#scoring-many-rows)).
The full walk-through with real output and metrics: [Getting Started](docs/guides/01-getting-started.md);
runnable files in [`examples/`](examples/README.md).

## 🧠 Models

| Model | Provider | Runs | Needs |
|---|---|---|---|
| TypeSafe Jev | `typesafe` | hosted API | `TYPESAFE_API_KEY` |
| Liquid AI D1 | `liquid` | hosted API | `LIQUID_API_KEY` |
| Cloudflare Clef, Clef-flash | `cloudflare` | hosted API (Workers AI) | `CLOUDFLARE_API_TOKEN`, `CLOUDFLARE_ACCOUNT_ID` |
| strands-decider 2B | `strands` | local server | no key |
| Kev, or any System One server | `systemone` | your server | optional key |
| Julia-1, Laya multilingual, Laya typed-decisions | `local` | in-process, offline | `decide_download` once |
| built-in test model | `stub` | in-process | nothing |

Which one to pick, and how they did on a 200-ticket evaluation: [Choosing a model](docs/guides/02-choosing-a-model.md).

## 📚 Documentation

### Guides

| Guide | Description |
|---|---|
| [Getting Started](docs/guides/01-getting-started.md) | Install, `decide_doctor()`, a first real decision, thresholds and metrics |
| [Choosing a model](docs/guides/02-choosing-a-model.md) | What you have and what to do; hosted vs local vs in-process; evaluation results; comparing models on your data |
| [Evaluate and calibrate](docs/guides/03-evaluate-and-calibrate.md) | Metrics, ROC and calibration curves, plotting, Platt scaling, abstaining |

### API reference

| Topic | Description |
|---|---|
| [Functions](docs/api/01-functions.md) | Every function, question kinds, `decide_table` columns, NULLs, scoring many rows |
| [Providers](docs/api/02-providers.md) | Registering models, per-model options, each hosted provider, API keys and secrets, local servers |
| [Local models](docs/api/03-local-models.md) | Julia-1 and Laya in-process, your own exported ONNX model, long text, batching |
| [GPU acceleration](docs/api/04-gpu.md) | CUDA, ROCm and MLX plugins, devices, verification, measurements |
| [Settings](docs/api/05-settings.md) | Every setting with its default |
| [Troubleshooting](docs/api/06-troubleshooting.md) | Every error message, what it means, what to run |

### Background

[docs/EVALUATION.md](docs/EVALUATION.md) (the 200-ticket evaluation),
[docs/SPIKE_RESULTS.md](docs/SPIKE_RESULTS.md) (provider contracts),
[docs/CALIBRATION_PERF.md](docs/CALIBRATION_PERF.md) (calibration and performance),
[docs/REVIEW_FOLLOWUP.md](docs/REVIEW_FOLLOWUP.md) (design decisions and model investigations),
[docs/GPU_BACKENDS.md](docs/GPU_BACKENDS.md) and [docs/GPU_ROCM.md](docs/GPU_ROCM.md) (GPU backends),
[CHANGELOG.md](CHANGELOG.md).

## Status

The extension is calendar-versioned (`2026.10.26` is the twenty-sixth of October 2026) and is built and
tested on Linux (amd64, arm64), macOS (arm64) and Windows (amd64) against DuckDB 1.5.6, and on DuckDB 2.0.
What works today and what does not yet: [Choosing a model](docs/guides/02-choosing-a-model.md#status-and-scope).

## 📦 Development

```bash
git clone --recurse-submodules https://github.com/DataZooDE/anofox-decide.git
cd anofox-decide
GEN=ninja make release      # needs vcpkg: see docs/BUILDING.md
make test_release           # offline suite: SQL tests + Catch2 (no network, no weights)
```

Prerequisites, faster local builds, live tests and tests that need weights: [docs/BUILDING.md](docs/BUILDING.md).

## 📄 License

- **This extension's code:** MIT, see [LICENSE](LICENSE).
- **Models:** the extension ships **no model weights**, and never redistributes any: `decide_download` fetches the
  Julia-1 and Laya weights from their authors' Hugging Face repositories on your request, at a pinned revision
  (the extension embeds only weight-free graphs, which carry no weights). Check a model's license before you
  build on it, especially one you export or serve yourself:

| Model | Provider | License / terms | Where it runs |
|---|---|---|---|
| Jev | TypeSafe | the provider's API terms | hosted; receives the text you score |
| D1 | Liquid AI | the provider's API terms | hosted; receives the text you score |
| Clef, Clef-flash | Cloudflare | Apache 2.0 weights; the hosted service under Cloudflare's terms | hosted on Workers AI; receives the text you score |
| strands-decider | Strands Labs | Apache 2.0 | your machine, as a local server |
| Kev | Jared Palmer | Apache 2.0 | your machine, as a local server |
| Julia-1 | SupersonicLabs ([SupersonicLabs/Julia-1](https://huggingface.co/SupersonicLabs/Julia-1)) | Apache 2.0 | your machine, in-process; weights downloaded from Hugging Face on request |
| Laya multilingual, Laya typed-decisions | ConvAI Innovations ([convaiinnovations/laya](https://huggingface.co/convaiinnovations/laya)) | Apache 2.0 | your machine, in-process; weights downloaded from Hugging Face on request |

Licenses are as published by each project at the time of writing.

## Telemetry

Sends **anonymous** usage telemetry (extension load and per-function call counts: no state, questions,
answers, model ids, endpoints, keys or SQL) to PostHog EU, through the shared
[`posthog-telemetry`](https://github.com/DataZooDE/posthog-telemetry) library, matching the other anofox
extensions. CI environments are auto-detected and telemetry is disabled there. Opt out any time with
`export DATAZOO_DISABLE_TELEMETRY=1` or `SET anofox_telemetry_enabled = false;`. The full list of what is
collected is in [TELEMETRY.md](TELEMETRY.md).

## Feedback

If `anofox_decide` misbehaves — a model that will not answer, a probability that looks wrong, an error
that does not tell you what to do — please
[open an issue](https://github.com/DataZooDE/anofox-decide/issues). Providers, keys and networks differ in
ways we cannot reproduce here, so a report with the output of `SELECT * FROM decide_doctor();` and the
failing call (without your key) is the fastest path to a fix. Unexpected errors end with that link.

If it saved you time, a star on the repo helps other people find it.

The first time you load the extension in an interactive terminal each day, a small banner says the same.
It never prints when output is piped, in notebooks, or in CI. Silence it with `SET datazoo_banner = false;`
or `DATAZOO_NO_BANNER=1`.
