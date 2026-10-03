# anofox-decide

**Natural-language decisions on your data, inside DuckDB.** Ask a yes/no question, pick one of the
options *you* define, or rate a text on an ordered scale — as a SQL function, over any text or
structured state. Purpose-built **decision models** answer with calibrated probabilities, not just
labels, so you choose the threshold and can measure how good the answers are. No training, no prompt
engineering, no glue code: the questions and the answer sets are written in the query.

```sql
SELECT id,
       decide_probability(text, 'A refund is requested.')                            AS p_refund,
       decide_choice(text, 'Which team owns this?', ['billing', 'defect', 'other'])  AS team,
       decide_score(text, 'How frustrated is the writer?', ['calm', 'frustrated', 'angry']) AS frustration
FROM tickets;
```

**Pick the model that fits: hosted or on your machine.** Hosted: TypeSafe **Jev**, Liquid AI **D1** and Cloudflare **Clef**.
Local servers: **strands-decider** and **Kev** (or any System One compatible server). In-process and
fully offline: **Julia-1** and **Laya** on ONNX Runtime. All of them answer the same questions through
the same functions, so you can swap the model without touching the query — and compare them on your
own data in one statement.

---

## Quickstart

### 1. Install & load

```sql
INSTALL anofox_decide FROM community;
LOAD anofox_decide;
```

The community build is signed, so no extra flags are needed, but it is published by a separate submission
and can trail a release by days. To take a release the moment it is out, use the anofox repository instead.
Those builds are **unsigned**, so DuckDB must be started with `-unsigned`:

```bash
duckdb -unsigned
```
```sql
INSTALL httpfs; LOAD httpfs;   -- FIRST: see below
SET custom_extension_repository = 'https://get.anofox.com';
INSTALL anofox_decide;
LOAD anofox_decide;
```

Two things that are easy to hit and give unhelpful errors:

* Without `-unsigned`, installing from `get.anofox.com` fails with *"Attempting to install an extension file
  that doesn't have a valid signature"*. The community repository needs no flag.
* `custom_extension_repository` applies to **every** install, including ones DuckDB triggers for you. If it is
  set before `httpfs` is present, DuckDB autoloads `httpfs` from `get.anofox.com`, which does not serve it, and
  the failure names `httpfs` rather than the setting. Install `httpfs` first.

Check which build you got:

```sql
SELECT extension_version FROM duckdb_extensions() WHERE extension_name = 'anofox_decide';
```

The extension ships no model weights and calls nothing until you register a model and, for hosted
models, opt in.

### 2. Pick a model

`SELECT * FROM decide_doctor();` tells you what is ready and how to fix what is not. Then choose one:

| You have | Do this |
|---|---|
| Nothing yet | Use the built-in `stub` to try the SQL: `model := 'stub'`. It returns constants (0.5, the first option, the middle level) so it can never pass for a real answer. |
| A Liquid AI key ([D1](https://docs.liquid.ai/lfm/models/decision-models)) | `export LIQUID_API_KEY=...`, then register `d1:free` (below). |
| A Cloudflare account ([Clef](https://blog.cloudflare.com/clef-decision-models/) on Workers AI) | `export CLOUDFLARE_API_TOKEN=...` and `CLOUDFLARE_ACCOUNT_ID=...`, then `decide_register_model('clef', 'cloudflare')` ([Cloudflare Clef](#cloudflare-clef)). |
| A TypeSafe key (Jev) | `export TYPESAFE_API_KEY=...`, then `decide_register_model('jev-latest', 'typesafe')`. |
| A machine that can run a 2B model | Run [strands-decider](https://github.com/strands-labs/strands-decider) locally, no key: [Models and providers](#models-and-providers). |
| An exported ONNX model | Run it inside DuckDB, offline: [Local models](#local-models). |

On a fresh install the doctor says what is missing and the fix for each item:

```
┌───────────────────┬────────┬───────────────────────────────────────────────────────────────────────────┐
│       item        │ status │                                  detail                                   │
├───────────────────┼────────┼───────────────────────────────────────────────────────────────────────────┤
│ default model     │ warn   │ no default model is set, so every call must name one with model := '<id>' │
│ registered models │ warn   │ only the built-in test model 'stub' is registered (it returns constants)  │
│ remote calls      │ ok     │ no remote models registered                                               │
│ telemetry         │ ok     │ anonymous usage telemetry is off                                          │
└───────────────────┴────────┴───────────────────────────────────────────────────────────────────────────┘
```

(plus a `fix` column with the statement to run for each warning; telemetry reads "on" unless you opted out).
`SELECT model, ready, hint FROM decide_models();` lists every registered model, whether it can be called
now, and why not.

A hosted model sends the text you score to the provider, so it sits behind an explicit opt-in that is off
by default. Register it once per database instance (registrations and settings are not stored: repeat them
in each new process), then name it per call (`model := 'd1:free'`) or make it the default:

```sql
SET anofox_decide_allow_remote = true;                 -- explicit opt-in: the text goes to the provider
SELECT decide_register_model('d1:free', 'liquid');     -- the id is the model name on the wire
SET anofox_decide_model = 'd1:free';                   -- the default for calls that name no model
```

A call that names no model, with no default set, fails and says how to choose one; it never quietly
returns a constant.

### 3. Decide

Eight support tickets with known answers, scored for three questions at once. This is real output from
Liquid D1, so it needs the model from step 2 (`SET anofox_decide_model = 'd1:free'`); on the `stub` you get
the same columns filled with constants.

```sql
CREATE TABLE tickets AS SELECT * FROM (VALUES
  (1, 'I was charged twice for my March invoice, please refund the extra 49 EUR.', true,  'billing'),
  (2, 'The app crashes every time I open the export dialog on version 3.2.',       false, 'defect'),
  (3, 'Thanks for the quick help yesterday, everything works now!',                false, 'other'),
  (4, 'I want my money back, this product does not do what was advertised.',       true,  'billing'),
  (5, 'Login button does nothing on Safari, error 500 in the console.',            false, 'defect'),
  (6, 'Our invoice shows the wrong VAT rate; please issue a corrected one.',       false, 'billing'),
  (7, 'Do you have an office in Munich? We would like to visit.',                  false, 'other'),
  (8, 'Cancel my subscription and reimburse the unused months.',                   true,  'billing')
) t(id, text, refund, team);                                  -- refund and team are the known answers

CREATE TABLE scored AS
SELECT id, text, refund,
       decide_probability(text, 'A refund is requested.')                                   AS p_refund,
       decide_choice(text, 'Which team owns this?', ['billing', 'defect', 'other'])         AS team,
       decide_score(text, 'How frustrated is the writer?', ['calm', 'frustrated', 'angry']) AS frustration
FROM tickets;

SELECT id, round(p_refund, 3) AS p_refund, team, round(frustration, 2) AS frustration, left(text, 44) AS text
FROM scored ORDER BY id;
```
┌───────┬──────────┬─────────┬─────────────┬──────────────────────────────────────────────┐
│  id   │ p_refund │  team   │ frustration │                     text                     │
├───────┼──────────┼─────────┼─────────────┼──────────────────────────────────────────────┤
│     1 │    0.998 │ billing │         0.2 │ I was charged twice for my March invoice, pl │
│     2 │    0.321 │ defect  │        0.94 │ The app crashes every time I open the export │
│     3 │    0.107 │ other   │         0.0 │ Thanks for the quick help yesterday, everyth │
│     4 │    0.998 │ billing │         1.2 │ I want my money back, this product does not  │
│     5 │    0.033 │ defect  │        0.64 │ Login button does nothing on Safari, error 5 │
│     6 │    0.002 │ billing │        0.06 │ Our invoice shows the wrong VAT rate; please │
│     7 │    0.064 │ other   │        0.01 │ Do you have an office in Munich? We would li │
│     8 │    0.995 │ billing │        0.39 │ Cancel my subscription and reimburse the unu │
└───────┴──────────┴─────────┴─────────────┴──────────────────────────────────────────────┘
```

Probabilities become actions with a threshold *you* pick, and the same table says how good they are:

```sql
SELECT id FROM scored WHERE p_refund >= 0.7;                       -- escalate to the refunds team

SELECT decide_accuracy(p_refund, refund)    AS accuracy,           -- 1.0     (with D1)
       decide_brier_score(p_refund, refund) AS brier,              -- 0.015   (lower is better)
       decide_ece(p_refund, refund)         AS calibration_error   -- 0.067
FROM scored;
```

The complete, runnable version is [`examples/02_support_triage.sql`](examples/02_support_triage.sql); it
starts on the `stub` so it runs offline, and the lines marked `REAL MODEL` switch it to a real one. The
example files read their data from the repository (`examples/data/`), so run them from a clone of it; the
other examples are listed in [`examples/`](examples/README.md).

---

## What you can ask

Every question has a **kind**. You write the question and, for the kinds that need them, the answers —
at query time, per row if you like.

| Kind | You provide | You get | Function |
|---|---|---|---|
| `binary` | a statement | the probability (0 to 1) that it holds for the text | `decide_probability`, and `decide_decision` with a threshold |
| `choice` | a question and a list of options | the best option, plus the probability of each | `decide_choice` |
| `score` | a question and an ordered rubric of 2 to 10 levels, lowest first | the expected level (0-based, e.g. `1.4` between `frustrated` and `angry`), plus the probability of each level | `decide_score` |

Several questions about the same text go in one call, as a JSON array of
`{id, kind, instruction[, options | levels]}`:

```sql
SELECT question_id, kind, probability, choice, score, distribution
FROM decide_table('The bill is wrong and I want my money back.',
  '[{"id": "refund", "kind": "binary", "instruction": "A refund is requested."},
    {"id": "team",   "kind": "choice", "instruction": "Which team owns this?", "options": ["billing", "defect", "other"]},
    {"id": "mood",   "kind": "score",  "instruction": "How frustrated is the writer?", "levels": ["calm", "frustrated", "angry"]}]',
  model := 'd1:free');
```

### Output columns of `decide_table`

| Column | Type | Meaning |
|---|---|---|
| `question_id` | `VARCHAR` | the `id` you gave the question |
| `kind` | `VARCHAR` | `binary`, `choice` or `score` |
| `probability` | `DOUBLE` | `binary`: P(yes). `choice`: probability of the chosen option. `score`: probability of the most likely level |
| `choice` | `VARCHAR` | the chosen option (`choice` only) |
| `confidence` | `DOUBLE` | the model's own confidence when the service reports one, otherwise `NULL` |
| `model` | `VARCHAR` | the model that answered |
| `score` | `DOUBLE` | the expected 0-based level (`score` only) |
| `distribution` | `MAP(VARCHAR, DOUBLE)` | probability per option or level (`choice`, `score`) |

`decide_many` returns the same answers as one JSON document per text:
`{"model": "...", "results": [{"id", "kind", "probability", ...}]}`.

**NULLs.** A NULL text, question, option list or threshold gives a NULL result: no model is resolved and
nothing is sent. A NULL `model` argument falls back to the session default.

---

## Functions

Every function is available as `anofox_decide_<name>` and as the short alias `decide_<name>`.
`duckdb_functions()` carries a description, parameter names and a runnable example for each.

| Function | Returns | Purpose |
|---|---|---|
| `decide_probability(state, question[, model])` | `DOUBLE` | P(`question` holds for `state`) |
| `decide_decision(state, question, threshold[, model])` | `BOOLEAN` | `true` when that probability reaches an explicit `threshold` (0 to 1); never guessed |
| `decide_choice(state, question, options[, model])` | `VARCHAR` | the best of a list of options |
| `decide_choice_distribution(state, question, options[, model])` | `MAP(VARCHAR, DOUBLE)` | the probability of every option (sums to 1), so you can threshold or abstain in SQL ([Calibration](#calibration)) |
| `decide_score(state, question, levels[, model])` | `DOUBLE` | the expected 0-based level on an ordered rubric |
| `decide_many(state, questions[, model])` | `VARCHAR` (JSON) | several questions about one text in one request |
| `decide_table(state, questions[, model])` | table | the same, one row per question |
| `decide_accuracy(p, outcome[, threshold])` | `DOUBLE` | share of rows where thresholding `p` reproduces the label (aggregate) |
| `decide_brier_score(p, outcome)` | `DOUBLE` | mean squared error of the probabilities; 0 is perfect, 0.25 a constant 0.5 (aggregate) |
| `decide_ece(p, outcome)` | `DOUBLE` | expected calibration error over ten equal-width bins (aggregate) |
| `decide_fit_calibration(p, outcome)` | `VARCHAR` | fits Platt scaling to a model's raw probabilities and returns the `'platt:a,b'` spec for `decide_register_model` ([Calibration](#calibration)) (aggregate) |
| `decide_register_model(id[, provider[, ...]])` | `BOOLEAN` | register a model for this database instance |
| `decide_unregister_model(id)` | `BOOLEAN` | remove a registered model so it can be registered again |
| `decide_token_count(text[, model])` | `BIGINT` | tokens `text` has for a local model, to find rows that do not fit its window ([Local models](#local-models)) |
| `decide_models()` | table | the registered models, whether each can be called now, and what to do if not |
| `decide_doctor()` | table | `item`, `status` (`ok` / `warn` / `fail`), `detail`, `fix` for the whole setup |

---

## Models and providers

Hosted and local models are profiles over one wire format (System One: `binary`, `choice` and `score`
questions, probabilities back), so a model is just a registered id, and several can be used side by side
in one session with no `SET` in between.

| Provider | Service | Runs | Default endpoint | Key |
|---|---|---|---|---|
| `typesafe` | TypeSafe Jev | hosted API | `https://api.typesafe.ai` | `TYPESAFE_API_KEY` |
| `liquid` | Liquid AI D1 | hosted API | `https://api.liquid.ai` | `LIQUID_API_KEY` |
| `cloudflare` | Cloudflare Clef, Clef-flash | hosted API (Workers AI) | `https://api.cloudflare.com` | `CLOUDFLARE_API_TOKEN` |
| `strands` | [strands-decider](https://github.com/strands-labs/strands-decider) 2B | local server | `http://127.0.0.1:8000` | none |
| `systemone` | any compatible server, e.g. [Kev](https://github.com/jaredpalmer/kev) | your server | you set it | optional (`key_env`) |
| `local` | Julia-1, Laya | in-process, offline | — | none |
| `stub` | built-in test model | in-process | — | none |

```sql
SET anofox_decide_allow_remote = true;                       -- hosted and server models need the opt-in
SELECT decide_register_model('jev-latest', 'typesafe');
SELECT decide_register_model('d1:free', 'liquid');
SELECT decide_register_model('clef', 'cloudflare');             -- Cloudflare Clef; 'clef-flash' is the smaller one
SELECT decide_probability('...', '...', model := 'jev-latest');
SELECT decide_probability('...', '...', model := 'd1:free');
```

Per-model options go in a `MAP` (keys `endpoint`, `path`, `model`, `key_env`, `criteria`, `calibration`, `account_id`): a self-hosted
server, a different model name on the wire, an explicit key variable for a custom endpoint, or
`criteria: 'name'` for servers whose schema wants a string description per choice option instead of `null`, or a fitted `calibration` ([Calibration](#calibration)):

```sql
SELECT decide_register_model('kev-latest', 'systemone', MAP {'endpoint': 'http://127.0.0.1:8009'});
SELECT decide_register_model('d1', 'liquid', MAP {'model': 'd1:free'});
SELECT decide_register_model('mine', 'systemone', MAP {'endpoint': 'https://llm.internal', 'key_env': 'MY_KEY_VAR'});
```

### Cloudflare Clef

[Clef](https://blog.cloudflare.com/clef-decision-models/) is Cloudflare's open-weight family of decision models
(Apache 2.0 weights on Hugging Face), served on Workers AI: `clef` (27B) and the smaller `clef-flash` (9B). It
speaks the same questions as the other hosted models, so the call is the same:

```bash
export CLOUDFLARE_API_TOKEN=...      # before starting DuckDB: a token with "Workers AI - Read" and "Workers AI - Edit"
export CLOUDFLARE_ACCOUNT_ID=...     # the 32-character account id from your Cloudflare dashboard
```
```sql
SET anofox_decide_allow_remote = true;
SELECT decide_register_model('clef', 'cloudflare');          -- or 'clef-flash'; the id is the model name in the URL
SELECT decide_choice('Checkout has been failing for every customer.', 'Which team should handle this?',
                     ['billing', 'technical', 'sales'], model := 'clef');
-- the account id can also be given per model: decide_register_model('clef', 'cloudflare', MAP {'account_id': '<id>'})
```

Things worth knowing:

- **Cost and speed:** $0.24 per million input tokens for `clef` and $0.09 for `clef-flash`; one call took about
  0.3 to 1 s in our runs. A request takes at most **64 questions** (a call with more is refused before anything is
  sent) and a 64k-token text.
- **Quality:** on the 200-ticket evaluation `clef` is the third-best router (88%) and a solid refund detector (85%);
  `clef-flash` is much weaker on refund (70%). `clef` errs towards "no" at the 0.5 cut-off (recall 62%, specificity
  95%); [Calibration](#calibration) shows how to fit the cut-off to your own data. See
  [Evaluate and compare models](#evaluate-and-compare-models).
- **Errors:** Cloudflare answers a wrong token and a valid token of a different account with the same
  "Authentication error", so the message names both causes. A wrong model name is an HTTP 400 "No route for
  that URI"; the message says the models are `clef` and `clef-flash`.
- **Not supported:** Clef also accepts images (up to four per request); the functions here take text only. The
  27B and 9B models are too large to run in-process like Julia-1 or Laya; to run them yourself, serve them behind a
  System One compatible endpoint (see the model card for serving options) and register that endpoint with
  provider `systemone`.

### API keys

Environment variables work out of the box (`export LIQUID_API_KEY=...`), and a DuckDB secret always
overrides them. Precedence, highest first:

1. A stored secret (write-only, redacted in `duckdb_secrets()`), matched by the endpoint host:
   ```sql
   CREATE SECRET (TYPE anofox_decide, API_KEY 'sk-...', SCOPE 'api.liquid.ai');
   CREATE SECRET (TYPE anofox_decide, API_KEY getenv('LIQUID_API_KEY'), SCOPE 'api.liquid.ai');  -- copy from the env
   ```
2. `anofox_decide_api_key` (legacy, `typesafe` only; visible through `current_setting`, so prefer a secret).
3. The model's explicit `key_env` variable.
4. The provider's own env var (table above), only on that provider's default host.

A key is never sent to a host it was not configured for: a redirected endpoint gets nothing from the
environment, and a secret scoped to one host is not used for another. Cleartext `http://` works for
loopback hosts only.

### Local servers

[strands-decider](https://github.com/strands-labs/strands-decider) (Apache 2.0; a 1.9B Qwen3.5 torso with
a pointer head) needs no key. A choice question may have at most 24 options on this model. It listens on
loopback without authentication, so keep it local:

```bash
python3.12 -m venv .venv && . .venv/bin/activate
pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cpu   # or a CUDA/MPS build
pip install strands-decider
strands-decider serve StrandsAgents/strands-decider-2B-hobson-v19 --device cpu --port 8000   # first run downloads ~4.5 GB
```
```sql
SET anofox_decide_allow_remote = true;
SELECT decide_register_model('strands-decider', 'strands');   -- default endpoint http://127.0.0.1:8000
SELECT decide_choice('Help! My payouts have been failing.', 'Which team?', ['billing', 'sales', 'retail'],
                     model := 'strands-decider');
```

Kev (Qwen3.5 + LoRA, Apache 2.0) is run the same way: `python -m kev.serve --run jaredpalmer/kev-0.8b
--port 8009`, register it as `systemone` (above) and give it any key, since it ignores the value but the
provider requires one: `CREATE SECRET (TYPE anofox_decide, API_KEY 'local', SCOPE '127.0.0.1')`. Kev-4B,
9B and 27B are more accurate than 0.8B and need a GPU or a large Mac.

---

## Scoring many rows

A remote call takes one text plus any number of questions, so the cost is **one request per row**
whatever the SQL looks like. What you control is how many run at the same time.

| You write | Requests | Notes |
|---|---|---|
| `decide_probability` / `decide_choice` / `decide_score` / `decide_decision` / `decide_many` over a table | one per distinct row, **up to 8 at a time** | The fast path. A chunk of rows is evaluated together: identical rows are sent once, connections are reused, and a `429` makes the query back off |
| `decide_table(text, questions)` with constant arguments | one request for all questions | One document, many questions |
| `LATERAL decide_table(t.text, ...)` over a table | one per row, **one after another** | DuckDB gives the function one row per call in a lateral join, so it cannot be concurrent (measured: with 8 DuckDB threads the peak is still 1 request in flight). Fine for a few rows |

Measured on the real Liquid D1 service (8 tickets, three questions each): 176.7 s with
`SET anofox_decide_max_concurrency = 1` against 23.1 s with the default, same answers (D1 itself varies
slightly from call to call). The triage example above takes about 11 s.

```sql
SET anofox_decide_max_concurrency = 4;   -- 0 = automatic (default): 8 hosted, 1 for the local strands server; 1 = one at a time
SET anofox_decide_timeout_ms = 60000;    -- slow models: more time per request
SET anofox_decide_max_retries = 5;       -- retries after a 429, a 5xx or a connection failure
```

Lower the limit if your provider rate limits you. The query already halves its window after a `429`,
honours `Retry-After`, and stops starting new requests after the first error (the error of the first
failing row is the one reported). Two identical `(text, question)` pairs in one chunk cost one request;
nothing is shared across chunks or queries, because a service need not answer the same question the same
way twice. Local models run one after another. See
[`examples/03_scoring_many_rows.sql`](examples/03_scoring_many_rows.sql).

---

## Evaluate and compare models

The metric aggregates work on any `(probability, label)` pairs, so comparing models is one query:

```sql
SELECT model,
       decide_accuracy(p, refund)    AS accuracy,
       decide_brier_score(p, refund) AS brier,
       decide_ece(p, refund)         AS ece
FROM runs GROUP BY model ORDER BY brier;     -- see examples/04_compare_models.sql
```

`decide_accuracy` thresholds at 0.5 unless you pass a threshold. NULL rows are skipped and empty input
returns NULL; probabilities outside 0 to 1, NaN, or labels that are not 0/1 raise an error that says what
to fix.

Measured on 200 labelled tickets from the public Bitext customer-support dataset (60 refund requests,
140 not; routing into billing / orders / account / other). Full method, caveats and per-class results:
[docs/EVALUATION.md](docs/EVALUATION.md).

| Model | Runs | Refund accuracy | Refund AUROC | Routing accuracy |
|---|---|---|---|---|
| TypeSafe Jev (`typesafe`) | hosted API | 93.0% | 0.984 | 92.0% |
| Laya multilingual | local, in-process | 92.0% | 0.970 | 62.0% |
| Laya typed-decisions | local, in-process | 88.5% | 0.978 | 76.0% |
| Cloudflare Clef (`cloudflare`) | hosted API | 85.0% | 0.951 | 88.0% |
| strands-decider 2B (`strands`) | local server | 81.0% | 0.918 | 65.0% |
| Liquid D1 (`liquid`) | hosted API | 77.0% | 0.970 | 91.0% |
| Kev-0.8B (`systemone`) | local server | 70.5% | 0.850 | 44.0% |
| Cloudflare Clef-flash (`cloudflare`) | hosted API | 70.0% | 0.769 | 78.5% |
| Julia-1 | local, in-process | 29.5% | 0.407 | 37.5% |

Always-no scores 70% on refund and always-billing 45% on routing. Jev is the most accurate overall;
Clef is the third-best router (not significantly different from Jev) and a solid, conservative refund detector
(recall 62% at the 0.5 cut-off), while the smaller Clef-flash is no better than always-no on refund;
strands-decider is conservative on refund (recall 48% at the 0.5 cut-off) and a mid-table router; D1
routes as well as Jev, but at the 0.5 cut-off over-predicts refunds (its ranking is fine, AUROC 0.970);
Laya multilingual matches Jev on refund and is the best in-process model, with typed-decisions the best
in-process router. This is English, template-generated data: a smoke test of relative strength, not a
benchmark of your tickets, so evaluate on your own (how to reproduce this run: [docs/EVALUATION.md](docs/EVALUATION.md#reproduce)). Each local ONNX model
reproduces its upstream Python reference to within 5e-5 in probability on a small check set; the hosted
models are called as-is.

---

## Calibration

A model's yes/no probability is only as good as its cut-off. Some models say "yes" too often at 0.5 (Liquid
D1 flagged every complaint as a refund request) and others too rarely. Per-model **Platt scaling**
(`p' = sigmoid(a * logit(p) + b)`, `a > 0`, so the ranking never changes) fixes the cut-off and the
sharpness for yes/no questions; choice and score answers are untouched. Fit it from labelled data, register
the model with it:

```sql
-- 1. fit on a labelled sample of that model's raw probabilities
SELECT decide_fit_calibration(p, y) FROM labelled;                      -- 'platt:1.92,-0.72'
-- 2. register the model with the fitted spec (any provider; local models take the MAP as 6th argument)
SELECT decide_register_model('d1-cal', 'liquid', MAP {'model': 'd1:free', 'calibration': 'platt:1.92,-0.72'});
SELECT decide_register_model('laya-cal', 'local', '/m/laya.onnx', '/m/tokenizer.json', 'laya', MAP {'calibration': 'platt:1.1,-0.3'});
-- 3. every yes/no probability from that model (decide_probability, decide_decision, decide_many, decide_table) is calibrated
SELECT calibration FROM decide_models();                                   -- the spec in use, NULL when none
```

`decide_fit_calibration` needs both outcomes and non-constant probabilities, ignores NULL rows, and refuses a
model whose probabilities rank outcomes no better than chance (a negative slope would invert it). Fit on one
sample and check on another with `decide_brier_score` and `decide_ece`; about 50 or more labelled rows give a
stable fit. On the 200-ticket evaluation, fitting on one half and scoring the other (20 random splits)
cut D1's refund error from 23% to 9% and its Brier score from 0.157 to 0.065; see
[docs/EVALUATION.md](docs/EVALUATION.md#calibration) (`tools/eval/crossfit_calibration.py` reproduces it).
A calibration is fitted to one data distribution: refit it for your own tickets.

`decide_choice_distribution(state, question, options[, model])` returns the probability of every option as a
`MAP(VARCHAR, DOUBLE)` (it sums to 1), so you can threshold the winner's probability and abstain in plain SQL:

```sql
SELECT body, decide_choice_distribution(body, 'Which team owns this?', ['billing','defect','other'], model := 'jev-latest') AS dist
FROM tickets;
-- abstain below 60% confidence
SELECT CASE WHEN list_max(map_values(d)) >= 0.6 THEN list_extract(map_keys(d), list_position(map_values(d), list_max(map_values(d)))) END AS team
FROM (SELECT decide_choice_distribution(body, 'Which team owns this?', ['billing','defect','other'], model := 'jev-latest') AS d FROM tickets);
```

---

## Local models

A local model runs inside DuckDB on ONNX Runtime, on the CPU and fully offline after setup. The extension
ships no weights and no graphs: you export a checkpoint to ONNX once with
[`tools/export_julia`](tools/export_julia/README.md), then register the files.

```sql
SELECT decide_register_model('julia-1', 'local', '<path>/julia1.onnx', '<path>/tokenizer/tokenizer.json');
SELECT decide_probability('...', '...', model := 'julia-1');
```

The optional 5th argument is the **profile**, `julia-1` (default) or `laya`. It fixes how questions are
rendered for that model family, the sequence limits and the calibration. Laya
([convaiinnovations/laya](https://huggingface.co/convaiinnovations/laya), Apache 2.0, ModernBERT / mmBERT
encoders) needs `rl_agent_config.json` from its checkpoint next to the graph:

```sql
SELECT decide_register_model('laya', 'local', '<dir>/julia1.onnx',
                             '<snapshot>/multilingual/tokenizer/tokenizer.json', 'laya');
```

Both SentencePiece-style tokenizers (Julia-1, Laya multilingual) and ByteLevel BPE (Laya English and
typed-decisions) are supported. Registration checks the files, not only that they exist: a weights file
(`.safetensors`), a text file or a zip as the graph, a tokenizer that is not BPE, and a Laya config
without its keys are rejected with a message that says what the file is. `decide_models()` and
`decide_doctor()` run the same checks.

**Long text is an error, not a silent cut.** A local model reads a fixed window (8192 tokens for
`julia-1`, adjustable with `anofox_decide_max_length` and `anofox_decide_head_length`; Laya models use the
`max_len` and `head_max_len` of their `rl_agent_config.json` and ignore those two settings). When the text,
the question or an option does not fit, the call fails and says how much would be ignored, and for which
question. Find the long rows first with `decide_token_count`, or accept the cut with
`SET anofox_decide_on_truncate = 'ignore';`:

```sql
SELECT id, decide_token_count(body, model := 'julia-1') AS tokens FROM tickets ORDER BY tokens DESC LIMIT 10;
SET anofox_decide_on_truncate = 'ignore';  -- score the shortened text instead of failing
```

See [`examples/05_local_model.sql`](examples/05_local_model.sql).

---

## Settings

Settings apply to the whole database instance (`SET name = value;` affects every connection to it, and is
gone when the process ends) and are validated: a bad value fails with the allowed range, the default and an
example.

| Setting | Default | Meaning |
|---|---|---|
| `anofox_decide_model` | *(unset)* | Model id used when a call names none. Unset: name one per call or set it |
| `anofox_decide_allow_remote` | `false` | Opt-in for hosted and server models; the text you score goes to their endpoint |
| `anofox_decide_max_concurrency` | `0` | Remote requests in flight at once, 0 to 64. `0` = automatic: 8 hosted, 1 for the local strands server |
| `anofox_decide_timeout_ms` | `30000` | Per-request timeout for remote calls |
| `anofox_decide_max_retries` | `3` | Retries after a `429`, a `5xx` or a connection failure, 0 to 10 |
| `anofox_decide_max_questions` | `100` | Most questions in one `decide_many` / `decide_table` call, up to 1000 |
| `anofox_decide_max_length` | `8192` | Tokens a `julia-1` local model reads per question (Laya uses its own config) |
| `anofox_decide_head_length` | `512` | Tokens reserved for the question and its options in local models |
| `anofox_decide_on_truncate` | `error` | `error`: a local model refuses text, question or options that do not fit its window; `ignore`: score the shortened input |
| `anofox_decide_api_key` | *(unset)* | Legacy key for the `typesafe` provider, kept in plain text; prefer `CREATE SECRET` or the env var |
| `anofox_decide_endpoint` | `https://api.typesafe.ai` | Legacy endpoint for `typesafe`; other providers take a per-model `endpoint` |
| `anofox_telemetry_enabled` | `true` | Anonymous usage telemetry, see [Telemetry](#telemetry) |
| `datazoo_banner` | `true` | The once-a-day feedback banner in interactive terminals |

---

## Status & scope

The extension is calendar-versioned (`2026.10.03` is the third of October 2026) and is built and tested
on Linux (amd64, arm64), macOS (arm64) and Windows (amd64) against DuckDB 1.5.6. What works today:

| Provider | Models | Questions | Runs where |
|---|---|---|---|
| `stub` | deterministic placeholder (0.5 / first option / middle level) | `binary`, `choice`, `score` | in-process, for tests and demos |
| `typesafe` | TypeSafe Jev (`jev-latest`) | `binary`, `choice`, `score` | hosted API; opt-in via `anofox_decide_allow_remote` |
| `liquid` | Liquid AI D1 (`d1:free`) | `binary`, `choice`, `score` | hosted API; same opt-in |
| `cloudflare` | Cloudflare Clef (`clef`), Clef-flash (`clef-flash`) | `binary`, `choice`, `score` | hosted API (Workers AI); same opt-in |
| `systemone` | any System One compatible server, e.g. Kev | `binary`, `choice`, `score` | your endpoint (`https://`, or `http://` on loopback) |
| `strands` | strands-decider 2B | `binary`, `choice`, `score` | local server on loopback, no key |
| `local` | Julia-1, Laya multilingual, Laya typed-decisions | `binary`, `choice`, `score` | in-process on CPU, offline after setup |

`score` is verified for parity with each model's upstream implementation and against the live services;
unlike refund and routing it has not been scored for accuracy on labelled data.

**Not yet:** images (Cloudflare Clef accepts them, our functions take text only); GPU execution for local models; the Von model (it needs order-invariant attention in the
export); a built-in download for local model files (today you export them with `tools/export_julia`);
on macOS only Apple silicon is built (ONNX Runtime ships no Intel macOS archive after v1.23.2), and
WebAssembly, musl and MinGW are not built.

**Which model?** Jev was the most accurate in the 200-ticket evaluation, with Clef and D1 close behind on routing. Among models that run in-process,
Laya multilingual matched it on refund detection and Laya typed-decisions routes best; there is no single
best local model yet. Start with whichever you have a key or the hardware for, then compare on your own
labelled data.

Further reading: [docs/EVALUATION.md](docs/EVALUATION.md) (the 200-ticket evaluation),
[docs/SPIKE_RESULTS.md](docs/SPIKE_RESULTS.md) (provider contracts),
[docs/CALIBRATION_PERF.md](docs/CALIBRATION_PERF.md) (calibration and performance),
[docs/REVIEW_FOLLOWUP.md](docs/REVIEW_FOLLOWUP.md) (design decisions and model investigations),
[CHANGELOG.md](CHANGELOG.md).

---

## Troubleshooting

Every error follows one shape, `<function>: <what went wrong>. Fix: <what to run>`, and echoes the value
that caused it. Run `SELECT * FROM decide_doctor();` first when something does not work.

| You see | It means | Do this |
|---|---|---|
| `no model selected` | the call names no model and no default is set | `decide_probability(..., model := '<id>')` or `SET anofox_decide_model = '<id>'`; register one first with `decide_register_model` |
| `model '<id>' is not registered (given by ...)` | the id is a typo, or never registered, or removed | the message lists the registered ids and a close match; `SELECT * FROM decide_models()` |
| `model '<id>' is a ... model: calling it sends your text to ..., and remote calls are off` | remote models send your text to an endpoint and are off by default | `SET anofox_decide_allow_remote = true;` |
| `no API key for ... at <host>` | no secret, setting or env var supplies a key | export the provider's variable (`LIQUID_API_KEY`, `TYPESAFE_API_KEY`) or `CREATE SECRET (TYPE anofox_decide, API_KEY '...', SCOPE '<host>')`; a keyless local server uses provider `strands` |
| `... rejected the request (HTTP 401): "Authentication error (code 10000)"` (Cloudflare) | Cloudflare gives this answer for a wrong token and for a token that belongs to another account | check the token has Workers AI - Read and Edit and that `CLOUDFLARE_ACCOUNT_ID` is the account the token belongs to |
| `no Cloudflare account id for model '...'` | a `cloudflare` model needs the account id for its URL | `export CLOUDFLARE_ACCOUNT_ID=<id>` or `MAP {'account_id': '<id>'}` |
| `... accepts at most 64 questions per request` | Cloudflare Clef takes at most 64 questions in one request | split the questions across several calls |
| `... rejected the API key (HTTP 401)` | the service refused the key; the message says where the key came from | replace it: `CREATE OR REPLACE SECRET (TYPE anofox_decide, API_KEY '<key>', SCOPE '<host>')` (a secret overrides env vars) |
| `... does not know the model '<x>' ... (HTTP 404)` | the model name sent to the service is wrong (it is the registered id unless you set one) | `decide_register_model('<id>', '<provider>', MAP {'model': '<provider model name>'})` |
| `... rejected the request (HTTP 422)` | the service refused the question or options; its own reason is quoted | for a `criteria` complaint register the model with `MAP {'criteria': 'name'}` |
| `... is rate limiting requests (HTTP 429)` | too many calls; retried with the service's Retry-After | `SET anofox_decide_max_retries = 8;` or `SET anofox_decide_max_concurrency = 2;` |
| `... had a server error (HTTP 5xx)` | the service failed; not caused by your query | retry later or use another model |
| `... is not reachable: nothing is listening on 127.0.0.1:<port>` | a local server (strands-decider, Kev) is not running | start it; connection refused on loopback is not retried |
| `... did not answer within N ms` | a slow model or server | `SET anofox_decide_timeout_ms = 60000;` (or `SET anofox_decide_max_retries = 0;` to fail fast) |
| `... answered HTTP 200 but not with JSON (an HTML page)` | the endpoint or path is not a System One server (a login or error page) | check the endpoint and path in `decide_models()` |
| `endpoint '<x>' contains a path` / `has an invalid port` | an endpoint is only `scheme://host[:port]` | `MAP {'endpoint': 'https://host', 'path': '/v1/...'}` |
| `the graph file '...' does not exist` | the path is wrong; relative paths resolve against the DuckDB working directory | pass the full path of the exported `.onnx`; see [Local models](#local-models) |
| `the text is N tokens but the local model reads at most M ...` | the text does not fit the local model's window, and the end would be ignored | shorten the text, raise `anofox_decide_max_length` (`julia-1` only), or `SET anofox_decide_on_truncate = 'ignore';`; `decide_token_count(text)` measures it |
| `... the question is N tokens but only M fit in the head budget` / `option N (...) of question '...' is N tokens but ...` | the question or an option is too long for the part of the window they share | shorten it, raise `anofox_decide_head_length`, or `SET anofox_decide_on_truncate = 'ignore';` |
| `the graph file '...' is a safetensors weights file, not an ONNX graph` (also: text, JSON, HTML, zip, pickle) | the graph argument is not the exported `.onnx` | pass the `.onnx` file that `tools/export_julia` wrote; see [Local models](#local-models) |
| `ONNX Runtime could not load the graph '...'` / `the graph '...' is not a readable ONNX model` | the file passed the registration checks but ONNX Runtime cannot use it (damaged, incomplete, or an unsupported ONNX version); the reason from ONNX Runtime is in parentheses | export the model again with `tools/export_julia` and register that file |
| `the graph '...' has the inputs ..., but a local model is fed ...` | the graph is not a scores graph from `tools/export_julia`; the message names the missing and unused inputs | export it again with `tools/export_julia` |
| `the tokenizer file '...' is a Unigram tokenizer, but local models need a BPE tokenizer` | the tokenizer belongs to another model family | pass the `tokenizer.json` of a Julia-1 or Laya checkpoint |
| `the Laya config file '...' lacks the keys ...` | `rl_agent_config.json` is not the one of a Laya checkpoint | copy it from the checkpoint folder next to the graph |
| `'.../rl_agent_config.json' sits next to the graph ..., so it comes from a Laya checkpoint` | a Laya graph registered without the profile would be scored as `julia-1` | add the 5th argument: `decide_register_model('<id>', 'local', '<graph>', '<tokenizer>', 'laya')` |
| `provider '<x>' takes no file paths` | a remote provider was given a path argument | remote providers take an options `MAP`: `decide_register_model('m', 'typesafe', MAP {'endpoint': 'https://host'})` |
| `model '<id>' is already registered` | ids are unique per database instance | `decide_unregister_model('<id>')`, then register again |
| `<setting>: must be between A and B, got X` | a setting is out of range | the message shows the default and an example `SET` |
| `threshold must be between 0 and 1, got 70` | thresholds are fractions, not percents | use `0.7` for 70% |
| `probability must be between 0 and 1` (metrics) | the first argument is not a probability | pass probabilities, not percents or logits (`p / 100.0`); NaN: `NULLIF(p, 'NaN'::DOUBLE)` |
| `No function matches ... explicit type casts` | DuckDB's own message for argument types | cast the arguments (`x::VARCHAR`, `y = 1`); `decide_choice` options must be a list of strings `['a','b']` |

Local models print many `Schema error: ... already registered` lines on stderr the first time one loads;
they come from the bundled ONNX Runtime and are harmless.

---

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

## License

- **This extension's code:** MIT.
- **Models:** the extension ships **no model weights and no graphs**, and never redistributes any. Check a
  model's license before you build on it, especially one you export or serve yourself:

| Model | Provider | License / terms | Where it runs |
|---|---|---|---|
| Jev | TypeSafe | the provider's API terms | hosted; receives the text you score |
| D1 | Liquid AI | the provider's API terms | hosted; receives the text you score |
| Clef, Clef-flash | Cloudflare | Apache 2.0 weights; the hosted service under Cloudflare's terms | hosted on Workers AI; receives the text you score |
| strands-decider | Strands Labs | Apache 2.0 | your machine, as a local server |
| Kev | Jared Palmer | Apache 2.0 | your machine, as a local server |
| Julia-1 | SupersonicLabs | Apache 2.0 | your machine, in-process |
| Laya | ConvAI Innovations | Apache 2.0 | your machine, in-process |

Licenses are as published by each project at the time of writing.

## Telemetry

Sends **anonymous** usage telemetry (extension load and per-function call counts: no state, questions,
answers, model ids, endpoints, keys or SQL) to PostHog EU, through the shared
[`posthog-telemetry`](https://github.com/DataZooDE/posthog-telemetry) library, matching the other anofox
extensions. Opt out any time:

```bash
export DATAZOO_DISABLE_TELEMETRY=1        # environment
```
```sql
SET anofox_telemetry_enabled = false;      -- SQL
```

CI environments are auto-detected and telemetry is disabled there. The full list of what is collected is in
[TELEMETRY.md](TELEMETRY.md).

## Building

Clone with submodules (`git clone --recurse-submodules`; they include the shared `posthog-telemetry` and
`datazoo-banner` libraries), then:

```bash
# vcpkg is required: vcpkg.json depends on openssl, and ONNX Runtime is built from the vcpkg overlay port.
git clone https://github.com/microsoft/vcpkg && ./vcpkg/bootstrap-vcpkg.sh -disableMetrics
export VCPKG_TOOLCHAIN_PATH=$PWD/vcpkg/scripts/buildsystems/vcpkg.cmake

GEN=ninja make release                # release build; ONNX Runtime built via vcpkg (the first build is slow)
GEN=ninja make release DECIDE_ORT_VCPKG=0 CMAKE_PREFIX_PATH=<ort-install>   # faster local build against an existing ONNX Runtime
make test_release                     # offline suite: SQL tests + Catch2 (no network, no weights)
make test-live                        # live TypeSafe + Liquid D1 tests; SKIP without TYPESAFE_API_KEY / LIQUID_API_KEY
./build/release/test/unittest test/sql/decide_contract.test       # a single file
```

Without `VCPKG_TOOLCHAIN_PATH` the configure step stops with a cascade of CMake errors in which only the
first line is the real cause (`Could not find toolchain file: .../vcpkg_installed//share/vcpkg/...`); the
"no build program" and "compiler not set" lines after it are consequences, reported even when Ninja and the
compilers are installed. You need CMake (3.19 or newer to build the bundled C++ unit tests), a C++17
toolchain, and vcpkg.

A locally built extension is unsigned: start DuckDB with `duckdb -unsigned` and
`LOAD '/path/to/anofox_decide.duckdb_extension';`, or use the `duckdb` binary from `build/release`, which
has the extension linked in.

Some tests need model weights or a network and warn-and-pass without them: Catch2 `[tokenizer]` goldens
(`JULIA_WEIGHTS_DIR`, `LAYA_TYPED_DIR`), `[local]` real graphs (`JULIA_ONNX`, `LAYA_MULTILINGUAL_DIR` +
`LAYA_ONNX`, `LAYA_TYPED_DIR` + `LAYA_TYPED_ONNX`), and the export pytest in `tools/export_julia`
(`JULIA_WEIGHTS_DIR`, `JULIA_SRC_MODEL`, `JULIA_ONNX`). Run the suite with `TYPESAFE_API_KEY` and
`LIQUID_API_KEY` unset, as CI does, so tests do not depend on your own key. Tests always run with
`DATAZOO_DISABLE_TELEMETRY=1` (the Makefile does this).

Releases are calendar-versioned tags (`vYYYY.MM.DD`); see [CHANGELOG.md](CHANGELOG.md).
