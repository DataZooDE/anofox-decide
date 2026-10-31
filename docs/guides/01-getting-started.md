# Getting Started

Install the extension, check the setup with `decide_doctor()`, register a model and make a first real
decision over a small table of support tickets.

## 1. Install and load

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
models, opt in. Local models are downloaded only when you run `decide_download`.

## 2. Check the setup and pick a model

`SELECT * FROM decide_doctor();` tells you what is ready and how to fix what is not. On a fresh install
the doctor says what is missing and the fix for each item:

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

With nothing set up yet, use the built-in `stub` to try the SQL: `model := 'stub'`. It returns constants
(0.5, the first option, the middle level) so it can never pass for a real answer. For a real model, see
[Choosing a model](02-choosing-a-model.md): it has a "You have / Do this" table for every provider.

The rest of this guide uses Liquid AI D1, a hosted model. Its key comes from the environment, so set it
before starting DuckDB:

```bash
export LIQUID_API_KEY=...            # before starting DuckDB
```

(Another provider's key: [Providers](../api/02-providers.md). No key at all: `CALL decide_download('laya-multilingual');`
and `SET anofox_decide_model = 'laya-multilingual';` instead of the three statements below; see
[Local models](../api/03-local-models.md).)

A hosted model sends the text you score to the provider, so it sits behind an explicit opt-in that is off
by default. Register it once per database instance (registrations and settings are not stored: repeat them
in each new process), then name it per call (`model := 'd1'`) or make it the default:

```sql
SET anofox_decide_allow_remote = true;                 -- explicit opt-in: the text goes to the provider
SELECT decide_register_model('d1', 'liquid');     -- the id is the model name on the wire
SET anofox_decide_model = 'd1';                   -- the default for calls that name no model
```

A call that names no model, with no default set, fails and says how to choose one; it never quietly
returns a constant.

## 3. Decide

Eight support tickets with known answers, scored for three questions at once. This is real output from
Liquid D1, so it needs the model from step 2 (`SET anofox_decide_model = 'd1'`); on Laya the numbers
differ, and on the `stub` you get the same columns filled with constants.

```sql
CREATE OR REPLACE TABLE tickets AS SELECT * FROM (VALUES
  (1, 'I was charged twice for my March invoice, please refund the extra 49 EUR.', true,  'billing'),
  (2, 'The app crashes every time I open the export dialog on version 3.2.',       false, 'defect'),
  (3, 'Thanks for the quick help yesterday, everything works now!',                false, 'other'),
  (4, 'I want my money back, this product does not do what was advertised.',       true,  'billing'),
  (5, 'Login button does nothing on Safari, error 500 in the console.',            false, 'defect'),
  (6, 'Our invoice shows the wrong VAT rate; please issue a corrected one.',       false, 'billing'),
  (7, 'Do you have an office in Munich? We would like to visit.',                  false, 'other'),
  (8, 'Cancel my subscription and reimburse the unused months.',                   true,  'billing')
) t(id, text, refund, team);                                  -- refund and team are the known answers

CREATE OR REPLACE TABLE scored AS
SELECT id, text, refund,
       decide_probability(text, 'A refund is requested.')                                   AS p_refund,
       decide_choice(text, 'Which team owns this?', ['billing', 'defect', 'other'])         AS team,
       decide_score(text, 'How frustrated is the writer?', ['calm', 'frustrated', 'angry']) AS frustration
FROM tickets;

SELECT id, round(p_refund, 3) AS p_refund, team, round(frustration, 2) AS frustration, left(text, 44) AS text
FROM scored ORDER BY id;
```
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

The complete, runnable version is [`examples/02_support_triage.sql`](../../examples/02_support_triage.sql); it
starts on the `stub` so it runs offline, and the lines marked `REAL MODEL` switch it to a real one. The
example files read their data from the repository (`examples/data/`), so run them from a clone of it; the
other examples are listed in [`examples/`](../../examples/README.md).

## Next steps

- Every question kind, `decide_table` and scoring a whole table fast: [Functions](../api/01-functions.md)
- Which model to use, and how to compare them on your data: [Choosing a model](02-choosing-a-model.md)
- How good are the probabilities, and how to fix the cut-off: [Evaluate and calibrate](03-evaluate-and-calibrate.md)
- Something fails: [Troubleshooting](../api/06-troubleshooting.md)
