-- examples/01_quickstart.sql
-- The whole SQL surface in one file, on the built-in 'stub' test model, so it runs anywhere with
-- no key and no network. The stub returns constants (0.5, the first option, the middle level): it
-- checks that your SQL is right, not that an answer is. To get real answers, see the lines marked
-- "REAL MODEL" and examples/02_support_triage.sql.
--
--   run it:  duckdb -unsigned < examples/01_quickstart.sql
-- (use `duckdb < ...` once the extension is installed from the community repository)
-- offline: yes

LOAD anofox_decide;

-- Is everything ready? One row per check, with the fix for each problem.
SELECT * FROM decide_doctor();

-- REAL MODEL: register one, allow remote calls, and name it instead of 'stub' below.
--   SET anofox_decide_allow_remote = true;                  -- remote models send your text to the provider
--   SELECT decide_register_model('d1:free', 'liquid');      -- key: export LIQUID_API_KEY=... first
SET anofox_decide_model = 'stub';

-- Binary: how likely is it that a statement holds? (0 to 1)
SELECT decide_probability('The customer requests a refund.', 'A refund is requested.') AS p;

-- A decision needs an explicit threshold: it is never guessed for you. NULL in, NULL out.
SELECT decide_decision('The customer requests a refund.', 'A refund is requested.', 0.7) AS refund;

-- Choice: one answer from a list you define at query time.
SELECT decide_choice('Invoice charged twice', 'Which team owns this?', ['billing', 'defect', 'other']) AS team;

-- Score: where does the text sit on an ordered rubric? The expected level, 0-based.
SELECT decide_score('Help! My payouts have failed for 3 days!', 'How frustrated is the writer?',
                    ['calm', 'frustrated', 'angry']) AS frustration;

-- Several questions about one text, as rows: probability, choice, score, per-level distribution.
SELECT question_id, kind, probability, choice, score
FROM decide_table('The bill is wrong and I want my money back.',
  '[{"id": "refund", "kind": "binary", "instruction": "A refund is requested."},
    {"id": "team",   "kind": "choice", "instruction": "Which team owns this?", "options": ["billing", "defect", "other"]},
    {"id": "mood",   "kind": "score",  "instruction": "How frustrated is the writer?", "levels": ["calm", "frustrated", "angry"]}]');

-- The same questions as one JSON document per text.
SELECT decide_many('The bill is wrong.',
  '[{"id": "refund", "kind": "binary", "instruction": "A refund is requested."}]') AS answers;

-- Which models exist, and can each one be called right now?
SELECT model, provider, mode, ready, hint FROM decide_models();
