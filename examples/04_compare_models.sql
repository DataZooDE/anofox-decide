-- examples/04_compare_models.sql
-- Score the same labelled tickets with several models and compare them side by side. Needs real
-- models, so this file is not run by the offline tests; edit the registrations to the keys you have.
--
--   export LIQUID_API_KEY=...        # Liquid AI D1
--   export TYPESAFE_API_KEY=...      # TypeSafe Jev
--   duckdb -unsigned < examples/04_compare_models.sql
-- Run from the repository root (the data path is relative to it).
-- offline: no (calls hosted models)

LOAD anofox_decide;
SET anofox_decide_allow_remote = true;

SELECT decide_register_model('jev-latest', 'typesafe');
SELECT decide_register_model('d1:free', 'liquid');
-- A local server, keyless (see "Models and providers" in the README):
--   SELECT decide_register_model('strands-decider', 'strands');

CREATE OR REPLACE TABLE tickets AS SELECT * FROM read_csv('examples/data/support_tickets.csv');

CREATE OR REPLACE TABLE runs AS
SELECT 'jev-latest' AS model, id, refund,
       decide_probability(text, 'A refund is requested.', model := 'jev-latest') AS p FROM tickets
UNION ALL
SELECT 'd1:free', id, refund,
       decide_probability(text, 'A refund is requested.', model := 'd1:free') FROM tickets;

-- Accuracy, calibration and the share of confident answers per model. Judge on YOUR tickets:
-- a handful of rows says little; use a few hundred labelled ones.
SELECT model,
       count(*)                          AS n,
       decide_accuracy(p, refund)        AS accuracy,
       decide_brier_score(p, refund)     AS brier,
       decide_ece(p, refund)             AS ece
FROM runs GROUP BY model ORDER BY brier;
