-- examples/02_support_triage.sql
-- Triage support tickets: is a refund requested, which team owns it, how frustrated is the writer,
-- and how good is that against labels you already have. Works with any model; it starts on the
-- built-in 'stub' so it runs offline (the numbers are then meaningless).
--
-- REAL MODEL: replace the three lines below. A hosted model needs the opt-in and a key:
--   export LIQUID_API_KEY=...                       -- or CREATE SECRET (TYPE anofox_decide, API_KEY '...', SCOPE 'api.liquid.ai')
--   SET anofox_decide_allow_remote = true;
--   SELECT decide_register_model('d1:free', 'liquid');
--   SET anofox_decide_model = 'd1:free';
-- Run from the repository root (the data path is relative to it).
-- offline: yes (with the stub)

LOAD anofox_decide;
SET anofox_decide_model = 'stub';

CREATE OR REPLACE TABLE tickets AS
SELECT * FROM read_csv('examples/data/support_tickets.csv');   -- id, text, refund (label), team (label)

-- One scalar call per question. A table of rows is scored together: identical texts are sent once and
-- remote requests run concurrently (see "Scoring many rows" in the README).
CREATE OR REPLACE TABLE scored AS
SELECT id, text, refund AS refund_label, team AS team_label,
       decide_probability(text, 'A refund is requested.')                                  AS p_refund,
       decide_choice(text, 'Which team owns this?', ['billing', 'defect', 'other'])        AS team_pred,
       decide_score(text, 'How frustrated is the writer?', ['calm', 'frustrated', 'angry']) AS frustration
FROM tickets;

SELECT id, round(p_refund, 3) AS p_refund, team_pred, round(frustration, 2) AS frustration, left(text, 50) AS text
FROM scored ORDER BY id;

-- Turn probabilities into actions with an explicit threshold you choose and can defend.
SELECT id, p_refund >= 0.7 AS escalate_to_refunds FROM scored ORDER BY id;

-- How good is it? Accuracy of the 0.5 cut-off, Brier score (lower is better) and calibration error.
SELECT decide_accuracy(p_refund, refund_label)   AS refund_accuracy,
       decide_brier_score(p_refund, refund_label) AS brier,
       decide_ece(p_refund, refund_label)         AS ece,
       avg((team_pred = team_label)::INTEGER)    AS team_accuracy
FROM scored;
