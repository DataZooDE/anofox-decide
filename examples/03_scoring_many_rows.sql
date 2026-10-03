-- examples/03_scoring_many_rows.sql
-- Scoring a whole table with a hosted model: how many requests run at once, and what the shapes cost.
-- Written against the stub so the file runs offline; the settings are what matter.
-- Run from the repository root (the data path is relative to it).
-- offline: yes (with the stub)

LOAD anofox_decide;
SET anofox_decide_model = 'stub';

CREATE OR REPLACE TABLE tickets AS SELECT * FROM read_csv('examples/data/support_tickets.csv');

-- 0 (the default) = automatic: 8 requests at a time for hosted providers, 1 for the local strands
-- server. 1 sends them one after another. Lower it if your provider rate limits you; a 429 already
-- halves the window, and Retry-After is honoured.
SET anofox_decide_max_concurrency = 4;
SET anofox_decide_timeout_ms = 60000;     -- slow models: give each request more time
SET anofox_decide_max_retries = 5;        -- retries after a 429, a 5xx or a connection failure

-- The fast shape: scalar calls over the table. One request per distinct (text, question).
SELECT id,
       decide_probability(text, 'A refund is requested.')                           AS refund,
       decide_choice(text, 'Which team owns this?', ['billing', 'defect', 'other']) AS team
FROM tickets ORDER BY id;

-- Several questions per text in ONE request per text: decide_many returns one JSON document per row.
-- Read it with DuckDB's json functions (INSTALL json; LOAD json; autoloaded in the CLI), e.g.
--   json_extract(answers, '$.results[0].probability')::DOUBLE      json_extract_string(answers, '$.results[1].choice')
SELECT id,
       decide_many(text,
         '[{"id":"refund","kind":"binary","instruction":"A refund is requested."},
           {"id":"team","kind":"choice","instruction":"Which team owns this?","options":["billing","defect","other"]}]') AS answers
FROM tickets ORDER BY id;

-- One text, many questions, as rows. Constant arguments: a single request for all questions.
SELECT question_id, probability, choice
FROM decide_table((SELECT text FROM tickets WHERE id = 1),
  '[{"id":"refund","kind":"binary","instruction":"A refund is requested."},
    {"id":"vat","kind":"binary","instruction":"The writer disputes a tax amount."}]');

-- The same rows for a whole table: decide_answers returns decide_table's rows as a list per text, and
-- unnest(..., recursive := true) turns them into columns. It runs in the scalar pipeline, so the texts
-- are scored concurrently, identical texts are sent once and calibration applies. Prefer it to a
-- lateral join over decide_table (DuckDB gives that one row per call, so it runs one request after another).
SELECT id, question_id, kind, probability, choice
FROM (SELECT id, unnest(decide_answers(text,
  '[{"id":"refund","kind":"binary","instruction":"A refund is requested."},
    {"id":"team","kind":"choice","instruction":"Which team owns this?","options":["billing","defect","other"]}]'),
  recursive := true) FROM tickets)
ORDER BY id, question_id;
