---
name: anofox-decide-questions
description: >
  Asking natural-language questions about text or structured rows in SQL with the anofox_decide
  DuckDB extension. Covers the three question kinds (binary, choice, score) and every decision
  function: decide_probability (P(statement holds)), decide_decision (true/false at an explicit
  threshold), decide_choice and decide_choice_distribution (the best of options you define, and
  the probability of each), decide_score (the expected 0-based level on an ordered rubric), and
  several questions per text with decide_table (one row per question), decide_answers (the same
  rows as a LIST(STRUCT), unnested for whole tables) and decide_many (JSON). Also covers the
  questions JSON format, the decide_table output columns, NULLs and refusals, the model := argument
  and the anofox_decide_model default, structured state via to_json, installing and loading the
  extension, and the cheapest way to score a whole table (one decide_answers request per distinct
  text, joined back). Use when writing decide_probability / decide_decision /
  decide_choice / decide_choice_distribution / decide_score / decide_table / decide_answers /
  decide_many calls: classifying tickets, routing, flagging, rating, or tagging rows with an LLM
  decision model in SQL.
version: 2026.10.31
user-invocable: false
---

# Anofox Decide — Questions & Decision Functions

**Extension:** `anofox_decide` v2026.10.31 | **DuckDB:** v1.5.6 / v2.0 | **Dual naming:** every
function is `anofox_decide_<name>` and the short alias `decide_<name>` (identical)

You write the question and the possible answers in the query. A purpose-built decision model answers
with **probabilities**, not just labels. Picking and registering a model: `anofox-decide-models`.
Measuring the answers: `anofox-decide-evaluate`.

```sql
-- needs: network, once per machine (a community extension)
INSTALL anofox_decide FROM community;
LOAD anofox_decide;   -- in every new session
```

## Critical gotchas

1. **No model answers until one is chosen.** A call needs `model := '<id>'` or a default set with
   `SET anofox_decide_model = '<id>'`. Otherwise it fails with `no model selected`. Hosted models must be
   registered first and need `SET anofox_decide_allow_remote = true` (see `anofox-decide-models`).
   The built-in `stub` model needs nothing. Use it to check the SQL.
2. **`stub` returns constants:** probability 0.5 for yes/no, the first option (with probability 1.0) for a
   choice, a uniform distribution (so the middle level) for a score. It never passes for a real answer. Do
   not read results produced with it.
3. **`decide_decision` never guesses a threshold.** Its third argument is required and is a fraction
   (`0.7`, not `70`). To pick a cut-off from data, see `anofox-decide-evaluate`.
4. **`decide_score` returns an expected level, not a label.** The result is a 0-based `DOUBLE`. `1.4` means
   between level 1 and level 2. Levels are ordered **lowest first**, 2 to 10 of them. Round or bucket the
   value yourself, or read the `distribution` from `decide_table`.
5. **The model argument is named:** `model := 'd1'`, as the last argument. A NULL `model` falls back to
   the session default.
6. **The question kinds differ in their arguments:** `binary` takes a statement (phrase it as a claim:
   `'A refund is requested.'`). `choice` takes a question plus `['a', 'b', ...]` (a `VARCHAR[]` list
   literal). `score` takes a question plus ordered levels.
7. **Many rows: do not use `LATERAL decide_table(...)`.** DuckDB calls a table function once per row in a
   lateral join, so the rows are scored one after another. Use
   `unnest(decide_answers(text, questions), recursive := true)`, which scores the rows concurrently.
8. **NULL in, NULL out.** A NULL text, question, option list or threshold gives NULL, and nothing is sent.
   A **refusal** (OpenAI Decisions, Gemini safety blocks) is also NULL in the scalar functions, and
   `kind = 'refusal'` in `decide_table` / `decide_answers`. The query keeps running.
9. **Cost is one request per row, per `decide_*` call.** Three scalar calls in one `SELECT` send three
   requests per text. For several questions about the same text in **one** request, use `decide_answers`
   (or `decide_many`). Identical rows are sent once only **within one chunk** (2048 rows), and nothing is
   cached across queries: when texts repeat, score `SELECT DISTINCT text` and join the answers back (see
   the recipe at the end). Store paid answers in a table instead of re-running the query.

## The three kinds

| Kind | You provide | You get | Function |
|---|---|---|---|
| `binary` | a statement | P(statement holds), 0 to 1 | `decide_probability`, `decide_decision` |
| `choice` | a question + list of options | the best option, and P(each option) | `decide_choice`, `decide_choice_distribution` |
| `score` | a question + 2 to 10 ordered levels, lowest first | the expected 0-based level, and P(each level) | `decide_score` |

## Signatures

```text
decide_probability(state VARCHAR, question VARCHAR[, model := VARCHAR]) -> DOUBLE
decide_decision(state, question, threshold DOUBLE[, model := ...])     -> BOOLEAN   -- p >= threshold
decide_choice(state, question, options VARCHAR[][, model := ...])      -> VARCHAR
decide_choice_distribution(state, question, options[, model := ...])   -> MAP(VARCHAR, DOUBLE)  -- sums to 1
decide_score(state, question, levels VARCHAR[][, model := ...])        -> DOUBLE    -- 0-based expected level
decide_table(state, questions_json VARCHAR[, model := ...])            -> TABLE(question_id, kind, probability,
                                                                          choice, confidence, model, score, distribution)
decide_answers(state, questions_json[, model := ...])                  -> LIST(STRUCT(<the decide_table columns>))
decide_many(state, questions_json[, model := ...])                     -> VARCHAR (JSON {"model", "results": [...]})
```

## One question per call

```sql
CREATE OR REPLACE TABLE tickets AS SELECT * FROM (VALUES
  (1, 'I was charged twice for my March invoice, please refund the extra 49 EUR.', true,  'billing'),
  (2, 'The app crashes every time I open the export dialog on version 3.2.',       false, 'defect'),
  (3, 'Cancel my subscription and reimburse the unused months.',                   true,  'billing')
) t(id, text, refund, team);

SET anofox_decide_model = 'stub';     -- the default for calls that name no model; use a real id for real answers

SELECT id,
       round(decide_probability(text, 'A refund is requested.'), 3)                    AS p_refund,
       decide_decision(text, 'A refund is requested.', 0.7)                            AS refund_at_70,
       decide_choice(text, 'Which team owns this?', ['billing', 'defect', 'other'])    AS team,
       round(decide_score(text, 'How frustrated is the writer?',
                          ['calm', 'frustrated', 'angry']), 2)                         AS frustration
FROM tickets ORDER BY id;

-- the probability of every option, e.g. to abstain when the winner is unsure
SELECT id, decide_choice_distribution(text, 'Which team owns this?', ['billing', 'defect', 'other']) AS dist
FROM tickets;

-- name the model per call instead of a default (several models side by side in one query)
SELECT decide_probability('I want my money back.', 'A refund is requested.', model := 'stub') AS p;
```

Questions can be built per row, for example `decide_choice(text, 'Which team owns this?', allowed_teams)`
with a `VARCHAR[]` column.

## Structured state

The state is any text. For a row, pass JSON so that the model sees the field names:

```sql
-- needs: the json extension for to_json (the DuckDB CLI autoloads it)
SELECT id, decide_probability(to_json({ticket: text, team: team})::VARCHAR,
                              'The customer reports a product defect.', model := 'stub') AS p_defect
FROM tickets;
```

OpenAI Decisions sends JSON state as structured input (`MAP {'input': 'auto' | 'text' | 'json'}` at
registration). Other providers receive it as text.

## Several questions about one text

The questions are a JSON array of `{id, kind, instruction[, options | levels]}`. That is one request per
text. At most `anofox_decide_max_questions` (default 100) go in one call. Raise the setting to send more:
OpenAI takes up to 400 and Gemini up to 1000. Cloudflare Clef takes at most 64.

```sql
SELECT question_id, kind, probability, choice, score, distribution
FROM decide_table('The bill is wrong and I want my money back.',
  '[{"id": "refund", "kind": "binary", "instruction": "A refund is requested."},
    {"id": "team",   "kind": "choice", "instruction": "Which team owns this?", "options": ["billing", "defect", "other"]},
    {"id": "mood",   "kind": "score",  "instruction": "How frustrated is the writer?", "levels": ["calm", "frustrated", "angry"]}]',
  model := 'stub');
```

| Column | Type | Meaning |
|---|---|---|
| `question_id` | `VARCHAR` | the `id` you gave |
| `kind` | `VARCHAR` | `binary`, `choice`, `score`, or `refusal` (then every answer column is NULL) |
| `probability` | `DOUBLE` | binary: P(yes). choice: P(chosen option). score: P(most likely level) |
| `choice` | `VARCHAR` | the chosen option (choice only) |
| `confidence` | `DOUBLE` | the service's own confidence when it reports one, else NULL |
| `model` | `VARCHAR` | the model that answered |
| `score` | `DOUBLE` | the expected 0-based level (score only) |
| `distribution` | `MAP(VARCHAR, DOUBLE)` | P per option or level (choice, score) |

### Every row of a table: `decide_answers` + `unnest`

```sql
-- one output row per (ticket, question), scored concurrently
SELECT id, question_id, probability, choice, score
FROM (SELECT id, unnest(decide_answers(text,
        '[{"id": "refund", "kind": "binary", "instruction": "A refund is requested."},
          {"id": "team", "kind": "choice", "instruction": "Which team owns this?", "options": ["billing", "defect", "other"]}]',
        model := 'stub'), recursive := true)
      FROM tickets)
ORDER BY id, question_id;

-- wide: one row per ticket, one column per question
SELECT id,
       max(probability) FILTER (WHERE question_id = 'refund') AS p_refund,
       max(choice)      FILTER (WHERE question_id = 'team')   AS team
FROM (SELECT id, unnest(decide_answers(text,
        '[{"id": "refund", "kind": "binary", "instruction": "A refund is requested."},
          {"id": "team", "kind": "choice", "instruction": "Which team owns this?", "options": ["billing", "defect", "other"]}]',
        model := 'stub'), recursive := true)
      FROM tickets)
GROUP BY id ORDER BY id;
```

`decide_many` returns the same answers as one JSON document per text, for when you store or forward JSON:

```sql
SELECT id, decide_many(text, '[{"id": "refund", "kind": "binary", "instruction": "A refund is requested."}]',
                       model := 'stub') AS answers
FROM tickets;
```

## Refusals and NULLs in results

Metrics skip NULLs. To count a refusal as a wrong answer, coalesce it to the wrong side:

```sql
SELECT decide_accuracy(coalesce(p, CASE WHEN refund THEN 0.0 ELSE 1.0 END), refund, 0.5) AS accuracy,
       count(*) FILTER (WHERE p IS NULL) AS refusals
FROM (SELECT refund, decide_probability(text, 'A refund is requested.', model := 'stub') AS p FROM tickets);
```

## Scoring many rows

| You write | Requests |
|---|---|
| a scalar function (`decide_probability`, `decide_choice`, `decide_score`, `decide_decision`, `decide_many`) over a table | one per row **for each such call** (repeats within a chunk are sent once), up to 8 at a time per chunk |
| `decide_table(...)` with constant arguments | one request for all of its questions |
| `unnest(decide_answers(t.text, ...), recursive := true)` over a table | one per row for all questions (repeats within a chunk are sent once), up to 8 at a time per chunk |
| `LATERAL decide_table(t.text, ...)` | one per row, **one after another**. Fine for a handful of rows only |

```sql
SET anofox_decide_max_concurrency = 4;   -- requests in flight (0 = automatic: 8 hosted, 1 for strands)
SET anofox_decide_timeout_ms = 60000;    -- per request
SET anofox_decide_max_retries = 5;       -- after a 429, a 5xx or a connection failure
```

The limit applies per chunk of rows. On a large table DuckDB scores several chunks in parallel threads, so
up to `threads × max_concurrency` requests can be in flight. For a strict rate limit, lower both (e.g.
`SET threads = 1;` for the scoring query). A `429` halves the window and honours `Retry-After`. The first
failing row's error is reported, and no new requests start after it, so on a very large table insert in
chunks (`... WHERE text NOT IN (SELECT text FROM answers) LIMIT 10000`) and keep what was already paid for. Local models score one text at a time, or in batches with
`anofox_decide_batch_tokens` (see `anofox-decide-local`).

## Recipe: triage a ticket table

The cheapest form: every distinct text is sent **once**, with all three questions in that one request.
Three scalar calls (`decide_probability` + `decide_choice` + `decide_score`) give the same answers for
three times the requests.

```sql
SET anofox_decide_model = 'stub';        -- replace with a registered real model
CREATE OR REPLACE TABLE triage_answers AS
SELECT text,
       max(probability) FILTER (WHERE question_id = 'refund')  AS p_refund,
       max(choice)      FILTER (WHERE question_id = 'team')    AS team,
       max(score)       FILTER (WHERE question_id = 'urgency') AS urgency   -- 0 = low ... 3 = critical
FROM (SELECT text, unnest(decide_answers(text,
        '[{"id": "refund",  "kind": "binary", "instruction": "A refund is requested."},
          {"id": "team",    "kind": "choice", "instruction": "Which team owns this?", "options": ["billing", "defect", "other"]},
          {"id": "urgency", "kind": "score",  "instruction": "How urgent is this?", "levels": ["low", "normal", "high", "critical"]}]'),
        recursive := true)
      FROM (SELECT DISTINCT text FROM tickets WHERE text IS NOT NULL))
GROUP BY text;

-- join back: repeated texts share one answer, and tickets without text stay (with NULL answers)
CREATE OR REPLACE TABLE triaged AS
SELECT t.id, t.text, a.p_refund, a.team, a.urgency
FROM tickets t LEFT JOIN triage_answers a USING (text);

SELECT team, count(*) AS n, round(avg(urgency), 2) AS mean_urgency
FROM triaged GROUP BY team ORDER BY n DESC;
```

`unnest` drops a row whose `decide_answers` is NULL (NULL text), which is why the result is joined back to
`tickets`. A refusal keeps its rows with `kind = 'refusal'` and NULL answers.

Reference pages: [Functions](https://github.com/DataZooDE/anofox-decide/blob/main/docs/api/01-functions.md),
[Getting started](https://github.com/DataZooDE/anofox-decide/blob/main/docs/guides/01-getting-started.md).
