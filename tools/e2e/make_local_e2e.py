#!/usr/bin/env python3
"""Generate test/sql/decide_local_e2e.test: realistic support-ticket decisions on the three catalog models,
with the expected answers captured from a reference build: decide v2026.11.24 or earlier, whose only engine is ONNX Runtime
(formerly decide's engine; it was removed when the models moved to ggml).

The tickets are written for this repository (no third-party data). Short tickets (20-60 tokens) and long ones
(about 250-850 tokens: longer than ModernBERT's 128-token local-attention window, so the sliding-window layers
matter) are answered by each model: a refund probability, a routing choice, a frustration score, and a
thresholded decision; plus several questions at once through decide_answers and decide_many, with and without
cross-row batching. julia-1 (8192-token window) also reads a ~3000-token thread.

The test compares against the captured values: probabilities to 1e-4 absolute AND 2e-3 in log-odds (so a
saturated 1e-8 cannot hide a wrong logit), scores to 1e-4, labels exactly where the reference has a clear margin
(argmax margin and |p - 0.5| above 1e-3; closer calls are compared through their probabilities only). The engine
under test is the ggml engine on the CPU, proven by __anofox_decide_local_sessions().

  python3 tools/e2e/make_local_e2e.py --duckdb <a DuckDB CLI that loads it> \
      --extension <the anofox_decide.duckdb_extension of v2026.11.24>

The committed suites keep the values captured that way; --gpu-from derives the GPU suite without any reference build.

The models must be downloaded (CALL decide_download(...)) into the default cache.
"""
import argparse
import csv
import io
import os
import subprocess

# id, text, refund (known), team (known), mood (known: 0 calm, 1 frustrated, 2 angry), lang
TICKETS = [
    (1, "I was charged twice for my March invoice, please refund the extra 49 EUR.", True, "billing", 1, "en"),
    (2, "The app crashes every time I open the export dialog on version 3.2.", False, "defect", 1, "en"),
    (3, "Thanks for the quick help yesterday, everything works now!", False, "other", 0, "en"),
    (4, "I want my money back, this product does not do what was advertised.", True, "billing", 2, "en"),
    (5, "Login button does nothing on Safari, error 500 in the console.", False, "defect", 1, "en"),
    (6, "Our invoice shows the wrong VAT rate; please issue a corrected one.", False, "billing", 0, "en"),
    (7, "Do you have an office in Munich? We would like to visit.", False, "other", 0, "en"),
    (8, "Cancel my subscription and reimburse the unused months.", True, "billing", 1, "en"),
    (9, "My parcel was supposed to arrive on Monday and the tracking has not moved for a week.", False, "shipping", 1, "en"),
    (10, "The package arrived crushed and the glass inside is broken. I need a replacement or my money back.", True, "shipping", 2, "en"),
    (11, "How do I add a second user to our team account?", False, "account", 0, "en"),
    (12, "I cannot reset my password, the reset email never arrives.", False, "account", 1, "en"),
    (13, "This is the third time I am writing. Nobody answers. Refund me now or I will dispute the charge.", True, "billing", 2, "en"),
    (14, "The dashboard shows yesterday's numbers twice since the last update.", False, "defect", 0, "en"),
    (15, "Could you send me the invoice for order 88231 as a PDF?", False, "billing", 0, "en"),
    (16, "You billed me after I cancelled in January. Please return the two payments.", True, "billing", 1, "en"),
    (17, "The CSV import silently drops every row that contains a comma in a quoted field.", False, "defect", 1, "en"),
    (18, "Please change the delivery address of my open order to our new warehouse.", False, "shipping", 0, "en"),
    (19, "Absolutely unacceptable. The device died after two days and support hung up on me.", True, "defect", 2, "en"),
    (20, "Is there a discount for non-profit organisations?", False, "other", 0, "en"),
    (21, "My account was locked after three login attempts, please unlock it, I have a demo in an hour.", False, "account", 1, "en"),
    (22, "I ordered the blue jacket but received a red one. I would like to return it for a full refund.", True, "shipping", 1, "en"),
    (23, "Great product, just wanted to say the new search is much faster.", False, "other", 0, "en"),
    (24, "The mobile app logs me out every few minutes on Android 14.", False, "defect", 1, "en"),
    (25, "We were charged the enterprise price but we are on the team plan. Please credit the difference.", True, "billing", 1, "en"),
    (26, "Where can I download the data processing agreement?", False, "account", 0, "en"),
    (27, "The courier left the box in the rain, everything is soaked. I am furious.", False, "shipping", 2, "en"),
    (28, "Please delete my account and all my data.", False, "account", 0, "en"),
    (29, "I paid for express shipping and it took twelve days. I expect the shipping fee back.", True, "shipping", 1, "en"),
    (30, "After the update the report PDF has empty pages.", False, "defect", 1, "en"),
    (31, "Hello, I am interested in a partnership, who can I talk to?", False, "other", 0, "en"),
    (32, "You took the money but the order was never confirmed. I want it refunded today.", True, "billing", 2, "en"),
    # multilingual (laya-multilingual reads these too; the English models answer them as best they can)
    (33, "Ich wurde zweimal belastet, bitte erstatten Sie mir den doppelten Betrag zurück.", True, "billing", 1, "de"),
    (34, "Die App stürzt beim Öffnen der Einstellungen jedes Mal ab.", False, "defect", 1, "de"),
    (35, "Mon colis n'est jamais arrivé et le suivi ne bouge plus depuis dix jours.", False, "shipping", 1, "fr"),
    (36, "Je veux être remboursé, le produit ne correspond pas à la description.", True, "billing", 2, "fr"),
    (37, "No puedo iniciar sesión en mi cuenta desde ayer.", False, "account", 1, "es"),
    (38, "Gracias por la ayuda, todo funciona perfectamente.", False, "other", 0, "es"),
    (39, "Il pacco è arrivato danneggiato, voglio il rimborso completo.", True, "shipping", 2, "it"),
    (40, "Kunnen jullie de factuur op onze nieuwe bedrijfsnaam zetten?", False, "billing", 0, "nl"),
]


# Long tickets: realistic multi-paragraph support threads, each built from distinct paragraphs.
PARAGRAPHS = [
    "Hello support team, I am writing on behalf of our finance department about the subscription we took out in March.",
    "Since then we have received three invoices for the same period, each with a slightly different amount, and our "
    "accountant cannot reconcile them with the bank statements.",
    "The first invoice listed twelve seats at the team price, the second one fourteen seats at the enterprise price, and "
    "the third one again twelve seats but with an additional charge for premium support that we never ordered.",
    "We tried to change the billing contact in the portal last week, but the page shows an error and the old contact "
    "keeps receiving all the emails, including the payment reminders.",
    "Our procurement policy requires that every invoice matches the purchase order exactly, so at the moment none of "
    "these invoices can be approved and your system keeps sending reminders that payment is overdue.",
    "In addition, two of our users report that the desktop application logs them out several times per hour since the "
    "latest update, which makes it very hard to finish the monthly reports on time.",
    "I have attached the purchase order and the three invoices to this message so that you can compare them yourself.",
    "Could you please check what went wrong, cancel the incorrect invoices, and send us one corrected invoice for the "
    "twelve seats at the team price that we actually ordered?",
    "If an amount was already collected by direct debit for the wrong invoices, we expect the difference to be credited "
    "back to our account before the end of the month.",
    "We have been happy customers for two years and would like to keep it that way, but this situation has cost our team "
    "several hours already and our CFO is asking whether we should look at other vendors.",
    "Please also let us know whether the logout problem is a known issue and when a fix can be expected, because the "
    "workaround suggested in your help center does not work for us.",
    "Thank you in advance for a quick reply; you can reach me by email or by phone during business hours.",
]


def long_ticket(n_paragraphs, offset):
    return " ".join(PARAGRAPHS[(offset + k) % len(PARAGRAPHS)] for k in range(n_paragraphs))


# id, text, refund, team, mood, lang: the long tickets are billing disputes with a credit request (refund-like)
LONG_TICKETS = [
    (101, long_ticket(5, 0), False, "billing", 1, "en"),
    (102, long_ticket(9, 3), True, "billing", 1, "en"),
    (103, long_ticket(12, 1), True, "billing", 2, "en"),
    (104, long_ticket(12, 7) + " " + long_ticket(6, 2), True, "billing", 2, "en"),
]
# julia-1 reads 8192 tokens: a ~3000-token thread (the Laya models read at most 1024)
JULIA_THREAD = (201, " ".join(long_ticket(12, k) for k in range(5)), True, "billing", 2, "en")

MODELS = ["julia-1", "laya-multilingual", "laya-typed-decisions"]
TEAMS = "['billing', 'defect', 'account', 'shipping', 'other']"
TEAM_LIST = ["billing", "defect", "account", "shipping", "other"]
MOODS = "['calm', 'frustrated', 'angry']"
QUESTIONS_JSON = ('[{"id": "refund", "kind": "binary", "instruction": "A refund is requested."},'
                  ' {"id": "team", "kind": "choice", "instruction": "Which team owns this?", "options": ["billing", "defect", "account", "shipping", "other"]},'
                  ' {"id": "mood", "kind": "score", "instruction": "How frustrated is the writer?", "levels": ["calm", "frustrated", "angry"]}]')
MARGIN = 1e-3


def sql_str(s):
    return "'" + s.replace("'", "''") + "'"


def tickets_sql(model):
    rows = TICKETS + LONG_TICKETS + ([JULIA_THREAD] if model == "julia-1" else [])
    vals = ",\n  ".join(f"({i}, {sql_str(t)}, {str(r).lower()}, '{team}', {mood}, '{lang}')" for i, t, r, team, mood, lang in rows)
    return f"CREATE OR REPLACE TABLE tickets AS SELECT * FROM (VALUES\n  {vals}\n) t(id, text, refund, team, mood, lang);", len(rows)


def answers_sql(model, table):
    return (f"CREATE OR REPLACE TABLE {table} AS SELECT id,\n"
            f"  decide_probability(text, 'A refund is requested.', model := '{model}') AS p_refund,\n"
            f"  decide_choice(text, 'Which team owns this?', {TEAMS}, model := '{model}') AS team,\n"
            f"  decide_choice_distribution(text, 'Which team owns this?', {TEAMS}, model := '{model}') AS team_dist,\n"
            f"  decide_score(text, 'How frustrated is the writer?', {MOODS}, model := '{model}') AS mood,\n"
            f"  decide_probability(text, 'The writer is angry.', model := '{model}') AS p_angry,\n"
            f"  decide_decision(text, 'The writer is angry.', 0.5, model := '{model}') AS angry\n"
            f"FROM tickets;")


def run(duckdb, ext, sql):
    # The reference build answers on the CPU, whatever GPU plugin is installed.
    full = f"LOAD '{ext}';\nSET anofox_decide_device = 'cpu';\n{sql}"
    env = dict(os.environ)
    out = subprocess.run([duckdb, "-unsigned", "-csv"], input=full, text=True, capture_output=True,
                         env={**env, "DATAZOO_DISABLE_TELEMETRY": "1"})
    if out.returncode != 0:
        raise SystemExit(out.stderr)
    return list(csv.DictReader(io.StringIO(out.stdout)))


def num(x):
    return repr(float(x))



def gpu_variant(cpu_test):
    """The same suite on a GPU plugin: the same tickets and the same ONNX Runtime answers, served by the device named in
    DECIDE_E2E_DEVICE (cuda, rocm or mlx; the plugin from DECIDE_E2E_EP_PATH), to the GPU bar: probabilities and scores within 1e-3,
    labels exact where the reference is not a near tie. Each model block proves the plugin served it (engine 'plugin:<device>')."""
    out = cpu_test

    def sub(a, b):
        nonlocal out
        assert out.count(a) == 1, a[:70]
        out = out.replace(a, b)

    sub("# name: test/sql/decide_local_e2e.test", "# name: test/sql/decide_local_e2e_gpu.test")
    sub("# description: End-to-end support-ticket triage on the three in-process catalog models",
        "# description: decide_local_e2e.test on a GPU plugin (DECIDE_E2E_DEVICE = cuda | rocm | mlx, the plugin in\n"
        "#              DECIDE_E2E_EP_PATH), to the GPU bar: probabilities and scores within 1e-3 of ONNX Runtime on the CPU, labels\n"
        "#              exact where the reference is not a near tie, and every block served by the plugin. Below: the CPU suite's text.\n"
        "#              End-to-end support-ticket triage on the three in-process catalog models")
    sub("#              Runs on the ggml engine on the CPU (anofox_decide_device = 'cpu') and checks that it, and only it, answered.\n",
        "#              Runs on the GPU plugin of DECIDE_E2E_DEVICE and checks that it, and only it, answered.\n")
    sub("require-env DECIDE_CACHE_DIR\n", "require-env DECIDE_E2E_DEVICE\n\nrequire-env DECIDE_E2E_EP_PATH\n\nrequire-env DECIDE_CACHE_DIR\n")
    sub("SET anofox_decide_device = 'cpu';\n",
        "SET anofox_decide_device = '${DECIDE_E2E_DEVICE}';\n\nstatement ok\nSET anofox_decide_ep_path = '${DECIDE_E2E_EP_PATH}';\n")
    sub("CREATE MACRO same_p(a, b) AS coalesce(abs(a - b) <= 1e-4 AND\n"
        "    (a <= 0 OR a >= 1 OR b <= 0 OR b >= 1 OR abs(ln(a / (1 - a)) - ln(b / (1 - b))) <= 2e-3), false);",
        "CREATE MACRO same_p(a, b) AS coalesce(abs(a - b) <= 1e-3, false);")
    sub("CREATE MACRO same_x(a, b) AS coalesce(abs(a - b) <= 1e-4, false);",
        "CREATE MACRO same_x(a, b) AS coalesce(abs(a - b) <= 1e-3, false);")
    sub("# the same probability: 1e-4 absolute and 2e-3 in log-odds (both sides strictly inside (0, 1)); NULL is never the same",
        "# the same probability on a GPU: 1e-3 absolute (the GPU bar); NULL is never the same")
    old = "SELECT count(*) FILTER (engine = 'ggml' AND runs > 0), count(*) FILTER (engine <> 'ggml')"
    new = ("SELECT count(*) FILTER (engine = 'plugin:${DECIDE_E2E_DEVICE}' AND runs > 0), count(*) FILTER (engine <> "
           "'plugin:${DECIDE_E2E_DEVICE}')")
    assert out.count(old) == 3, out.count(old)
    out = out.replace(old, new).replace("# the ggml engine answered, and only it", "# the GPU plugin answered, and only it")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--duckdb")
    ap.add_argument("--extension")
    ap.add_argument("--out", default="test/sql/decide_local_e2e.test")
    ap.add_argument("--gpu-out", default="test/sql/decide_local_e2e_gpu.test")
    ap.add_argument("--gpu-from", help="only derive --gpu-out from this existing CPU suite (no reference capture)")
    a = ap.parse_args()
    if a.gpu_from:
        with open(a.gpu_out, "w") as f:
            f.write(gpu_variant(open(a.gpu_from).read()))
        print(f"wrote {a.gpu_out}")
        return

    blocks = []
    for model in MODELS:
        t = "got_" + model.replace("-", "_")
        e = "exp_" + model.replace("-", "_")
        tsql, n = tickets_sql(model)
        cols = ", ".join(f"team_dist['{k}'] AS d_{k}" for k in TEAM_LIST)
        rows = run(a.duckdb, a.extension, tsql + "\n" + answers_sql(model, t) +
                   f"\nSELECT id, p_refund, team, {cols}, mood, p_angry, angry, (SELECT max(decide_token_count(text, '{model}')) FROM tickets) AS max_tokens FROM {t} ORDER BY id;")
        metrics = run(a.duckdb, a.extension, tsql + "\n" + answers_sql(model, t) +
                      f"\nSELECT round(decide_accuracy(g.p_refund, k.refund), 4) AS acc, round(avg((g.team = k.team)::INTEGER), 4) AS routing"
                      f" FROM {t} g JOIN tickets k USING (id);")[0]
        exp_rows, close_calls = [], 0
        for r in rows:
            d = sorted((float(r[f"d_{k}"]) for k in TEAM_LIST), reverse=True)
            team_clear = (d[0] - d[1]) > MARGIN
            angry_clear = abs(float(r["p_angry"]) - 0.5) > MARGIN
            close_calls += (not team_clear) + (not angry_clear)
            dist = ", ".join(num(r[f"d_{k}"]) for k in TEAM_LIST)
            exp_rows.append(f"({r['id']}, {num(r['p_refund'])}, '{r['team']}', {str(team_clear).lower()}, [{dist}], "
                            f"{num(r['mood'])}, {num(r['p_angry'])}, {r['angry']}, {str(angry_clear).lower()})")
        print(f"{model}: {n} tickets, longest {rows[0]['max_tokens']} tokens, {close_calls} close calls compared by probability only")
        blocks.append(f"""
# ---------------------------------------------------------------- {model} ({n} tickets, the longest {rows[0]['max_tokens']} tokens)

statement ok
{tsql}

statement ok
{answers_sql(model, t)}

statement ok
CREATE OR REPLACE TABLE {e} AS SELECT * FROM (VALUES
  {(","+chr(10)+"  ").join(exp_rows)}
) v(id, p_refund, team, team_clear, team_dist, mood, p_angry, angry, angry_clear);

# every answer matches the reference engine (NULL counts as a mismatch); labels where the reference has a clear margin
query IIIIII
SELECT count(*),
       count(*) FILTER (NOT same_p(g.p_refund, x.p_refund)),
       count(*) FILTER (NOT coalesce(g.team = x.team, false) AND x.team_clear),
       count(*) FILTER (NOT (same_p(g.team_dist['billing'], x.team_dist[1]) AND same_p(g.team_dist['defect'], x.team_dist[2])
                             AND same_p(g.team_dist['account'], x.team_dist[3]) AND same_p(g.team_dist['shipping'], x.team_dist[4])
                             AND same_p(g.team_dist['other'], x.team_dist[5]))),
       count(*) FILTER (NOT (same_x(g.mood, x.mood) AND same_p(g.p_angry, x.p_angry))),
       count(*) FILTER (NOT coalesce(g.angry = x.angry, false) AND x.angry_clear)
FROM {t} g FULL JOIN {e} x USING (id)
----
{n}	0	0	0	0	0

# what a user reads off it: refund accuracy at the default 0.5 threshold, routing accuracy, against the known labels
query II
SELECT round(decide_accuracy(g.p_refund, k.refund), 4), round(avg((g.team = k.team)::INTEGER), 4) FROM {t} g JOIN tickets k USING (id)
----
{metrics['acc']}	{metrics['routing']}

# several questions at once (decide_answers, one padded batch per text) give the scalar answers
query IIIIII
-- one row per question per ticket: a renamed question_id would make the filters below vacuous
SELECT count(*) FILTER (a.question_id = 'refund'), count(*) FILTER (a.question_id = 'team'), count(*) FILTER (a.question_id = 'mood'),
       count(*) FILTER (a.question_id = 'refund' AND NOT same_p(a.probability, g.p_refund)),
       count(*) FILTER (a.question_id = 'team' AND NOT coalesce(a.choice = g.team, false)),
       count(*) FILTER (a.question_id = 'mood' AND NOT same_x(a.score, g.mood))
FROM (SELECT id, unnest(decide_answers(text, '{QUESTIONS_JSON}', model := '{model}'), recursive := true) FROM tickets) a
JOIN {t} g USING (id)
----
{n}	{n}	{n}	0	0	0

# decide_many returns one JSON document per text with all three answers
query I
SELECT count(*) FROM (SELECT decide_many(text, '{QUESTIONS_JSON}', model := '{model}') AS m FROM tickets)
WHERE (length(m) - length(replace(m, '"kind"', ''))) // 6 = 3
----
{n}

# cross-row batching (texts padded to the length ladder, several per forward pass) answers the same, every kind
statement ok
SET anofox_decide_batch_tokens = 16384;

query IIII
SELECT count(*),
       count(*) FILTER (NOT same_p(b.p, g.p_refund)),
       count(*) FILTER (NOT coalesce(b.team = g.team, false)),
       count(*) FILTER (NOT same_x(b.mood, g.mood))
FROM (SELECT id, decide_probability(text, 'A refund is requested.', model := '{model}') AS p,
             decide_choice(text, 'Which team owns this?', {TEAMS}, model := '{model}') AS team,
             decide_score(text, 'How frustrated is the writer?', {MOODS}, model := '{model}') AS mood FROM tickets) b
JOIN {t} g USING (id)
----
{n}	0	0	0

statement ok
RESET anofox_decide_batch_tokens;

# the ggml engine answered, and only it
query II
SELECT count(*) FILTER (engine = 'ggml' AND runs > 0), count(*) FILTER (engine <> 'ggml')
FROM __anofox_decide_local_sessions() WHERE model LIKE 'weights:graph_{model}|%'
----
1	0
""")

    header = f"""# name: test/sql/decide_local_e2e.test
# description: End-to-end support-ticket triage on the three in-process catalog models (julia-1, laya-multilingual,
#              laya-typed-decisions) with their real weights, short tickets and long threads (past the 128-token
#              local-attention window): refund probability, team routing, frustration score, a thresholded decision,
#              several questions at once (decide_answers, decide_many) and cross-row batching. Every answer must match
#              the reference (ONNX Runtime 1.29 in decide v2026.11.24, captured by tools/e2e/make_local_e2e.py): probabilities to 1e-4 and
#              2e-3 in log-odds, scores to 1e-4, labels exactly where the reference is not a near tie.
#
#              Runs on the ggml engine on the CPU (anofox_decide_device = 'cpu') and checks that it, and only it, answered.
#              Needs the weights in the default cache:
#                CALL decide_download('julia-1'); CALL decide_download('laya-multilingual'); CALL decide_download('laya-typed-decisions');
#                DECIDE_REAL_WEIGHTS=1 ./build/release/test/unittest test/sql/decide_local_e2e.test
#              Generated file: edit tools/e2e/make_local_e2e.py and regenerate, do not edit by hand.
# group: [anofox_decide]

require anofox_decide

require-env DECIDE_REAL_WEIGHTS

require-env DECIDE_CACHE_DIR

statement ok
LOAD anofox_decide

statement ok
SET anofox_decide_device = 'cpu';

# DuckDB's unittest gives every test its own HOME, so the downloaded models are found through DECIDE_CACHE_DIR
# (scripts/e2e_local.sh sets it to ~/.cache/anofox-decide)
statement ok
SET anofox_decide_cache_dir = '${{DECIDE_CACHE_DIR}}';

# all three models are downloaded and verified (offline: a cached file is checked, not fetched)
query I
SELECT count(*) FROM (SELECT * FROM decide_download('julia-1') UNION ALL SELECT * FROM decide_download('laya-multilingual')
                      UNION ALL SELECT * FROM decide_download('laya-typed-decisions')) WHERE status = 'cached'
----
8

# the same probability: 1e-4 absolute and 2e-3 in log-odds (both sides strictly inside (0, 1)); NULL is never the same
statement ok
CREATE MACRO same_p(a, b) AS coalesce(abs(a - b) <= 1e-4 AND
    (a <= 0 OR a >= 1 OR b <= 0 OR b >= 1 OR abs(ln(a / (1 - a)) - ln(b / (1 - b))) <= 2e-3), false);

statement ok
CREATE MACRO same_x(a, b) AS coalesce(abs(a - b) <= 1e-4, false);
"""
    with open(a.out, "w") as f:
        f.write(header + "".join(blocks))
    print(f"wrote {a.out}")
    with open(a.gpu_out, "w") as f:
        f.write(gpu_variant(header + "".join(blocks)))
    print(f"wrote {a.gpu_out}")


if __name__ == "__main__":
    main()
