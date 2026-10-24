#!/usr/bin/env python3
"""Does anofox-visualization pick up anofox_decide's contract producers? (#34)

anofox-visualization plots any table that follows the anofox integration
contract (anofox-visualization docs/plans/integration-contract.md): its
`anofox_plot(tbl)` picks the chart from the columns, and producers advertise
themselves with `duckdb_functions()` tags
`{'anofox.contract': '1', 'anofox.output': '<schema>'}`. This script is the
definition of done for decide's producers. It loads the decide artifact that
was just built next to a released anofox_visualization in a stock DuckDB CLI
and checks, each in a fresh process:

  1. both extensions load in either order, and `anofox_contract_version()`
     (registered by whichever loads first) exists once and returns '1';
  2. discovery: the tagged decide producers are exactly EXPECTED_PRODUCERS,
     and the visualization has a plot macro for each one's schema (and, when
     its build carries `anofox.consumes` tags, declares that schema);
  3. every producer's output renders: `anofox_plot(t)` is an SVG with no
     render warning, and it is the same SVG as the schema's own macro
     (`anofox_plot_curve`, `anofox_plot_summary`), so the auto-detection chose
     that schema; every model is drawn;
  4. the ROC legend's AUC (computed by the visualization from the curve)
     equals `decide_auc`, and the calibration curve's x spans P(yes);
  5. probabilities scored by decide itself (the offline `stub` model under
     Platt calibrations) plot the same way;
  6. with --statistics, decide's curves UNION with anofox-statistics'
     `roc_curve_by` into one plot.

    python3 scripts/visualization_pickup.py \\
        build/release/extension/anofox_decide/anofox_decide.duckdb_extension

The DuckDB CLI is downloaded (--duckdb-version, default v1.5.6) unless
--duckdb names one. The visualization comes from its project repository
(--viz-repo or VIZ_REPO; `community` for the community repository).
Telemetry is off. Exit status 1 when any check fails.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from smoke_test import _download_duckdb_cli  # noqa: E402  # pyright: ignore[reportMissingImports]

# The tagged producers this extension ships: function -> contract schema.
# Kept apart from the C++ list (src/decide_contract.cpp) on purpose, so a tag
# that goes missing or appears unplanned fails here.
EXPECTED_PRODUCERS = {
    name: schema
    for base, schema in (("roc_curve_by", "curve"), ("calibration_curve_by", "curve"), ("metrics_by", "summary"))
    for name in (f"decide_{base}", f"anofox_decide_{base}")
}

DEFAULT_VIZ_REPO = "https://datazoode.github.io/anofox-visualization/extensions"

# Two models with spread, a tied score and NULL rows (skipped by the producers).
FIXTURE = """
CREATE TABLE scored AS SELECT * FROM (VALUES
    ('a', 0.1, false), ('a', 0.4, false), ('a', 0.35, true), ('a', 0.8, true),
    ('a', 0.8, false), ('a', 0.62, true), ('a', 0.05, false), ('a', 0.93, true), ('a', NULL, true),
    ('b', 0.2, false), ('b', 0.2, true), ('b', 0.7, true), ('b', 0.55, false), ('b', 0.9, true),
    ('b', 0.3, false), ('b', 0.5, NULL)
) t(model, p, y);
"""

# Probabilities scored by decide itself, offline: five calibrated stubs s0 .. s4 answer
# sigmoid(-1.8) .. sigmoid(1.8); a row's stub follows its label plus a model-specific offset.
STUB_FIXTURE = """
SELECT bool_and(decide_register_model('s' || k, 'stub', MAP {'calibration': 'platt:1,' || ((k - 2) * 0.9)::VARCHAR}))
FROM range(5) r(k);
CREATE TABLE stub_scored AS
SELECT m AS model, refund AS y,
       decide_probability('ticket ' || i, 'A refund is requested.',
                          model := 's' || least(4, CASE WHEN refund THEN 2 ELSE 0 END + (i * o) % 3)) AS p
FROM (SELECT i, i % 3 = 0 AS refund FROM range(1, 61) t(i)), (VALUES ('fixture-a', 1), ('fixture-b', 2)) m(m, o);
"""

PRODUCER_CALLS = {
    "decide_roc_curve_by": "decide_roc_curve_by('{src}', model, p, y)",
    "decide_calibration_curve_by": "decide_calibration_curve_by('{src}', model, p, y, n_bins := 5)",
    "decide_metrics_by": "decide_metrics_by('{src}', model, p, y)",
}


class Pickup:
    def __init__(self, duckdb: str, ext: str, viz_repo: str, home: str):
        self.duckdb, self.ext, self.viz_repo, self.home = duckdb, ext, viz_repo, home
        self.failures: list[str] = []

    def run(self, sql: str, load: str = "decide,viz") -> tuple[bool, str, list[dict]]:
        """`sql` in a fresh `duckdb -unsigned -json` process after the LOADs → (ok, output, last result rows)."""
        names = {"decide": "anofox_decide", "viz": "anofox_visualization"}
        loads = "".join(f"LOAD {names[x]};\n" for x in load.split(",") if x)
        env = dict(os.environ, HOME=self.home, USERPROFILE=self.home, DATAZOO_DISABLE_TELEMETRY="1")
        p = subprocess.run([self.duckdb, "-unsigned", "-json", "-c", loads + sql],
                           capture_output=True, text=True, env=env, timeout=600)
        stdout = p.stdout or ""
        rows: list[dict] = []
        dec, i = json.JSONDecoder(), 0
        while (j := stdout.find("[", i)) >= 0:
            try:
                parsed, i = dec.raw_decode(stdout, j)
                if isinstance(parsed, list):
                    rows = parsed
            except ValueError:
                i = j + 1
        return p.returncode == 0 and not (p.stderr or "").strip(), stdout + (p.stderr or ""), rows

    def check(self, name: str, ok: bool, detail: str = "") -> bool:
        print(f"  {'ok  ' if ok else 'FAIL'} {name}" + (f"\n       {detail}" if detail and not ok else ""))
        if not ok:
            self.failures.append(name)
        return ok

    def install(self) -> None:
        src = "community" if self.viz_repo == "community" else f"'{self.viz_repo}'"
        ok, out, _ = self.run(f"INSTALL '{self.ext}';\nFORCE INSTALL anofox_visualization FROM {src};", load="")
        if not ok:
            raise SystemExit(f"Visualization pickup FAILED: could not install the extensions\n{out}")
        _, _, rows = self.run("SELECT extension_name, extension_version FROM duckdb_extensions() "
                              "WHERE extension_name IN ('anofox_decide', 'anofox_visualization') ORDER BY 1;")
        for r in rows:
            print(f"  {r['extension_name']} {r['extension_version']}")

    # ── 1 ──
    def load_order(self) -> None:
        print("[1] loading in either order, one shared anofox_contract_version()")
        for order in ("decide,viz", "viz,decide"):
            ok, out, rows = self.run("SELECT count(*) AS n, any_value(anofox_contract_version()) AS v "
                                     "FROM duckdb_functions() WHERE function_name = 'anofox_contract_version';",
                                     load=order)
            self.check(f"LOAD {order.replace(',', ' then ')}", ok and rows == [{"n": 1, "v": "1"}],
                       out.strip()[-500:])

    # ── 2 ──
    def discovery(self) -> None:
        print("[2] discovery through duckdb_functions() tags")
        ok, out, rows = self.run("SELECT function_name AS f, tags['anofox.output'] AS s FROM duckdb_functions() "
                                 "WHERE tags['anofox.contract'] = '1' AND (function_name LIKE 'decide\\_%' ESCAPE '\\' "
                                 "OR function_name LIKE 'anofox\\_decide\\_%' ESCAPE '\\');")
        found = {r["f"]: r["s"] for r in rows}
        self.check("tagged decide producers are exactly the expected ones", ok and found == EXPECTED_PRODUCERS,
                   f"expected {EXPECTED_PRODUCERS}\n       found    {found}\n       {out.strip()[-300:]}")
        ok, out, rows = self.run("SELECT function_name AS f, tags['anofox.consumes'] AS c FROM duckdb_functions() "
                                 "WHERE function_name LIKE 'anofox\\_plot%' ESCAPE '\\';")
        macros = {r["f"]: r["c"] for r in rows}
        consumes = macros.get("anofox_plot") or ""
        for schema in sorted(set(EXPECTED_PRODUCERS.values())):
            self.check(f"the visualization has anofox_plot_{schema}", ok and f"anofox_plot_{schema}" in macros, out)
            if consumes:  # the community (C++) build tags its consumers; the C-API build has no tags
                self.check(f"anofox_plot declares it consumes '{schema}'", schema in consumes.split(","), consumes)

    # ── 3 / 4 / 5 ──
    def render(self, src: str, setup: str, label: str) -> None:
        print(f"[{label}] rendering every producer's output ({src})")
        for producer, schema in (("decide_roc_curve_by", "curve"), ("decide_calibration_curve_by", "curve"),
                                 ("decide_metrics_by", "summary")):
            call = PRODUCER_CALLS[producer].format(src=src)
            ok, out, rows = self.run(
                f"{setup}\nCREATE TABLE t AS SELECT * FROM {call};\n"
                f"SELECT anofox_plot(t) AS auto, anofox_plot_{schema}(t) AS specific, "
                f"(SELECT list(DISTINCT model_id ORDER BY model_id) FROM t) AS models FROM t;")
            if not self.check(f"{producer}: anofox_plot renders", ok and len(rows) == 1, out.strip()[-600:]):
                continue
            auto, specific, models = rows[0]["auto"] or "", rows[0]["specific"], rows[0]["models"]
            self.check(f"{producer}: an SVG", auto.startswith("<svg"), auto[:200])
            warn = re.search(r'data-warnings="([^"]*)"', auto)
            self.check(f"{producer}: no render warnings", warn is None, warn.group(1) if warn else "")
            self.check(f"{producer}: auto-detected as '{schema}' (same SVG as anofox_plot_{schema})",
                       auto == specific)
            if schema == "curve":
                series = re.findall(r'data-series="([^"]*)"', auto)
                drawn = sorted({s.split(" · ")[0] for s in series})
                self.check(f"{producer}: every model drawn as a series", drawn == models,
                           f"models {models}, series {series}")
            else:
                self.check(f"{producer}: every model drawn", all(f">{m}<" in auto for m in models), str(models))

        roc = PRODUCER_CALLS["decide_roc_curve_by"].format(src=src)
        ok, out, rows = self.run(
            f"{setup}\nCREATE TABLE t AS SELECT * FROM {roc};\n"
            f"SELECT anofox_plot(t) AS svg, (SELECT list(model || ' · AUC ' || printf('%.2f', a) ORDER BY model) "
            f"FROM (SELECT model, decide_auc(p, y) AS a FROM {src} GROUP BY model)) AS want FROM t;")
        if ok and rows:
            series = sorted(set(re.findall(r'data-series="([^"]*)"', rows[0]["svg"])))
            self.check("ROC legend AUC (the visualization's) equals decide_auc", series == rows[0]["want"],
                       f"svg {series}, decide {rows[0]['want']}")
        else:
            self.check("ROC legend AUC (the visualization's) equals decide_auc", False, out.strip()[-400:])
        cal = PRODUCER_CALLS["decide_calibration_curve_by"].format(src=src)
        ok, out, rows = self.run(f"{setup}\nSELECT min(x) < 0.5 AS spans FROM {cal};")
        self.check("calibration x spans P(yes), not only the 0.5 .. 1 confidence", ok and rows == [{"spans": True}],
                   out.strip()[-300:])

    # ── 6 ──
    def statistics(self, repo: str) -> None:
        print("[6] decide and anofox-statistics curves in one plot")
        src = "community" if repo == "community" else f"'{repo}'"
        ok, out, _ = self.run(f"FORCE INSTALL anofox_statistics FROM {src};", load="")
        if not self.check("install anofox_statistics", ok, out.strip()[-300:]):
            return
        ok, out, rows = self.run(
            f"LOAD anofox_statistics;\n{FIXTURE}\n"
            "CREATE TABLE t AS SELECT * EXCLUDE (model) REPLACE ('decide ' || model_id AS model_id) "
            "FROM decide_roc_curve_by('scored', model, p, y) UNION ALL "
            "SELECT * EXCLUDE (model) REPLACE ('statistics ' || model_id AS model_id) "
            "FROM roc_curve_by('scored', model, y, p);\n"
            "SELECT anofox_plot(t) AS svg FROM t;")
        svg = rows[0]["svg"] if ok and rows else ""
        self.check("UNION of decide_roc_curve_by and roc_curve_by plots",
                   svg.startswith("<svg") and "data-warnings" not in svg, out.strip()[-500:])


def main() -> int:
    ap = argparse.ArgumentParser(description="anofox-visualization picks up anofox_decide's contract producers")
    ap.add_argument("extension", help="the built anofox_decide.duckdb_extension")
    ap.add_argument("--duckdb", help="DuckDB CLI to use (default: download the official one)")
    ap.add_argument("--duckdb-version", default="v1.5.6")
    ap.add_argument("--arch", default="linux_amd64")
    ap.add_argument("--viz-repo", default=os.environ.get("VIZ_REPO", DEFAULT_VIZ_REPO),
                    help="extension repository of anofox_visualization ('community' for the community one)")
    ap.add_argument("--statistics", metavar="REPO", help="also check a UNION with anofox_statistics from REPO")
    args = ap.parse_args()

    ext = os.path.abspath(args.extension).replace("\\", "/")
    if not os.path.isfile(ext):
        print(f"Visualization pickup FAILED: artifact not found: {ext}")
        return 1
    with tempfile.TemporaryDirectory() as tmp:
        duckdb = args.duckdb or _download_duckdb_cli(args.duckdb_version, args.arch, tmp)
        home = os.path.join(tmp, "home")
        os.makedirs(home)
        print(f"\nVisualization pickup\n  decide : {ext}\n  duckdb : {duckdb}\n  viz    : {args.viz_repo}")
        t = Pickup(duckdb, ext, args.viz_repo, home)
        t.install()
        t.load_order()
        t.discovery()
        t.render("scored", FIXTURE, "3-4")
        t.render("stub_scored", STUB_FIXTURE, "5")
        if args.statistics:
            t.statistics(args.statistics)
    if t.failures:
        print(f"\nVisualization pickup FAILED ({len(t.failures)}):\n  " + "\n  ".join(t.failures))
        return 1
    print("\nVisualization pickup PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
