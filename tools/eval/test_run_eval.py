#!/usr/bin/env python3
"""Tests for tools/eval/run_eval.py --device. They need no GPU, no DuckDB build and no network:

  * device_refusal(): the policy, case by case;
  * main() end to end against a stub `duckdb` executable that writes what the real session would write, so the
    flow (SET anofox_decide_device, the serving-device query, refusal, no report, exit status) is exercised.
    The stub reports the serving device it is told to through STUB_SERVING, and a hard error like the extension's
    when STUB_FAIL is set (an explicit GPU device with no plugin installed).

Run: python3 -m unittest tools/eval/test_run_eval.py   (or: python3 tools/eval/test_run_eval.py)
"""
import json
import os
import re
import stat
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import run_eval  # noqa: E402

STUB = '''#!/usr/bin/env python3
import os, re, sys
sql = sys.argv[-1]
m = re.search(r"SET anofox_decide_device = '(\\w+)'", sql)
device = m.group(1) if m else None
if os.environ.get("STUB_FAIL") and device not in (None, "auto", "cpu"):
    sys.stderr.write("Invalid Input Error: decide_answers: anofox_decide_device is '%s' but no '%s' plugin is "
                     "installed. Fix: CALL decide_accelerate();\\n" % (device, device))
    sys.exit(1)
res = re.search(r"COPY \\(\\s*SELECT id::INTEGER.*?TO '([^']+)'", sql, re.S).group(1)
with open(res, "w") as f:
    f.write("id,qid,probability,choice\\n")
    for i, (refund, route) in enumerate([("true", "billing"), ("false", "orders")]):
        f.write("%d,refund,%s,\\n" % (i, 0.9 if refund == "true" else 0.1))
        f.write("%d,routing,,%s\\n" % (i, route))
sv = re.search(r"decide_models\\(\\) WHERE model = '[^']*'\\) TO '([^']+)'", sql)
if sv:
    with open(sv.group(1), "w") as f:
        f.write("device\\n" + os.environ.get("STUB_SERVING", "") + "\\n")
'''

TICKETS = "id,text,refund,routing\n0,I want my money back,true,billing\n1,Where is my parcel,false,orders\n"


class DeviceRefusal(unittest.TestCase):
    def test_default_and_auto_accept_anything(self):
        for requested in (None, "auto"):
            for serving in ("cpu", "cuda:0", "", None):
                self.assertIsNone(run_eval.device_refusal(requested, serving))

    def test_cpu_request(self):
        self.assertIsNone(run_eval.device_refusal("cpu", "cpu"))
        self.assertIn("served on 'cuda:0'", run_eval.device_refusal("cpu", "cuda:0"))

    def test_gpu_request_served_by_that_gpu(self):
        for dev, serving in (("cuda", "cuda:0"), ("rocm", "rocm:0"), ("mlx", "mlx")):
            self.assertIsNone(run_eval.device_refusal(dev, serving))

    def test_gpu_request_that_fell_back_to_cpu_is_refused(self):
        why = run_eval.device_refusal("cuda", "cpu")
        self.assertIn("fell back to the CPU", why)

    def test_gpu_request_served_by_another_gpu_is_refused(self):
        self.assertIn("served on 'rocm:0'", run_eval.device_refusal("cuda", "rocm:0"))

    def test_no_device_at_all_is_refused_for_an_explicit_request(self):
        for serving in ("", None):
            for requested in ("cpu", "cuda", "rocm", "mlx"):
                self.assertIn("empty", run_eval.device_refusal(requested, serving))


class EndToEnd(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        tmp = Path(self.tmp.name)
        self.stub = tmp / "duckdb"
        self.stub.write_text(STUB)
        self.stub.chmod(self.stub.stat().st_mode | stat.S_IXUSR)
        self.tickets = tmp / "tickets.csv"
        self.tickets.write_text(TICKETS)
        self.out = tmp / "out"

    def tearDown(self):
        self.tmp.cleanup()

    def run_eval(self, *extra, **env):
        e = {k: v for k, v in os.environ.items() if not k.startswith("STUB_")}
        e.update(env)
        return subprocess.run(
            [sys.executable, str(HERE / "run_eval.py"), "--tickets", str(self.tickets), "--out", str(self.out),
             "--duckdb", str(self.stub), "--extension", "unused.duckdb_extension", "--models", "laya-multilingual",
             *extra], capture_output=True, text=True, env=e)

    def test_default_run_is_unchanged(self):
        p = self.run_eval()
        self.assertEqual(p.returncode, 0, p.stderr)
        report = json.loads((self.out / "report.json").read_text())
        self.assertNotIn("serving_device", report["laya-multilingual"])
        self.assertNotIn("Device", (self.out / "report.md").read_text())

    def test_cpu_device_is_accepted_and_recorded(self):
        p = self.run_eval("--device", "cpu", STUB_SERVING="cpu")
        self.assertEqual(p.returncode, 0, p.stderr)
        r = json.loads((self.out / "report.json").read_text())["laya-multilingual"]
        self.assertEqual((r["requested_device"], r["serving_device"]), ("cpu", "cpu"))
        self.assertIsNotNone(r["wall_seconds"])
        self.assertIn("| Laya multilingual (catalog) | cpu |", (self.out / "report.md").read_text())

    def test_gpu_device_without_plugin_is_refused_and_nothing_is_reported(self):
        p = self.run_eval("--device", "cuda", STUB_FAIL="1")
        self.assertEqual(p.returncode, 2, p.stderr)
        self.assertIn("REFUSED laya-multilingual", p.stderr)
        self.assertFalse((self.out / "report.md").exists())
        self.assertFalse((self.out / "report.json").exists())
        self.assertFalse((self.out / "results_laya-multilingual.csv").exists())

    def test_gpu_request_served_on_cpu_is_refused(self):
        p = self.run_eval("--device", "mlx", STUB_SERVING="cpu")
        self.assertEqual(p.returncode, 2, p.stderr)
        self.assertIn("fell back to the CPU", p.stderr)
        self.assertFalse((self.out / "report.md").exists())
        self.assertFalse((self.out / "results_laya-multilingual.csv").exists())

    def test_gpu_request_served_on_that_gpu_is_reported(self):
        p = self.run_eval("--device", "cuda", STUB_SERVING="cuda:0")
        self.assertEqual(p.returncode, 0, p.stderr)
        r = json.loads((self.out / "report.json").read_text())["laya-multilingual"]
        self.assertEqual(r["serving_device"], "cuda:0")

    def test_device_with_analyse_only_is_rejected(self):
        p = self.run_eval("--device", "cpu", "--analyse-only")
        self.assertNotEqual(p.returncode, 0)
        self.assertIn("--device needs scoring", p.stderr)

    def test_stale_results_are_not_reported_after_a_refusal(self):
        self.out.mkdir()
        (self.out / "results_laya-multilingual.csv").write_text("id,qid,probability,choice\n")
        p = self.run_eval("--device", "rocm", STUB_SERVING="cpu")
        self.assertEqual(p.returncode, 2, p.stderr)
        self.assertFalse((self.out / "results_laya-multilingual.csv").exists())



class RefusalTest(unittest.TestCase):
    """A refused answer (empty probability / choice in the results) is scored as wrong and reported."""

    def test_refusal_counts_as_wrong_and_is_listed(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            (d / "tickets.csv").write_text(
                "id,text,refund,routing\n1,a,true,billing\n2,b,false,orders\n3,c,true,billing\n4,d,false,other\n")
            (d / "results_m.csv").write_text(
                "id,qid,probability,choice\n"
                "1,refund,,\n1,routing,,billing\n2,refund,0.1,\n2,routing,,orders\n"
                "3,refund,0.9,\n3,routing,,billing\n4,refund,0.2,\n4,routing,,\n")
            report, truth = run_eval.analyse(d / "tickets.csv", d, ["m"], {"m": "Model"})
            r = report["m"]
            self.assertEqual(r["refusals"], {"refund": [1], "routing": [4]})
            self.assertAlmostEqual(r["refund_acc"], 0.75)   # ticket 1 refused -> wrong
            self.assertAlmostEqual(r["routing_acc"], 0.75)  # ticket 4 refused -> wrong
            md = run_eval.render(report, truth, {})
            self.assertIn("## Refusals", md)
            self.assertIn("| Model | 1 | 1 | 1, 4 |", md)

if __name__ == "__main__":
    unittest.main()
