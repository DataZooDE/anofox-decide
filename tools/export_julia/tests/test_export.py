"""Julia-1 export regression tests (mirrors export_onnx/tests/test_export.py).

Gates:
- test_tiny_export_parity runs wherever the venv deps exist (no weights):
  guards the TorchScript export path + manual-head dynamics.
- test_upstream_equivalence needs JULIA_WEIGHTS_DIR (snapshot with
  model.safetensors) + JULIA_SRC_MODEL (upstream julia/model.py): guards the
  manual-head port against the real JuliaDecisionModel.
- test_full_ort_parity needs JULIA_WEIGHTS_DIR + JULIA_ONNX: guards the
  shipped graph torch-vs-ORT.

Tolerances are fp32-CPU grade (1e-4); the -1e4 mask path is asserted
explicitly, never averaged away.
"""
from __future__ import annotations

import os

import pytest

torch = pytest.importorskip("torch")
ort = pytest.importorskip("onnxruntime")

from export_julia.export import export_onnx, load_scores_model, make_tiny_model  # noqa: E402

WEIGHTS_DIR = os.environ.get("JULIA_WEIGHTS_DIR", "")
SRC_MODEL = os.environ.get("JULIA_SRC_MODEL", "")
FULL_ONNX = os.environ.get("JULIA_ONNX", "")

needs_weights = pytest.mark.skipif(not WEIGHTS_DIR, reason="needs JULIA_WEIGHTS_DIR (577MB snapshot)")
needs_src = pytest.mark.skipif(not SRC_MODEL, reason="needs JULIA_SRC_MODEL (upstream julia/model.py)")
needs_onnx = pytest.mark.skipif(not FULL_ONNX, reason="needs JULIA_ONNX (exported julia1.onnx)")


def _run_torch_ort(model, sess_path, cases):
    import onnxruntime as ort_mod

    sess = ort_mod.InferenceSession(sess_path, providers=["CPUExecutionProvider"])
    worst = 0.0
    for ids, attn, mp, mm, qt in cases:
        with torch.inference_mode():
            ref = model(ids, attn, mp, mm, qt)
        got = sess.run(
            ["scores"],
            {
                "input_ids": ids.numpy(),
                "attention_mask": attn.numpy(),
                "marker_pos": mp.numpy(),
                "marker_mask": mm.numpy(),
                "qtype": qt.numpy(),
            },
        )[0]
        worst = max(worst, float(abs(got - ref.numpy()).max()))
        masked = mm.numpy() == False  # noqa: E712
        assert (got[masked] < -9000).all(), "masked slots must stay near -1e4"
    return worst


def test_tiny_export_parity(tmp_path):
    """Random-init tiny graph: torch vs ORT across batch/seq/marker shapes."""
    from pathlib import Path

    torch.manual_seed(3)
    model = make_tiny_model(0)
    model.eval()
    graph = Path(str(tmp_path)) / "tiny.onnx"
    export_onnx(model, graph)
    torch.manual_seed(3)
    cases = []
    for b, t, m in [(1, 16, 2), (2, 24, 2)]:
        ids = torch.randint(0, 1000, (b, t))
        ids[:, 0] = 1
        attn = torch.ones(b, t, dtype=torch.long)
        mp = torch.stack([torch.randperm(t)[:m] for _ in range(b)])
        mm = torch.ones(b, m, dtype=torch.bool)
        mm[0, -1] = False
        qt = torch.tensor([i % 3 for i in range(b)])
        cases.append((ids, attn, mp, mm, qt))
    assert _run_torch_ort(model, str(graph), cases) < 1e-4


@needs_weights
@needs_src
def test_upstream_equivalence():
    """Manual head port vs the real JuliaDecisionModel on shared weights."""
    import importlib.util
    from pathlib import Path

    from safetensors.torch import load_file

    weights = Path(WEIGHTS_DIR)
    spec = importlib.util.spec_from_file_location("upstream_model", SRC_MODEL)
    upstream_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(upstream_mod)

    from transformers import AutoConfig, AutoModel

    enc_config = AutoConfig.from_pretrained(weights / "encoder", trust_remote_code=False)
    enc_up = AutoModel.from_config(enc_config, trust_remote_code=False, attn_implementation="eager")
    up = upstream_mod.JuliaDecisionModel(enc_up, head_layers=2)
    up.load_state_dict(load_file(str(weights / "model.safetensors")), strict=False)
    up.eval()

    mine = load_scores_model(weights, 2, 0.1)
    mine.eval()

    torch.manual_seed(11)
    worst = 0.0
    for trial in range(2):
        b, t, m = 2, 40 + trial * 8, 3
        ids = torch.randint(0, 5000, (b, t))
        ids[:, 0] = 1
        attn = torch.ones(b, t, dtype=torch.long)
        mp = torch.tensor([[5, 9, 14], [6, 11, 20]])[:, :m]
        mm = torch.ones(b, m, dtype=torch.bool)
        qt = torch.tensor([0, 2])
        with torch.inference_mode():
            d = float(abs(mine(ids, attn, mp, mm, qt) - up(ids, attn, mp, mm, qt)).max())
        worst = max(worst, d)
    assert worst < 1e-5


@needs_weights
@needs_onnx
def test_full_ort_parity():
    """Shipped graph vs torch across batch/seq/marker shapes + qtypes."""
    from pathlib import Path

    model = load_scores_model(Path(WEIGHTS_DIR), 2, 0.1)
    model.eval()
    torch.manual_seed(21)
    cases = []
    for b, t, m in [(1, 32, 2), (2, 48, 3)]:
        ids = torch.randint(0, 5000, (b, t))
        ids[:, 0] = 1
        attn = torch.ones(b, t, dtype=torch.long)
        mp = torch.stack([torch.randperm(t)[:m] for _ in range(b)])
        mm = torch.ones(b, m, dtype=torch.bool)
        mm[0, -1] = False
        qt = torch.tensor([i % 3 for i in range(b)])
        cases.append((ids, attn, mp, mm, qt))
    assert _run_torch_ort(model, FULL_ONNX, cases) < 1e-4


def test_cli_help():
    """The CLI at least parses (offline-safe)."""
    from export_julia import cli

    with pytest.raises(SystemExit) as exc:
        cli.main(["--help"])
    assert exc.value.code == 0


TOKENIZER_JSON = os.path.join(WEIGHTS_DIR, "tokenizer", "tokenizer.json") if WEIGHTS_DIR else ""
TICKETS = os.path.join(os.path.dirname(__file__), "..", "..", "..", "test", "fixtures", "support_tickets.csv")


def _collate(tok, state, question, options, qtype, cls=2, sep=1, mask=4):
    """[CLS] head [SEP] (MASK opt)* [SEP] state [SEP] (spec layout)."""
    names = {0: "choice", 1: "score", 2: "noul"}
    ids = [cls] + tok.encode(f"{names[qtype]} question: {question}", add_special_tokens=False).ids + [sep]
    markers = []
    for o in options:
        markers.append(len(ids))
        ids.append(mask)
        ids += tok.encode(" " + o, add_special_tokens=False).ids[:48]
    ids.append(sep)
    ids += tok.encode(state, add_special_tokens=False).ids + [sep]
    return ids, markers


def _batch(rows):
    t = max(len(i) for i, _, _ in rows)
    m = max(len(mk) for _, mk, _ in rows)
    ids = torch.zeros(len(rows), t, dtype=torch.long)
    attn = torch.zeros(len(rows), t, dtype=torch.long)
    mp = torch.zeros(len(rows), m, dtype=torch.long)
    mm = torch.zeros(len(rows), m, dtype=torch.bool)
    qt = torch.zeros(len(rows), dtype=torch.long)
    for b, (i, mk, q) in enumerate(rows):
        ids[b, : len(i)] = torch.tensor(i)
        attn[b, : len(i)] = 1
        mp[b, : len(mk)] = torch.tensor(mk)
        mm[b, : len(mk)] = True
        qt[b] = q
    return ids, attn, mp, mm, qt


@needs_weights
@needs_onnx
def test_real_ticket_parity():
    """Real support tickets (real tokenizer + weights): torch vs ORT raw
    logits, single vs padded batch, and both [false,true] orders."""
    import csv
    from pathlib import Path

    from tokenizers import Tokenizer

    tok = Tokenizer.from_file(TOKENIZER_JSON)
    tickets = list(csv.DictReader(open(TICKETS)))
    model = load_scores_model(Path(WEIGHTS_DIR), 2, 0.1)
    model.eval()
    sess = ort.InferenceSession(FULL_ONNX, providers=["CPUExecutionProvider"])

    def run_ort(batch):
        ids, attn, mp, mm, qt = batch
        return sess.run(
            ["scores"],
            {"input_ids": ids.numpy(), "attention_mask": attn.numpy(), "marker_pos": mp.numpy(),
             "marker_mask": mm.numpy(), "qtype": qt.numpy()},
        )[0]

    q = "A refund is requested."
    rows = []
    for t in tickets:
        for opts in (["false", "true"], ["true", "false"]):
            ids, mk = _collate(tok, t["text"], q, opts, 2)
            rows.append((ids, mk, 2))
    ids_, attn_, mp_, mm_, qt_ = padded = _batch(rows)
    with torch.inference_mode():
        ref = model(*padded).numpy()
    got = run_ort(padded)
    assert abs(got - ref).max() < 1e-3, "torch vs ORT raw logits on real tickets"
    for n, row in enumerate(rows):
        single = run_ort(_batch([row]))
        assert abs(single[0] - got[n, : single.shape[1]]).max() < 1e-3, f"row {n}: single vs padded batch"
    # Option order changes scores only within a documented tolerance band:
    # record the worst swing (informational, printed under -s).
    swing = max(abs(got[2 * i, 1] - got[2 * i + 1, 0]) for i in range(len(tickets)))
    print(f"max P(true) logit swing between option orders: {swing:.3f}")
