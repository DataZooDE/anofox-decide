"""Weight-free export: tiny random-init checkpoint, no real weights needed."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("onnxruntime")
onnx = pytest.importorskip("onnx")

from safetensors.torch import save_file  # noqa: E402

from export_julia.export import make_tiny_model  # noqa: E402
from export_julia.weight_free import (  # noqa: E402
    assert_weight_free,
    checkpoint_f32,
    export_weight_free,
    inject,
)

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


def _feed(b, t, m, seed):
    g = np.random.default_rng(seed)
    ids = g.integers(5, 900, (b, t)).astype(np.int64)
    mp = np.stack([g.permutation(t)[:m] for _ in range(b)]).astype(np.int64)
    return {
        "input_ids": ids,
        "attention_mask": np.ones((b, t), np.int64),
        "marker_pos": mp,
        "marker_mask": np.ones((b, m), bool),
        "qtype": np.arange(b, dtype=np.int64) % 3,
    }


@pytest.fixture(scope="module")
def tiny(tmp_path_factory):
    d = tmp_path_factory.mktemp("wf")
    model = make_tiny_model(0).eval()
    # fp16 on disk like the Laya checkpoints: the map must work on the upcast values.
    sd = {k: v.detach().half().contiguous() for k, v in model.state_dict().items()}
    ck = d / "model.safetensors"
    save_file(sd, str(ck))
    # reference model carries exactly the values the runtime will inject
    model.load_state_dict({k: v.float() for k, v in sd.items()})
    info = export_weight_free(model, ck, d / "out", "tiny", {"hidden_size": 32}, 17)
    return model, ck, d / "out", info


def test_map_covers_every_checkpoint_tensor(tiny):
    _, ck, out, info = tiny
    tmap = json.loads((out / "tensor_map_tiny.json").read_text())
    assert info["mapped"] == len(tmap["initializers"]) > 10
    assert set(tmap["initializers"].values()) <= set(checkpoint_f32(ck))
    assert info["inline_small"] == [] or all(isinstance(n, str) for n in info["inline_small"])


def test_graph_is_weight_free(tiny):
    _, _, out, _ = tiny
    tmap = json.loads((out / "tensor_map_tiny.json").read_text())
    assert_weight_free(out / "graph_tiny.onnx", tmap)
    r = subprocess.run([sys.executable, str(SCRIPTS / "check_graph_invariants.py"), str(out / "graph_tiny.onnx")],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr


def test_injected_session_matches_torch(tiny):
    model, ck, out, _ = tiny
    tmap = json.loads((out / "tensor_map_tiny.json").read_text())
    sess = inject(out / "graph_tiny.onnx", tmap, checkpoint_f32(ck))
    for b, t, m, seed in [(1, 16, 2, 1), (3, 40, 4, 2)]:
        feed = _feed(b, t, m, seed)
        got = sess.run(["scores"], feed)[0]
        with torch.inference_mode():
            ref = model(*(torch.from_numpy(feed[k]) for k in
                          ("input_ids", "attention_mask", "marker_pos", "marker_mask", "qtype"))).numpy()
        assert np.abs(got - ref).max() < 1e-4


def test_invariants_reject_inline_weights(tmp_path):
    """A graph with weight bytes inline must fail the CI check."""
    model = make_tiny_model(1).eval()
    from export_julia.export import export_onnx

    g = tmp_path / "graph_bad.onnx"
    export_onnx(model, g, 17, constant_folding=False)
    (tmp_path / "tensor_map_bad.json").write_text(
        json.dumps({"initializers": {"encoder.embeddings.tok_embeddings.weight": "x"}}))
    r = subprocess.run([sys.executable, str(SCRIPTS / "check_graph_invariants.py"), str(g)],
                       capture_output=True, text=True)
    assert r.returncode == 1 and "FAIL" in r.stdout
