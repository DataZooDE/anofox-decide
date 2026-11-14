"""Half-ready graph passes on the committed tiny fixture (offline, no real weights)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

onnx = pytest.importorskip("onnx")
pytest.importorskip("onnxruntime")
pytest.importorskip("torch")
from onnx import TensorProto, helper  # noqa: E402

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

import _gate_common as C  # noqa: E402
import half_ready as H  # noqa: E402

FIX = C.tiny_files()


def _load():
    return onnx.load(str(FIX["graph"]), load_external_data=False)


@pytest.fixture(scope="module")
def env():
    tmap = json.loads(Path(FIX["tmap"]).read_text())
    return tmap, C.load_ckpt(FIX["weights"])


def _float_constants(proto):
    for n in proto.graph.node:
        if n.op_type == "Constant":
            for a in n.attribute:
                if a.name == "value" and a.t.data_type in (TensorProto.FLOAT, TensorProto.DOUBLE):
                    yield n.name, onnx.numpy_helper.to_array(a.t)


def test_exact_counts_on_the_tiny_graph():
    p = _load()
    # the tiny graph has default attention: two -inf mask constants, no finfo.min (documented difference)
    assert sum(1 for _, v in _float_constants(p) if (v <= -65504).any()) == 2
    assert not any(float(v) == np.finfo(np.float32).min for _, v in _float_constants(p) if v.ndim == 0)
    assert H.half_ready(p) == {"masks": 2, "rotary": 1}
    assert all(np.isfinite(v).all() and (v > -65504).all() for _, v in _float_constants(p))
    assert not [n for n in p.graph.node if n.op_type == "MatMul" and "rotary_emb" in n.name]
    assert [n for n in p.graph.node if n.op_type == "Mul" and n.name.endswith("rotary_emb/MatMul_as_Mul")]


def test_idempotent():
    p = _load()
    H.half_ready(p)
    once = p.SerializeToString()
    assert H.half_ready(p) == {"masks": 0, "rotary": 0}
    assert p.SerializeToString() == once


def test_finfo_min_and_inf_become_minus_1e4():
    """The real graphs carry -3.4028235e38 (finfo.min) and -inf; both must land on exactly -1e4."""
    consts = [helper.make_node("Constant", [], [f"c{i}"], name=f"k{i}",
                               value=helper.make_tensor("v", TensorProto.FLOAT, [], [x]))
              for i, x in enumerate([np.finfo(np.float32).min, -np.inf, -65504.0, -65503.0, 0.5])]
    g = helper.make_graph(consts, "g", [], [helper.make_tensor_value_info(f"c{i}", TensorProto.FLOAT, []) for i in range(5)])
    m = helper.make_model(g)
    assert H.sanitize_masks(m) == 3
    vals = [float(onnx.numpy_helper.to_array(n.attribute[0].t)) for n in m.graph.node]
    assert vals == [-10000.0, -10000.0, -10000.0, -65503.0, 0.5]


@pytest.mark.parametrize("bad", [np.inf, np.nan])
def test_refuses_positive_inf_and_nan(bad):
    n = helper.make_node("Constant", [], ["c"], name="k", value=helper.make_tensor("v", TensorProto.FLOAT, [], [bad]))
    m = helper.make_model(helper.make_graph([n], "g", [], [helper.make_tensor_value_info("c", TensorProto.FLOAT, [])]))
    with pytest.raises(ValueError):
        H.sanitize_masks(m)


@pytest.mark.parametrize("variant", [dict(masks=True, rotary=False), dict(masks=False, rotary=True),
                                     dict(masks=True, rotary=True)])
def test_bit_identical_on_padded_batches(env, variant):
    """Valid-marker scores are bit-identical, incl. a batch whose short row is padded > 64 tokens (fully
    masked local-window rows) and single-row batches."""
    tmap, ckpt = env
    base = C.make_session(FIX["graph"], tmap, ckpt, 2)
    p = _load()
    H.half_ready(p, **variant)
    patched = C.make_session(p, tmap, ckpt, 2)
    scen = C.random_scenarios()
    assert any(f["attention_mask"].shape[1] - f["attention_mask"].sum(axis=1).min() > 64 for _, f in scen)
    for name, feed in scen:
        a = base.run(["scores"], feed)[0]
        b = patched.run(["scores"], feed)[0]
        mm = feed["marker_mask"]
        assert np.array_equal(a[mm], b[mm]), name


def test_cli_writes_copies_and_never_resources(tmp_path):
    out = tmp_path / "ready"
    c = H.patch_file(FIX["graph"], out)
    assert c == {"masks": 2, "rotary": 1}
    assert (out / "graph_tiny.onnx").exists() and (out / "tensor_map_tiny.json").exists()
    with pytest.raises(SystemExit):
        H.patch_file(FIX["graph"], tmp_path / "resources")
