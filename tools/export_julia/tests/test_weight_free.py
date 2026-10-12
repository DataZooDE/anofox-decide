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


def _stub_ext(graph):
    proto = onnx.load(str(graph), load_external_data=False)
    return {i.name: {e.key: e.value for e in i.external_data}
            for i in proto.graph.initializer if i.data_location == onnx.TensorProto.EXTERNAL}, proto


def test_pack_layout_is_sorted_aligned_and_on_every_stub(tiny):
    _, _, out, _ = tiny
    tmap = json.loads((out / "tensor_map_tiny.json").read_text())
    pack = tmap["pack"]
    assert pack["alignment"] == 64 and pack["itemsize"] == 4
    assert sorted(pack["initializers"]) == sorted(tmap["initializers"])
    stubs, _ = _stub_ext(out / "graph_tiny.onnx")
    end = 0
    for name in sorted(tmap["initializers"]):
        ent = pack["initializers"][name]
        numel = int(np.prod(tmap["shapes"][name], dtype=np.int64))
        assert ent["length"] == numel * 4
        assert ent["offset"] % 64 == 0 and ent["offset"] >= end
        end = ent["offset"] + ent["length"]
        # Every stub addresses the same packed file.
        assert stubs[name] == {"location": "weights.bin", "offset": str(ent["offset"]), "length": str(ent["length"])}
    assert pack["total_bytes"] == (end + 63) // 64 * 64


def test_blob_rebuilt_from_the_pack_section_equals_the_checkpoint(tiny):
    from export_julia.weight_free import build_packed_blob

    _, ck, out, _ = tiny
    tmap = json.loads((out / "tensor_map_tiny.json").read_text())
    blob = build_packed_blob(tmap, ck)
    ref = checkpoint_f32(ck)
    assert blob.size == tmap["pack"]["total_bytes"]
    for name, ent in tmap["pack"]["initializers"].items():
        a = ref[tmap["initializers"][name]]
        if tmap["transforms"].get(name) == "transpose":
            a = a.T
        got = blob[ent["offset"]:ent["offset"] + ent["length"]].view(np.float32).reshape(tmap["shapes"][name])
        assert got.tobytes() == a.astype("<f4").tobytes()
    r = subprocess.run([sys.executable, str(SCRIPTS / "verify_pack.py"), str(out / "graph_tiny.onnx"), str(ck)],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr


def test_repack_of_a_legacy_graph_equals_a_fresh_export(tiny, tmp_path):
    """A graph exported before the pack layout existed (stub = location only) gains exactly what a fresh export writes."""
    from export_julia.weight_free import externalize, repack

    _, _, out, _ = tiny
    tmap = json.loads((out / "tensor_map_tiny.json").read_text())
    proto = onnx.load(str(out / "graph_tiny.onnx"), load_external_data=False)
    externalize(proto, set(tmap["initializers"]), None)  # legacy stubs: location only
    legacy = tmp_path / "graph_tiny.onnx"
    onnx.save(proto, str(legacy))
    legacy_map = {k: v for k, v in tmap.items() if k != "pack"}
    (tmp_path / "tensor_map_tiny.json").write_text(json.dumps(legacy_map, indent=1, sort_keys=True) + "\n")
    stubs, _ = _stub_ext(legacy)
    assert all(set(e) == {"location"} for e in stubs.values())
    repack(legacy, tmp_path / "tensor_map_tiny.json")
    assert legacy.read_bytes() == (out / "graph_tiny.onnx").read_bytes()
    assert (tmp_path / "tensor_map_tiny.json").read_text() == (out / "tensor_map_tiny.json").read_text()
    repack(legacy, tmp_path / "tensor_map_tiny.json")  # idempotent
    assert legacy.read_bytes() == (out / "graph_tiny.onnx").read_bytes()


def test_invariants_reject_a_stub_that_disagrees_with_the_pack(tiny, tmp_path):
    _, _, out, _ = tiny
    tmap = json.loads((out / "tensor_map_tiny.json").read_text())
    proto = onnx.load(str(out / "graph_tiny.onnx"), load_external_data=False)
    victim = next(i for i in proto.graph.initializer if i.data_location == onnx.TensorProto.EXTERNAL)
    for e in victim.external_data:
        if e.key == "offset":
            e.value = str(int(e.value) + 64)
    bad = tmp_path / "graph_tiny.onnx"
    onnx.save(proto, str(bad))
    (tmp_path / "tensor_map_tiny.json").write_text(json.dumps(tmap))
    r = subprocess.run([sys.executable, str(SCRIPTS / "check_graph_invariants.py"), str(bad)],
                       capture_output=True, text=True)
    assert r.returncode == 1 and "does not match pack" in r.stdout
    # and a map without a pack section is rejected with the fix named
    (tmp_path / "tensor_map_tiny.json").write_text(json.dumps({k: v for k, v in tmap.items() if k != "pack"}))
    r = subprocess.run([sys.executable, str(SCRIPTS / "check_graph_invariants.py"), str(out / "graph_tiny.onnx")],
                       capture_output=True, text=True)
    assert r.returncode == 0  # the real map next to the real graph still passes
    r = subprocess.run([sys.executable, str(SCRIPTS / "check_graph_invariants.py"), str(bad)],
                       capture_output=True, text=True)
    assert r.returncode == 1 and "no `pack` section" in r.stdout and "repack_offsets.py" in r.stdout


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


def test_external_file_load_matches_cpu_injection(tiny, tmp_path):
    import onnxruntime as ort
    from export_julia.weight_free import build_packed_blob, externalize

    _, ck, out, _ = tiny
    graph = tmp_path / "graph_tiny.onnx"
    graph.write_bytes((out / graph.name).read_bytes())
    tmap = json.loads((out / "tensor_map_tiny.json").read_text())
    blob = build_packed_blob(tmap, ck)
    (tmp_path / tmap["pack"]["location"]).write_bytes(blob.tobytes())
    # A standard ONNX reader must resolve every tensor from this ONE file.
    loaded = onnx.load(str(graph))
    ref = checkpoint_f32(ck)
    for init in loaded.graph.initializer:
        if init.name in tmap["initializers"]:
            a = ref[tmap["initializers"][init.name]]
            if tmap["transforms"].get(init.name) == "transpose":
                a = a.T
            assert init.raw_data == a.astype("<f4").tobytes()
    file_session = ort.InferenceSession(str(graph), providers=["CPUExecutionProvider"])
    injected = inject(graph, tmap, ref)
    proto = onnx.load(str(graph), load_external_data=False)
    externalize(proto, set(tmap["initializers"]))
    legacy = tmp_path / "legacy.onnx"
    onnx.save(proto, str(legacy))
    legacy_session = inject(legacy, tmap, ref)
    for b, t, m, seed in [(1, 16, 2, 1), (3, 40, 4, 2)]:
        feed = _feed(b, t, m, seed)
        got = injected.run(["scores"], feed)[0].tobytes()
        assert file_session.run(["scores"], feed)[0].tobytes() == got
        assert legacy_session.run(["scores"], feed)[0].tobytes() == got


@pytest.mark.parametrize("corruption", ["location", "alignment", "total_bytes", "extra", "shape", "gap"])
def test_invariants_reject_invalid_pack(tiny, tmp_path, corruption):
    _, _, out, _ = tiny
    tmap = json.loads((out / "tensor_map_tiny.json").read_text())
    name = sorted(tmap["initializers"])[0]
    pack = tmap["pack"]
    if corruption == "location":
        pack["location"] = "wrong.bin"
    elif corruption == "alignment":
        pack["alignment"] = 0
    elif corruption == "total_bytes":
        pack["total_bytes"] += 64
    elif corruption == "extra":
        pack["initializers"]["extra"] = {"offset": 0, "length": 4}
    elif corruption == "shape":
        tmap["shapes"][name] = [1]
    else:
        pack["initializers"][name]["offset"] += 64
    graph = tmp_path / "graph_tiny.onnx"
    graph.write_bytes((out / graph.name).read_bytes())
    (tmp_path / "tensor_map_tiny.json").write_text(json.dumps(tmap))
    r = subprocess.run([sys.executable, str(SCRIPTS / "check_graph_invariants.py"), str(graph)],
                       capture_output=True, text=True)
    assert r.returncode == 1 and "FAIL" in r.stdout, r.stdout + r.stderr


def test_pack_transposed_tensor_bytes(tmp_path):
    from export_julia.weight_free import build_packed_blob, pack_layout

    a = torch.tensor([[0.0, -0.0, 2.0], [3.0, 4.0, 5.0]], dtype=torch.float16)
    ck = tmp_path / "model.safetensors"
    save_file({"weight": a}, str(ck))
    init = onnx.numpy_helper.from_array(a.float().numpy().T, "transposed")
    proto = onnx.helper.make_model(onnx.helper.make_graph([], "transpose", [], [], [init]))
    tmap = {"initializers": {"transposed": "weight"}, "transforms": {"transposed": "transpose"},
            "shapes": {"transposed": [3, 2]}, "pack": pack_layout(proto, {"transposed"})}
    blob = build_packed_blob(tmap, ck)
    assert blob[:24].tobytes() == a.float().numpy().T.astype("<f4").tobytes()
    assert not blob[24:].any()  # deterministic zero padding
