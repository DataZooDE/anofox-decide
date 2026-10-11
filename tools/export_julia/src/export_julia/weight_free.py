"""Weight-free graph + tensor map (the shipping artifact of the local models).

The extension embeds a graph whose checkpoint tensors are external-data stubs
and a map `initializer name -> safetensors key`. The upstream safetensors file
is downloaded from Hugging Face at run time (`decide_download`) and injected
into ONNX Runtime with AddExternalInitializers. Nothing in the repository
contains a weight byte.

Pipeline (ported from anofox-tabfm tools/export_onnx):
  1. export with do_constant_folding=False so Linear weights keep their
     parameter names (folding renames them to onnx::MatMul_N),
  2. strip doc_string / metadata_props,
  3. build_tensor_map: name match first, then (shape, sha1) and transposed
     hash matching on float32-upcast values; EVERY match is value-checked
     against the checkpoint tensor, so a wrong mapping cannot ship,
  4. force-externalize every mapped initializer (threshold 0),
  5. assert_weight_free.

Initializers that are not checkpoint tensors (rotary tables, scalar
constants) stay inline: they come from code, are tiny, and cannot be
injected from the checkpoint. The invariants check bounds their total size.
"""
from __future__ import annotations

import hashlib
import json
import math
import struct
from pathlib import Path

import numpy as np
import onnx
import onnx.numpy_helper
import torch

from .export import TENSOR_MAP

# An unmatched initializer at or above this many bytes is a mapping bug.
SMALL_INLINE_LIMIT = 1024
# Total inline bytes allowed in a shipped graph (checked by the invariants).
INLINE_BUDGET = 256 * 1024
# Packed f32 layout of the injected weights, for backends that read a real file
# (MIGraphX reads ONNX external data only from a file with offset/length). Each
# mapped initializer lives at a 64-byte aligned offset of ONE f32 blob, in sorted
# name order, so a plugin can rebuild the blob from the tensor map alone.
PACK_LOCATION = "weights.bin"
PACK_ALIGNMENT = 64
PACK_ITEMSIZE = 4  # float32: what the runtime injects, whatever the checkpoint dtype


def read_safetensors_header(path: Path) -> dict:
    with open(path, "rb") as f:
        (n,) = struct.unpack("<Q", f.read(8))
        return json.loads(f.read(n))


def checkpoint_f32(path: Path) -> dict[str, np.ndarray]:
    """All safetensors tensors, floats upcast to float32 (what the runtime injects)."""
    from safetensors.torch import load_file

    out = {}
    for k, v in load_file(str(path)).items():
        out[k] = (v.float() if v.is_floating_point() else v).numpy()
    return out


def _sig(arr: np.ndarray) -> tuple:
    return (tuple(arr.shape), str(arr.dtype), hashlib.sha1(np.ascontiguousarray(arr).tobytes()).hexdigest())


def build_tensor_map(proto: onnx.ModelProto, ckpt: dict[str, np.ndarray]) -> dict:
    by_sig: dict[tuple, list[str]] = {}
    by_sig_t: dict[tuple, list[str]] = {}
    hashes_ready = False

    def ensure_hashes():
        nonlocal hashes_ready
        if hashes_ready:
            return
        for k, a in ckpt.items():
            by_sig.setdefault(_sig(a), []).append(k)
            if a.ndim == 2:
                by_sig_t.setdefault(_sig(a.T), []).append(k)
        hashes_ready = True

    mapping, transforms, shapes = {}, {}, {}
    unmatched_small, unmatched_large = [], []
    for init in proto.graph.initializer:
        arr = onnx.numpy_helper.to_array(init)
        key, transform = None, None
        cand = ckpt.get(init.name)
        if cand is not None and cand.shape == arr.shape and np.array_equal(cand, arr):
            key = init.name
        if key is None and cand is not None and cand.ndim == 2 and cand.T.shape == arr.shape \
                and np.array_equal(cand.T, arr):
            key, transform = init.name, "transpose"
        if key is None:
            ensure_hashes()
            hit = by_sig.get(_sig(arr))
            if hit is None:
                hit = by_sig_t.get(_sig(arr))
                transform = "transpose" if hit is not None else None
            if hit is not None:
                if len(hit) > 1:
                    raise RuntimeError(
                        f"initializer {init.name} {tuple(arr.shape)} matches several checkpoint tensors {hit}; "
                        "ambiguous by value, give the graph stable parameter names"
                    )
                key = hit[0]
        if key is None:
            (unmatched_large if arr.nbytes >= SMALL_INLINE_LIMIT else unmatched_small).append(init.name)
            continue
        mapping[init.name] = key
        shapes[init.name] = list(arr.shape)
        if transform:
            transforms[init.name] = transform
    if unmatched_large:
        raise RuntimeError(f"unmatched large (>= {SMALL_INLINE_LIMIT} B) initializers: {unmatched_large[:12]}")
    return {"initializers": mapping, "transforms": transforms, "shapes": shapes,
            "unmatched_small": sorted(unmatched_small)}


def _strip_graph(g) -> None:
    for node in g.node:
        node.doc_string = ""
        del node.metadata_props[:]
        for attr in node.attribute:
            if attr.type == onnx.AttributeProto.GRAPH:
                _strip_graph(attr.g)
            elif attr.type == onnx.AttributeProto.GRAPHS:
                for sub in attr.graphs:
                    _strip_graph(sub)
    for vi in list(g.value_info) + list(g.input) + list(g.output):
        vi.doc_string = ""
        del vi.metadata_props[:]


def _align(n: int, alignment: int = PACK_ALIGNMENT) -> int:
    return (n + alignment - 1) // alignment * alignment


def pack_layout(proto: onnx.ModelProto, mapped: set[str]) -> dict:
    """Deterministic packed f32 layout of the mapped initializers.

    Sorted by name, each at a 64-byte aligned offset, length = numel * 4. Depends only on
    names and shapes, so it is reproducible from the tensor map without the graph.
    """
    dims = {i.name: list(i.dims) for i in proto.graph.initializer if i.name in mapped}
    missing = sorted(mapped - set(dims))
    if missing:
        raise RuntimeError(f"mapped initializers missing from the graph: {missing[:5]}")
    for i in proto.graph.initializer:
        if i.name in mapped and i.data_type != onnx.TensorProto.FLOAT:
            raise RuntimeError(f"mapped initializer {i.name} is not float32 (data_type {i.data_type})")
    entries, offset = {}, 0
    for name in sorted(dims):
        length = math.prod(dims[name]) * PACK_ITEMSIZE
        entries[name] = {"offset": offset, "length": length}
        offset = _align(offset + length)
    return {"location": PACK_LOCATION, "alignment": PACK_ALIGNMENT, "itemsize": PACK_ITEMSIZE, "total_bytes": offset,
            "initializers": entries}


def externalize(proto: onnx.ModelProto, mapped: set[str], pack: dict | None = None) -> None:
    """Replace every mapped initializer's payload by an external-data stub.

    `location` names the shared blob when a layout is supplied. CPU injection still
    uses initializer names; it never opens the external file.
    """
    for init in proto.graph.initializer:
        if init.name not in mapped:
            continue
        for fld in ("raw_data", "float_data", "int32_data", "int64_data", "double_data"):
            init.ClearField(fld)
        init.data_location = onnx.TensorProto.EXTERNAL
        del init.external_data[:]
        entries = [("location", pack["location"] if pack is not None else init.name)]
        if pack is not None:
            ent = pack["initializers"][init.name]
            entries += [("offset", str(ent["offset"])), ("length", str(ent["length"]))]
        for k, v in entries:
            e = init.external_data.add()
            e.key, e.value = k, v


def postprocess(full_graph: Path, ckpt: dict[str, np.ndarray], out_graph: Path) -> dict:
    """full_graph (weights inline) -> out_graph (weight-free); returns the tensor map."""
    proto = onnx.load(str(full_graph), load_external_data=True)
    del proto.metadata_props[:]
    proto.doc_string = ""
    proto.graph.doc_string = ""
    _strip_graph(proto.graph)
    for fn in proto.functions:
        for node in fn.node:
            node.doc_string = ""
            del node.metadata_props[:]
        del fn.metadata_props[:]
    tmap = build_tensor_map(proto, ckpt)
    tmap["pack"] = pack_layout(proto, set(tmap["initializers"]))
    externalize(proto, set(tmap["initializers"]), tmap["pack"])
    onnx.save(proto, str(out_graph))
    return tmap


def repack(graph: Path, tmap_path: Path) -> dict:
    """Add the packed layout to an existing weight-free graph + tensor map, in place.

    Only external-data metadata and the pack section change. The result is
    what `postprocess` + `write_tensor_map` produce for a fresh export.
    """
    tmap = json.loads(tmap_path.read_text())
    proto = onnx.load(str(graph), load_external_data=False)
    pack = pack_layout(proto, set(tmap["initializers"]))
    externalize(proto, set(tmap["initializers"]), pack)
    onnx.save(proto, str(graph))
    tmap["pack"] = pack
    tmap_path.write_text(json.dumps(tmap, indent=1, sort_keys=True) + "\n")
    return pack


def build_packed_blob(tmap: dict, safetensors_path: Path) -> np.ndarray:
    """Rebuild the packed f32 blob from the tensor map's `pack` section and the checkpoint only.

    Uses nothing from the graph: the python mirror of what a GPU plugin does at create time.
    """
    from safetensors import safe_open

    pack = tmap["pack"]
    blob = np.zeros(pack["total_bytes"], dtype=np.uint8)
    with safe_open(str(safetensors_path), "np") as f:
        for name, ent in pack["initializers"].items():
            a = f.get_tensor(tmap["initializers"][name]).astype("<f4")
            if tmap["transforms"].get(name) == "transpose":
                a = a.T
            if list(a.shape) != tmap["shapes"][name] or a.nbytes != ent["length"]:
                raise RuntimeError(f"{name}: checkpoint shape {a.shape} does not fit the pack entry")
            blob[ent["offset"]:ent["offset"] + ent["length"]] = np.frombuffer(a.tobytes(), dtype=np.uint8)
    return blob


def assert_weight_free(graph: Path, tmap: dict) -> None:
    if graph.with_name(graph.name + ".data").exists():
        raise RuntimeError("weight data file next to the graph")
    proto = onnx.load(str(graph), load_external_data=False)
    mapped = set(tmap["initializers"])
    inline = 0
    for init in proto.graph.initializer:
        external = init.data_location == onnx.TensorProto.EXTERNAL
        if init.name in mapped:
            if not external or init.raw_data or init.float_data or init.int32_data or init.int64_data:
                raise RuntimeError(f"mapped initializer {init.name} carries weight bytes")
            pack = tmap.get("pack")
            if pack is not None:
                ext = {e.key: e.value for e in init.external_data}
                want = pack["initializers"][init.name]
                if ext != {"location": pack["location"], "offset": str(want["offset"]),
                           "length": str(want["length"])}:
                    raise RuntimeError(f"stub {init.name} does not match the pack layout: {ext}")
        else:
            inline += len(init.raw_data) or 4 * len(init.float_data)
    if inline > INLINE_BUDGET:
        raise RuntimeError(f"{inline} inline initializer bytes exceed the budget {INLINE_BUDGET}")


def write_tensor_map(path: Path, tmap: dict, *, arch: str, encoder: dict, opset: int,
                     checkpoint_keys: int) -> None:
    payload = {
        "format": 1,
        "arch": arch,
        "encoder": encoder,
        "opset": opset,
        **{k: TENSOR_MAP[k] for k in ("inputs", "outputs", "masked_fill")},
        "checkpoint_tensors": checkpoint_keys,
        # ONNX initializer name -> safetensors key (floats are upcast to f32 by the loader)
        "initializers": tmap["initializers"],
        "transforms": tmap["transforms"],
        # expected shape of every injected initializer (the loader verifies it)
        "shapes": tmap["shapes"],
        # packed f32 layout (alignment, total_bytes, per-initializer offset/length) that the
        # graph stubs carry too; GPU plugins build their weight blob from this alone
        "pack": tmap["pack"],
    }
    path.write_text(json.dumps(payload, indent=1, sort_keys=True) + "\n")


def export_weight_free(model: torch.nn.Module, ckpt_path: Path, out_dir: Path, arch: str,
                       encoder_cfg: dict, opset: int = 17) -> dict:
    """Export `model` (weights loaded from ckpt_path) as graph_<arch>.onnx + tensor_map_<arch>.json."""
    import tempfile

    from .export import export_onnx

    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt = checkpoint_f32(ckpt_path)
    graph = out_dir / f"graph_{arch}.onnx"
    with tempfile.TemporaryDirectory() as tmp:
        full = Path(tmp) / "full.onnx"
        export_onnx(model, full, opset, constant_folding=False)
        tmap = postprocess(full, ckpt, graph)
    assert_weight_free(graph, tmap)
    mp = out_dir / f"tensor_map_{arch}.json"
    write_tensor_map(mp, tmap, arch=arch, encoder=encoder_cfg, opset=opset, checkpoint_keys=len(ckpt))
    return {"graph": str(graph), "tensor_map": str(mp), "mapped": len(tmap["initializers"]),
            "transposed": len(tmap["transforms"]), "inline_small": tmap["unmatched_small"],
            "graph_bytes": graph.stat().st_size}


def inject(graph: Path, tmap: dict, ckpt: dict[str, np.ndarray]):
    """onnxruntime session over the weight-free graph with the checkpoint injected (python mirror of the C++ loader)."""
    import onnxruntime as ort

    opts = ort.SessionOptions()
    names, values = [], []
    for name, key in tmap["initializers"].items():
        a = ckpt[key]
        if tmap["transforms"].get(name) == "transpose":
            a = a.T
        a = np.ascontiguousarray(a)
        if list(a.shape) != tmap["shapes"][name]:
            raise RuntimeError(f"{name}: checkpoint shape {a.shape} != mapped shape {tmap['shapes'][name]}")
        names.append(name)
        values.append(ort.OrtValue.ortvalue_from_numpy(a))
    opts.add_external_initializers(names, values)
    sess = ort.InferenceSession(graph.read_bytes(), opts, providers=["CPUExecutionProvider"])
    sess._keepalive = (opts, values)  # the injected buffers must outlive the session
    return sess
