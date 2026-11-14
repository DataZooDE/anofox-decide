"""Shared helpers of the half-precision evidence gates (G0 identity, G1 NaN, G2 ranges).

Run the gate scripts with the exporter venv python and `-I`; they put this directory on sys.path.
"""
from __future__ import annotations

import csv
import glob
import json
import os
from pathlib import Path

import numpy as np
import onnx
import onnx.numpy_helper
import onnxruntime as ort

from export_julia.weight_free import checkpoint_f32

REPO = Path(__file__).resolve().parents[3]
MODELS = ("laya-multilingual", "laya-typed-decisions", "julia-1")
CACHE = Path(os.environ.get("ANOFOX_DECIDE_CACHE", Path.home() / ".cache" / "anofox-decide"))

QUESTIONS = [
    (2, "noul question: A refund is requested.", [" no", " yes"]),
    (0, "choice question: Which team should handle this request?", [" billing", " orders", " account", " other"]),
    (0, "choice question: Pick one.", [f" option {i}" for i in range(20)]),  # M = 20
]


def model_files(model: str, resources: Path | None = None) -> dict:
    """graph, tensor map, checkpoint and tokenizer paths of one real model (cache layout of decide_download)."""
    res = resources or REPO / "resources"
    revs = sorted(glob.glob(str(CACHE / model / "*")))
    if not revs:
        raise SystemExit(f"no cached weights for {model} under {CACHE / model}; run the extension once to download them")
    rev = Path(revs[-1])
    return {"graph": res / f"graph_{model}.onnx", "tmap": res / f"tensor_map_{model}.json",
            "weights": rev / "model.safetensors", "tokenizer": rev / "tokenizer.json"}


def load_ckpt(path: Path) -> dict:
    return checkpoint_f32(path)


def make_session(graph, tmap: dict, ckpt: dict, threads: int = 8, optimize: bool = True):
    """CPU session over a weight-free graph (path or ModelProto) with the checkpoint injected (like inject()).

    optimize=False disables ORT graph optimizations: needed to test Cast-wrapped fp16 MatMuls, because
    the default level removes the Cast pairs and silently runs them in fp32."""
    opts = ort.SessionOptions()
    if not optimize:
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    opts.intra_op_num_threads = threads
    opts.inter_op_num_threads = 1
    names, values = [], []
    for name, key in tmap["initializers"].items():
        a = ckpt[key]
        if tmap["transforms"].get(name) == "transpose":
            a = a.T
        names.append(name)
        values.append(ort.OrtValue.ortvalue_from_numpy(np.ascontiguousarray(a)))
    opts.add_external_initializers(names, values)
    data = graph.SerializeToString() if isinstance(graph, onnx.ModelProto) else Path(graph).read_bytes()
    sess = ort.InferenceSession(data, opts, providers=["CPUExecutionProvider"])
    sess._keepalive = (opts, values)
    return sess


def fill_weights(proto: onnx.ModelProto, tmap: dict, ckpt: dict) -> onnx.ModelProto:
    """Weight-free proto -> proto with every mapped initializer inline (for the fp16 converter)."""
    out = onnx.ModelProto()
    out.CopyFrom(proto)
    for init in out.graph.initializer:
        key = tmap["initializers"].get(init.name)
        if key is None:
            continue
        a = ckpt[key]
        if tmap["transforms"].get(init.name) == "transpose":
            a = a.T
        init.CopyFrom(onnx.numpy_helper.from_array(np.ascontiguousarray(a), init.name))
    return out


def specials(tok):
    def pick(*names):
        for n in names:
            if tok.token_to_id(n) is not None:
                return tok.token_to_id(n)
        raise SystemExit(f"none of {names} in the tokenizer")

    return (pick("[CLS]", "<bos>", "<cls>"), pick("[SEP]", "<eos>", "<sep>"), pick("[MASK]", "<mask>"),
            pick("[PAD]", "<pad>"))


def collate(tok, sp, state, qtype, head, options, max_len):
    """One row, same layout as DecideCollateRow: [CLS] head [SEP] (MASK option)* [SEP] state [SEP]."""
    cls, sep, mask, _ = sp
    ids = [cls] + tok.encode(head, add_special_tokens=False).ids + [sep]
    markers = []
    for o in options:
        markers.append(len(ids))
        ids.append(mask)
        ids += tok.encode(o, add_special_tokens=False).ids[:48]
    ids.append(sep)
    room = max_len - len(ids) - 1
    ids += tok.encode(state, add_special_tokens=False).ids[: max(room, 0)] + [sep]
    return ids, markers, qtype


def pad_batch(rows, T, pad_id):
    """rows -> the model feed padded to [B, T] (masked marker slots carry position 0)."""
    b, m = len(rows), max(len(r[1]) for r in rows)
    ids = np.full((b, T), pad_id, np.int64)
    att = np.zeros((b, T), np.int64)
    mp = np.zeros((b, m), np.int64)
    mm = np.zeros((b, m), bool)
    qt = np.zeros(b, np.int64)
    for i, (x, mk, q) in enumerate(rows):
        assert len(x) <= T
        ids[i, : len(x)] = x
        att[i, : len(x)] = 1
        mp[i, : len(mk)] = mk
        mm[i, : len(mk)] = True
        qt[i] = q
    return {"input_ids": ids, "attention_mask": att, "marker_pos": mp, "marker_mask": mm, "qtype": qt}


def read_tickets(path: Path | None) -> list[str]:
    if path is not None and Path(path).exists():
        return [r["text"] for r in csv.DictReader(open(path, encoding="utf-8"))]
    fx = REPO / "test" / "fixtures" / "support_tickets.csv"
    return [r["text"] for r in csv.DictReader(open(fx, encoding="utf-8"))]


def long_text(tickets, n_chars, offset=0):
    out, i = "", offset
    while len(out) < n_chars:
        out += tickets[i % len(tickets)] + " "
        i += 1
    return out


def scenarios(tok, sp, tickets, ladder=(128, 256, 1024), max_b=32):
    """Padded batches over the bucket ladder: (name, feed). Every multi-row feed has one row padded
    far (> 64 tokens) beyond its last valid token, so a fully masked local-window row exists."""
    pad = sp[3]
    qs = QUESTIONS
    out = []
    for T in ladder:
        short = [collate(tok, sp, t, *qs[i % 2], min(T, 256)) for i, t in enumerate(tickets[:max_b * 2])]
        out.append((f"B1_T{T}_single", pad_batch(short[:1], T, pad)))  # one row padded up to the bucket
        longrow = collate(tok, sp, long_text(tickets, T * 5), *qs[1], T)  # fills the bucket
        out.append((f"B2_T{T}_short+long", pad_batch([short[1], longrow], T, pad)))
        rows = [short[i] for i in range(min(8, max_b))] + [collate(tok, sp, long_text(tickets, T * 4, 3), *qs[0], T)]
        out.append((f"B{len(rows)}_T{T}_mixed", pad_batch(rows, T, pad)))
        mixed20 = [collate(tok, sp, tickets[i], *qs[2], T) for i in range(2)] + [short[2]]
        out.append((f"B3_T{T}_m20", pad_batch(mixed20, T, pad)))
    if max_b >= 32:
        s256 = [collate(tok, sp, t, *qs[i % 2], 256) for i, t in enumerate(tickets[:32])]
        out.append(("B32_T256_tickets", pad_batch(s256, 256, pad)))
    return out


def valid(scores, feed):
    return scores[feed["marker_mask"]]


def random_scenarios(vocab=900, ladder=(32, 160), seed=0):
    """Offline scenarios for the tiny fixture (random in-vocab ids, markers on valid positions).
    Rows have different lengths; the T=160 batches have a row padded > 64 tokens beyond its last token."""
    g = np.random.default_rng(seed)

    def row(n, m, q):
        ids = [int(x) for x in g.integers(5, vocab, n)]
        return ids, sorted(int(x) for x in g.permutation(n)[:m]), q

    out = [("B1_single", pad_batch([row(20, 3, 0)], ladder[0], 0))]
    for T in ladder:
        rows = [row(T, 4, 1), row(12, 2, 2), row(T // 2, 3, 0)]
        out.append((f"B3_T{T}_padded", pad_batch(rows, T, 0)))
    out.append(("B2_T160_fully_masked_row", pad_batch([row(8, 2, 0), row(150, 4, 1)], 160, 0)))
    return out


def tiny_files() -> dict:
    d = REPO / "test" / "fixtures" / "local_tiny"
    return {"graph": d / "graph_tiny.onnx", "tmap": d / "tensor_map_tiny.json", "weights": d / "tiny.safetensors"}
