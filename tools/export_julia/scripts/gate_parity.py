#!/usr/bin/env python3
"""Gate G1: weight-free graph + injected safetensors vs the full-weights ONNX path.

Both sessions are fed identical collated rows built from real tickets (the same
layout as DecideCollateRow: [CLS] head [SEP] (MASK option)* [SEP] state [SEP]).
Reports the worst absolute score delta, the worst probability delta (softmax
over the valid options) and the number of rows whose chosen option differs.

  gate_parity.py --full julia1.onnx --graph graph_julia-1.onnx --map tensor_map_julia-1.json \
      --weights model.safetensors --encoder-config encoder/config.json \
      --tokenizer tokenizer/tokenizer.json --tickets tickets.csv [--limit 200]
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer

from export_julia.weight_free import checkpoint_f32, inject

QUESTIONS = [
    (2, "noul question: A refund is requested.", [" no", " yes"]),
    (0, "choice question: Which team should handle this request?", [" billing", " orders", " account", " other"]),
]


def collate(tok, specials, state, qtype, head, options, max_len=1024):
    cls, sep, mask = specials
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


def batch(rows):
    b = len(rows)
    t = max(len(r[0]) for r in rows)
    m = max(len(r[1]) for r in rows)
    ids = np.zeros((b, t), np.int64)
    att = np.zeros((b, t), np.int64)
    mp = np.zeros((b, m), np.int64)
    mm = np.zeros((b, m), bool)
    qt = np.zeros(b, np.int64)
    for i, (x, mk, q) in enumerate(rows):
        ids[i, : len(x)] = x
        att[i, : len(x)] = 1
        mp[i, : len(mk)] = mk
        mm[i, : len(mk)] = True
        qt[i] = q
    return {"input_ids": ids, "attention_mask": att, "marker_pos": mp, "marker_mask": mm, "qtype": qt}


def softmax(s, valid):
    s = np.where(valid, s, -np.inf)
    e = np.exp(s - s.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--full", required=True)
    ap.add_argument("--graph", required=True)
    ap.add_argument("--map", required=True)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--encoder-config", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--tickets", required=True)
    ap.add_argument("--limit", type=int, default=200)
    a = ap.parse_args(argv)

    cfg = json.loads(Path(a.encoder_config).read_text())
    tok = Tokenizer.from_file(a.tokenizer)

    def special(key, *names):
        if key in cfg:
            return cfg[key]
        for n in names:
            if tok.token_to_id(n) is not None:
                return tok.token_to_id(n)
        raise SystemExit(f"no {key} in the encoder config and none of {names} in the tokenizer")

    specials = (special("cls_token_id", "[CLS]", "<cls>"), special("sep_token_id", "[SEP]", "<sep>"),
                special("mask_token_id", "[MASK]", "<mask>"))
    tickets = [r["text"] for r in csv.DictReader(open(a.tickets, encoding="utf-8"))][: a.limit]

    full = ort.InferenceSession(a.full, providers=["CPUExecutionProvider"])
    tmap = json.loads(Path(a.map).read_text())
    wf = inject(Path(a.graph), tmap, checkpoint_f32(Path(a.weights)))

    worst_s = worst_p = 0.0
    rows_n = diff_choice = 0
    chunks = []
    for text in tickets:
        for qtype, head, opts in QUESTIONS:
            chunks.append(collate(tok, specials, text, qtype, head, opts))
    # single-row and padded 8-row batches
    for size in (1, 8):
        for i in range(0, len(chunks), size):
            feed = batch(chunks[i : i + size])
            s1 = full.run(["scores"], feed)[0]
            s2 = wf.run(["scores"], feed)[0]
            valid = feed["marker_mask"]
            worst_s = max(worst_s, float(np.abs(s1 - s2)[valid].max()))
            p1, p2 = softmax(s1, valid), softmax(s2, valid)
            worst_p = max(worst_p, float(np.abs(p1 - p2)[valid].max()))
            if size == 1:
                rows_n += len(s1)
                diff_choice += int((p1.argmax(1) != p2.argmax(1)).sum())
    result = {"rows": rows_n, "tickets": len(tickets), "worst_score_delta": worst_s,
              "worst_prob_delta": worst_p, "choice_mismatches": diff_choice,
              "pass": worst_s <= 1e-4 and worst_p <= 1e-3 and diff_choice == 0}
    print(json.dumps(result))
    return 0 if result["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
