#!/usr/bin/env python3
"""Where the expected scores in test_cuda_plugin_cpu.cpp come from.

Runs the committed tiny fixture (test/fixtures/local_tiny) through Python onnxruntime with the checkpoint's
tensors injected via SessionOptions.add_external_initializers (upcast to float32 with numpy), for the batch the
C++ test uses. That is an implementation of the weight feeding independent of the plugin's, so the test checks
the plugin against ORT itself rather than against its own code. Regenerate (and paste the printed values into the
test) only if the fixture or the batch changes:

    pip install onnx onnxruntime==1.29.0 numpy
    python3 plugins/cuda/test/tiny_reference.py          # from the repository root
"""
import json
import struct
from pathlib import Path

import numpy as np
import onnxruntime as ort

D = Path(__file__).resolve().parents[3] / "test/fixtures/local_tiny"
with open(D / "tiny.safetensors", "rb") as f:
    header_len = struct.unpack("<Q", f.read(8))[0]
    header = json.loads(f.read(header_len))
    blob = f.read()
tensor_map = json.loads((D / "tensor_map_tiny.json").read_text())

names, values = [], []
for name, key in tensor_map["initializers"].items():
    t = header[key]
    b0, b1 = t["data_offsets"]
    dtype = {"F16": np.float16, "F32": np.float32}[t["dtype"]]
    array = np.frombuffer(blob[b0:b1], dtype=dtype).reshape(t["shape"]).astype(np.float32)
    names.append(name)
    values.append(ort.OrtValue.ortvalue_from_numpy(array))
options = ort.SessionOptions()
options.add_external_initializers(names, values)
session = ort.InferenceSession(str(D / "graph_tiny.onnx"), options, providers=["CPUExecutionProvider"])

rng = np.random.RandomState(7)
ids = rng.randint(1, 1000, size=(2, 12)).astype(np.int64)
mask = np.ones((2, 12), dtype=np.int64)
mask[1, 9:] = 0
marker_pos = np.array([[2, 5, 8], [1, 4, 0]], dtype=np.int64)
marker_mask = np.array([[True, True, True], [True, True, False]])
qtype = np.array([0, 1], dtype=np.int64)
scores = session.run(["scores"], {"input_ids": ids, "attention_mask": mask, "marker_pos": marker_pos,
                                  "marker_mask": marker_mask, "qtype": qtype})[0]
print("input_ids", ids.tolist())
print("scores", scores.tolist())
