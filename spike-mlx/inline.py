"""Spike 1: inline safetensors weights into the weight-free graph (via tensor map), keep dynamic axes for now."""
import sys, json, onnx, numpy as np
from onnx import numpy_helper
from safetensors import safe_open
graph, tmap, st, out = sys.argv[1:5]
m = onnx.load(graph, load_external_data=False)
tm = json.load(open(tmap))['initializers']
f = safe_open(st, 'np')
n = 0
for init in m.graph.initializer:
    if init.name in tm:
        arr = f.get_tensor(tm[init.name]).astype(np.float32)
        new = numpy_helper.from_array(arr, init.name)
        init.CopyFrom(new); n += 1
print("inlined", n, "of", len(m.graph.initializer))
onnx.save(m, out)
