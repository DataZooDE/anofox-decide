import sys, time, numpy as np, onnxruntime as ort
g, B, T, M, inp, ref = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), sys.argv[5], sys.argv[6]
raw = open(inp,'rb').read(); o = 0
def take(dt, n):
    global o; a = np.frombuffer(raw, dt, n, o); o += a.nbytes; return a
ids = take(np.int64,B*T).reshape(B,T); am = take(np.int64,B*T).reshape(B,T); pos = take(np.int64,B*M).reshape(B,M)
mm = take(np.uint8,B*M).reshape(B,M).astype(bool); q = take(np.int64,B)
s = ort.InferenceSession(g, providers=['CPUExecutionProvider'])
f = dict(input_ids=ids, attention_mask=am, marker_pos=pos, marker_mask=mm, qtype=q)
t=[]
for i in range(5):
    a=time.time(); r=s.run(None,f)[0]; t.append((time.time()-a)*1000)
print("ORT CPU run ms:", " ".join(f"{x:.0f}" for x in t))
ref_arr = np.fromfile(ref, np.float32).reshape(B,M); v = ref_arr > -9000
print("diff vs linux ORT ref (valid):", float(np.abs(r[v]-ref_arr[v]).max()))
