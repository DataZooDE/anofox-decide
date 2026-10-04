import sys, json, struct
st, tmap, out = sys.argv[1:4]
f = open(st,'rb'); n = struct.unpack('<Q', f.read(8))[0]; hdr = json.loads(f.read(n)); base = 8+n
tm = json.load(open(tmap))['initializers']
with open(out,'w') as o:
    for name, key in tm.items():
        e = hdr[key]; a,b = e['data_offsets']; o.write(f"{name}\t{e['dtype']}\t{base+a}\t{b-a}\n")
print("index entries", len(tm))
