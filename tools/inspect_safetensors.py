import json, struct, sys, collections, re
path = sys.argv[1]
with open(path, "rb") as f:
    n = struct.unpack("<Q", f.read(8))[0]
    header = json.loads(f.read(n))
header.pop("__metadata__", None)
keys = sorted(header.keys())
print(f"# tensors: {len(keys)}")

# group by top-level prefix
groups = collections.Counter(k.split(".")[0] for k in keys)
print("\n## top-level prefixes (count):")
for p, c in groups.most_common():
    print(f"  {p:30s} {c}")

# dtype tally
dt = collections.Counter(v["dtype"] for v in header.values())
print("\n## dtypes:", dict(dt))

def show(prefix, limit=40):
    sub = [k for k in keys if k.startswith(prefix)]
    print(f"\n## {prefix}*  ({len(sub)} tensors, first {min(limit,len(sub))}):")
    for k in sub[:limit]:
        v = header[k]
        print(f"  {k:60s} {str(v['shape']):20s} {v['dtype']}")

for pfx in sys.argv[2:]:
    show(pfx)
