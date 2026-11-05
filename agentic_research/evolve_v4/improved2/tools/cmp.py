import json, sys
a, b = sys.argv[1], sys.argv[2]
for k in "VR":
    g = json.load(open(f"res/{a}.{k}.json"))["per_case"]; h = json.load(open(f"res/{b}.{k}.json"))["per_case"]
    d = [(i, x[0].split('|')[0] + x[0][-3:], x[1], x[2], y[2], y[2] - x[2]) for i, (x, y) in enumerate(zip(g, h)) if x[2] != y[2]]
    print(k, sum(t[5] for t in d), d)
