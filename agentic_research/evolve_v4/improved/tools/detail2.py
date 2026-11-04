import sys
import numpy as np
from trace import *
m = load(sys.argv[1]); i = int(sys.argv[2])
case = load_cases()[i]
if hasattr(m, 'LOG'): m.LOG.clear()
qs = run(case, m.choose); pr = prune(case, qs)
tags = list(getattr(m, 'LOG', ['']*len(qs)))
labels, lower, upper = case["labels"], case["lower"], case["upper"]
print(case['group'], case['epsilon'], case['expected'], len(qs), len(pr))
print('pruned:', sorted(pr))
partial = np.full(labels.shape, -1, dtype=np.int8)
cls = [0, 1, 2]  # representative rules for v classes (u=0.7)
for t, (c, r) in enumerate(qs):
    partial[c, :, r] = labels[c, :, r]
    poss = possible_starts(partial)
    cand = poss & np.isfinite(lower)
    G = np.where(cand, lower, np.inf).min(-1)
    H = np.where(cand, upper, -np.inf).max(-1)
    gm = G.max(0); hm = H.min(0)
    alive = cand.any(-1).all(0) & (gm <= hm)
    lab = ''.join(str(x) for x in labels[c, cls, r])
    gap = ' '.join(f"{(gm[k]-hm[k]):+.2f}({int(G[:,k].argmax())},{int(H[:,k].argmin())})" for k in cls)
    print(t, (c, r), lab, tags[t] if t < len(tags) else '', ''.join('1' if a else '.' for a in alive), gap, '*' if (c, r) in pr else '')
