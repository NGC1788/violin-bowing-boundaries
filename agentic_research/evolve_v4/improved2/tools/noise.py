"""Analysis only (full labels): where are the zeros above positives in each benchmark?"""
import sys
import numpy as np
sys.path.insert(0, sys.argv[1])
from agentic_research.evolve_v4.colab_task import load_cases
cs = load_cases()
seen = set()
H1 = np.zeros(50); H2 = np.zeros(50); nrow = 0; nviol = 0
for c in cs:
    g = c['group'].split('|')[0] + c['group'][-3:]
    if g in seen: continue
    seen.add(g)
    L = c['labels']; C, R, N = L.shape
    for cc in range(C):
        for r in range(R):
            row = L[cc, r]
            ones = np.flatnonzero(row == 1)
            if len(ones) == 0: continue
            nrow += 1
            pair = np.flatnonzero((row[:-1] == 1) & (row[1:] == 1))
            fp = pair[0] if len(pair) else N
            # positives strictly below first pair: rank histogram
            for k in ones[ones < fp]: H1[k] += 1
            lz = np.flatnonzero(row == 0).max() if (row == 0).any() else -1
            if lz > ones[0]: nviol += 1
            # zeros above first positive
            for k in np.flatnonzero(row == 0):
                if k > ones[0]: H2[k] += 1
print('rows with a positive', nrow, 'non-monotone', nviol)
print('isolated positives below first pair by rank:', ' '.join(str(int(x)) for x in H1))
print('zeros above first positive by rank:', ' '.join(str(int(x)) for x in H2))
