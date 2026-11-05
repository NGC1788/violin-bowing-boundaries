"""Analysis only: chance that one read reveals a monotonicity violation, by test type."""
import sys
import numpy as np
sys.path.insert(0, sys.argv[1])
from agentic_research.evolve_v4.colab_task import load_cases
cs = load_cases(); seen = set()
up = [0, 0]; dn = [0, 0]; bot = [0, 0]
for c in cs:
    g = c['group'].split('|')[0] + c['group'][-3:]
    if g in seen: continue
    seen.add(g)
    L = c['labels']; C, R, N = L.shape
    for cc in range(C):
        for k in range(N - 1):
            pos = L[cc, :, k] == 1
            if pos.any():  # read above a positive: violation if a rule positive at k is zero at k+1
                up[0] += 1; up[1] += bool((pos & (L[cc, :, k + 1] == 0)).any())
            zer = L[cc, :, k + 1] == 0
            if zer.any():  # read below a zero
                dn[0] += 1; dn[1] += bool((zer & (L[cc, :, k] == 1)).any())
        for k in range(N):  # bottom read (rank 1) given a zero somewhere above at rank k>1 in same rule
            zr = L[cc, :, k] == 0
            if k > 2 and zr.any():
                bot[0] += 1; bot[1] += bool((zr & (L[cc, :, 1] == 1)).any())
print('P(viol | read k+1 above pos k) = %.3f (n=%d)' % (up[1] / up[0], up[0]))
print('P(viol | read k below zero k+1) = %.3f (n=%d)' % (dn[1] / dn[0], dn[0]))
print('P(viol | read rank1 below a zero at k>2) = %.3f (n=%d)' % (bot[1] / bot[0], bot[0]))
