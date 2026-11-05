"""Analysis only: alive rule set and pass cost over time for selected cases."""
import sys
import numpy as np
from tr import load
from agentic_research.evolve_v4.colab_task import load_cases
from agentic_research.evolve_v4.core import possible_starts, certificates, View
m = load(sys.argv[2]); cs = load_cases()
for i in map(int, sys.argv[3].split(',')):
    c = cs[i]; labels, lower, upper = c['labels'], c['lower'], c['upper']
    partial = np.full(labels.shape, -1, dtype=np.int8); rev = {k: np.full(v.shape, np.nan) for k, v in c['hidden'].items()}
    prev = None; q = 0
    # final passing rules (full data)
    P = possible_starts(labels.astype(np.int8))
    while True:
        poss = possible_starts(partial)
        if certificates(poss, lower, upper)[0] is not None: break
        L = lower[:, 0, :]; H = upper[:, 0, :]; fin = np.isfinite(L) & np.isfinite(H)
        cand = poss & fin[:, None, :]
        left = np.where(cand, np.where(fin, L, -np.inf)[:, None, :], np.inf).min(-1)
        right = np.where(cand, np.where(fin, H, np.inf)[:, None, :], -np.inf).max(-1)
        alive = tuple(np.flatnonzero(cand.any(-1).all(0) & (left.max(0) <= right.min(0))))
        pc = m._pass_cost(partial, poss, fin, np.where(fin, L, -np.inf), np.where(fin, H, np.inf), partial[:, 0, :] == -1, labels.shape[2])
        if alive != prev:
            print(i, 'read', q, 'alive', alive, 'pc', [int(pc[r]) for r in alive])
            prev = alive
        cc, rr = m.choose(View(partial, poss, c, rev, q)); partial[cc, :, rr] = labels[cc, :, rr]; q += 1
    print(i, 'end', q)
