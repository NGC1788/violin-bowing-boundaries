"""Analysis only: which rules pass on full data, and which rule the pin phase targets."""
import sys
import numpy as np
from tr import load
from agentic_research.evolve_v4.colab_task import load_cases
from agentic_research.evolve_v4.core import possible_starts, certificates, View
m = load(sys.argv[2]); cs = load_cases()
for i, c in enumerate(cs):
    if not c['expected']: continue
    labels, lower, upper = c['labels'], c['lower'], c['upper']
    P = possible_starts(labels)
    valid = P & np.isfinite(lower)
    lo = np.where(valid, lower, np.inf).min(-1).max(0); hi = np.where(valid, upper, -np.inf).max(-1).min(0)
    passing = [r for r in range(9) if valid[:, r].any(-1).all() and lo[r] <= hi[r]]
    partial = np.full(labels.shape, -1, dtype=np.int8); rev = {k: np.full(v.shape, np.nan) for k, v in c['hidden'].items()}
    q = 0; pins = {}
    while True:
        poss = possible_starts(partial)
        if certificates(poss, lower, upper)[0] is not None: break
        m.TAG[0] = '?'
        L = lower[:, 0, :]; H = upper[:, 0, :]; fin = np.isfinite(L) & np.isfinite(H)
        cc, rr = m.choose(View(partial, poss, c, rev, q))
        if m.TAG[0] == 'pin':
            pc = m._pass_cost(partial, poss, fin, np.where(fin, L, -np.inf), np.where(fin, H, np.inf), partial[:, 0, :] == -1, labels.shape[2])
            cand = poss & fin[:, None, :]
            left = np.where(cand, np.where(fin, L, -np.inf)[:, None, :], np.inf).min(-1); right = np.where(cand, np.where(fin, H, np.inf)[:, None, :], -np.inf).max(-1)
            alive = np.flatnonzero(cand.any(-1).all(0) & (left.max(0) <= right.min(0)))
            b = min(alive, key=lambda r: (pc[r], r)); pins[int(b)] = pins.get(int(b), 0) + 1
        partial[cc, :, rr] = labels[cc, :, rr]; q += 1
    print(i, c['group'].split('|')[0] + c['group'][-3:], c['epsilon'], 'reads', q, 'passing', passing, 'pinned', pins)
