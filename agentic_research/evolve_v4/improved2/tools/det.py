"""Analysis only: read index at which a zero-above-positive is first seen, per case."""
import sys, json
import numpy as np
sys.argv += []
from tr import load, run
from agentic_research.evolve_v4.colab_task import load_cases
m = load(sys.argv[2]); cases = load_cases()
res = json.load(open(sys.argv[3]))['per_case'] if len(sys.argv) > 3 else None
for i, case in enumerate(cases):
    qs = run(case, m)
    C, R, N = case['labels'].shape
    P = np.full((C, R, N), -1)
    det = None
    for k, (c, r, t, lab) in enumerate(qs):
        P[c, :, r] = lab
        idx = np.arange(N)
        fo = np.where(P == 1, idx, N).min(-1); lz = np.where(P == 0, idx, -1).max(-1)
        if det is None and (lz > fo).any():
            det = k
    lab = case['labels']
    fo = np.where(lab == 1, np.arange(N), N).min(-1); lz = np.where(lab == 0, np.arange(N), -1).max(-1)
    print(i, case['group'].split('|')[0] + case['group'][-3:], case['epsilon'], case['expected'], len(qs), 'det', det, 'frac_nonmono_rows %.2f' % (lz > fo).mean())
