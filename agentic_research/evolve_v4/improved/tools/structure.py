import os, sys
import numpy as np
from trace_r import *
cases = load_cases()
fs, iso, nposlow = [], [], []
for case in cases:
    lab = case['labels'].astype(np.int8); C, R, N = lab.shape
    f = possible_starts(lab).argmax(-1)  # first pair start (N-1 = none)
    fs.extend(f[:, [0, 4, 8]].ravel().tolist())
    # isolated positives below the first pair (noise) per column
    for c in range(C):
        for r in [0, 4, 8]:
            row = lab[c, r]; k = f[c, r]
            iso.append(int(row[:k].sum()))
fs = np.array(fs); iso = np.array(iso)
print(os.environ.get('BUNDLE', 'violin').split('/')[-1], 'cases', len(cases), 'first-pair rank: median', np.median(fs[fs < 49]), 'p25', np.percentile(fs[fs < 49], 25), 'p75', np.percentile(fs[fs < 49], 75), 'none-frac', (fs == 49).mean().round(2))
print('   positives below the first pair per column/rule: mean', iso.mean().round(2), 'frac>0', (iso > 0).mean().round(2))
