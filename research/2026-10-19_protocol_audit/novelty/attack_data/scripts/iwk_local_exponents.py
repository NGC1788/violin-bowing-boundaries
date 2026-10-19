"""Local speed exponents of the IWK 2025 lower boundary (v1->v2 vs v2->v3) from the final-analysis cache
(read-only). Per column: first run of K=2 Helmholtz windows in force order at t_eval -> interval
(f[s-1], f[s]); geometric midpoint. Per specimen: ln F_low = c_v + q ln beta (common q), OLS on columns with
finite intervals; column bootstrap."""
import numpy as np
from collections import defaultdict
CACHE = '/Users/jeongjihoon/violin-rne/final/cache/'
K = 2
rng = np.random.default_rng(3)
for t_eval in (0.0, 0.2):
    print(f'=== t = {t_eval} s')
    for tag in ('A1', 'A2', 'A_T2', 'B1', 'C1', 'D1', 'B_T2r1', 'B_T2r2'):
        try:
            d = dict(np.load(CACHE + tag + '.npz'))
        except FileNotFoundError:
            continue
        m = np.isclose(d['t'], t_eval)
        groups = defaultdict(list)
        for i in np.flatnonzero(m):
            groups[(int(d['folder'][i]), int(d['column'][i]))].append(i)
        cols = []
        for key, idx in groups.items():
            idx = sorted(idx, key=lambda i: d['force'][i])
            f = np.log(np.maximum(d['force'][idx], 1e-3)); h = d['H'][idx]
            runs = [i for i in range(len(h) - K + 1) if all(h[i:i + K])]
            if not runs or runs[0] == 0:
                continue
            s = runs[0]
            cols.append((d['speed'][idx[0]], np.log(np.median(d['beta'][idx])), 0.5 * (f[s - 1] + f[s])))
        cols = np.array(cols)
        sp = np.unique(np.round(cols[:, 0], 3))
        if len(sp) != 3:
            print(tag, 'speeds', sp); continue
        def fitc(c):
            X = np.column_stack([np.isclose(np.round(c[:, 0], 3), v).astype(float) for v in sp] + [c[:, 1]])
            th = np.linalg.lstsq(X, c[:, 2], rcond=None)[0]
            return (th[1] - th[0]) / np.log(sp[1] / sp[0]), (th[2] - th[1]) / np.log(sp[2] / sp[1]), (th[2] - th[0]) / np.log(sp[2] / sp[0]), th[3]
        e12, e23, e13, q = fitc(cols)
        bs = []
        for _ in range(1000):
            c = cols[rng.integers(0, len(cols), len(cols))]
            if all(np.isclose(np.round(c[:, 0], 3), v).sum() > 2 for v in sp):
                bs.append(fitc(c))
        bs = np.array(bs)
        ci = np.percentile(bs, [2.5, 97.5], axis=0)
        d23_12 = bs[:, 1] - bs[:, 0]
        print(f'{tag:7s} n_cols={len(cols)} beta-slope {q:+.2f} | e12 {e12:+.2f} [{ci[0,0]:+.2f},{ci[1,0]:+.2f}]  e23 {e23:+.2f} [{ci[0,1]:+.2f},{ci[1,1]:+.2f}]  e13 {e13:+.2f}  | e23-e12 {e23-e12:+.2f} [{np.percentile(d23_12,2.5):+.2f},{np.percentile(d23_12,97.5):+.2f}]')
