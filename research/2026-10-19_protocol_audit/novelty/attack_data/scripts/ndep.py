"""N-dependence of the lower force limit at the lowest accelerations (a = 0.15-0.25 m/s^2, columns pooled).
Authors' labels. P = pmax*sigmoid((lnF - l)/w) on strokes with F <= median success force of the column."""
import sys, numpy as np, scipy.io as sio
from scipy.optimize import minimize
BASE = '/Users/jeongjihoon/violin-rne/novelty/attack_data/raw/zenodo/raw_data/'
rng = np.random.default_rng(7)
def nll(p, lF, y):
    l, lw, lpm = p
    z = np.clip((lF - l) / np.exp(lw), -40, 40)
    pr = np.clip(1 / (1 + np.exp(-lpm)) / (1 + np.exp(-z)), 1e-9, 1 - 1e-9)
    return -np.sum(y * np.log(pr) + (1 - y) * np.log(1 - pr))
def fit(lF, y, st=None):
    best = None
    for s0 in ([st] if st is not None else [[np.log(x), np.log(0.1), 2.0] for x in (0.4, 0.8, 1.5)]):
        r = minimize(nll, s0, args=(lF, y), method='Nelder-Mead', options=dict(maxiter=4000))
        if best is None or r.fun < best.fun: best = r
    return best
sessions = ['first_repetitions/2023-10-18', 'first_repetitions/2023-10-20', 'after_dismounting/2023-11-10', 'different_betas/55/2023-11-14']
for amax in (0.2, 0.3):
    print(f'--- columns with a <= {amax}')
    for s in sessions:
        m = sio.loadmat(BASE + s + '/matx.mat')['matx']; a, F, tt = m.T
        out = []
        for name, succ in (('N5', tt < 5), ('N10', tt < 10), ('N15', tt < 15), ('N20', tt < 20), ('N30', (tt < 30)), ('N50', (tt < 50) & (tt != 30)), ('any', tt != 30)):
            na = 30; nF = len(a) // na
            A = a.reshape(nF, na); FF = F.reshape(nF, na); S = succ.reshape(nF, na)
            lF, y = [], []
            for j in range(na):
                if np.median(A[:, j]) > amax: continue
                fs = FF[S[:, j], j]
                if len(fs) < 2: continue
                sel = FF[:, j] <= np.median(fs)
                lF += list(np.log(FF[sel, j])); y += list(S[sel, j].astype(float))
            lF, y = np.array(lF), np.array(y)
            if len(y) < 6 or y.sum() < 2:
                out.append(f'{name}: none'); continue
            r = fit(lF, y)
            bs = [fit(lF[i], y[i], list(r.x)).x[0] for i in (rng.integers(0, len(y), len(y)) for _ in range(300))]
            lo, hi = np.exp(np.percentile(bs, [2.5, 97.5]))
            cens = '<' if np.exp(r.x[0]) <= np.exp(lF.min()) * 1.02 else ''
            out.append(f'{name}: {cens}{np.exp(r.x[0]):.2f} [{lo:.2f},{hi:.2f}]')
        print(f'{s:32s} (lowest F {F.min():.2f}) ' + ' | '.join(out))
