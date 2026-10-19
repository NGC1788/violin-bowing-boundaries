"""Band-wise lower-boundary logistic fit, robust to the upper boundary and to speckle.
Per acceleration column keep strokes with F <= median force of that column's successes (lower half of
the playable band); pool columns within an acceleration band; MLE of
P(success) = pmax * sigmoid((lnF - l0 - s*ln a)/w);  s = d lnF_low / d ln a.  Stroke bootstrap."""
import sys, numpy as np, scipy.io as sio
from scipy.optimize import minimize
BASE = '/Users/jeongjihoon/violin-rne/novelty/attack_data/raw/zenodo/raw_data/'

def select(a, F, succ, na=30, amin=0, amax=99):
    nF = len(a) // na
    A = a.reshape(nF, na); FF = F.reshape(nF, na); S = succ.reshape(nF, na)
    keep_a, keep_F, keep_s = [], [], []
    for j in range(na):
        am = np.median(A[:, j])
        if not (amin <= am <= amax):
            continue
        fs = FF[S[:, j], j]
        if len(fs) < 2:
            continue
        sel = FF[:, j] <= np.median(fs)
        keep_a += list(A[sel, j]); keep_F += list(FF[sel, j]); keep_s += list(S[sel, j])
    return np.array(keep_a), np.array(keep_F), np.array(keep_s, float)

def nll(p, la, lF, y):
    l0, s, lw, lpm = p
    w = np.exp(lw); pm = 1 / (1 + np.exp(-lpm))
    z = np.clip((lF - l0 - s * la) / w, -40, 40)
    pr = np.clip(pm / (1 + np.exp(-z)), 1e-9, 1 - 1e-9)
    return -np.sum(y * np.log(pr) + (1 - y) * np.log(1 - pr))

def fit(a, F, y, start=None):
    la, lF = np.log(a), np.log(F)
    best = None
    starts = [start] if start is not None else [[np.log(1.2), s0, np.log(0.08), 1.5] for s0 in (0.0, 0.5, 1.0)]
    for st in starts:
        r = minimize(nll, st, args=(la, lF, y), method='Nelder-Mead', options=dict(maxiter=8000, xatol=1e-6, fatol=1e-8))
        if best is None or r.fun < best.fun:
            best = r
    return best

def run(sessions, crits, bands, nb=300, seed=0, verbose=True):
    rng = np.random.default_rng(seed)
    res = {}
    for crit in crits:
        for s in sessions:
            m = sio.loadmat(BASE + s + '/matx.mat')['matx']
            a, F, tt = m.T
            succ = {'N5': tt < 5, 'N10': tt < 10, 'N15': tt < 15, 'N20': tt < 20, 'N50': (tt < 50) & (tt != 30), 'any': tt != 30}[crit]
            for (b0, b1) in bands:
                aa, FF, yy = select(a, F, succ, amin=b0, amax=b1)
                if len(aa) < 20 or yy.sum() < 5 or np.ptp(np.log(aa)) < 0.3:
                    res[(crit, s, (b0, b1))] = None
                    if verbose: print(f'{crit:4s} {s:32s} a[{b0},{b1}] insufficient')
                    continue
                r = fit(aa, FF, yy)
                l0, sl, lw, lpm = r.x
                # censoring diagnostics: is the boundary inside the force range?
                Fb_mid = np.exp(l0 + sl * np.log(np.sqrt(b0 * min(b1, 3.2))))
                bs = []
                for _ in range(nb):
                    idx = rng.integers(0, len(aa), len(aa))
                    rb = fit(aa[idx], FF[idx], yy[idx], start=list(r.x))
                    bs.append(rb.x[1])
                lo, hi = np.percentile(bs, [2.5, 97.5])
                res[(crit, s, (b0, b1))] = (sl, lo, hi, np.exp(l0), np.exp(lw), 1 / (1 + np.exp(-lpm)), len(aa), FF.min(), Fb_mid)
                if verbose:
                    print(f'{crit:4s} {s:32s} a[{b0},{b1}] n={len(aa):4d} slope s={sl:+.2f} [{lo:+.2f},{hi:+.2f}]  F_low(1 m/s2)={np.exp(l0):.2f} N  width={np.exp(lw):.2f} pmax={1/(1+np.exp(-lpm)):.2f}  F_low@band-mid={Fb_mid:.2f} (lowest F {FF.min():.2f})', flush=True)
    return res

if __name__ == '__main__':
    sessions = ['first_repetitions/2023-10-18', 'first_repetitions/2023-10-20', 'after_dismounting/2023-11-10',
                'different_betas/55/2023-11-14', 'first_repetitions/2023-10-19_r', 'first_repetitions/2023-10-20_r',
                'after_dismounting/2023-11-13_r', 'different_betas/85/2023-11-14', 'different_betas/115/2023-11-15',
                'different_betas/30/2023-11-15']
    crits = sys.argv[1].split(',') if len(sys.argv) > 1 else ['N10', 'N20', 'any']
    bands = [(0.14, 0.56), (0.5, 1.6), (1.5, 3.5), (0.14, 3.5)]
    res = run(sessions, crits, bands, nb=int(sys.argv[2]) if len(sys.argv) > 2 else 200)
    import pickle
    pickle.dump(res, open('/Users/jeongjihoon/violin-rne/novelty/attack_data/work/lowerfit_' + '_'.join(crits) + '.pkl', 'wb'))
