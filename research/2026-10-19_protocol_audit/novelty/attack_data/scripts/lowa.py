"""Low-acceleration part of the lower force limit (a <= 0.55 m/s^2, the range that overlaps the IWK 2025
ramp accelerations), authors' labels; stroke bootstrap within columns."""
import sys, numpy as np, scipy.io as sio
sys.path.insert(0, '/Users/jeongjihoon/violin-rne/novelty/attack_data/scripts')
import colbound as cb
BASE = '/Users/jeongjihoon/violin-rne/novelty/attack_data/raw/zenodo/raw_data/'
sessions = ['first_repetitions/2023-10-18', 'first_repetitions/2023-10-20', 'after_dismounting/2023-11-10',
            'different_betas/55/2023-11-14', 'first_repetitions/2023-10-19_r', 'first_repetitions/2023-10-20_r',
            'after_dismounting/2023-11-13_r']
rng = np.random.default_rng(5)
NB = 1000
def curves(a, F, succ, na=30):
    nF = len(a) // na
    A = a.reshape(nF, na); FF = F.reshape(nF, na); S = succ.reshape(nF, na)
    co = np.argsort(np.median(A, 0))
    return A[:, co], FF[:, co], S[:, co]
def col_low(Fc, Sc):
    return cb.column_lower(Fc, Sc)
summary = {}
for crit in ('N10', 'N20', 'any'):
    print('====', crit)
    for s in sessions:
        m = sio.loadmat(BASE + s + '/matx.mat')['matx']
        a, F, tt = m.T
        succ = {'N10': tt < 10, 'N20': tt < 20, 'any': tt != 30}[crit]
        A, FF, S = curves(a, F, succ)
        cols = [j for j in range(30) if np.median(A[:, j]) <= 0.56]
        av = np.array([np.median(A[:, j]) for j in cols])
        est = [col_low(FF[:, j], S[:, j]) for j in cols]
        txt = ' '.join(f'{v:.2f}:{e[0]:.2f}{"<" if e[1]=="left" else ("?" if e[1]=="none" else "")}' for v, e in zip(av, est))
        ok = [i for i, e in enumerate(est) if e[1] == 'ok']
        if len(ok) >= 3:
            sl = np.polyfit(np.log(av[ok]), np.log([est[i][0] for i in ok]), 1)[0]
            bs = []
            for _ in range(NB):
                xs, ys = [], []
                for i, j in enumerate(cols):
                    idx = rng.integers(0, FF.shape[0], FF.shape[0])
                    e = col_low(FF[idx, j], S[idx, j])
                    if e[1] == 'ok':
                        xs.append(np.log(av[i])); ys.append(np.log(e[0]))
                if len(xs) >= 3 and np.ptp(xs) > 0.5:
                    bs.append(np.polyfit(xs, ys, 1)[0])
            lo, hi = np.percentile(bs, [2.5, 97.5])
            summary.setdefault(crit, []).append((s, sl, lo, hi))
            print(f'{s:32s} Fmin(F_lowest={FF.min():.2f}) {txt} | slope a<=0.55: {sl:+.2f} [{lo:+.2f},{hi:+.2f}] -> attack-only speed exp 2s = {2*sl:+.2f} [{2*lo:+.2f},{2*hi:+.2f}]')
        else:
            print(f'{s:32s} Fmin(F_lowest={FF.min():.2f}) {txt} | too few uncensored columns')
