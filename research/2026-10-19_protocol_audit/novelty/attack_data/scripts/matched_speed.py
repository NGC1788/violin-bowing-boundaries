"""Pure attack effect: lower force limit vs acceleration at MATCHED bow speed.
success(v*) = sustained Helmholtz (authors' TT != 30) that begins while the bow speed is still <= v*,
v_H = a * (t_slip + TT*T0), t_slip = kappa * sqrt(2 x_s / a), x_s = mu_s F beta (1-beta) L / T
(mu_s, T per stroke from the authors' stat.mat / tns.mat; kappa = 0.58 calibrated on our own
first-slip detections, sensitivity 0.45 and 1.0). Lower boundary per acceleration band via lowerfit."""
import sys, numpy as np, scipy.io as sio
sys.path.insert(0, '/Users/jeongjihoon/violin-rne/novelty/attack_data/scripts')
import lowerfit as lf
BASE = '/Users/jeongjihoon/violin-rne/novelty/attack_data/raw/zenodo/raw_data/'
T0 = 1 / 97.86; L = 0.7
sessions = {'first_repetitions/2023-10-18': 5.5 / 70, 'first_repetitions/2023-10-20': 5.5 / 70,
            'after_dismounting/2023-11-10': 5.5 / 70, 'different_betas/55/2023-11-14': 5.5 / 70,
            'first_repetitions/2023-10-19_r': 5.5 / 70, 'first_repetitions/2023-10-20_r': 5.5 / 70,
            'after_dismounting/2023-11-13_r': 5.5 / 70, 'different_betas/85/2023-11-14': 8.5 / 70}
AMAX = float(sys.argv[3]) if len(sys.argv) > 3 else 3.5
kappa = float(sys.argv[1]) if len(sys.argv) > 1 else 0.58
NB = int(sys.argv[2]) if len(sys.argv) > 2 else 100
rng = np.random.default_rng(11)
out = {}
for vstar in (0.06, 0.08, 0.10, 0.12):
    print(f'=== v* = {vstar} m/s  (kappa {kappa})')
    for s, beta in sessions.items():
        m = sio.loadmat(BASE + s + '/matx.mat')['matx']; st = sio.loadmat(BASE + s + '/stat.mat')['stat']
        tn = sio.loadmat(BASE + s + '/tns.mat')['tns']
        a, F, tt = m.T
        mus = np.where(np.isfinite(st[:, 2]) & (st[:, 2] > 0), st[:, 2], np.nanmedian(st[:, 2]))
        T = tn[:, 2]
        tsl = kappa * np.sqrt(2 * mus * F * beta * (1 - beta) * L / T / a)
        vH = a * (tsl + tt * T0)
        succ = (tt != 30) & (vH <= vstar)
        # deadline must lie inside the analysed acceleration (v_end = sqrt(0.12 a) >= v*): true for a >= v*^2/0.12
        aa, FF, yy = lf.select(a, F, succ, amin=max(0.14, vstar ** 2 / 0.12), amax=AMAX)
        if len(aa) < 20 or yy.sum() < 8 or np.ptp(np.log(aa)) < 0.5:
            print(f'{s:32s} insufficient (n={len(aa)}, succ={int(yy.sum())})'); continue
        r = lf.fit(aa, FF, yy)
        bs = []
        for _ in range(NB):
            idx = rng.integers(0, len(aa), len(aa))
            bs.append(lf.fit(aa[idx], FF[idx], yy[idx], start=list(r.x)).x[1])
        lo, hi = np.percentile(bs, [2.5, 97.5])
        arange = (aa.min(), aa.max())
        out[(vstar, s)] = (r.x[1], lo, hi, np.exp(r.x[0]), arange, len(aa), int(yy.sum()))
        print(f'{s:32s} a in [{arange[0]:.2f},{arange[1]:.2f}] n={len(aa)} succ={int(yy.sum())}  s_att={r.x[1]:+.2f} [{lo:+.2f},{hi:+.2f}]  F_low(1)={np.exp(r.x[0]):.2f} N  F_low(0.3)={np.exp(r.x[0]+r.x[1]*np.log(0.3)):.2f}  width={np.exp(r.x[2]):.2f} pmax={1/(1+np.exp(-r.x[3])):.2f}', flush=True)
import pickle
pickle.dump(out, open(f'/Users/jeongjihoon/violin-rne/novelty/attack_data/work/matched_speed_k{kappa}_amax{AMAX}.pkl', 'wb'))
