"""Lower force limit vs acceleration from the dataset authors' own per-stroke transient times (matx.mat)."""
import sys, glob, numpy as np, scipy.io as sio
sys.path.insert(0, '/Users/jeongjihoon/violin-rne/novelty/attack_data/scripts')
import boundary_fit as bf
BASE = '/Users/jeongjihoon/violin-rne/novelty/attack_data/raw/zenodo/raw_data/'
sessions = ['first_repetitions/2023-10-18', 'first_repetitions/2023-10-19_r', 'first_repetitions/2023-10-20',
            'first_repetitions/2023-10-20_r', 'after_dismounting/2023-11-10', 'after_dismounting/2023-11-13_r',
            'different_betas/30/2023-11-15', 'different_betas/55/2023-11-14', 'different_betas/85/2023-11-14',
            'different_betas/115/2023-11-15']
beta = {'30': 3 / 70, '55': 5.5 / 70, '85': 8.5 / 70, '115': 11.5 / 70}
rng = np.random.default_rng(1)
NB = int(sys.argv[1]) if len(sys.argv) > 1 else 100
quad = True
print('session beta N | n succ | L(0.15) L(1) L(3) N | slope@0.15 @0.5 @1 @3 [boot 95%] | authors F_R=c a+k (c,k) | attack-only speed exp (v .05->.2)')
rows = []
for s in sessions:
    m = sio.loadmat(BASE + s + '/matx.mat')['matx']
    lc = sio.loadmat(BASE + s + '/lim_coeffs.mat')['lim_coeffs'][0]
    a, F, tt = m[:, 0], m[:, 1], m[:, 2]
    b = beta.get(s.split('/')[1], 5.5 / 70)
    for N in (5, 10, 20):
        y = (tt < N).astype(float)
        ok = (a > 0) & (F > 0)
        r = bf.fit(F[ok], a[ok], y[ok], quad=quad)
        d = bf.describe(r, quad)
        A = np.array([0.15, 0.5, 1.0, 3.0])
        Ls = bf.lower_force(r, quad, A); sl = bf.lower_slope(r, quad, A)
        # attack-only exponent: a = v^2/(2*0.048)
        av = np.array([0.05, 0.2]) ** 2 / 0.096
        Lv = bf.lower_force(r, quad, av)
        ex = np.log(Lv[1] / Lv[0]) / np.log(4)
        boots = []
        for _ in range(NB):
            idx = rng.integers(0, ok.sum(), ok.sum())
            rb = bf.fit(F[ok][idx], a[ok][idx], y[ok][idx], quad=quad, start=list(r.x))
            boots.append(np.r_[bf.lower_slope(rb, quad, A), np.log(bf.lower_force(rb, quad, av)[1] / bf.lower_force(rb, quad, av)[0]) / np.log(4)])
        boots = np.array(boots)
        lo, hi = np.percentile(boots, [2.5, 97.5], axis=0) if NB > 0 else (np.full(5, np.nan), np.full(5, np.nan))
        print(f'{s:32s} {b:.4f} N={N:2d} | {ok.sum()} {int(y[ok].sum()):3d} | ' +
              ' '.join(f'{v:.2f}' for v in Ls) + ' | ' +
              ' '.join(f'{v:+.2f}[{l:+.2f},{h:+.2f}]' for v, l, h in zip(sl, lo[:4], hi[:4])) +
              f' | c_R={lc[2]:.3f} k_R={lc[3]:.3f} | attack-only {ex:+.2f} [{lo[4]:+.2f},{hi[4]:+.2f}] (extrapolated a 0.026-0.42)', flush=True)
        rows.append((s, b, N, *Ls, *sl, *lo[:4], *hi[:4], ex, lo[4], hi[4], lc[2], lc[3], d['u1'], d['pmax']))
np.save('/Users/jeongjihoon/violin-rne/novelty/attack_data/work/authors_labels_fits.npy', np.array(rows, dtype=object), allow_pickle=True)
