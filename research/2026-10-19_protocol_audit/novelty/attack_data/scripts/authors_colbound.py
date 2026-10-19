import sys, numpy as np, scipy.io as sio
sys.path.insert(0, '/Users/jeongjihoon/violin-rne/novelty/attack_data/scripts')
import colbound as cb
BASE = '/Users/jeongjihoon/violin-rne/novelty/attack_data/raw/zenodo/raw_data/'
sessions = ['first_repetitions/2023-10-18', 'first_repetitions/2023-10-19_r', 'first_repetitions/2023-10-20',
            'first_repetitions/2023-10-20_r', 'after_dismounting/2023-11-10', 'after_dismounting/2023-11-13_r',
            'different_betas/30/2023-11-15', 'different_betas/55/2023-11-14', 'different_betas/85/2023-11-14',
            'different_betas/115/2023-11-15']
rng = np.random.default_rng(2)
def P_acc(a):  # periods available during the 6 cm acceleration (f0 97.86 Hz)
    return np.sqrt(0.12 / a) * 97.86
for N in (5, 10, 15, 20, 'any'):
    print(f'=== success: TT < {N}' if N != 'any' else '=== success: any sustained HM within the analysed stroke (TT != 30)')
    for s in sessions:
        m = sio.loadmat(BASE + s + '/matx.mat')['matx']
        lc = sio.loadmat(BASE + s + '/lim_coeffs.mat')['lim_coeffs'][0]
        a, F, tt = m.T
        succ = (tt != 30) if N == 'any' else (tt < N)
        cur = cb.lower_curve(a, F, succ)
        nleft = sum(c[2] == 'left' for c in cur); nnone = sum(c[2] == 'none' for c in cur)
        sl, ic, n = cb.slope_fit(cur)
        ci = cb.boot_slope(cur, rng=rng)
        sl_lo, _, n_lo = cb.slope_fit(cur, amax=1.0)
        sl_hi, _, n_hi = cb.slope_fit(cur, amin=1.0)
        c, k = cb.lin_fit(cur)
        def at(x):
            pts = [(cc[0], cc[1]) for cc in cur if cc[2] == 'ok']
            if not pts: return np.nan
            pa = np.array(pts); j = np.argmin(abs(np.log(pa[:, 0] / x)))
            return np.median(pa[max(0, j - 1):j + 2, 1])
        print(f'{s:32s} ok/left/none {n}/{nleft}/{nnone} | Flow(0.15,0.5,1,2,3)= ' + ' '.join(f'{at(x):.2f}' for x in (0.15, 0.5, 1, 2, 3)) +
              f' | loglog slope {sl:+.2f} [{ci[0]:+.2f},{ci[1]:+.2f}]  a<=1: {sl_lo:+.2f}({n_lo})  a>=1: {sl_hi:+.2f}({n_hi}) | lin F=c a+k: c={c:.2f} k={k:.2f} | authors c_R={lc[2]:.2f} k_R={lc[3]:.2f}')
