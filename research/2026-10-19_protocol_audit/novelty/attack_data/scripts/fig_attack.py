import sys, pickle, numpy as np, scipy.io as sio
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
sys.path.insert(0, '/Users/jeongjihoon/violin-rne/novelty/attack_data/scripts')
import colbound as cb
BASE = '/Users/jeongjihoon/violin-rne/novelty/attack_data/raw/zenodo/raw_data/'
W = '/Users/jeongjihoon/violin-rne/novelty/attack_data/work/'
C1, C2, C3, GR = '#2a78d6', '#eb6834', '#1baf7a', '#8a8984'
plt.rcParams.update({'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False, 'axes.edgecolor': '#52514e',
                     'axes.labelcolor': '#0b0b0b', 'xtick.color': '#52514e', 'ytick.color': '#52514e'})
fig, ax = plt.subplots(1, 3, figsize=(14, 5.0), gridspec_kw=dict(width_ratios=[1.2, 1, 1]))
# (a) Guettler diagram with lower limits for several N
s = 'first_repetitions/2023-10-18'
m = sio.loadmat(BASE + s + '/matx.mat')['matx']; a, F, tt = m.T
lc = sio.loadmat(BASE + s + '/lim_coeffs.mat')['lim_coeffs'][0]
col = np.where(tt == 30, 1.0, np.where(tt >= 20, 0.7, np.clip(tt / 20 * 0.5, 0, 0.5)))
ax[0].scatter(a, F, c=[str(0.15 + 0.8 * (1 - v)) if False else plt.cm.Greys(0.15 + 0.8 * v) for v in col], s=9, linewidths=0)
for crit, c, lab in (('N10', C2, 'within 10 periods'), ('N20', C1, 'within 20 periods (paper)'), ('any', C3, 'sustained by end of 6 cm')):
    succ = {'N10': tt < 10, 'N20': tt < 20, 'any': tt != 30}[crit]
    cur = [x for x in cb.lower_curve(a, F, succ) if x[2] == 'ok']
    ax[0].plot([x[0] for x in cur], [x[1] for x in cur], '-o', color=c, lw=2, ms=3.5, label=lab)
xa = np.linspace(0.15, 3.15, 50)
ax[0].plot(xa, lc[2] * xa + lc[3], '--', color='#0b0b0b', lw=1, label=f'paper fit F_R = {lc[2]:.2f}a + {lc[3]:.2f}')
ax[0].set_xscale('log'); ax[0].set_yscale('log')
ax[0].set_xlabel('bow acceleration a (m/s$^2$)'); ax[0].set_ylabel('bow force F (N)')
ax[0].set_title('(a) Lower force limit vs attack, Dominant G, β=0.079\n(dots: authors\' transient time, dark = slow/failed)', fontsize=9, loc='left')
ax[0].legend(fontsize=7, frameon=True, framealpha=0.9, edgecolor='none', loc='upper left')
ax[0].set_ylim(0.1, 5)
from matplotlib.ticker import FixedLocator, FixedFormatter, NullLocator
for axis, ticks in ((ax[0].xaxis, [0.15, 0.3, 0.5, 1, 2, 3]), (ax[0].yaxis, [0.1, 0.2, 0.5, 1, 2, 4])):
    axis.set_major_locator(FixedLocator(ticks)); axis.set_major_formatter(FixedFormatter([str(t) for t in ticks])); axis.set_minor_locator(NullLocator())
# (b) slopes per session
sessions = ['first_repetitions/2023-10-18', 'first_repetitions/2023-10-19_r', 'first_repetitions/2023-10-20', 'first_repetitions/2023-10-20_r',
            'after_dismounting/2023-11-10', 'after_dismounting/2023-11-13_r', 'different_betas/55/2023-11-14']
short = ['10-18', '10-19r', '10-20', '10-20r', '11-10', '11-13r', '11-14']
rN = pickle.load(open(W + 'lowerfit_N5_N10_N15_N20.pkl', 'rb'))
rA = pickle.load(open(W + 'lowerfit_any.pkl', 'rb'))
rM = pickle.load(open(W + 'matched_speed_k0.58_amax3.5.pkl', 'rb')) if __import__('os').path.exists(W + 'matched_speed_k0.58_amax3.5.pkl') else pickle.load(open(W + 'matched_speed_k0.58.pkl', 'rb'))
rML = pickle.load(open(W + 'matched_speed_k0.58_amax0.8.pkl', 'rb'))
y = np.arange(len(sessions))
for off, c, lab, getter in ((-0.27, C2, 'fixed N=10, a 0.15-3.15', lambda s: rN.get(('N10', s, (0.14, 3.5)))),
                            (-0.09, C3, 'sustained by end of 6 cm', lambda s: rA.get(('any', s, (0.14, 3.5)))),
                            (0.09, C1, 'matched speed v*=0.10 m/s', lambda s: rM.get((0.10, s))),
                            (0.27, GR, 'matched speed, a ≤ 0.8 only', lambda s: rML.get((0.10, s)))):
    for i, s in enumerate(sessions):
        r = getter(s)
        if r is None: continue
        ax[1].plot([r[1], r[2]], [i + off] * 2, color=c, lw=2, solid_capstyle='round')
        ax[1].plot(r[0], i + off, 'o', color=c, ms=5, label=lab if i == 0 else None)
ax[1].axvline(1, color='#0b0b0b', lw=0.8, ls='--'); ax[1].text(1.03, len(sessions) - 0.7, 'Guettler A/D: F ∝ a', fontsize=7)
ax[1].set_yticks(y); ax[1].set_yticklabels(short); ax[1].set_xlim(-0.1, 1.9); ax[1].set_ylim(-0.9, len(sessions) - 0.4)
ax[1].set_xlabel('s = d ln F$_{low}$ / d ln a  (95% stroke bootstrap)')
ax[1].set_title('(b) Log-log slope of the lower limit, 7 sessions at β=0.079', fontsize=9, loc='left')
ax[1].legend(fontsize=7, frameon=False, loc='upper center', bbox_to_anchor=(0.5, -0.16), ncol=2)
# (c) attack-only speed exponent for the IWK design vs measured IWK exponents
cats = ['Guettler\ntheory', 'matched\nspeed,\nall a', 'matched\nspeed,\na ≤ 0.8', 'paper fits\nF=ca+k,\nextrap.']
mm = np.array([rM[(0.10, s)][0] for s in sessions]); ml = np.array([rML[(0.10, s)][0] for s in sessions])
fits = [sio.loadmat(BASE + s + '/lim_coeffs.mat')['lim_coeffs'][0][2:4] for s in sessions]
av = lambda v: v ** 2 / 0.096
e_lin = np.array([np.log((c * av(0.2) + k) / (c * av(0.05) + k)) / np.log(4) for c, k in fits])
vals = [np.array([2.0]), 2 * mm, 2 * ml, e_lin]
for i, v in enumerate(vals):
    ax[2].plot([i] * len(v), v, 'o', color=C1, ms=4, alpha=0.8)
    ax[2].plot([i - 0.25, i + 0.25], [np.median(v)] * 2, color=C1, lw=2)
iwk = {'A1': 0.55, 'A2': 0.72, 'A_T2': 1.02, 'B1': 0.83, 'C1': 0.88, 'D1': 0.88, 'B_T2 r1': 1.18, 'B_T2 r2': 0.76}
ax[2].axhspan(min(iwk.values()), max(iwk.values()), color=C2, alpha=0.15, lw=0)
ax[2].text(3.5, np.mean(list(iwk.values())), 'IWK 2025 measured\nlower-boundary\nspeed exponents\n(8 specimens, t=0.2 s)', fontsize=7, color='#52514e', va='center')
ax[2].set_xticks(range(4)); ax[2].set_xticklabels(cats, fontsize=7); ax[2].set_xlim(-0.5, 4.6)
ax[2].set_ylabel('speed exponent from the attack alone (a = v²/0.096 m)')
ax[2].set_title('(c) Attack-only exponent vs measured IWK exponents', fontsize=9, loc='left')
fig.tight_layout()
fig.savefig(W + 'fig_attack.png', dpi=160)
print('saved')
