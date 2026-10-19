"""Summary figure: measured friction characteristics (IWK type B T2, reconstructed friction) and the boundary tests.
Usage: python -I fig_summary.py <reconstructions dir> <results dir> <out.png>"""
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

rec, res, out = sys.argv[1], sys.argv[2], sys.argv[3]
SURF, INK, INK2, GRID = '#fcfcfb', '#0b0b0b', '#52514e', '#e4e3df'
S1, S2, S3 = '#2a78d6', '#eb6834', '#1baf7a'
SEQ = {0.05: '#86b6ef', 0.10: '#3987e5', 0.15: '#1c5cab', 0.20: '#0d366b'}
MK = {0.05: 'o', 0.10: 's', 0.15: '^', 0.20: 'D'}
plt.rcParams.update({'font.size': 9, 'axes.edgecolor': INK2, 'axes.labelcolor': INK, 'xtick.color': INK2, 'ytick.color': INK2,
                     'axes.facecolor': SURF, 'figure.facecolor': SURF, 'axes.grid': True, 'grid.color': GRID, 'grid.linewidth': 0.6,
                     'axes.spines.top': False, 'axes.spines.right': False, 'legend.frameon': False})
A = json.load(open(os.path.join(res, 'analysis.json')))


def logfmt(axis, ticks):
    axis.xaxis.set_major_locator(matplotlib.ticker.FixedLocator(ticks))
    axis.xaxis.set_major_formatter(matplotlib.ticker.FixedFormatter([f'{t:g}' for t in ticks]))
    axis.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())

W = A['windows']
fig, ax = plt.subplots(2, 3, figsize=(15, 9.2))

# (a) cycle-averaged friction loops at three forces, beta 0.0405, v 0.05
a_ = ax[0, 0]
for (n, lab), col in zip(((2, '0.58 N'), (4, '1.54 N'), (9, '5.26 N')), (S1, S2, S3)):
    d = np.loadtxt(os.path.join(rec, f'friction_velocity_reconstruction_{n}_1.csv'), delimiter=',', skiprows=1)
    Fb = [w['Fb'] for w in W if w['n'] == n][0]
    mu, s = d[:, 0] / Fb, 0.05 - d[:, 1]
    sel = slice(0, int(30 / 0.02))  # first 30 ms
    a_.plot(s[sel], mu[sel], color=col, lw=0.8, alpha=0.9, label=f'Fb {lab}')
a_.legend(loc='lower right', fontsize=8)
a_.set_xlabel('sliding speed v0 − V (m/s)'); a_.set_ylabel('μ = F_friction / Fb')
a_.set_title('(a) friction–velocity loops, β 0.0405, v0 0.05 m/s', color=INK, fontsize=9.5, loc='left')

# (b) mu_peak and mu_d vs Fb*v
a_ = ax[0, 1]
P = np.array([w['Fb'] * w['v'] for w in W])
a_.scatter(P, [w['mu_peak'] for w in W], s=22, color=S2, edgecolor=SURF, linewidth=0.8, zorder=3)
a_.scatter(P, [w['mu_d'] for w in W], s=22, color=S1, edgecolor=SURF, linewidth=0.8, zorder=3)
sl = A['structured_law']
pp = np.exp(np.linspace(np.log(P.min()), np.log(P.max()), 50))
a_.plot(pp, np.exp(sl['a']) * pp ** sl['c'], color=S1, lw=1.5)
a_.axhline(sl['mu_peak'], color=S2, lw=1.5)
a_.set_xscale('log')
a_.text(P.max(), sl['mu_peak'] + 0.02, f"peak μ ≈ {sl['mu_peak']:.2f} (flat)", ha='right', va='bottom', color=INK, fontsize=8)
a_.text(P.max(), np.exp(sl['a']) * P.max() ** sl['c'] - 0.03, f"slip-core μ_d ∝ (Fb·v)^{sl['c']:.2f}", ha='right', va='top', color=INK, fontsize=8)
a_.set_xlabel('Fb · v0 (W)'); a_.set_ylabel('μ'); a_.set_ylim(0.35, 1.0)
logfmt(a_, [0.02, 0.05, 0.1, 0.2, 0.5])
a_.set_title('(b) peak vs slip-core friction (31 Helmholtz windows)', color=INK, fontsize=9.5, loc='left')

# (c) contrast vs Fb by speed
a_ = ax[0, 2]
for vv in (0.05, 0.10, 0.15, 0.20):
    ww = [w for w in W if abs(w['v'] - vv) < 1e-6]
    a_.scatter([w['Fb'] for w in ww], [w['dmu'] for w in ww], s=26, color=SEQ[vv], marker=MK[vv], edgecolor=SURF, linewidth=0.8,
               label=f'v0 {vv:.2f} m/s', zorder=3)
a_.set_xscale('log'); a_.set_xlabel('bow force Fb (N)'); a_.set_ylabel('contrast Δμ = μ_peak − μ_d')
logfmt(a_, [0.3, 0.5, 1, 2, 5])
a_.legend(loc='upper left', fontsize=8)
a_.set_title('(c) friction contrast grows with force, saturates ≈0.35', color=INK, fontsize=9.5, loc='left')

# (d) hysteresis vs Fb
a_ = ax[1, 0]
for vv in (0.05, 0.10, 0.15, 0.20):
    ww = [w for w in W if abs(w['v'] - vv) < 1e-6]
    a_.scatter([w['Fb'] for w in ww], [w['hyst'] for w in ww], s=26, color=SEQ[vv], marker=MK[vv], edgecolor=SURF, linewidth=0.8,
               label=f'v0 {vv:.2f} m/s', zorder=3)
a_.axhline(0, color=INK2, lw=0.8)
a_.set_xscale('log'); a_.set_xlabel('bow force Fb (N)'); a_.set_ylabel('μ(accelerating) − μ(decelerating) at same slip speed')
logfmt(a_, [0.3, 0.5, 1, 2, 5])
a_.legend(loc='upper left', fontsize=8)
a_.set_title('(d) loop hysteresis (thermal-like) appears above ~1 N', color=INK, fontsize=9.5, loc='left')

# (e) measured vs Schelleng-required contrast at the B_T2 upper boundary
a_ = ax[1, 1]
L = A['absolute_upper_local']
lab = [f"β {r['beta']:.3f}\nv {r['v']:.2f}" for r in L]
x = np.arange(len(L))
for i, r in enumerate(L):
    a_.plot([i, i], [r['dmu'], r['dmu_required']], color=GRID, lw=2, zorder=1)
a_.scatter(x, [r['dmu'] for r in L], s=34, color=S1, zorder=3, label='measured Δμ (nearest Helmholtz window)')
a_.scatter(x, [r['dmu_required'] for r in L], s=34, color=S2, marker='D', zorder=3, label='required: 2Zv / (β F_up,meas)')
a_.set_xticks(x); a_.set_xticklabels(lab, fontsize=7)
a_.set_ylabel('friction contrast'); a_.set_ylim(0, 1.05)
a_.legend(loc='upper left', fontsize=8)
rat = [r['ratio'] for r in L]
a_.set_title(f'(e) Schelleng F_max with measured Δμ: {min(rat):.1f}–{max(rat):.1f}× too high', color=INK, fontsize=9.5, loc='left')

# (f) cross-string: contrast change required by upper vs lower levels
a_ = ax[1, 2]
cr = A['cross']
du, dl = np.array(cr['d_up']), np.array(cr['d_lo'])
lim = (-0.4, 0.9)
a_.plot(lim, lim, color=INK2, lw=0.8, ls='--')
a_.text(0.62, 0.70, 'one per-string Δμ factor\n(Schelleng) ⇒ points on this line', fontsize=7.5, color=INK2, ha='right', va='bottom', rotation=0)
a_.axhline(0, color=GRID, lw=1); a_.axvline(0, color=GRID, lw=1)
a_.scatter(du, dl, s=30, color=S1, edgecolor=SURF, linewidth=0.8, zorder=3)
OFF = {'C1': (5, -9), 'B_T2r1': (-44, 5), 'B_T2r2': (5, -4), 'A_T2': (5, 3)}
for t, xx, yy in zip(cr['tags'], du, dl):
    a_.annotate(t, (xx, yy), xytext=OFF.get(t, (4, 3)), textcoords='offset points', fontsize=7.5, color=INK)
a_.set_xlim(lim); a_.set_ylim(-0.25, 0.9)
a_.set_xlabel('log(Δμ_X/Δμ_A) required by upper level'); a_.set_ylabel('log(Δμ_X/Δμ_A) required by lower level')
a_.set_title('(f) 6/6 non-A diagrams need opposite contrast changes', color=INK, fontsize=9.5, loc='left')

fig.suptitle('Reconstructed bowing-point friction (IWK, Zenodo 15879128; one string: type B, T2) vs playability-boundary levels',
             color=INK, fontsize=11, x=0.01, ha='left')
fig.tight_layout(rect=(0, 0, 1, 0.96))
fig.savefig(out, dpi=130)
