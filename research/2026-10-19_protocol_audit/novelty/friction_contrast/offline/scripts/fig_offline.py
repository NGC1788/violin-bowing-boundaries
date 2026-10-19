"""Figure for the offline friction-contrast continuation (Claude, 2026-10-19).
Inputs: offline/results/friction_offline.json, offline/results/levels.json. Usage: python -I fig_offline.py <results_dir> <out.png>"""
import json
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

R, OUT = sys.argv[1], sys.argv[2]
F = json.load(open(os.path.join(R, 'friction_offline.json')))
LV = json.load(open(os.path.join(R, 'levels.json')))
SURF, INK, INK2, GRID = '#fcfcfb', '#0b0b0b', '#52514e', '#e4e3df'
SER = {0.05: ('#2a78d6', 'o'), 0.10: ('#eb6834', 's'), 0.15: ('#1baf7a', '^'), 0.20: ('#eda100', 'D')}
plt.rcParams.update({'font.size': 9, 'axes.edgecolor': INK2, 'axes.labelcolor': INK, 'xtick.color': INK2, 'ytick.color': INK2,
                     'axes.facecolor': SURF, 'figure.facecolor': SURF, 'axes.grid': True, 'grid.color': GRID, 'grid.linewidth': 0.6,
                     'axes.spines.top': False, 'axes.spines.right': False, 'legend.frameon': False})
H = [w for w in F['windows'] if w['k'] == 1 and w['helmholtz'] and np.isfinite(w.get('dmu', np.nan))]
Fb = np.array([w['Fb'] for w in H]); v = np.array([w['v0'] for w in H]); be = np.array([w['beta'] for w in H])
mud = np.array([w['mu_d'] for w in H]); mpk = np.array([w['mu_peak'] for w in H]); s = np.array([w['s_core'] for w in H])
hy = np.array([w['hyst'] for w in H]); hci = np.array([w['hyst_ci'] for w in H])
fig, ax = plt.subplots(2, 3, figsize=(15, 9.2))
fig.suptitle('Friction at the bow (IWK reconstructions, one string: type B T2) and what it implies for the boundaries — offline continuation',
             x=0.01, ha='left', fontsize=11, color=INK)


def by_speed(a, x, y, yerr=None, label=True):
    for vv, (c, m) in SER.items():
        k = np.isclose(v, vv)
        if k.any():
            a.scatter(x[k], y[k], s=36, color=c, marker=m, edgecolor=SURF, linewidth=0.8, zorder=3, label=f'v {vv:.2f} m/s' if label else None)
            if yerr is not None:
                a.errorbar(x[k], y[k], yerr=yerr[:, k], fmt='none', ecolor=c, elinewidth=1, zorder=2)


# (a) mu_d vs power
a = ax[0, 0]
by_speed(a, Fb * v, mud)
c = F['control_variable']['mu_d']['power Fb*v']['coef']
xx = np.geomspace((Fb * v).min(), (Fb * v).max(), 50)
a.plot(xx, np.exp(c[0]) * xx ** c[1], color=INK2, lw=1.2, zorder=1)
a.set_xscale('log'); a.set_yscale('log')
a.set_xlabel('bow force × bow speed  Fb·v  (W)'); a.set_ylabel('slip-core friction coefficient μ_d')
a.set_title(f'(a) μ_d collapses on Fb·v: μ_d ∝ (Fb v)^{c[1]:.2f}, R² 0.95', loc='left', fontsize=9.5)
a.legend(loc='upper right', fontsize=8)
# (b) residual vs measured slip speed
a = ax[0, 1]
res = np.log(mud) - (c[0] + c[1] * np.log(Fb * v))
by_speed(a, s, res, label=False)
a.axhline(0, color=INK2, lw=1)
a.set_xscale('log')
pc = F['partial_corr_mu_d_s_given_power']
A = F['control_variable']['mu_d']
a.set_xlabel('measured sliding speed in the slip core  (m/s)'); a.set_ylabel('log μ_d − power-law fit')
a.set_title(f'(b) no slip-speed dependence left (partial r {pc:+.2f})', loc='left', fontsize=9.5)
a.text(0.42, 0.04, f"AIC: power {A['power Fb*v']['aic']:.0f}, slip speed {A['slip speed s_core (measured)']['aic']:.0f}, "
       f"force {A['force Fb']['aic']:.0f}\nholds for Fb ≥ 1 N and for ±0.05 N force offsets", transform=a.transAxes, fontsize=8, color=INK2)
# (c) hysteresis vs force
a = ax[0, 2]
by_speed(a, Fb, hy, yerr=np.abs(hci.T - hy), label=False)
a.axhline(0, color=INK2, lw=1)
a.set_xscale('log')
a.set_xlabel('bow force Fb (N)'); a.set_ylabel('μ(sliding speeds up) − μ(slows down)')
a.set_title('(c) friction–velocity loop grows with force, not speed', loc='left', fontsize=9.5)
a.text(0.36, 0.04, 'per-cycle median, 95 % cycle-bootstrap CI\nlog-force slope +0.14/ln N, speed −0.01±0.02', transform=a.transAxes, fontsize=8, color=INK2)
# (d) absolute F_max vs same-specimen brackets
a = ax[1, 0]
D = [d for d in F['absolute_upper_same_specimen'] if np.isfinite(d['F_first_x_above'])]
lab_done = set()
for i, d in enumerate(D):
    a.plot([i, i], [d['F_last_H'], d['F_first_x_above']], color=INK, lw=6, alpha=0.18, solid_capstyle='butt', zorder=1)
    a.errorbar(i - 0.12, d['F_pred'], yerr=[[d['F_pred'] - d['F_pred_ci'][0]], [d['F_pred_ci'][1] - d['F_pred']]], fmt='o', color='#eb6834',
               ms=7, mec=SURF, zorder=3, label='steady in-cycle contrast μ_peak − μ_d' if i == 0 else None)
    e = [x for x in F['attack'] if abs(x['beta'] - d['beta']) < 2e-3 and x['v0'] == d['v'] and abs(x['Fb1'] - d['F_last_H']) < 1e-6]
    if e:
        cc = e[0]['mu_first'] - e[0]['mu_d']
        a.scatter(i + 0.12, 2 * 1.1899 * d['v'] / (d['beta'] * cc), marker='s', s=44, color='#1baf7a', edgecolor=SURF, zorder=3,
                  label=None if 'att' in lab_done else 'first-breakaway (attack) μ_first − μ_d')
        lab_done.add('att')
a.text(0.02, 0.03, 'grey bar = last Helmholtz → first non-Helmholtz steady window\n(same strokes as the friction data); Z = 1.1899 kg/s',
       transform=a.transAxes, fontsize=8, color=INK2)
a.set_xticks(range(len(D))); a.set_xticklabels([f"β {d['beta']:.3f}\nv {d['v']:.2f}" for d in D])
a.set_yscale('log'); a.set_ylabel('bow force (N)'); a.set_xlim(-0.6, len(D) - 0.4)
a.set_title('(d) Schelleng F_max = 2Zv/(βΔμ) vs the same string\'s bracket (grey)', loc='left', fontsize=9.5)
a.legend(loc='upper right', fontsize=8)
# (e) friction-only test across strings
a = ax[1, 1]
T = LV['schelleng_test']
tags = ['A1', 'A2', 'B1', 'C1', 'D1', 'A_T2', 'B_T2s1', 'B_T2r1', 'B_T2r2', 'C_T2']
y = np.arange(len(tags))[::-1]
for yi, t in zip(y, tags):
    r = T[t]['friction_only_residual']; ci = T[t]['friction_only_residual_ci']
    a.plot(ci, [yi, yi], color='#2a78d6', lw=2, solid_capstyle='round')
    a.scatter(r, yi, color='#2a78d6', s=36, edgecolor=SURF, zorder=3)
a.axvline(0, color=INK2, lw=1)
a.set_yticks(y); a.set_yticklabels(tags)
a.set_xlabel('(L_lower − L_upper) − log(Z/Z_A)   (log units)')
a.set_title('(e) a per-string Δμ moves both boundaries together (0);\n     every non-A string widens the band instead', loc='left', fontsize=9.5)
# (f) upper speed exponent: implied vs measured
a = ax[1, 2]
items = []
imp = F['implied_upper']['selfconsistent']['const_peak']
items.append(('friction law (structured)', imp['speed'], imp['speed_ci'], '#eb6834'))
for nm, lab in (('upper, Fb >= 1.2 N', 'friction law, Δμ fit Fb≥1.2 N'), ('upper, Fb >= 2.4 N', 'friction law, Δμ fit Fb≥2.4 N')):
    d = F['implied_direct'][nm]
    items.append((lab, d['speed'], d['speed_ci'], '#eb6834'))
for t in ['A1', 'A2', 'B1', 'C1', 'D1', 'A_T2', 'B_T2s1', 'B_T2r1', 'B_T2r2', 'C_T2']:
    L = LV['free_laws'][t]['upper']
    items.append((f'measured {t}', L['speed'], L['speed_ci'], '#2a78d6'))
y = np.arange(len(items))[::-1]
for yi, (lab, e, ci, col) in zip(y, items):
    a.plot(ci, [yi, yi], color=col, lw=2, solid_capstyle='round')
    a.scatter(e, yi, color=col, s=36, edgecolor=SURF, zorder=3)
a.axvline(1, color=INK2, lw=1, ls='--')
a.text(1.01, y[0] + 0.3, 'Schelleng, constant Δμ', fontsize=8, color=INK2)
a.set_yticks(y); a.set_yticklabels([i[0] for i in items], fontsize=8)
a.set_xlabel('upper-boundary speed exponent  d log F_max / d log v')
a.set_title('(f) upper speed exponent: friction law vs measured', loc='left', fontsize=9.5)
from matplotlib.ticker import NullFormatter, ScalarFormatter  # noqa: E402
for a in ax.flat:
    for axis in (a.xaxis, a.yaxis):
        if axis.get_scale() == 'log':
            f = ScalarFormatter(); f.set_scientific(False)
            axis.set_major_formatter(f); axis.set_minor_formatter(NullFormatter())
ax[0, 0].set_xticks([0.02, 0.05, 0.1, 0.2, 0.5]); ax[0, 0].set_yticks([0.45, 0.5, 0.6, 0.7, 0.8])
ax[0, 1].set_xticks([0.5, 1, 2, 3]); ax[0, 2].set_xticks([0.3, 0.5, 1, 2, 5]); ax[1, 0].set_yticks([1, 2, 3, 5, 8])
plt.tight_layout(rect=(0, 0, 1, 0.96))
plt.savefig(OUT, dpi=110)
