"""Offline continuation of the friction-contrast analysis (Claude, 2026-10-19). Data: already-downloaded IWK
friction reconstructions (Zenodo 15879128), one string = type B, T2 (diameter 0.97282 mm = POMA B_T2; session 03-07-2024,
inferred to be B_T2 sample 2). No network. Run with python -I; the data folder is passed as an argument.

Per window (same definitions as ../scripts/friction_metrics.py, re-implemented, plus cycle-bootstrap CIs):
  s = v0 - V_rec (sliding speed), mu = Fr_rec / Fb (Fb = window-mean bow force; no bow-force time series is published).
  slip episodes: s > thr * v0 / beta (thr 0.15; Helmholtz flag also required at thr 0.30), gaps < 0.3 ms merged.
  phase-averaged cycle (aligned at slip onset, 1 ms pre-roll, one nominal period):
    mu_peak = max mu up to the sliding-speed peak (friction at release = effective static coefficient in the cycle)
    mu_d    = mean mu where s > 0.5 s_max during the slip (dynamic coefficient)
    dmu     = mu_peak - mu_d (friction contrast)
    hyst    = mean over s/s_max in {0.3,0.5,0.7} of mu(rising s) - mu(falling s)   (per cycle, median over cycles)
    s_core  = mean measured sliding speed in the slip core; P = cycle mean of Fr * s (frictional power, W)
    stick_err = rms of s during the stick part of the averaged cycle / (v0/beta)   (reconstruction quality; ideal 0)
  CIs: 400 bootstrap resamples of the cycles of the window (cycle-to-cycle variability only; not systematic error).
  Attack window (k = 0): mu_first = max mu up to the first slip's sliding-speed peak (cold 'breakaway' coefficient;
  uses the window-mean force, so it is biased if the force was still ramping).

Cross-window analyses on the steady Helmholtz windows:
  [A] which control variable organises the friction: frictional power (Fb v), slip speed (measured s_core or nominal
      v/beta), force, speed, beta -> OLS on logs, AIC and leave-one-out RMS.
  [B] hysteresis: sign and dependence.
  [C] Schelleng upper limit F (mu_peak - mu_d(F v)) = C v / beta with the measured friction law -> implied boundary
      exponents (window bootstrap CIs) vs measured B_T2/A exponents (from ../results/boundary_laws.json and
      ../offline/results/levels.json if present).
  [D] absolute Schelleng F_max = 2 Z v / (beta dmu) vs the same-specimen brackets (last Helmholtz / first
      non-Helmholtz steady window of the reconstruction strokes at the same beta and v).
  [E] attack windows: cold breakaway coefficient vs steady in-cycle peak.
Usage: python -I friction_offline.py <reconstructions_dir> <out_dir>
"""
import csv
import glob
import json
import os
import sys
from itertools import product

import numpy as np
from scipy.optimize import brentq

F0 = 97.45
Z_PUB, Z_DATA = 1.1899, 1.1786  # POMA B_T2; the dataset's own pluck value (assumed T 160.8 N)
NB = 400
RNG = np.random.default_rng(20261019)


def load(root, n, k):
    a = np.loadtxt(os.path.join(root, f'friction_velocity_reconstruction_{n}_{k}.csv'), delimiter=',', skiprows=1)
    with open(os.path.join(root, f'bowing_parameters_{n}_{k}.csv')) as f:
        p = list(csv.DictReader(f))[0]
    return a, float(p['beta']), float(p['Fb']), float(p['v0'])


def episodes(mask, dt_ms, gap_ms=0.3):
    idx = np.flatnonzero(np.diff(np.r_[0, mask.astype(int), 0]))
    out = []
    for s, e in zip(idx[0::2], idx[1::2]):
        if out and (s - out[-1][1]) * dt_ms < gap_ms:
            out[-1][1] = e
        else:
            out.append([s, e])
    return [e for e in out if e[0] > 0 and e[1] < len(mask)]


def helm_flag(s, t, dt, v0, beta, thr):
    eps = episodes(s > thr * v0 / beta, dt)
    if len(eps) < 2:
        return False, eps
    per_ms = 1000.0 / F0
    onsets = np.array([e[0] for e in eps]); iv = np.diff(t[onsets])
    smax = np.array([s[e[0]:e[1]].max() for e in eps])
    epp = len(eps) / ((t[-1] - t[0]) / per_ms)
    ok = 0.9 <= epp <= 1.1 and np.std(iv) / np.mean(iv) < 0.1 and np.std(smax) / np.mean(smax) < 0.35
    return bool(ok), eps


def interp_branch(sb, mb, x):
    o = np.argsort(sb); sb, mb = sb[o], mb[o]
    if sb.size < 3 or x < sb[0] or x > sb[-1]:
        return np.nan
    return float(np.interp(x, sb, mb))


def cycle_hyst(s_seg, mu_seg):
    ipk = int(np.argmax(s_seg)); sm = s_seg[ipk]
    d = [interp_branch(s_seg[:ipk + 1], mu_seg[:ipk + 1], q * sm) - interp_branch(s_seg[ipk:], mu_seg[ipk:], q * sm) for q in (0.3, 0.5, 0.7)]
    return float(np.nanmean(d)) if np.isfinite(d).any() else np.nan


def avg_metrics(SM, SS, Fb, v0, beta, pre, per_n, thr):
    am, as_ = SM.mean(axis=0), SS.mean(axis=0)
    ipk = int(np.argmax(as_[:pre + per_n // 2])); smx = as_[ipk]
    j1 = pre
    while j1 < len(as_) and as_[j1] > thr * v0 / beta:
        j1 += 1
    core = np.arange(pre, max(j1, pre + 1)); core = core[as_[core] > 0.5 * smx]
    mu_pk = float(am[:ipk + 1].max()); mu_d = float(am[core].mean()) if core.size else np.nan
    stick = slice(min(j1 + 5, len(as_) - 1), len(as_))
    return dict(mu_peak=mu_pk, mu_d=mu_d, dmu=mu_pk - mu_d, s_core=float(as_[core].mean()) if core.size else np.nan,
                P=float(np.mean(am * Fb * as_)), stick_err=float(np.sqrt(np.mean(as_[stick] ** 2)) / (v0 / beta)),
                hyst_avg=cycle_hyst(as_[pre:j1], am[pre:j1]) if j1 - pre > 6 else np.nan)


def window(root, n, k):
    a, beta, Fb, v0 = load(root, n, k)
    t, Fr, V = a[:, 4], a[:, 0], a[:, 1]
    dt = float(np.median(np.diff(t)))
    mu, s = Fr / Fb, v0 - V
    r = dict(n=n, k=k, beta=beta, Fb=Fb, v0=v0)
    h15, eps = helm_flag(s, t, dt, v0, beta, 0.15)
    h30, _ = helm_flag(s, t, dt, v0, beta, 0.30)
    r['helmholtz'] = bool(h15 and h30)
    if k == 0 and eps:
        a0, a1 = eps[0]
        ip = a0 + int(np.argmax(s[a0:a1]))
        r['mu_first'] = float(mu[:ip + 1].max()); r['t_first_ms'] = float(t[a0] - t[0])
        r['mu_d_first'] = float(mu[a0:a1][s[a0:a1] > 0.5 * s[ip]].mean())
    if len(eps) < 3:
        return r
    per_n = int(round(1000.0 / F0 / dt)); pre = int(round(1.0 / dt))
    SM, SS = [], []
    for a0, a1 in eps:
        if a0 - pre >= 0 and a0 - pre + per_n <= len(mu):
            SM.append(mu[a0 - pre:a0 - pre + per_n]); SS.append(s[a0 - pre:a0 - pre + per_n])
    if len(SM) < 3:
        return r
    SM, SS = np.array(SM), np.array(SS)
    r.update(avg_metrics(SM, SS, Fb, v0, beta, pre, per_n, 0.15)); r['n_cycles'] = len(SM)
    hy = np.array([cycle_hyst(s[a0:a1], mu[a0:a1]) for a0, a1 in eps])
    r['hyst'] = float(np.nanmedian(hy))
    boots = {key: [] for key in ('mu_peak', 'mu_d', 'dmu', 'hyst')}
    for _ in range(NB):
        i = RNG.integers(0, len(SM), len(SM))
        m = avg_metrics(SM[i], SS[i], Fb, v0, beta, pre, per_n, 0.15)
        for key in ('mu_peak', 'mu_d', 'dmu'):
            boots[key].append(m[key])
        boots['hyst'].append(np.nanmedian(hy[RNG.integers(0, len(hy), len(hy))]))
    for key, b in boots.items():
        b = np.array(b, float)
        r[key + '_ci'] = [float(np.nanpercentile(b, 2.5)), float(np.nanpercentile(b, 97.5))]
        r[key + '_se'] = float(np.nanstd(b))
    return r


def ols(X, y):
    X = np.asarray(X, float); y = np.asarray(y, float)
    b, *_ = np.linalg.lstsq(X, y, rcond=None)
    res = y - X @ b; n, p = X.shape
    s2 = res @ res / max(1, n - p)
    se = np.sqrt(np.diag(s2 * np.linalg.inv(X.T @ X)))
    aic = n * np.log(res @ res / n) + 2 * (p + 1)
    H = X @ np.linalg.inv(X.T @ X) @ X.T
    loo = np.sqrt(np.mean((res / (1 - np.diag(H))) ** 2))
    r2 = 1 - res @ res / np.sum((y - y.mean()) ** 2)
    return b, se, aic, loo, r2


def main():
    root, outd = sys.argv[1], sys.argv[2]
    rows = []
    for p in sorted(glob.glob(os.path.join(root, 'bowing_parameters_*_*.csv'))):
        n, k = map(int, os.path.basename(p)[:-4].split('_')[-2:])
        rows.append(window(root, n, k))
    rows.sort(key=lambda r: (r['n'], r['k']))
    out = dict(windows=rows)
    W = {(r['n'], r['k']): r for r in rows}
    H = [r for r in rows if r['k'] == 1 and r['helmholtz'] and np.isfinite(r.get('dmu', np.nan))]
    print(f'steady Helmholtz windows: {len(H)} (of {sum(r["k"] == 1 for r in rows)} steady windows)')
    # cross-check against the earlier run
    try:
        prev = {(r['n'], r['k']): r for r in json.load(open(os.path.join(os.path.dirname(outd.rstrip('/')), '..', 'results', 'friction_metrics.json')))}
        d = [abs(r['dmu'] - prev[(r['n'], 1)]['avg_dmu']) for r in H]
        print(f'  reproduction of the earlier avg_dmu: max |diff| {max(d):.2e}')
    except (OSError, KeyError):
        pass
    # per-condition table
    with open(os.path.join(outd, 'friction_per_window.csv'), 'w', newline='') as f:
        w = csv.writer(f)
        cols = ['n', 'beta', 'v0', 'Fb', 'helmholtz', 'n_cycles', 'mu_peak', 'mu_peak_lo', 'mu_peak_hi', 'mu_d', 'mu_d_lo', 'mu_d_hi',
                'dmu', 'dmu_lo', 'dmu_hi', 'hyst', 'hyst_lo', 'hyst_hi', 's_core', 'P_W', 'stick_err', 'mu_first_attack']
        w.writerow(cols)
        for r in rows:
            if r['k'] != 1:
                continue
            g = lambda key, i=None: (r.get(key + '_ci', [np.nan, np.nan])[i] if i is not None else r.get(key, np.nan))
            w.writerow([r['n'], f"{r['beta']:.4f}", r['v0'], f"{r['Fb']:.3f}", int(r['helmholtz']), r.get('n_cycles', 0)] +
                       [f"{x:.4f}" for x in (g('mu_peak'), g('mu_peak', 0), g('mu_peak', 1), g('mu_d'), g('mu_d', 0), g('mu_d', 1),
                                             g('dmu'), g('dmu', 0), g('dmu', 1), g('hyst'), g('hyst', 0), g('hyst', 1),
                                             r.get('s_core', np.nan), r.get('P', np.nan), r.get('stick_err', np.nan),
                                             W[(r['n'], 0)].get('mu_first', np.nan))])
    Fb = np.array([r['Fb'] for r in H]); v = np.array([r['v0'] for r in H]); be = np.array([r['beta'] for r in H])
    met = {k: np.array([r[k] for r in H]) for k in ('mu_peak', 'mu_d', 'dmu', 'hyst', 's_core', 'P', 'stick_err', 'hyst_avg')}
    se = {k: np.array([r[k + '_se'] for r in H]) for k in ('mu_peak', 'mu_d', 'dmu', 'hyst')}
    print('\n[0] per-window precision (cycle bootstrap) and reconstruction quality')
    for k in ('mu_peak', 'mu_d', 'dmu', 'hyst'):
        print(f'  {k:8s}: median value {np.median(met[k]):+.3f}, median bootstrap SE {np.median(se[k]):.4f}, '
              f'between-window SD {np.std(met[k]):.3f}')
    print(f'  stick-phase sliding speed rms / (v0/beta): median {np.median(met["stick_err"]):.3f}, max {met["stick_err"].max():.3f}')
    print(f'  dmu > 0 with 95 % CI excluding 0: {int(np.sum([r["dmu_ci"][0] > 0 for r in H]))}/{len(H)}; '
          f'hyst CI > 0: {int(np.sum([r["hyst_ci"][0] > 0 for r in H]))}, CI < 0: {int(np.sum([r["hyst_ci"][1] < 0 for r in H]))}')
    print('  per condition (beta, v0, Fb: mu_peak / mu_d / dmu [95 % CI] / hyst):')
    for r in sorted(H, key=lambda r: (r['v0'], r['beta'], r['Fb'])):
        print(f"    {r['beta']:.4f} {r['v0']:.2f} {r['Fb']:5.2f} N: {r['mu_peak']:.3f} / {r['mu_d']:.3f} / {r['dmu']:.3f} "
              f"[{r['dmu_ci'][0]:.3f}, {r['dmu_ci'][1]:.3f}] / {r['hyst']:+.3f} [{r['hyst_ci'][0]:+.3f}, {r['hyst_ci'][1]:+.3f}]")

    # [A] which variable controls the friction
    print('\n[A] control variable of the friction coefficients (OLS on logs; AIC lower = better; LOO = leave-one-out RMS)')
    L = np.log
    designs = {
        'const': np.ones((len(H), 1)),
        'power Fb*v': np.c_[np.ones(len(H)), L(Fb * v)],
        'slip speed s_core (measured)': np.c_[np.ones(len(H)), L(met['s_core'])],
        'slip speed v/beta (nominal)': np.c_[np.ones(len(H)), L(v / be)],
        'force Fb': np.c_[np.ones(len(H)), L(Fb)],
        'speed v': np.c_[np.ones(len(H)), L(v)],
        'Fb, v, beta (free)': np.c_[np.ones(len(H)), L(Fb), L(v), L(be)],
        'power + s_core': np.c_[np.ones(len(H)), L(Fb * v), L(met['s_core'])],
        'power + beta': np.c_[np.ones(len(H)), L(Fb * v), L(be)],
    }
    A = {}
    for y in ('mu_d', 'mu_peak', 'dmu'):
        A[y] = {}
        print(f'  log {y}:')
        for nm, X in designs.items():
            b, s_, aic, loo, r2 = ols(X, L(met[y]))
            A[y][nm] = dict(coef=b.tolist(), se=s_.tolist(), aic=aic, loo=loo, r2=r2)
            print(f'    {nm:30s} AIC {aic:7.1f}  LOO {loo:.3f}  R2 {r2:.2f}  coef ' + ' '.join(f'{c:+.3f}±{e:.3f}' for c, e in zip(b[1:], s_[1:])))
        # measurement floor: relative SE of the window values
        print(f'    (median relative bootstrap SE of {y}: {np.median(se[y] / np.abs(met[y])):.3f})')
    out['control_variable'] = A
    # direct test: at fixed Fb*v, does slip speed matter? partial correlation
    rp = np.corrcoef(np.linalg.lstsq(designs['power Fb*v'], L(met['mu_d']), rcond=None)[0] @ designs['power Fb*v'].T - L(met['mu_d']),
                     L(met['s_core']) - designs['power Fb*v'] @ np.linalg.lstsq(designs['power Fb*v'], L(met['s_core']), rcond=None)[0])[0, 1]
    print(f'  partial corr(log mu_d, log s_core | log Fb v) = {-rp:+.2f}')
    out['partial_corr_mu_d_s_given_power'] = float(-rp)
    # robustness: load-cell zero offsets of tens of mN (handoff 10.16) and low-force windows
    print('  robustness of "power, not slip speed" for mu_d (AIC power / s_core / v-beta; coefficient on log v/beta in the free model):')
    rob = {}
    for nm, msk, dF in (('all', np.ones(len(H), bool), 0.0), ('Fb >= 1 N', Fb >= 1.0, 0.0),
                        ('true Fb = Fb + 0.05 N', np.ones(len(H), bool), 0.05), ('true Fb = Fb - 0.05 N', np.ones(len(H), bool), -0.05)):
        Fx = Fb[msk] + dF; y = L(met['mu_d'][msk] * Fb[msk] / Fx); vx, bx, sx = v[msk], be[msk], met['s_core'][msk]
        one = np.ones(msk.sum())
        a_p = ols(np.c_[one, L(Fx * vx)], y)[2]; a_s = ols(np.c_[one, L(sx)], y)[2]; a_vb = ols(np.c_[one, L(vx / bx)], y)[2]
        bf, sf, *_ = ols(np.c_[one, L(Fx * vx), L(vx / bx)], y)
        rob[nm] = dict(n=int(msk.sum()), aic_power=a_p, aic_s=a_s, aic_vb=a_vb, power_coef=bf[1], vb_coef=bf[2], vb_se=sf[2])
        print(f'    {nm:22s} n {msk.sum():2d}: AIC {a_p:7.1f} / {a_s:7.1f} / {a_vb:7.1f}; mu_d ~ (Fb v)^{bf[1]:+.3f} (v/beta)^{bf[2]:+.3f}±{sf[2]:.3f}')
    out['robustness_mu_d'] = rob

    # [B] hysteresis
    print('\n[B] friction-velocity hysteresis (mu on accelerating minus decelerating branch at equal sliding speed)')
    for nm, X in (('Fb, v, beta', designs['Fb, v, beta (free)']), ('power Fb*v', designs['power Fb*v']), ('force Fb', designs['force Fb'])):
        b, s_, aic, loo, r2 = ols(X, met['hyst'])
        print(f'  hyst ~ log {nm:12s}: coef ' + ' '.join(f'{c:+.3f}±{e:.3f}' for c, e in zip(b[1:], s_[1:])) + f'  AIC {aic:.1f} R2 {r2:.2f}')
    pos = met['hyst'] > 0
    print(f'  hyst > 0 in {pos.sum()}/{len(H)} windows; Fb < 1 N: {np.round(met["hyst"][Fb < 1], 3)}; Fb > 2.4 N median {np.median(met["hyst"][Fb > 2.4]):.3f}')
    print(f'  relative to contrast: median hyst/dmu {np.median(met["hyst"] / met["dmu"]):.2f}')
    out['hyst'] = dict(values=met['hyst'].tolist(), Fb=Fb.tolist())

    # [C] implied Schelleng upper exponents with the measured law, window bootstrap
    print('\n[C] Schelleng upper limit with the measured friction law, F*(mu_pk - mu_d(F v)) = C v/beta^m')
    bl = json.load(open(os.path.join(os.path.dirname(outd.rstrip('/')), '..', 'results', 'boundary_laws.json')))
    Fu_ref = np.sqrt(bl['B_T2r1']['upper']['at']['0.1044_0.1'] * bl['B_T2r2']['upper']['at']['0.1044_0.1'])
    grid_b = np.exp(np.linspace(np.log(0.04), np.log(0.12), 7)); grid_v = np.array([0.05, 0.1, 0.2])

    def implied(idx, variant):
        """variant: 'const_peak' (mu_pk constant) or 'beta_peak' (mu_pk = p0 beta^p1)."""
        xF = L(Fb[idx] * v[idx]); b_d, *_ = np.linalg.lstsq(np.c_[np.ones(len(idx)), xF], L(met['mu_d'][idx]), rcond=None)
        if variant == 'const_peak':
            pk = lambda bb: float(np.median(met['mu_peak'][idx]))
        else:
            b_p, *_ = np.linalg.lstsq(np.c_[np.ones(len(idx)), L(be[idx])], L(met['mu_peak'][idx]), rcond=None)
            pk = lambda bb: float(np.exp(b_p[0] + b_p[1] * np.log(bb)))
        dmu = lambda F, vv, bb: pk(bb) - np.exp(b_d[0]) * (F * vv) ** b_d[1]
        C = Fu_ref * dmu(Fu_ref, 0.1, 0.1044) * 0.1044 / 0.1
        if not np.isfinite(C) or C <= 0:
            return np.nan, np.nan
        pts = []
        for bb, vv in product(grid_b, grid_v):
            f = lambda F: F * dmu(F, vv, bb) - C * vv / bb
            Fz = brentq(lambda F: dmu(F, vv, bb), 1e-6, 1e4) if dmu(1e-6, vv, bb) < 0 < dmu(1e4, vv, bb) else 1e-6
            try:
                pts.append((np.log(bb), np.log(vv), np.log(brentq(f, Fz * 1.0001, 1e4))))
            except ValueError:
                pass
        P_ = np.array(pts)
        if len(P_) < 10:
            return np.nan, np.nan
        bf, *_ = np.linalg.lstsq(np.c_[np.ones(len(P_)), P_[:, 1], P_[:, 0]], P_[:, 2], rcond=None)
        return bf[1], bf[2]
    imp = {}
    for variant in ('const_peak', 'beta_peak'):
        e0 = implied(np.arange(len(H)), variant)
        bs = np.array([implied(RNG.integers(0, len(H), len(H)), variant) for _ in range(300)])
        ci = np.nanpercentile(bs, [2.5, 97.5], axis=0)
        imp[variant] = dict(speed=float(e0[0]), beta=float(e0[1]), speed_ci=ci[:, 0].tolist(), beta_ci=ci[:, 1].tolist())
        print(f'  {variant:10s}: upper ~ v^{e0[0]:+.2f} [{ci[0, 0]:+.2f}, {ci[1, 0]:+.2f}]  beta^{e0[1]:+.2f} [{ci[0, 1]:+.2f}, {ci[1, 1]:+.2f}]'
              f'   (constant contrast: v^+1, beta^-1)')
    # local elasticity at the measured B_T2 upper boundary points (no calibration constant needed)
    b_d, *_ = np.linalg.lstsq(designs['power Fb*v'], L(met['mu_d']), rcond=None)
    pk0 = float(np.median(met['mu_peak']))
    el = []
    for key in ('0.0405_0.05', '0.0405_0.1', '0.0484_0.05', '0.1044_0.05', '0.1044_0.1', '0.1175_0.05', '0.1175_0.2'):
        bb, vv = map(float, key.split('_'))
        Fm = np.sqrt(bl['B_T2r1']['upper']['at'][key] * bl['B_T2r2']['upper']['at'][key])
        mud = np.exp(b_d[0]) * (Fm * vv) ** b_d[1]
        e = -b_d[1] * mud / (pk0 - mud)
        el.append(dict(beta=bb, v=vv, F_up=Fm, mu_d=mud, dmu=pk0 - mud, e=e, speed_exp=(1 - e) / (1 + e), beta_exp=-1 / (1 + e)))
        print(f'  local at B_T2 upper point beta {bb:.4f} v {vv:.2f} (F_up {Fm:.2f} N, Fb v {Fm * vv:.3f} W): contrast elasticity {e:.2f} '
              f'-> v^{(1 - e) / (1 + e):+.2f} beta^{-1 / (1 + e):+.2f}')
    out['implied_upper'] = dict(selfconsistent=imp, local=el)
    # direct power-law regressions of dmu in a force range relevant to each boundary (systematic spread of the method)
    print('  direct log-dmu regressions (dmu ~ Fb^g v^k beta^l) -> F^(1+g) ~ v^(1-k) beta^-(m+l); window bootstrap 95 % CI')
    dirr = {}
    for nm, msk, m in (('upper, Fb >= 1.2 N', Fb >= 1.2, 1), ('upper, Fb >= 2.4 N', Fb >= 2.4, 1), ('lower, Fb <= 1.6 N', Fb <= 1.6, 2)):
        idx0 = np.flatnonzero(msk)
        def ex(ii):
            X = np.c_[np.ones(len(ii)), L(Fb[ii]), L(v[ii]), L(be[ii])]
            b = np.linalg.lstsq(X, L(met['dmu'][ii]), rcond=None)[0]
            return (1 - b[2]) / (1 + b[1]), (-m - b[3]) / (1 + b[1])
        e0 = ex(idx0)
        bs = []
        for _ in range(400):
            ii = idx0[RNG.integers(0, len(idx0), len(idx0))]
            if len(set(np.round(v[ii], 3))) > 1 and len(set(np.round(be[ii], 3))) > 1:
                bs.append(ex(ii))
        ci = np.nanpercentile(np.array(bs), [2.5, 97.5], axis=0)
        dirr[nm] = dict(n=len(idx0), speed=e0[0], beta=e0[1], speed_ci=ci[:, 0].tolist(), beta_ci=ci[:, 1].tolist())
        print(f'    {nm:20s} n {len(idx0):2d}: v^{e0[0]:+.2f} [{ci[0, 0]:+.2f}, {ci[1, 0]:+.2f}]  beta^{e0[1]:+.2f} [{ci[0, 1]:+.2f}, {ci[1, 1]:+.2f}]'
              f'  (constant contrast: v^+1, beta^-{m})')
    out['implied_direct'] = dirr
    P0 = (pk0 / np.exp(b_d[0])) ** (1 / b_d[1])
    print(f'  NOTE the structured law mu_d = {np.exp(b_d[0]):.3f}(Fb v)^{b_d[1]:+.3f} reaches mu_peak {pk0:.3f} (zero contrast) at Fb v = {P0:.4f} W '
          f'(lowest measured {np.min(Fb * v):.3f} W): a lower limit set by vanishing contrast would scale as F ~ 1/v (speed exponent -1).')
    meas = {}
    for tag in ('A1', 'A2', 'B_T2r1', 'B_T2r2'):
        meas[tag] = (bl[tag]['upper']['b_speed'], bl[tag]['upper']['c_beta'])
    lv = os.path.join(outd, 'levels.json')
    if os.path.exists(lv):
        LJ = json.load(open(lv))
        for tag, d in LJ.get('free_laws', {}).items():
            meas[tag] = (d['upper']['speed'], d['upper']['beta'])
    print('  measured upper exponents (free censored law per diagram, t = 0.2 s): ' +
          '; '.join(f'{t} v^{a:+.2f} beta^{b:+.2f}' for t, (a, b) in meas.items()))
    out['measured_upper_exponents'] = meas

    # [D] absolute F_max vs same-specimen brackets
    print('\n[D] absolute Schelleng F_max = 2 Z v/(beta dmu) vs same-specimen brackets (reconstruction strokes, steady windows)')
    conds = {}
    for r in rows:
        if r['k'] == 1:
            conds.setdefault((round(r['beta'], 3), r['v0']), []).append(r)
    D_ = []
    for (bb, vv), lst in sorted(conds.items()):
        lst = sorted(lst, key=lambda r: r['Fb'])
        hs = [r for r in lst if r['helmholtz'] and np.isfinite(r.get('dmu', np.nan))]
        if not hs:
            continue
        last = hs[-1]
        above = [r for r in lst if r['Fb'] > last['Fb'] and not r['helmholtz']]
        hi = above[0]['Fb'] if above else np.inf
        for Z, zn in ((Z_PUB, 'POMA'), (Z_DATA, 'pluck')):
            Fp = 2 * Z * vv / (last['beta'] * last['dmu'])
            Fp_ci = [2 * Z * vv / (last['beta'] * x) for x in last['dmu_ci'][::-1]]
            req = [2 * Z * vv / (last['beta'] * hi) if np.isfinite(hi) else 0.0, 2 * Z * vv / (last['beta'] * last['Fb'])]
            if zn == 'POMA':
                D_.append(dict(beta=last['beta'], v=vv, F_last_H=last['Fb'], F_first_x_above=hi, dmu=last['dmu'], dmu_ci=last['dmu_ci'],
                               mu_peak=last['mu_peak'], mu_d=last['mu_d'], F_pred=Fp, F_pred_ci=Fp_ci, dmu_required=req,
                               consistent=bool(Fp_ci[0] <= hi and Fp_ci[1] >= last['Fb'])))
                print(f"  beta {last['beta']:.4f} v {vv:.2f}: Helmholtz up to {last['Fb']:.2f} N, next non-H above {hi:.2f} N | dmu {last['dmu']:.3f} "
                      f"[{last['dmu_ci'][0]:.3f}, {last['dmu_ci'][1]:.3f}] -> F_max {Fp:.2f} N [{Fp_ci[0]:.2f}, {Fp_ci[1]:.2f}] "
                      f"| dmu needed to put F_max in the bracket: {req[0]:.2f}-{req[1]:.2f}")
    out['absolute_upper_same_specimen'] = D_
    print('  (only brackets with a non-Helmholtz window above the last Helmholtz window test F_max; others are lower bounds on F_max)')

    # [E] attack windows
    print('\n[E] attack windows (k = 0): first breakaway coefficient vs steady in-cycle peak of the same stroke')
    E = []
    for r in rows:
        if r['k'] == 0 and 'mu_first' in r and (r['n'], 1) in W and np.isfinite(W[(r['n'], 1)].get('mu_peak', np.nan)) and W[(r['n'], 1)]['helmholtz']:
            s1 = W[(r['n'], 1)]
            E.append(dict(n=r['n'], beta=r['beta'], v0=r['v0'], Fb0=r['Fb'], Fb1=s1['Fb'], mu_first=r['mu_first'], t_first_ms=r['t_first_ms'],
                          mu_peak=s1['mu_peak'], mu_d=s1['mu_d']))
    if E:
        mf = np.array([e['mu_first'] for e in E]); mp = np.array([e['mu_peak'] for e in E]); md = np.array([e['mu_d'] for e in E])
        F0_ = np.array([e['Fb0'] for e in E]); F1_ = np.array([e['Fb1'] for e in E])
        print(f'  n {len(E)}: mu_first median {np.median(mf):.2f} [{mf.min():.2f}, {mf.max():.2f}]; steady mu_peak {np.median(mp):.2f}; '
              f'ratio {np.median(mf / mp):.2f}; attack/steady window force ratio {np.median(F0_ / F1_):.2f} (force still ramping)')
        b, s_, *_ = ols(np.c_[np.ones(len(E)), L(F0_)], L(mf))
        print(f'  log mu_first ~ log Fb(attack window): slope {b[1]:+.2f}±{s_[1]:.2f}')
        tf = np.array([e['t_first_ms'] for e in E])
        mdf = np.array([W[(e['n'], 0)].get('mu_d_first', np.nan) for e in E])
        print(f'  first slip at {np.median(tf):.0f} ms [{tf.min():.0f}, {tf.max():.0f}] after the start of the attack window; '
              f'first-slip contrast mu_first - mu_d(first slip) median {np.nanmedian(mf - mdf):.2f} [{np.nanmin(mf - mdf):.2f}, {np.nanmax(mf - mdf):.2f}]')
        cold = mf - md
        print(f'  "cold" contrast mu_first - mu_d(steady): median {np.median(cold):.2f} [{cold.min():.2f}, {cold.max():.2f}] '
              f'vs steady contrast {np.median(mp - md):.2f}; if mu_first is corrected by the force ratio: {np.median(mf * F0_ / F1_ - md):.2f}')
        for d in D_:
            e = [x for x in E if abs(x['beta'] - d['beta']) < 2e-3 and x['v0'] == d['v'] and abs(x['Fb1'] - d['F_last_H']) < 1e-6]
            if e:
                c = e[0]['mu_first'] - e[0]['mu_d']
                print(f"    beta {d['beta']:.4f} v {d['v']:.2f}: cold contrast {c:.2f} -> F_max {2 * Z_PUB * d['v'] / (d['beta'] * c):.2f} N "
                      f"(bracket {d['F_last_H']:.2f}-{d['F_first_x_above']:.2f} N)")
    out['attack'] = E
    json.dump(out, open(os.path.join(outd, 'friction_offline.json'), 'w'), indent=1, default=float)


if __name__ == '__main__':
    main()
