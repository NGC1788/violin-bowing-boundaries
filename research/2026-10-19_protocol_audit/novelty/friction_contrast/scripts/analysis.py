"""Friction contrast (measured on one string, IWK type B T2 sample 2 = 'Helicore T2') vs playability-boundary levels.
(Claude, 2026-10-19). Inputs: results/friction_metrics.json (thr 0.15), results/friction_metrics_thr0.30.json,
results/boundary_laws.json. Output: results/analysis.json + printed report.

Parts
 1. Within-string regression of friction metrics on bow force, speed, beta (steady Helmholtz windows only).
 2. Exponents implied by Schelleng's boundary laws when the measured, force/speed-dependent contrast is inserted:
    F * dmu(F, v, beta) = C v / beta^m  (m = 1 upper, 2 lower)  =>  F ~ v^((1-k)/(1+g)) beta^((-m-l)/(1+g))
 3. Absolute Schelleng upper limit F_max = 2 Z v / (beta dmu) with the measured dmu, vs the measured B_T2 boundary.
 4. Cross-string: levels (log offsets vs the type-A law, given by the project) vs log Z, log EI and the
    'common friction factor' implied by Schelleng (a per-string dmu shifts both boundaries by the same -log dmu).
"""
import json
from itertools import product

import numpy as np

R = 'results/'


def ols(X, y):
    X = np.asarray(X, float); y = np.asarray(y, float)
    b, *_ = np.linalg.lstsq(X, y, rcond=None)
    res = y - X @ b
    dof = max(1, len(y) - X.shape[1])
    s2 = res @ res / dof
    cov = s2 * np.linalg.inv(X.T @ X)
    r2 = 1 - res @ res / np.sum((y - y.mean()) ** 2)
    return b, np.sqrt(np.diag(cov)), r2, np.sqrt(s2)


def main():
    out = {}
    m15 = {(r['n'], r['k']): r for r in json.load(open(R + 'friction_metrics.json'))}
    m30 = {(r['n'], r['k']): r for r in json.load(open(R + 'friction_metrics_thr0.30.json'))}
    H = [r for key, r in m15.items() if key[1] == 1 and r.get('helmholtz') and m30[key].get('helmholtz')]
    H.sort(key=lambda r: r['n'])
    print(f'steady Helmholtz windows (both thresholds): {len(H)}')
    Fb = np.array([r['Fb'] for r in H]); v = np.array([r['v0'] for r in H]); be = np.array([r['beta'] for r in H])
    met = {k: np.array([r[k] for r in H]) for k in ('avg_mu_peak', 'avg_mu_d', 'avg_dmu', 'hyst', 'mu_peak', 'mu_d', 'dmu', 'avg_mu_stick')}
    X = np.c_[np.ones(len(H)), np.log(Fb), np.log(v), np.log(be)]
    print('\n[1] descriptive (phase-averaged cycle)')
    for k in ('avg_mu_peak', 'avg_mu_d', 'avg_dmu', 'hyst'):
        x = met[k]
        print(f'  {k:12s} median {np.median(x):.3f}  range [{x.min():.3f}, {x.max():.3f}]')
    lowF = Fb < 1.0; highF = Fb > 2.4
    for name, msk in (('Fb<1.0 N', lowF), ('Fb>2.4 N', highF)):
        print(f'  {name}: n {msk.sum()}, mu_peak {np.median(met["avg_mu_peak"][msk]):.3f}, mu_d {np.median(met["avg_mu_d"][msk]):.3f}, '
              f'dmu {np.median(met["avg_dmu"][msk]):.3f}, hyst {np.median(met["hyst"][msk]):.3f}')
    out['n_windows'] = len(H)
    out['windows'] = [dict(n=r['n'], Fb=r['Fb'], v=r['v0'], beta=r['beta'], mu_peak=r['avg_mu_peak'], mu_d=r['avg_mu_d'],
                           dmu=r['avg_dmu'], hyst=r['hyst'], mu_stick=r['avg_mu_stick']) for r in H]
    print('\n[1b] OLS: y = a + g log Fb + k log v + l log beta')
    reg = {}
    for k, ylab in (('avg_dmu', 'log dmu'), ('avg_mu_d', 'log mu_d'), ('avg_mu_peak', 'log mu_peak')):
        b, se, r2, s = ols(X, np.log(met[k]))
        reg[k] = dict(coef=b.tolist(), se=se.tolist(), r2=r2, resid_sd=s)
        print(f'  {ylab:12s}: g(Fb) {b[1]:+.3f}±{se[1]:.3f}  k(v) {b[2]:+.3f}±{se[2]:.3f}  l(beta) {b[3]:+.3f}±{se[3]:.3f}  R2 {r2:.2f}  sd {s:.3f}')
    print('  power collapse: log y = a + c log(Fb*v) + d log v + l log beta  (d = 0 <=> y depends on Fb, v only via Fb*v)')
    Xp = np.c_[np.ones(len(H)), np.log(Fb * v), np.log(v), np.log(be)]
    for k in ('avg_mu_d', 'avg_dmu', 'avg_mu_peak'):
        b, se, r2, s_ = ols(Xp, np.log(met[k]))
        b1, se1, r21, s1 = ols(np.c_[np.ones(len(H)), np.log(Fb * v)], np.log(met[k]))
        reg[k + '_power'] = dict(coef=b.tolist(), se=se.tolist(), r2=r2, single=dict(coef=b1.tolist(), se=se1.tolist(), r2=r21, sd=s1))
        print(f'    {k:12s}: c {b[1]:+.3f}±{se[1]:.3f}  d {b[2]:+.3f}±{se[2]:.3f}  l {b[3]:+.3f}±{se[3]:.3f} | single log(Fb v): {b1[1]:+.3f}±{se1[1]:.3f} R2 {r21:.2f} sd {s1:.3f}')
    b, se, r2, s = ols(X, met['hyst'])
    reg['hyst'] = dict(coef=b.tolist(), se=se.tolist(), r2=r2)
    print(f'  hyst (lin)  : Fb {b[1]:+.3f}±{se[1]:.3f}  v {b[2]:+.3f}±{se[2]:.3f}  beta {b[3]:+.3f}±{se[3]:.3f}  R2 {r2:.2f}')
    # subsets: force dependence at low vs high force (beta, v fixed design rows)
    sub = {}
    for name, msk in (('low (Fb<=1.6)', Fb <= 1.6), ('high (Fb>=1.2)', Fb >= 1.2), ('top (Fb>=2.4)', Fb >= 2.4)):
        Xs = X[msk]
        b, se, r2, s = ols(Xs, np.log(met['avg_dmu'][msk]))
        sub[name] = dict(coef=b.tolist(), se=se.tolist(), n=int(msk.sum()))
        print(f'  log dmu {name:14s} n {msk.sum():2d}: g {b[1]:+.3f}±{se[1]:.3f}  k {b[2]:+.3f}±{se[2]:.3f}  l {b[3]:+.3f}±{se[3]:.3f}')
    out['regression'] = reg; out['regression_subsets'] = sub
    # same-beta, same-v force series
    print('  force series (dmu):')
    for bb, vv in ((0.0405, 0.05), (0.1044, 0.05)):
        sel = [r for r in H if abs(r['beta'] - bb) < 2e-3 and abs(r['v0'] - vv) < 1e-6]
        print(f'    beta {bb}, v {vv}: ' + ', '.join(f"{r['Fb']:.2f}N->{r['avg_dmu']:.3f}" for r in sorted(sel, key=lambda r: r['Fb'])))
    print('  speed series (dmu, mu_d):')
    for bb in (0.0484, 0.1175):
        sel = [r for r in H if abs(r['beta'] - bb) < 2e-3]
        print(f'    beta {bb}: ' + ', '.join(f"v{r['v0']:.2f}/{r['Fb']:.2f}N->{r['avg_dmu']:.3f},{r['avg_mu_d']:.3f}" for r in sorted(sel, key=lambda r: r['v0'])))

    # [2] implied exponents
    print('\n[2] Schelleng with measured dmu(F, v, beta): implied boundary exponents (speed, beta)')
    bl = json.load(open(R + 'boundary_laws.json'))
    imp = {}
    for label, coef in (('global', reg['avg_dmu']['coef']), ('low-F fit', sub['low (Fb<=1.6)']['coef']), ('high-F fit', sub['high (Fb>=1.2)']['coef']), ('top-F fit', sub['top (Fb>=2.4)']['coef'])):
        g, k, l = coef[1], coef[2], coef[3]
        up = ((1 - k) / (1 + g), (-1 - l) / (1 + g)); lo = ((1 - k) / (1 + g), (-2 - l) / (1 + g))
        imp[label] = dict(upper=up, lower=lo)
        print(f'  {label:10s}: upper v^{up[0]:+.2f} beta^{up[1]:+.2f} | lower v^{lo[0]:+.2f} beta^{lo[1]:+.2f}  (constant dmu: upper +1/-1, lower +1/-2)')
    for tag in ('B_T2r1', 'B_T2r2', 'A1', 'A2'):
        L = bl[tag]
        print(f'  measured {tag:6s}: upper v^{L["upper"]["b_speed"]:+.2f} beta^{L["upper"]["c_beta"]:+.2f} | lower v^{L["lower"]["b_speed"]:+.2f} beta^{L["lower"]["c_beta"]:+.2f}')
    out['implied_exponents'] = imp
    # structured law: mu_peak = const, mu_d = exp(a) (Fb v)^c  -> dmu(F, v) = mu_pk - mu_d(F v); solve F dmu = C v / beta^m
    from scipy.optimize import brentq
    mu_pk = float(np.median(met['avg_mu_peak']))
    a_d, c_d = reg['avg_mu_d_power']['single']['coef']
    dmu_law = lambda F, vv: mu_pk - np.exp(a_d) * (F * vv) ** c_d
    P0 = (mu_pk / np.exp(a_d)) ** (1 / c_d)  # Fb*v at which the law's contrast vanishes
    print(f'  structured law: mu_peak {mu_pk:.3f} (const), mu_d = {np.exp(a_d):.3f} (Fb v)^{c_d:+.3f}; contrast -> 0 at Fb*v = {P0:.4f} W '
          f'(lowest measured Fb*v {np.min(Fb * v):.3f})')
    sc = {}
    grid_b = np.exp(np.linspace(np.log(0.04), np.log(0.12), 7)); grid_v = np.array([0.05, 0.1, 0.2])
    Fu = np.sqrt(bl['B_T2r1']['upper']['at']['0.1044_0.1'] * bl['B_T2r2']['upper']['at']['0.1044_0.1'])
    for which, m, anchor in (('upper', 1, None), ('upper_calibrated', 1, None), ('lower', 2, None)):
        if which == 'upper':
            C = 2 * 1.1899
        elif which == 'upper_calibrated':  # constant chosen so that F_up(beta 0.1044, v 0.1) = measured B_T2 value
            C = Fu * dmu_law(Fu, 0.1) * 0.1044 / 0.1
        else:  # calibrate C so that F_lo(beta 0.1044, v 0.1) equals the measured B_T2 value (geometric mean of reps)
            Fl = np.sqrt(bl['B_T2r1']['lower']['at']['0.1044_0.1'] * bl['B_T2r2']['lower']['at']['0.1044_0.1'])
            C = Fl * dmu_law(Fl, 0.1) * 0.1044 ** 2 / 0.1 if dmu_law(Fl, 0.1) > 0 else np.nan
        rows_ = []
        for bb, vv in product(grid_b, grid_v):
            f = lambda F: F * dmu_law(F, vv) - C * vv / bb ** m
            lo_ = P0 / vv * 1.0001
            try:
                F = brentq(f, lo_, 1e3)
            except (ValueError, RuntimeError):
                F = np.nan
            rows_.append((np.log(bb), np.log(vv), np.log(F)))
        A_ = np.array(rows_); ok = np.isfinite(A_[:, 2])
        bfit, *_ = np.linalg.lstsq(np.c_[np.ones(ok.sum()), A_[ok, 1], A_[ok, 0]], A_[ok, 2], rcond=None)
        sc[which] = dict(speed=float(bfit[1]), beta=float(bfit[2]), C=float(C), n=int(ok.sum()))
        print(f'  structured law, self-consistent {which}: v^{bfit[1]:+.2f} beta^{bfit[2]:+.2f} (grid beta 0.04-0.12, v 0.05-0.2; C {C:.3g})')
    out['structured_law'] = dict(mu_peak=mu_pk, a=a_d, c=c_d, P0=P0, selfconsistent=sc)

    # [3] absolute upper limit
    print('\n[3] Absolute Schelleng upper limit F_max = 2 Z v/(beta dmu), Z = 1.1899 (POMA B_T2)')
    Z = 1.1899
    gl = reg['avg_dmu']['coef']
    absres = []
    for key in ('0.1044_0.05', '0.1044_0.1', '0.0405_0.05', '0.0405_0.1', '0.0484_0.05', '0.0484_0.2', '0.1175_0.05', '0.1175_0.2'):
        bb, vv = map(float, key.split('_'))
        Fm = np.sqrt(bl['B_T2r1']['upper']['at'][key] * bl['B_T2r2']['upper']['at'][key])
        dmu_at = float(np.exp(gl[0] + gl[1] * np.log(Fm) + gl[2] * np.log(vv) + gl[3] * np.log(bb)))
        need = 2 * Z * vv / (bb * Fm)
        pred = 2 * Z * vv / (bb * dmu_at)
        absres.append(dict(beta=bb, v=vv, F_upper_meas=Fm, dmu_measured_at_F=dmu_at, dmu_required=need, F_pred=pred, ratio=pred / Fm))
        print(f'  beta {bb:.4f} v {vv:.2f}: measured F_up {Fm:.2f} N | dmu(meas, at F_up) {dmu_at:.3f} -> Schelleng F_max {pred:.2f} N '
              f'(x{pred / Fm:.2f}) | dmu required {need:.2f}')
    out['absolute_upper'] = absres
    print('  local version (dmu of the measured steady Helmholtz window(s) of the same beta & v closest to F_up; no extrapolation):')
    loc = []
    for key in ('0.1044_0.05', '0.1044_0.1', '0.0405_0.05', '0.0484_0.05', '0.1175_0.05', '0.1175_0.2', '0.0484_0.2'):
        bb, vv = map(float, key.split('_'))
        Fm = np.sqrt(bl['B_T2r1']['upper']['at'][key] * bl['B_T2r2']['upper']['at'][key])
        cand = [r for r in H if abs(r['beta'] - bb) / bb < 0.03 and abs(r['v0'] - vv) < 1e-6]
        if not cand:
            continue
        r = min(cand, key=lambda r: abs(np.log(r['Fb'] / Fm)))
        if Fm > 5.5 or abs(np.log(r['Fb'] / Fm)) > np.log(1.5):
            print(f'    beta {bb:.4f} v {vv:.2f}: skipped (F_up {Fm:.2f} N outside measured range or nearest window {r["Fb"]:.2f} N too far)')
            continue
        need = 2 * Z * vv / (bb * Fm)
        pred = 2 * Z * vv / (bb * r['avg_dmu'])
        loc.append(dict(beta=bb, v=vv, F_up=Fm, F_window=r['Fb'], dmu=r['avg_dmu'], mu_peak=r['avg_mu_peak'], mu_d=r['avg_mu_d'],
                        dmu_required=need, F_pred=pred, ratio=pred / Fm))
        print(f'    beta {bb:.4f} v {vv:.2f}: F_up {Fm:.2f} N; window {r["Fb"]:.2f} N dmu {r["avg_dmu"]:.3f} (peak {r["avg_mu_peak"]:.2f}, mu_d {r["avg_mu_d"]:.2f}) '
              f'-> Schelleng {pred:.2f} N (x{pred / Fm:.2f}); required dmu {need:.2f}')
    out['absolute_upper_local'] = loc
    # own-string brackets from the reconstruction strokes (Helmholtz flag, steady windows)
    print('  same-specimen brackets from the reconstruction strokes (last Helmholtz / first non-Helmholtz, steady window):')
    allw = [r for key, r in m15.items() if key[1] == 1]
    br = {}
    for bb, vv in ((0.0405, 0.05), (0.0405, 0.10), (0.1044, 0.05), (0.1044, 0.10), (0.1777, 0.10)):
        sel = sorted([r for r in allw if abs(r['beta'] - bb) < 2e-3 and abs(r['v0'] - vv) < 1e-6], key=lambda r: r['Fb'])
        flags = [(round(r['Fb'], 2), bool(r.get('helmholtz') and m30[(r['n'], 1)].get('helmholtz'))) for r in sel]
        br[f'{bb}_{vv}'] = flags
        print(f'    beta {bb} v {vv}: ' + ' '.join(f"{f}{'H' if h else 'x'}" for f, h in flags))
    out['brackets'] = br

    # [4] cross-string
    print('\n[4] Cross-string levels (log offsets vs type-A law; given) vs string properties (POMA Table 1)')
    lev = {'A1': (0.115, 0.002), 'A2': (-0.118, -0.002), 'B1': (-0.357, 0.299), 'C1': (-0.561, -0.004), 'D1': (-0.248, 0.172),
           'A_T2': (-0.539, 0.197), 'B_T2r1': (-0.318, 0.240), 'B_T2r2': (-0.139, 0.249)}
    prop = {'A1': (1.0592, 3.032), 'A2': (1.0592, 3.032), 'B1': (1.0998, 4.397), 'C1': (1.0525, 3.142), 'D1': (1.0398, 5.246),
            'A_T2': (1.1884, 3.969), 'B_T2r1': (1.1899, 7.295), 'B_T2r2': (1.1899, 7.295)}
    grp = {'A1': 'A', 'A2': 'A', 'B1': 'B1', 'C1': 'C1', 'D1': 'D1', 'A_T2': 'A_T2', 'B_T2r1': 'B_T2', 'B_T2r2': 'B_T2'}
    tags = list(lev)
    Lo = np.array([lev[t][0] for t in tags]); Up = np.array([lev[t][1] for t in tags])
    zr = np.log([prop[t][0] / 1.0592 for t in tags]); er = np.log([prop[t][1] / 3.032 for t in tags])
    d_up = zr - Up          # Schelleng upper: Up = zr - log(dmu/dmu_A)  -> required log contrast ratio
    d_lo = 2 * zr - Lo      # Schelleng lower (R fixed): Lo = 2 zr - log(dmu/dmu_A)
    print('  string   L_lo    L_up   | log Z/ZA  log EI/EIA | required log(dmu/dmu_A): from upper, from lower | implied R/R_A (lower-upper)')
    for i, t in enumerate(tags):
        print(f'  {t:7s} {Lo[i]:+.3f} {Up[i]:+.3f} | {zr[i]:+.3f}   {er[i]:+.3f}   | {d_up[i]:+.3f} ({np.exp(d_up[i]):.2f}x)  {d_lo[i]:+.3f} ({np.exp(d_lo[i]):.2f}x) | {np.exp(zr[i] - Lo[i] + Up[i]):.2f}')
    nonA = np.array([g != 'A' for g in (grp[t] for t in tags)])
    out['cross'] = dict(tags=tags, d_up=d_up.tolist(), d_lo=d_lo.tolist())
    r_ud = np.corrcoef(d_up[nonA], d_lo[nonA])[0, 1]
    print(f'  corr(required contrast from upper, from lower), non-A diagrams (n {nonA.sum()}): {r_ud:+.2f}; '
          f'sign agreement {int(np.sum(np.sign(d_up[nonA]) == np.sign(d_lo[nonA])))}/{nonA.sum()}')
    print(f'  corr(L_lo, L_up) all 8: {np.corrcoef(Lo, Up)[0, 1]:+.2f}')
    for nm, x in (('log Z', zr), ('log EI', er)):
        print(f'  corr {nm:6s}: lower {np.corrcoef(x, Lo)[0, 1]:+.2f}, upper {np.corrcoef(x, Up)[0, 1]:+.2f}')

    # models, no intercept (type A is the reference); grouped leave-one-string-out
    gl_ = reg['avg_dmu']['coef']; sh = sub['top (Fb>=2.4)']['coef']; sl = sub['low (Fb<=1.6)']['coef']
    g_hi, g_lo = max(sh[1], 0.0), max(sl[1], 0.0)
    fixed = {
        'null (0)': (np.zeros(8), np.zeros(8)),
        'Schelleng Z (lo 2z, up z)': (2 * zr, zr),
        'Beigi Z lower (z)': (zr, zr),
        'Schelleng Z + measured dmu(F) common law': (2 * zr / (1 + g_lo), zr / (1 + g_hi)),
        'common per-string friction factor (Lo = Up + z)': (Up + zr, Lo - zr),
    }
    groups = sorted(set(grp.values()))

    def fitpred(cols, y, train):
        Xm = np.c_[cols][:, :] if cols.ndim > 1 else cols[:, None]
        b, *_ = np.linalg.lstsq(Xm[train], y[train], rcond=None)
        return Xm @ b, b
    free = {'a*logZ': zr[:, None], 'b*logEI': er[:, None], 'a*logZ + b*logEI': np.c_[zr, er]}
    print('\n  model RMS over 8 diagrams (in-sample) and grouped leave-one-string-out RMS (free models):')
    mres = {}
    for nm, (pl, pu) in fixed.items():
        rl = np.sqrt(np.mean((Lo - pl) ** 2)); ru = np.sqrt(np.mean((Up - pu) ** 2))
        mres[nm] = dict(lower=rl, upper=ru)
        print(f'    {nm:46s} lower {rl:.3f}  upper {ru:.3f}   (no free parameters)')
    for nm, Xf in free.items():
        row = {}
        for which, y in (('lower', Lo), ('upper', Up)):
            b, *_ = np.linalg.lstsq(Xf, y, rcond=None)
            ins = np.sqrt(np.mean((y - Xf @ b) ** 2))
            pred = np.zeros(8)
            for gname in groups:
                te = np.array([grp[t] == gname for t in tags]); tr = ~te
                bb, *_ = np.linalg.lstsq(Xf[tr], y[tr], rcond=None)
                pred[te] = Xf[te] @ bb
            loo = np.sqrt(np.mean((y - pred) ** 2))
            row[which] = dict(coef=b.tolist(), rms=ins, loo=loo)
        mres[nm] = row
        print(f'    {nm:46s} lower {row["lower"]["rms"]:.3f} (LOO {row["lower"]["loo"]:.3f}) coef {np.round(row["lower"]["coef"], 2)} | '
              f'upper {row["upper"]["rms"]:.3f} (LOO {row["upper"]["loo"]:.3f}) coef {np.round(row["upper"]["coef"], 2)}')
    out['models'] = mres
    print('  measurement-noise yardsticks (given): lower A1-A2 0.233, B_T2 rep-rep 0.179; upper A1-A2 0.004, B_T2 rep-rep 0.009')
    json.dump(out, open(R + 'analysis.json', 'w'), indent=1, default=float)


if __name__ == '__main__':
    main()
