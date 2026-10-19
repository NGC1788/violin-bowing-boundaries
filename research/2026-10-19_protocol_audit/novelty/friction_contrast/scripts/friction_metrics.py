"""Per-segment friction metrics from the IWK reconstructed bowing-point friction force
(Zenodo 15879128, Lampis 2025; string = 'Helicore T2', which we map to IWK type B, T2, sample 2).

Inputs (untrusted data folder, read only):
  reconstructions/friction_velocity_reconstruction_<n>_<k>.csv : Fr_rec, V_rec, F_b(bridge), F_n(nut), time[ms]
  reconstructions/bowing_parameters_<n>_<k>.csv               : beta, Fb (mean bow force of the window), v0
k = 0: first ~316 ms of the stroke (attack transient + start); k = 1: ~100 ms steady-state window.

Definitions (per slip cycle, then median over cycles of a segment):
  s = v0 - V_rec (sliding speed; > 0 while slipping), mu(t) = Fr_rec / Fb.
  slip episode: s > thr = 0.15 * v0 / beta (15 % of the ideal Helmholtz slip speed), merged if gaps < 0.3 ms.
  mu_peak  : max mu in [onset - 1 ms, time of max s]      (breakaway / effective static coefficient)
  mu_onset : mu at the onset sample                          (friction when sliding begins)
  mu_d     : mean mu over samples with s > 0.5 * s_max      (core of the slip; 'dynamic' coefficient)
  mu_slip  : mean mu over the whole slip episode
  mu_stick : mean mu over the stick interval preceding the episode (s < thr)
  dmu      : mu_peak - mu_d  (friction contrast)
  hyst     : mean over s/s_max in {0.3, 0.5, 0.7} of [mu on rising branch - mu on falling branch]
             (> 0: friction is higher while the sliding speed is increasing than at the same sliding speed
              while it is decreasing, i.e. the loop expected from thermal softening of rosin)
  Helmholtz flag: episodes per nominal period in [0.9, 1.1], CV of onset intervals < 0.1, CV of s_max < 0.35.
First-slip (k = 0 only): mu_first = max mu before and up to the peak of the first slip episode.
"""
import csv
import glob
import json
import os
import sys

import numpy as np

F0 = 97.45  # Hz, pluck-derived f0 of this string (string_properties 19-24.mat)


def load(root, n, k):
    a = np.loadtxt(os.path.join(root, f'friction_velocity_reconstruction_{n}_{k}.csv'), delimiter=',', skiprows=1)
    with open(os.path.join(root, f'bowing_parameters_{n}_{k}.csv')) as f:
        p = list(csv.DictReader(f))[0]
    return a, float(p['beta']), float(p['Fb']), float(p['v0'])


def episodes(mask, dt_ms, gap_ms=0.3):
    idx = np.flatnonzero(np.diff(np.r_[0, mask.astype(int), 0]))
    starts, ends = idx[0::2], idx[1::2]  # [start, end)
    out = []
    for s, e in zip(starts, ends):
        if out and (s - out[-1][1]) * dt_ms < gap_ms:
            out[-1][1] = e
        else:
            out.append([s, e])
    return out


def branch_interp(s_branch, mu_branch, x):
    o = np.argsort(s_branch)
    sb, mb = s_branch[o], mu_branch[o]
    if sb.size < 3 or x < sb[0] or x > sb[-1]:
        return np.nan
    return float(np.interp(x, sb, mb))


THR_FRAC = 0.15


def analyse(a, beta, Fb, v0, k):
    t, Fr, V = a[:, 4], a[:, 0], a[:, 1]
    dt = float(np.median(np.diff(t)))  # ms
    mu = Fr / Fb
    s = v0 - V
    w = max(1, int(round(0.5 / float(np.median(np.diff(t))))))  # 0.5 ms boxcar
    mu_sm = np.convolve(mu, np.ones(w) / w, mode='same')
    thr = THR_FRAC * v0 / beta
    eps = episodes(s > thr, dt)
    # drop episodes touching the segment edges
    eps = [e for e in eps if e[0] > 0 and e[1] < len(s)]
    res = dict(n_ep=len(eps))
    if len(eps) < 2:
        res.update(helmholtz=False)
        return res, None
    dur_ms = t[-1] - t[0]
    per_ms = 1000.0 / F0
    onsets = np.array([e[0] for e in eps])
    iv = np.diff(t[onsets])
    smax = np.array([s[e[0]:e[1]].max() for e in eps])
    ep_per_period = len(eps) / (dur_ms / per_ms)
    res.update(ep_per_period=ep_per_period, iv_cv=float(np.std(iv) / np.mean(iv)), smax_cv=float(np.std(smax) / np.mean(smax)),
               slip_duty=float(np.median([(e[1] - e[0]) * dt for e in eps]) / per_ms))
    res['helmholtz'] = bool(0.9 <= ep_per_period <= 1.1 and res['iv_cv'] < 0.1 and res['smax_cv'] < 0.35)
    cyc = []
    nb = int(round(1.0 / dt))
    for j, (a0, a1) in enumerate(eps):
        seg_s = s[a0:a1]
        ipk = a0 + int(np.argmax(seg_s))
        lo = max(0, a0 - nb)
        mu_peak = float(mu[lo:ipk + 1].max())
        mu_peak_sm = float(mu_sm[lo:ipk + 1].max())
        mu_onset = float(mu[a0])
        sm = s[ipk]
        core = (s[a0:a1] > 0.5 * sm)
        mu_d = float(mu[a0:a1][core].mean())
        mu_slip = float(mu[a0:a1].mean())
        prev_end = eps[j - 1][1] if j > 0 else max(0, a0 - int(per_ms / dt))
        st = slice(prev_end, a0)
        mu_stick = float(mu[st].mean()) if a0 - prev_end > 5 else np.nan
        noise = float(np.std(mu[st] - mu_sm[st])) if a0 - prev_end > 5 else np.nan
        rise_s, rise_m = s[a0:ipk + 1], mu[a0:ipk + 1]
        fall_s, fall_m = s[ipk:a1], mu[ipk:a1]
        hs = []
        for q in (0.3, 0.5, 0.7):
            r = branch_interp(rise_s, rise_m, q * sm)
            f = branch_interp(fall_s, fall_m, q * sm)
            hs.append(r - f)
        cyc.append(dict(mu_peak=mu_peak, mu_peak_sm=mu_peak_sm, dmu_sm=mu_peak_sm - mu_d, noise=noise, mu_onset=mu_onset, mu_d=mu_d, mu_slip=mu_slip, mu_stick=mu_stick,
                        dmu=mu_peak - mu_d, hyst=float(np.nanmean(hs)) if np.isfinite(hs).any() else np.nan,
                        s_max=float(sm), s_at_peak=float(s[lo + int(np.argmax(mu[lo:ipk + 1]))])))
    # phase-averaged cycle (aligned at onset): reduces random reconstruction noise before taking the peak
    L = int(round(per_ms / dt))
    pre = nb
    segs_mu, segs_s = [], []
    for (a0, a1) in eps:
        if a0 - pre >= 0 and a0 - pre + L <= len(mu):
            segs_mu.append(mu[a0 - pre:a0 - pre + L]); segs_s.append(s[a0 - pre:a0 - pre + L])
    if len(segs_mu) >= 3:
        am, as_ = np.mean(segs_mu, axis=0), np.mean(segs_s, axis=0)
        ipk = int(np.argmax(as_[:pre + int(round(0.5 * per_ms / dt))]))
        smx = as_[ipk]
        # slip episode of the averaged cycle
        j0 = pre
        j1 = j0
        while j1 < L and as_[j1] > THR_FRAC * v0 / beta:
            j1 += 1
        core = np.arange(j0, max(j1, j0 + 1))
        core = core[as_[core] > 0.5 * smx]
        res['avg_mu_peak'] = float(am[:ipk + 1].max())
        res['avg_mu_d'] = float(am[core].mean()) if core.size else np.nan
        res['avg_dmu'] = res['avg_mu_peak'] - res['avg_mu_d']
        res['avg_mu_stick'] = float(am[j1:].mean()) if j1 < L - 5 else np.nan
        res['n_avg'] = len(segs_mu)
    for key in cyc[0]:
        vals = np.array([c[key] for c in cyc], float)
        res[key] = float(np.nanmedian(vals))
        res[key + '_iqr'] = float(np.nanpercentile(vals, 75) - np.nanpercentile(vals, 25))
    if k == 0:
        a0, a1 = eps[0]
        ipk = a0 + int(np.argmax(s[a0:a1]))
        res['mu_first'] = float(mu[:ipk + 1].max())
        res['t_first_ms'] = float(t[a0])
        res['dmu_first'] = res['mu_first'] - float(mu[a0:a1][s[a0:a1] > 0.5 * s[ipk]].mean())
    return res, cyc


def main():
    global THR_FRAC
    root, out = sys.argv[1], sys.argv[2]
    if len(sys.argv) > 3:
        THR_FRAC = float(sys.argv[3])
    rows = []
    for p in sorted(glob.glob(os.path.join(root, 'bowing_parameters_*_*.csv'))):
        n, k = map(int, os.path.basename(p)[:-4].split('_')[-2:])
        a, beta, Fb, v0 = load(root, n, k)
        res, _ = analyse(a, beta, Fb, v0, k)
        res.update(n=n, k=k, beta=beta, Fb=Fb, v0=v0)
        rows.append(res)
    rows.sort(key=lambda r: (r['n'], r['k']))
    with open(out, 'w') as f:
        json.dump(rows, f, indent=1)
    keys = ['n', 'k', 'beta', 'Fb', 'v0', 'helmholtz', 'ep_per_period', 'slip_duty', 'mu_peak', 'mu_onset', 'mu_d', 'dmu',
            'hyst', 'mu_stick', 'mu_first']
    print(' '.join(f'{c:>8s}' for c in keys))
    for r in rows:
        print(' '.join(f'{r.get(c, np.nan):8.3f}' if isinstance(r.get(c, np.nan), float) else f'{str(r.get(c)):>8s}' for c in keys))


if __name__ == '__main__':
    main()
