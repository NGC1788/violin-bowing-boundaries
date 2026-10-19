"""Column-wise lower force limit F_low(a) of a Guettler diagram and its log-log slope.
For each acceleration column: rows sorted by force, success smoothed with a centred 3-row mean;
F_low = log-interpolated force where the smoothed success first reaches 0.5 from below
(left-censored if the lowest row already >= 0.5; None if never reached).
"""
import numpy as np

def column_lower(F, s, k=3, thr=0.5):
    o = np.argsort(F); F = F[o]; s = s[o].astype(float)
    if k > 1:
        pad = np.r_[s[0], s, s[-1]] if k == 3 else s
        m = np.convolve(pad, np.ones(k) / k, mode='valid') if k == 3 else s
    else:
        m = s
    if m[0] >= thr:
        return F[0], 'left'
    for i in range(1, len(m)):
        if m[i] >= thr:
            # interpolate in log F between i-1 and i
            f = (thr - m[i - 1]) / (m[i] - m[i - 1])
            return float(np.exp(np.log(F[i - 1]) + f * (np.log(F[i]) - np.log(F[i - 1])))), 'ok'
    return np.nan, 'none'

def grid(a, F, na=30):
    nF = len(a) // na
    A = a.reshape(nF, na); FF = F.reshape(nF, na)
    co = np.argsort(A.mean(0))
    return co, nF

def lower_curve(a, F, succ, na=30, k=3):
    nF = len(a) // na
    A = a.reshape(nF, na); FF = F.reshape(nF, na); S = succ.reshape(nF, na)
    out = []
    for j in range(na):
        fl, st = column_lower(FF[:, j], S[:, j], k=k)
        out.append((np.median(A[:, j]), fl, st))
    out.sort()
    return out

def slope_fit(curve, amin=0, amax=99):
    x = np.array([np.log(c[0]) for c in curve if c[2] == 'ok' and amin <= c[0] <= amax])
    y = np.array([np.log(c[1]) for c in curve if c[2] == 'ok' and amin <= c[0] <= amax])
    if len(x) < 3:
        return np.nan, np.nan, len(x)
    b = np.polyfit(x, y, 1)
    return b[0], b[1], len(x)

def lin_fit(curve):
    x = np.array([c[0] for c in curve if c[2] == 'ok'])
    y = np.array([c[1] for c in curve if c[2] == 'ok'])
    if len(x) < 3:
        return np.nan, np.nan
    b = np.polyfit(x, y, 1)
    return b[0], b[1]

def boot_slope(curve, nb=2000, rng=None, amin=0, amax=99):
    rng = rng or np.random.default_rng(0)
    pts = [(np.log(c[0]), np.log(c[1])) for c in curve if c[2] == 'ok' and amin <= c[0] <= amax]
    pts = np.array(pts)
    if len(pts) < 4:
        return np.nan, np.nan
    bs = []
    for _ in range(nb):
        idx = rng.integers(0, len(pts), len(pts))
        if np.ptp(pts[idx, 0]) < 1e-6:
            continue
        bs.append(np.polyfit(pts[idx, 0], pts[idx, 1], 1)[0])
    return np.percentile(bs, [2.5, 97.5])
