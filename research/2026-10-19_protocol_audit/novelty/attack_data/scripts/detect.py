"""Independent time-resolved Helmholtz detector for Guettler-diagram strokes (bridge force only).
Windows of W nominal periods (hop 1 period) after the first slip; each window labelled with the
flyback-counting classifier of violin-bowing-boundaries/scripts/regime_map.py (fold at the dominant
period, count flybacks >= frac of the tallest)."""
import sys, math, numpy as np
from scipy.signal import butter, sosfiltfilt, find_peaks
sys.path.insert(0, '/Users/jeongjihoon/violin-bowing-boundaries/scripts')
import regime_map as rm

FS = 50000
F0 = 97.86
T0 = 1 / F0
TH = dict(rm.DEFAULT_THRESHOLDS)

def accel_fit(y):
    t = np.arange(len(y)) / FS
    sel = (y > 2e-4) & (y < 0.055)
    i_last = np.flatnonzero(sel)
    # only the monotone rising part before the max
    imax = int(np.argmax(y))
    sel &= (np.arange(len(y)) < imax)
    p = np.polyfit(t[sel], np.sqrt(y[sel]), 1)
    a = 2 * p[0] ** 2
    t0 = -p[1] / p[0]
    resid = np.sqrt(y[sel]) - np.polyval(p, t[sel])
    return a, t0, float(np.sqrt(np.mean(resid ** 2)))

_sos1k = butter(9, 1000, 'low', fs=FS, output='sos')

def first_slip(fb, i_move):
    """Authors-like: |d/dt| of 1 kHz low-passed bridge force, normalised by its max within the first
    0.15 s of a cut starting 2000 samples before motion; first peak > 0.15 (prominence 0.1, 5 ms apart)."""
    c0 = max(0, i_move - 2000)
    x = sosfiltfilt(_sos1k, fb[c0:])
    d = np.abs(np.diff(x))
    d = d / d[:int(0.15 * FS)].max()
    d = sosfiltfilt(_sos1k, d)
    pk, _ = find_peaks(d, height=0.15, prominence=0.1, distance=int(0.005 * FS))
    if len(pk) == 0:
        return math.nan
    return c0 + int(pk[0])

def window_labels(fb, i_start, i_end, W=4, f0_range=(60.0, 160.0)):
    P = FS / F0
    out = []
    k = 0
    while True:
        a0 = int(round(i_start + k * P)); a1 = int(round(i_start + (k + W) * P))
        if a1 > i_end:
            break
        x = fb[a0:a1].astype(float)
        tt = np.arange(len(x))
        x = x - np.polyval(np.polyfit(tt, x, 1), tt)
        amp = float(np.percentile(x, 99.5) - np.percentile(x, 0.5))
        per, f0 = rm.dominant_period(x, FS, f0_range)
        if math.isfinite(f0):  # unbiased normalisation: the biased ACF of a W-period window peaks at (W-1)/W
            lag = FS / f0
            per = per * len(x) / (len(x) - lag)
        if math.isfinite(f0):
            fl = rm.flyback_structure(x, FS / f0, TH['flyback_frac'], TH['hysteresis_sigma'])
            nfl = fl['flybacks_per_period']; sign = fl['flyback_sign']; r2 = fl['second_flyback_ratio']
        else:
            nfl, sign, r2 = math.nan, 0, math.nan
        cls = rm.classify(amp, per, f0, nfl, F0, TH)
        hm = (cls == 'helmholtz') and math.isfinite(f0) and abs(f0 / F0 - 1) < 0.12
        out.append((k, amp, per, f0, nfl, sign, r2, cls, hm))
        k += 1
    return out

def entry_times(hm, run=3):
    """first k with hm[k..k+run-1] all True (entry); final sustained entry (last 'run' windows HM and
    no non-HM window after the returned k)."""
    hm = np.asarray(hm, bool)
    n = len(hm)
    first = next((k for k in range(n - run + 1) if hm[k:k + run].all()), None)
    final = None
    if n >= run and hm[-run:].all():
        k = n - 1
        while k > 0 and hm[k - 1]:
            k -= 1
        final = k
    return first, final
