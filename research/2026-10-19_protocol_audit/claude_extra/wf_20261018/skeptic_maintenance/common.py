"""Shared loading + statistic for the adversarial check of the A2 v=0.05 m/s maintenance claim.
Reads only cache_A1.npz / cache_A2.npz (built by build_cache.py from A1/A2 features)."""
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
FLOOR_TOL = 1.15
D_FLOOR = 0.048


class Stroke:
    __slots__ = ("folder", "trial", "beta", "col", "t", "F", "H", "lab", "accel", "f_onset", "ramp_s", "speed",
                 "floor", "rank", "plateau_on", "onset")


def load(tag, win=0.2, floor_tol=FLOOR_TOL):
    z = np.load(HERE / f"cache_{tag}.npz")
    m = z["win"] == win
    folder, trial, beta, t, F, lab = (z[k][m] for k in ("folder", "trial", "beta", "t", "force", "lab"))
    speed_all, folder_all = z["speed"], z["folder"]
    speed = {f: float(np.median(speed_all[(folder_all == f) & (z["win"] == 0.2)])) for f in np.unique(folder_all)}
    meta = {(int(f), int(tr)): row for f, tr, row in zip(z["meta_folder"], z["meta_trial"], z["meta"])}
    order = np.lexsort((t, trial, folder))
    folder, trial, beta, t, F, lab = (a[order] for a in (folder, trial, beta, t, F, lab))
    key = folder * 100000 + trial
    cut = np.flatnonzero(np.diff(key)) + 1
    strokes = []
    for idx in np.split(np.arange(len(key)), cut):
        s = Stroke()
        s.folder, s.trial = int(folder[idx[0]]), int(trial[idx[0]])
        s.beta = float(beta[idx[0]])
        s.t, s.F, s.lab = t[idx], F[idx], lab[idx]
        s.H = s.lab == 0
        s.speed = speed[s.folder]
        mm = meta.get((s.folder, s.trial))
        if mm is None:
            s.accel = s.f_onset = s.ramp_s = s.plateau_on = s.onset = np.nan
            s.floor = False
        else:
            s.accel, s.f_onset, s.ramp_s, _, s.plateau_on, s.onset = mm
            s.floor = bool(s.accel <= floor_tol * s.speed ** 2 / (2 * D_FLOOR))
        r = (s.trial - 1) % 50 + 1
        s.rank = 51 - r if tag == "A1" else r
        strokes.append(s)
    # beta columns: per folder sort all strokes by beta, 40 groups of 50
    by_f = defaultdict(list)
    for s in strokes:
        by_f[s.folder].append(s)
    for f, ss in by_f.items():
        ss.sort(key=lambda s: s.beta)
        for i, s in enumerate(ss):
            s.col = i // 50
    return strokes, speed


def runs_of(h, K):
    """start indices i such that h[i:i+K] are all True"""
    h = np.asarray(h, bool)
    if len(h) < K:
        return np.array([], int)
    c = np.convolve(h.astype(int), np.ones(K, int), "valid")
    return np.flatnonzero(c == K)


def entry_low(F, h, K=2, K_low=None, skip_entry_run=False):
    """entry = force at the last window of the first K-run; low = min force at the last window of any K_low-run
    starting at or after the entry run (K_low defaults to K). Returns (entry, low, i_entry_end, i_low_end) or None."""
    r = runs_of(h, K)
    if r.size == 0:
        return None
    ie = r[0] + K - 1
    KL = K if K_low is None else K_low
    rl = runs_of(h, KL)
    rl = rl[rl >= r[0]]
    if skip_entry_run:
        rl = rl[rl > ie]
    if rl.size == 0:
        return (F[ie], np.nan, ie, -1)
    ends = rl + KL - 1
    j = ends[np.argmin(F[ends])]
    return (F[ie], F[j], ie, j)


def column_ratios(strokes, K=2, q=0.1, K_low=None, min_n=5, labels=None, skip_entry_run=False, select=None):
    """Reproduces entry_vs_low.py. labels: optional dict id(stroke)->bool array replacing s.H.
    Returns dict speed -> list of (column, ratio, median(low/entry), frac lows below q-entry, n_entered)."""
    cells = defaultdict(list)
    for s in strokes:
        if select is not None and not select(s):
            continue
        h = s.H if labels is None else labels[id(s)]
        el = entry_low(s.F, h, K, K_low, skip_entry_run)
        if el is None or not np.isfinite(el[1]):
            continue
        cells[(s.folder, s.col)].append((el[0], el[1]))
    out = defaultdict(list)
    for (f, c), g in cells.items():
        if len(g) < min_n:
            continue
        e = np.array([x[0] for x in g]); lo = np.array([x[1] for x in g])
        qe = np.quantile(e, q)
        out[f].append((c, np.quantile(lo, q) / qe, np.median(lo / e), np.mean(lo < qe), len(g)))
    return out


def summary(out):
    return {f: (len(r), float(np.median([x[1] for x in r])) if r else np.nan,
                float(np.mean([x[1] < 1 for x in r])) if r else np.nan) for f, r in sorted(out.items())}
