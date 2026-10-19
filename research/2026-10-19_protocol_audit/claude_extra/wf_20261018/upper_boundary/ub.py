"""Shared helpers for the upper-boundary track (Claude, 2026-10-18).

Mirrors claude_extra/verify_astra_boundaries.py for the lower boundary (beta-column recovery, K-run
boundary, interval-censored normal likelihood) and adds the mirrored upper boundary:
  per (speed folder, beta column, plateau time t): sort the column's strokes by the CURRENT window force,
  scan from the HIGH-force side for the first run of K consecutive Helmholtz windows; the upper boundary
  lies between the top stroke of that run and the next stroke above it.  If the run includes the highest
  stroke -> right-censored (Fmax > highest force).  No Helmholtz run at all -> no upper boundary in range
  (excluded and counted).
"""
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.special import log_ndtr

HERE = Path(__file__).resolve().parent
CACHE = HERE / "cache"
TIMES = (0.0, 0.1, 0.2, 0.3)
H = 0  # label code of helmholtz in cache
LABELS = ["helmholtz", "multiple_slip", "aperiodic", "ambiguous", "subharmonic", "no_oscillation", "error"]


class Spec:
    """Per-specimen window table restricted to one window length, plus per-stroke metadata."""

    def __init__(self, tag, window_s=0.2):
        z = np.load(CACHE / f"windows_{tag}.npz")
        self.tag = tag
        self.folders = list(z["folders"])
        m = z["window_s"] == np.float32(window_s)
        self.w = {k: z[k][m] for k in z.files if k not in ("folders", "labels")}
        # stroke id
        self.sid = self.w["folder"].astype(np.int64) * 10000 + self.w["trial"]
        # per-stroke beta and column
        uniq, idx = np.unique(self.sid, return_index=True)
        self.stroke_beta = dict(zip(uniq.tolist(), self.w["beta"][idx].tolist()))
        self.col = columns(self.stroke_beta)
        # measured plateau speed per folder (median of window speeds)
        self.speed = {f: float(np.median(self.w["speed"][self.w["folder"] == f])) for f in range(3)}
        # A1 trial order reversed (nominal force rank)
        trial = np.array([s % 10000 for s in uniq])
        if tag == "A1":
            rank = 51 - ((trial - 1) % 50 + 1)
        else:
            rank = (trial - 1) % 50 + 1
        self.rank = dict(zip(uniq.tolist(), rank.tolist()))

    def at(self, t):
        """dict stroke id -> (force, label, row index) for windows starting at plateau time t."""
        m = np.flatnonzero(np.isclose(self.w["t"], t))
        return {int(self.sid[i]): (float(self.w["force"][i]), int(self.w["label"][i]), int(i)) for i in m}


def columns(stroke_beta):
    """40 beta columns per folder: sort the folder's 2000 strokes by beta, cut into groups of 50."""
    col = {}
    by_folder = defaultdict(list)
    for s, b in stroke_beta.items():
        by_folder[s // 10000].append((b, s))
    for f, items in by_folder.items():
        items.sort()
        assert len(items) == 2000, (f, len(items))
        for i, (_, s) in enumerate(items):
            col[s] = i // 50
        lb = np.log([b for b, _ in items]).reshape(40, 50)
        assert np.all(lb[1:, 0] > lb[:-1, -1]), f
    return col


def column_groups(spec, t, force_override=None, label_override=None):
    """(folder, col) -> list of (force, is_helmholtz, label, beta, stroke id) sorted by force."""
    d = spec.at(t)
    groups = defaultdict(list)
    for s, (f, lab, i) in d.items():
        if force_override is not None:
            f = force_override.get(s, np.nan)
            if not np.isfinite(f):
                continue
        if label_override is not None:
            lab = label_override(i, lab)
        groups[(s // 10000, spec.col[s])].append((f, lab == H, lab, spec.stroke_beta[s], s))
    for g in groups.values():
        g.sort()
    return groups


def upper_interval(f, h, K=2):
    """Upper boundary (log-force interval) from forces f (sorted ascending) and Helmholtz flags h."""
    n = len(h)
    lf = np.log(np.maximum(np.asarray(f), 1e-4))
    e = next((i for i in range(n - 1, K - 2, -1) if all(h[i - K + 1:i + 1])), None)
    if e is None:
        return None, None, "none"
    if e == n - 1:
        return lf[-1], np.inf, "right"
    return lf[e], lf[e + 1], "interval"


def lower_interval(f, h, K=2):
    n = len(h)
    lf = np.log(np.maximum(np.asarray(f), 1e-4))
    s = next((i for i in range(n - K + 1) if all(h[i:i + K])), None)
    if s is None:
        return lf[-1], np.inf, "right"
    if s == 0:
        return -np.inf, lf[0], "left"
    return lf[s - 1], lf[s], "interval"


def widest_upper(f, h, max_hole=2):
    """Repo-style: bridge Helmholtz ranks separated by <= max_hole others, take the widest run, upper end."""
    idx = [i for i, x in enumerate(h) if x]
    if not idx:
        return None, None, "none"
    runs, cur = [], [idx[0]]
    for i in idx[1:]:
        if i - cur[-1] - 1 <= max_hole:
            cur.append(i)
        else:
            runs.append(cur)
            cur = [i]
    runs.append(cur)
    best = max(runs, key=lambda r: (r[-1] - r[0], r[-1]))
    lf = np.log(np.maximum(np.asarray(f), 1e-4))
    e = best[-1]
    if e == len(h) - 1:
        return lf[-1], np.inf, "right"
    return lf[e], lf[e + 1], "interval"


def table(spec, t, kind="upper", K=2, groups=None, **kw):
    """Rows (folder, col, log v, log beta, lo, hi, status)."""
    groups = groups if groups is not None else column_groups(spec, t, **kw)
    rows = []
    for (folder, c), g in sorted(groups.items()):
        f = [x[0] for x in g]
        h = [x[1] for x in g]
        if kind == "upper":
            lo, hi, st = upper_interval(f, h, K)
        elif kind == "widest":
            lo, hi, st = widest_upper(f, h)
        else:
            lo, hi, st = lower_interval(f, h, K)
        lb = np.log(np.median([x[3] for x in g]))
        rows.append((folder, c, np.log(spec.speed[folder]), lb, lo if lo is not None else np.nan,
                     hi if hi is not None else np.nan, st))
    return rows


def as_array(rows, drop_none=True):
    r = [x for x in rows if not (drop_none and x[6] == "none")]
    D = np.array([[x[0], x[1], x[2], x[3], x[4], x[5]] for x in r], dtype=float)
    return D


def nll_interval(mu, s, lo, hi):
    """-log P(lo < Y < hi), Y ~ N(mu, s), stable for one-sided intervals."""
    a = (lo - mu) / s
    b = (hi - mu) / s
    out = np.empty_like(mu)
    both = np.isfinite(lo) & np.isfinite(hi)
    right = np.isfinite(lo) & ~np.isfinite(hi)
    left = ~np.isfinite(lo) & np.isfinite(hi)
    # interval: log(Phi(b) - Phi(a)) ; use symmetric form for stability
    if np.any(both):
        aa, bb = a[both], b[both]
        flip = aa > 0
        lo_ = np.where(flip, -bb, aa)
        hi_ = np.where(flip, -aa, bb)
        lp_hi = log_ndtr(hi_)
        lp_lo = log_ndtr(lo_)
        out[both] = -(lp_hi + np.log1p(-np.exp(np.minimum(lp_lo - lp_hi, -1e-300))))
    if np.any(right):
        out[right] = -log_ndtr(-a[right])
    if np.any(left):
        out[left] = -log_ndtr(b[left])
    return out


def design(D, covs=()):
    """Design matrix [1, log v, log beta, *extra]."""
    cols = [np.ones(len(D)), D[:, 2], D[:, 3]] + [np.asarray(c) for c in covs]
    return np.column_stack(cols)


def fit(D, X=None, x0=None, fixed=None):
    """Interval-censored normal regression of log boundary on X (default [1, lv, lb]).
    fixed: dict {coef index: value}. Returns (coefs incl. log sigma, nll)."""
    X = design(D) if X is None else X
    lo, hi = D[:, 4], D[:, 5]
    p = X.shape[1]
    fixed = fixed or {}
    free = [i for i in range(p) if i not in fixed]

    def unpack(z):
        beta = np.zeros(p)
        for i, v in fixed.items():
            beta[i] = v
        beta[free] = z[:-1]
        return beta, z[-1]

    def nll(z):
        beta, ls = unpack(z)
        ls = np.clip(ls, -6, 3)
        return float(np.sum(nll_interval(X @ beta, np.exp(ls), lo, hi)))

    # start: OLS on finite midpoints / endpoints
    if x0 is None:
        y = np.where(np.isfinite(lo) & np.isfinite(hi), 0.5 * (lo + hi), np.where(np.isfinite(lo), lo + 0.3, hi - 0.3))
        b0 = np.linalg.lstsq(X, y, rcond=None)[0]
        starts = [np.r_[b0[free], np.log(0.4)]]
        b1 = b0.copy(); b1[1] = 1.0 if 1 in free else b1[1]
        starts.append(np.r_[b1[free], np.log(0.4)])
    else:
        starts = [np.asarray(x0)]
    best = None
    for st in starts:
        r = minimize(nll, st, method="Nelder-Mead", options={"maxiter": 40000, "xatol": 1e-7, "fatol": 1e-9})
        r = minimize(nll, r.x, method="Nelder-Mead", options={"maxiter": 40000, "xatol": 1e-8, "fatol": 1e-10})
        if best is None or r.fun < best.fun:
            best = r
    beta, ls = unpack(best.x)
    return np.r_[beta, ls], best.fun, best.x


def boot(D, nboot=500, seed=0, X_fn=None, strat=True, stat=lambda c: c[1]):
    """Column bootstrap (stratified by speed folder) of a statistic of the fitted coefficients."""
    rng = np.random.default_rng(seed)
    full, _, z = fit(D, None if X_fn is None else X_fn(D))
    out = []
    folders = np.unique(D[:, 0])
    for _ in range(nboot):
        if strat:
            idx = np.concatenate([rng.choice(np.flatnonzero(D[:, 0] == f), np.sum(D[:, 0] == f)) for f in folders])
        else:
            idx = rng.integers(0, len(D), len(D))
        Db = D[idx]
        c, _, _ = fit(Db, None if X_fn is None else X_fn(Db), x0=z)
        out.append(stat(c))
    return full, np.array(out)


def ci(a, q=(2.5, 97.5)):
    a = np.asarray(a)
    return np.percentile(a, q[0], axis=0), np.percentile(a, q[1], axis=0)


# ---------------------------------------------------------------- relabelling and metadata
def _thresholds():
    import json
    import sys
    repo = Path.home() / "violin-bowing-boundaries"
    sys.path.insert(0, str(repo / "scripts"))
    summ = next((repo / "reports/research_20261013/lockbox_A2_preprocess/2024-04-19_TypeA_sample2/regime_map").glob("*/summary.json"))
    return json.loads(summ.read_text())["thresholds_applied"]


def relabel_fn(spec, frac=0.5, min_periodicity=None, max_aperiodic=None):
    """Return label_override(i, lab) re-classifying window i with a different flyback fraction / periodicity
    thresholds (one flyback  <=>  second_flyback_ratio < frac, as in regime_map.flyback_structure)."""
    import math
    thr = dict(_thresholds())
    if min_periodicity is not None:
        thr["min_periodicity"] = min_periodicity
    if max_aperiodic is not None:
        thr["max_aperiodic"] = max_aperiodic
    import regime_map as rm
    w = spec.w

    def f(i, lab):
        if lab == LABELS.index("error"):
            return lab
        fpp = w["fpp"][i]
        if not math.isfinite(fpp):
            fpp_i = None
        elif fpp >= 1:
            sfr = w["sfr"][i]
            fpp_i = 1 if (math.isfinite(sfr) and sfr < frac) else 2
        else:
            fpp_i = int(fpp)
        try:
            new = rm.classify(float(w["amp"][i]), float(w["per"][i]), float(w["f0"][i]), fpp_i, 98.08, thr)
        except TypeError:
            new = "ambiguous"
        return LABELS.index(new)
    return f


def ramp_meta(tag):
    import csv
    path = {"A1": Path.home() / "violin-rne/claude_extra/ramp/ramp_meta_2024-03-25_TypeA_sample1.csv",
            "A2": Path.home() / "violin-rne/claude_extra/ramp_meta_A2.csv"}[tag]
    out = {}
    with open(path) as fh:
        for r in csv.DictReader(fh):
            if r["error"]:
                continue
            out[(r["folder"], int(r["trial"]))] = {k: float(r[k]) for k in ("accel", "ramp_s", "force_at_onset", "force_plateau",
                                                                          "force_cv_plateau", "plateau_on_s", "onset_s", "plateau_off_s")}
    return out


def meta_by_sid(spec):
    m = ramp_meta(spec.tag)
    res = {}
    for s in spec.stroke_beta:
        key = (spec.folders[s // 10000], s % 10000)
        if key in m:
            res[s] = m[key]
    return res
