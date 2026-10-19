"""Analyse every archive processed by replicate.py, under several labelling rules.

For each archive (features/<archive>.csv.gz), window length and labelling rule:
  1. matched-time speed exponent  - speeds compared at equal time since plateau onset
  2. beta exponent versus time    - per speed, from the censored lower-boundary intervals
  3. steady-state extrapolation   - relaxation model per speed, with an identifiability flag
  4. critical slowing down        - lock-in time versus distance above the late boundary
Results: results/<archive>.json and a console summary. Rows are de-duplicated on
(folder, trial, window_s, t), because a crashed run can append a folder twice.

Usage: python analysis_multi.py [archive-stem ...]   (default: every file in features/)
"""

import csv
import gzip
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm, spearmanr

HERE = Path.home() / "violin-rne/timeres"
FEATURES, RESULTS = HERE / "features", HERE / "results"
RULES = {  # name: (min_periodicity, helmholtz run length)
    "base": (0.8, 2), "run1": (0.8, 1), "run3": (0.8, 3), "per0.7": (0.7, 2), "per0.9": (0.9, 2),
}
WINDOWS = (0.1, 0.2, 0.4)


# ------------------------------------------------------------------ loading and labelling

def load(stem: str) -> list[dict]:
    seen, rows = set(), []
    with gzip.open(FEATURES / f"{stem}.csv.gz", "rt", newline="") as handle:
        for r in csv.DictReader(handle):
            if r.get("error") or r.get("folder") == "folder":
                continue
            key = (r["folder"], r["trial"], r["window_s"], r["t"])
            if key in seen:
                continue
            seen.add(key)
            for k in ("beta", "window_s", "t", "plateau_s", "force", "speed", "amplitude_std", "periodicity",
                      "f0_hz", "flybacks_per_period"):
                try:
                    r[k] = float(r[k])
                except (TypeError, ValueError):
                    r[k] = math.nan
            r["trial"] = int(r["trial"])
            rows.append(r)
    return rows


def label(r: dict, floor: float, ref_f0: float, min_per: float) -> str:
    """regime_map.classify with a variable periodicity threshold."""
    a, p, f0, fb = r["amplitude_std"], r["periodicity"], r["f0_hz"], r["flybacks_per_period"]
    if not (a >= floor) and not (p >= min_per):
        return "no_oscillation"
    if not math.isfinite(p) or p < 0.5:
        return "aperiodic"
    if p < min_per:
        return "ambiguous"
    if math.isfinite(ref_f0) and math.isfinite(f0) and f0 < 0.75 * ref_f0:
        return "subharmonic"
    if fb == 1:
        return "helmholtz"
    if fb >= 2:
        return "multiple_slip"
    return "ambiguous"


def prepare(rows: list[dict]) -> dict:
    """Per-archive constants: noise floor, reference f0 and speed per folder, beta levels."""
    contact_free = [r["amplitude_std"] for r in rows if r["force"] <= 0 and math.isfinite(r["amplitude_std"])]
    floor = max(0.02, 3 * float(np.median(contact_free))) if len(contact_free) >= 10 else 0.02
    info = {"floor": floor, "folders": {}}
    for folder in sorted({r["folder"] for r in rows}):
        sub = [r for r in rows if r["folder"] == folder]
        strong = [r["f0_hz"] for r in sub if r["periodicity"] >= 0.8 and r["amplitude_std"] >= floor and math.isfinite(r["f0_hz"])]
        speed = float(np.median([r["speed"] for r in sub if math.isfinite(r["speed"])]))
        # beta levels: split sorted log-beta values at gaps larger than a third of the median spacing of distinct values
        uniq = np.unique(np.round(np.log([r["beta"] for r in sub if r["beta"] > 0]), 4))
        gaps = np.diff(uniq)
        cut = np.median(gaps[gaps > 0]) / 3 if (gaps > 0).any() else 0.01
        levels, current = {}, 0
        for i, value in enumerate(uniq):
            if i and value - uniq[i - 1] > cut:
                current += 1
            levels[value] = current
        info["folders"][folder] = {"ref_f0": float(np.median(strong)) if strong else math.nan, "speed": speed,
                                   "levels": {float(k): v for k, v in levels.items()}}
    return info


# ------------------------------------------------------------------ censored boundary intervals

def intervals(rows, info, t, window_s, rule, folders=None):
    """(speed, beta, log F_lo, log F_hi) per (folder, beta level) at window start t."""
    min_per, run = RULES[rule]
    cols = defaultdict(list)
    for r in rows:
        if r["window_s"] != window_s or abs(r["t"] - t) > 1e-6 or r["force"] <= 0:
            continue
        if folders and r["folder"] not in folders:
            continue
        f = info["folders"][r["folder"]]
        level = f["levels"].get(float(np.round(np.log(r["beta"]), 4)))
        cols[(r["folder"], level)].append(r)
    out = []
    for (folder, _), cells in cols.items():
        if len(cells) < 6:
            continue
        f = info["folders"][folder]
        cells.sort(key=lambda r: r["force"])
        h = [label(c, info["floor"], f["ref_f0"], min_per) == "helmholtz" for c in cells]
        lf = np.log([c["force"] for c in cells])
        start = next((i for i in range(len(h) - run + 1) if all(h[i:i + run])), None)
        if start is None:
            lo, hi = lf[-1], np.inf
        elif start == 0:
            lo, hi = -np.inf, lf[0]
        else:
            lo, hi = lf[start - 1], lf[start]
        out.append((f["speed"], float(np.median([c["beta"] for c in cells])), lo, hi))
    return out


def censored_fit(data, design):
    """Interval-censored normal MLE. design(data) -> X (n x k); returns coefficients and sigma."""
    X = design(data)
    lo = np.array([d[2] for d in data]); hi = np.array([d[3] for d in data])

    def nll(x):
        mu = X @ x[:-1]; s = np.exp(x[-1])
        up = np.where(np.isfinite(hi), norm.cdf((hi - mu) / s), 1.0)
        dn = np.where(np.isfinite(lo), norm.cdf((lo - mu) / s), 0.0)
        return -np.sum(np.log(np.clip(up - dn, 1e-300, None)))

    x0 = np.zeros(X.shape[1] + 1); x0[-1] = np.log(0.4)
    r = minimize(nll, x0, method="L-BFGS-B")
    return r.x[:-1], float(np.exp(r.x[-1]))


def law_design(data):
    return np.column_stack([np.ones(len(data)), np.log([d[0] for d in data]), np.log([d[1] for d in data])])


def beta_design(data):
    return np.column_stack([np.ones(len(data)), np.log([d[1] for d in data])])


def bootstrap(data, design, index, reps=60, seed=0):
    rng = np.random.default_rng(seed)
    vals = [censored_fit([data[i] for i in rng.integers(0, len(data), len(data))], design)[0][index] for _ in range(reps)]
    return [float(v) for v in np.percentile(vals, [2.5, 97.5])]


# ------------------------------------------------------------------ the four analyses

def matched_speed(rows, info, window_s, rule):
    folders = info["folders"]
    tmax = min(max(r["t"] for r in rows if r["folder"] == f and r["window_s"] == window_s) for f in folders)
    out = []
    for t in np.round(np.arange(0, tmax + 1e-9, 0.1), 2):
        data = intervals(rows, info, t, window_s, rule)
        if len({round(d[0], 3) for d in data}) < 2 or len(data) < 20:
            continue
        coef, _ = censored_fit(data, law_design)
        out.append({"t": float(t), "n": len(data), "speed_exp": float(coef[1]), "beta_exp": float(coef[2]),
                    "speed_ci": bootstrap(data, law_design, 1)})
    return out


def extrapolate(rows, info, window_s, rule, folder):
    sub = [r for r in rows if r["folder"] == folder and r["window_s"] == window_s]
    tmax = max(r["t"] for r in sub)
    step = 0.05 if tmax < 1 else 0.1 if tmax < 2 else 0.25
    data = []
    for t in np.round(np.arange(0, tmax + 1e-9, step), 3):
        data += [(d[1], t, d[2], d[3]) for d in intervals(sub, info, t, window_s, rule)]
    if len(data) < 40:
        return None
    lb = np.log([d[0] for d in data]); tt = np.array([d[1] for d in data])
    lo = np.array([d[2] for d in data]); hi = np.array([d[3] for d in data])

    def nll(x):
        c, p, d0, d1, ltau, ls = x
        mu = c + p * lb + (d0 + d1 * lb) * np.exp(-tt / np.exp(ltau)); s = np.exp(ls)
        up = np.where(np.isfinite(hi), norm.cdf((hi - mu) / s), 1.0)
        dn = np.where(np.isfinite(lo), norm.cdf((lo - mu) / s), 0.0)
        return -np.sum(np.log(np.clip(up - dn, 1e-300, None)))

    best = min((minimize(nll, [-5, -2, 1, 1, np.log(t0), np.log(0.3)], method="Nelder-Mead",
                         options={"maxiter": 30000, "maxfev": 30000}) for t0 in (0.15, 0.5, 1.5)), key=lambda r: r.fun)
    tau = float(np.exp(best.x[4]))
    return {"folder": folder, "p_inf": float(best.x[1]), "tau_s": tau, "t_obs_s": float(tmax),
            "identifiable": bool(tau < tmax / 3), "p_at_0": float(best.x[1] + best.x[3])}


def critical_slowing(rows, info, window_s, rule, folder):
    min_per, run = RULES[rule]
    sub = [r for r in rows if r["folder"] == folder and r["window_s"] == window_s]
    f = info["folders"][folder]
    late = max(r["t"] for r in sub) * 0.5
    late = min({r["t"] for r in sub}, key=lambda x: abs(x - late))
    bound = {}
    for (_, beta, lo, hi) in intervals(sub, info, late, window_s, rule):
        if np.isfinite(lo) and np.isfinite(hi):
            bound[round(np.log(beta), 2)] = hi
    strokes = defaultdict(list)
    for r in sub:
        strokes[r["trial"]].append(r)
    d, T, censored = [], [], []
    for seq in strokes.values():
        seq.sort(key=lambda r: r["t"])
        key = round(np.log(np.median([r["beta"] for r in seq])), 2)
        if key not in bound:
            continue
        dist = np.log(np.mean([r["force"] for r in seq])) - bound[key]
        if not 0 < dist < 1.2:
            continue
        h = np.array([label(r, info["floor"], f["ref_f0"], min_per) == "helmholtz" for r in seq])
        t = np.array([r["t"] for r in seq])
        lock = next((t[i] for i in range(len(h)) if h[i] and h[i:].mean() >= 0.8), None)
        d.append(dist); T.append(lock if lock is not None else t[-1]); censored.append(lock is None)
    if len(d) < 30:
        return None
    rho, p = spearmanr(d, T)
    return {"folder": folder, "n": len(d), "rho": float(rho), "p": float(p),
            "never_locked_near": float(np.mean([c for x, c in zip(d, censored) if x < 0.15])) if any(x < 0.15 for x in d) else None}


def analyse(stem: str) -> dict:
    rows = load(stem)
    info = prepare(rows)
    report = {"archive": stem, "windows": len(rows), "floor": info["floor"],
              "folders": {k: {"speed": v["speed"], "ref_f0": v["ref_f0"]} for k, v in info["folders"].items()},
              "by_setting": {}}
    for window_s in WINDOWS:
        for rule in RULES:
            if window_s != 0.2 and rule != "base":
                continue  # rules are varied at 0.2 s; window lengths at the base rule
            key = f"w{window_s}_{rule}"
            entry = {"matched_speed": matched_speed(rows, info, window_s, rule)}
            if rule == "base":
                entry["extrapolation"] = [x for x in (extrapolate(rows, info, window_s, rule, f) for f in info["folders"]) if x]
                entry["critical_slowing"] = [x for x in (critical_slowing(rows, info, window_s, rule, f) for f in info["folders"]) if x]
            report["by_setting"][key] = entry
            print(f"  {stem} {key}: " + "; ".join(f"t{m['t']:.1f} v^{m['speed_exp']:+.2f}" for m in entry["matched_speed"][:4]), flush=True)
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / f"{stem}.json").write_text(json.dumps(report, indent=1))
    return report


def main():
    stems = sys.argv[1:] or sorted(p.name[:-7] for p in FEATURES.glob("*.csv.gz"))
    for stem in stems:
        print(f"== {stem}", flush=True)
        r = analyse(stem)
        base = r["by_setting"].get("w0.2_base", {})
        for e in base.get("extrapolation", []):
            print(f"   정상상태 β {e['folder'][-6:]}: p∞={e['p_inf']:+.2f} τ={e['tau_s']:.2f}s 관측 {e['t_obs_s']:.2f}s "
                  f"{'식별 가능' if e['identifiable'] else '식별 불가'}")
        for e in base.get("critical_slowing", []):
            print(f"   임계감속 {e['folder'][-6:]}: ρ={e['rho']:+.2f} p={e['p']:.1e} n={e['n']}")


if __name__ == "__main__":
    main()
