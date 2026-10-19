"""Independent re-implementation of Astra's matched-time descriptive boundary fit (Claude, 2026-10-18).
Shares only the window classifier (repo regime_map.classify with the A2 thresholds, reference f0 98.08 Hz);
the beta-column recovery, boundary extraction and interval likelihood are written from scratch.
Usage: python verify_astra_boundaries.py
"""
import csv
import gzip
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm

REPO = Path.home() / "violin-bowing-boundaries"
sys.path.insert(0, str(REPO / "scripts"))
import regime_map as rm  # noqa: E402

SUMMARY = next((REPO / "reports/research_20261013/lockbox_A2_preprocess/2024-04-19_TypeA_sample2/regime_map").glob("*/summary.json"))
APPLIED = json.loads(SUMMARY.read_text())["thresholds_applied"]
FEAT = Path.home() / "violin-rne/timeres/features"
SPECIMENS = {"A1": "2024-03-25_TypeA_sample1", "A2": "2024-04-19_TypeA_sample2"}
TIMES = (0.0, 0.1, 0.2, 0.3)


def load(stem):
    rows = defaultdict(dict)  # (folder, trial) -> {t: (force, label)}
    beta, speed = {}, defaultdict(list)
    with gzip.open(FEAT / f"{stem}.csv.gz", "rt") as fh:
        for r in csv.DictReader(fh):
            if r["error"] or r["window_s"] != "0.2":
                continue
            t = round(float(r["t"]), 2)
            if t not in TIMES:
                continue
            fpp = r["flybacks_per_period"]
            label = rm.classify(float(r["amplitude_std"]), float(r["periodicity"]), float(r["f0_hz"]),
                                int(float(fpp)) if fpp not in ("", "nan", "None") else None, 98.08, APPLIED)
            key = (r["folder"], int(r["trial"]))
            rows[key][t] = (float(r["force"]), label == "helmholtz")
            beta[key] = float(r["beta"])
            speed[r["folder"]].append(float(r["speed"]))
    return rows, beta, {f: float(np.median(v)) for f, v in speed.items()}


def columns(beta):
    """40 beta columns per folder: sort the folder's 2000 strokes by beta, cut into 40 consecutive groups of 50."""
    col = {}
    by_folder = defaultdict(list)
    for k, b in beta.items():
        by_folder[k[0]].append((b, k))
    for folder, items in by_folder.items():
        items.sort()
        assert len(items) == 2000, (folder, len(items))
        for i, (_, k) in enumerate(items):
            col[k] = i // 50
        # sanity: the gap between columns must exceed the spread inside a column
        lb = np.log([b for b, _ in items]).reshape(40, 50)
        assert np.all(lb[1:, 0] > lb[:-1, -1]), folder
    return col


def intervals(rows, beta, col, speed, t, K=2):
    groups = defaultdict(list)
    for k, d in rows.items():
        if t in d:
            groups[(k[0], col[k])].append((d[t][0], d[t][1], beta[k]))
    out = []
    for (folder, c), g in groups.items():
        g.sort()
        f = np.array([x[0] for x in g]); h = [x[1] for x in g]
        s = next((i for i in range(len(h) - K + 1) if all(h[i:i + K])), None)
        lf = np.log(np.maximum(f, 1e-4))
        if s is None:
            lo, hi = lf[-1], np.inf
        elif s == 0:
            lo, hi = -np.inf, lf[0]
        else:
            lo, hi = lf[s - 1], lf[s]
        out.append((np.log(speed[folder]), np.log(np.median([x[2] for x in g])), lo, hi))
    return np.array(out)


def fit(D):
    lv, lb, lo, hi = D.T
    def nll(x):
        mu = x[0] + x[1] * lv + x[2] * lb
        s = np.exp(x[3])
        p = np.where(np.isfinite(hi), norm.cdf((hi - mu) / s), 1.0) - np.where(np.isfinite(lo), norm.cdf((lo - mu) / s), 0.0)
        return -np.sum(np.log(np.clip(p, 1e-300, None)))
    best = min((minimize(nll, [0, a, -1, np.log(0.5)], method="Nelder-Mead",
                         options={"maxiter": 20000, "xatol": 1e-8, "fatol": 1e-10}) for a in (0.0, 1.0, 2.0)),
               key=lambda r: r.fun)
    return best.x


def main():
    for tag, stem in SPECIMENS.items():
        rows, beta, speed = load(stem)
        col = columns(beta)
        print(f"{tag}: 긋기 {len(rows)}, 속도 {speed}")
        for t in TIMES:
            D = intervals(rows, beta, col, speed, t)
            x = fit(D)
            rng = np.random.default_rng(0)
            cols = np.arange(len(D))
            boots = [fit(D[rng.integers(0, len(D), len(D))])[1] for _ in range(100)]
            print(f"  t={t:.1f}s: 열 {len(D)} (왼쪽검열 {np.sum(~np.isfinite(D[:, 2]))}, 오른쪽검열 {np.sum(~np.isfinite(D[:, 3]))})  "
                  f"속도지수 {x[1]:+.3f} [{np.percentile(boots, 2.5):+.2f}, {np.percentile(boots, 97.5):+.2f}]  β지수 {x[2]:+.3f}")


if __name__ == "__main__":
    main()
