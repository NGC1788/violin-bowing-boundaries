"""Measured targets for the simulator track (Claude, 2026-10-18, wf_20261018/simulator_hair).

A1/A2 development data only. For the beta columns nearest 0.1 and every speed folder:
  * lower boundary in the 0.2 s window at plateau start (current window force, first K=2 consecutive
    Helmholtz windows in force order; same rule as verify_astra_boundaries.intervals);
  * lock-in time of each stroke = first window start t (0.1 s windows every 0.05 s, from plateau start) from
    which >= 90 % of the remaining windows of the plateau are Helmholtz (None = never);
    summarised for strokes whose force at t = 0 is >= 2x, and 1.3-3x, the column boundary;
  * ramp duration (bow motion onset -> plateau start) of the same strokes, from ramp_meta (A2) so that
    lock-in can also be expressed from motion onset.
Classifier: repo regime_map.classify, A2 thresholds, reference f0 98.08 Hz (as verify_astra_boundaries).
Usage: python -B measured_targets.py
"""
import csv
import gzip
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO = Path.home() / "violin-bowing-boundaries"
sys.path.insert(0, str(REPO / "scripts"))
import regime_map as rm  # noqa: E402

SUMMARY = next((REPO / "reports/research_20261013/lockbox_A2_preprocess/2024-04-19_TypeA_sample2/regime_map").glob("*/summary.json"))
APPLIED = json.loads(SUMMARY.read_text())["thresholds_applied"]
FEAT = Path.home() / "violin-rne/timeres/features"
SPEC = {"A2": "2024-04-19_TypeA_sample2", "A1": "2024-03-25_TypeA_sample1"}
RAMP = {"A2": Path.home() / "violin-rne/claude_extra/ramp_meta_A2.csv",
        "A1": Path.home() / "violin-rne/claude_extra/ramp/ramp_meta_2024-03-25_TypeA_sample1.csv"}
BETA_LO, BETA_HI = 0.088, 0.115


def load(stem):
    w01 = defaultdict(dict)   # key -> {t: helm}  (0.1 s windows)
    w02 = {}                  # key -> (force, helm) at t = 0, 0.2 s window
    beta, speed = {}, defaultdict(list)
    with gzip.open(FEAT / f"{stem}.csv.gz", "rt") as fh:
        for r in csv.DictReader(fh):
            if r["error"]:
                continue
            key = (r["folder"], int(r["trial"]))
            beta[key] = float(r["beta"])
            fpp = r["flybacks_per_period"]
            lab = rm.classify(float(r["amplitude_std"]), float(r["periodicity"]), float(r["f0_hz"]),
                              int(float(fpp)) if fpp not in ("", "nan", "None") else None, 98.08, APPLIED)
            t = round(float(r["t"]), 2)
            if r["window_s"] == "0.1":
                w01[key][t] = lab == "helmholtz"
            elif r["window_s"] == "0.2" and t == 0.0:
                w02[key] = (float(r["force"]), lab == "helmholtz")
                speed[r["folder"]].append(float(r["speed"]))
    return w01, w02, beta, {f: float(np.median(v)) for f, v in speed.items()}


def lock_time(d, frac=0.9):
    ts = sorted(d)
    h = np.array([d[t] for t in ts], bool)
    for i in range(len(ts)):
        if h[i] and h[i:].mean() >= frac:
            return ts[i]
    return None


def main():
    for tag, stem in SPEC.items():
        w01, w02, beta, speed = load(stem)
        ramp = {}
        with open(RAMP[tag]) as fh:
            for r in csv.DictReader(fh):
                if not r["error"]:
                    ramp[(r["folder"], int(r["trial"]))] = (float(r["ramp_s"]), float(r["accel"]))
        by_folder = defaultdict(list)
        for k, b in beta.items():
            by_folder[k[0]].append((b, k))
        print(f"== {tag} ==")
        for folder in sorted(by_folder):
            items = sorted(by_folder[folder])
            cols = [items[i * 50:(i + 1) * 50] for i in range(40)]
            sel = [c for c in cols if BETA_LO <= np.median([b for b, _ in c]) <= BETA_HI]
            bounds, locks2, locks13, ramps, locked0 = [], [], [], [], []
            for c in sel:
                g = sorted((w02[k][0], w02[k][1], k) for _, k in c if k in w02)
                h = [x[1] for x in g]
                s = next((i for i in range(len(h) - 1) if h[i] and h[i + 1]), None)
                if s is None or s == 0:
                    bounds.append(np.nan)
                    continue
                fb = float(np.sqrt(g[s - 1][0] * g[s][0]))
                bounds.append(fb)
                for f, _, k in g:
                    lt = lock_time(w01[k])
                    lt = np.inf if lt is None else lt
                    if f >= 2 * fb:
                        locks2.append(lt)
                        if k in ramp:
                            ramps.append(ramp[k][0])
                    if 1.3 * fb <= f <= 3 * fb:
                        locks13.append(lt)
            L2, L13 = np.array(locks2), np.array(locks13)
            def summ(L):
                fin = L[np.isfinite(L)]
                q = np.percentile(fin, [50, 75, 90]) if len(fin) else [np.nan] * 3
                return (f"n {len(L)}, locked by t=0 {np.mean(L <= 0.0):.2f}, by 0.2 s {np.mean(L <= 0.2):.2f}, by 0.4 s {np.mean(L <= 0.4):.2f}, "
                        f"never {np.mean(~np.isfinite(L)):.2f}; locked strokes: median {q[0]:.2f}, p75 {q[1]:.2f}, p90 {q[2]:.2f} s")
            print(f" {folder} v={speed[folder]:.3f}  cols {len(sel)} (beta {np.median([b for c in sel for b, _ in c]):.3f})  "
                  f"boundary t0 per col {np.round(bounds, 3).tolist()}  median {np.nanmedian(bounds):.3f} N")
            print(f"    F >= 2 F*:   {summ(L2)}")
            print(f"    1.3-3 F*:    {summ(L13)}")
            if ramps:
                print(f"    ramp (motion onset -> plateau) for F >= 2 F*: median {np.median(ramps):.3f} s, range {np.min(ramps):.3f}-{np.max(ramps):.3f} s")


if __name__ == "__main__":
    main()
