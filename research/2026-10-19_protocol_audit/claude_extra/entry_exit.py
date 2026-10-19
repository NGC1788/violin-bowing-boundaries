"""Establishment vs maintenance in the measured IWK data, using the force that falls with bow travel
(Claude, 2026-10-18; A1/A2 development data only).

Per stroke (0.2 s windows every 0.05 s over the plateau, classifier as in verify_astra_boundaries.py):
  entered   = a run of K=2 Helmholtz windows occurs somewhere in the plateau
  F_low_H   = the lowest current window force among Helmholtz windows that follow the entry run
              ("Helmholtz still present at this force, reached from a higher force")
Per (speed, beta column):
  F_E = establishment boundary in *onset* force (force before the bow moves): strokes sorted by onset force,
        outcome = entered (whole plateau available), first K=2 run of entered strokes -> interval midpoint
  F_M = lower 10 % quantile of F_low_H over entered strokes (how far down Helmholtz is kept)
Reported: ratio F_M / F_E per column (< 1: Helmholtz kept below the force from which it can be established),
speed exponents of F_E and F_M (OLS on columns with both finite), and the share of entered strokes whose
Helmholtz survives below F_E. Caveats: strokes starting near F_M also lose force while they try to establish;
attack acceleration and total bowing time differ between the two groups; this is description, not a causal test.
"""
import csv
import gzip
from collections import defaultdict

import numpy as np

import verify_astra_boundaries as V

RAMP = {"A1": "ramp/ramp_meta_2024-03-25_TypeA_sample1.csv", "A2": "ramp_meta_A2.csv"}


def load_all(stem):
    seq = defaultdict(list)
    beta = {}
    speed = defaultdict(list)
    with gzip.open(V.FEAT / f"{stem}.csv.gz", "rt") as fh:
        for r in csv.DictReader(fh):
            if r["error"] or r["window_s"] != "0.2":
                continue
            fpp = r["flybacks_per_period"]
            lab = V.rm.classify(float(r["amplitude_std"]), float(r["periodicity"]), float(r["f0_hz"]),
                                int(float(fpp)) if fpp not in ("", "nan", "None") else None, 98.08, V.APPLIED)
            k = (r["folder"], int(r["trial"]))
            seq[k].append((float(r["t"]), float(r["force"]), lab == "helmholtz", lab))
            beta[k] = float(r["beta"])
            speed[r["folder"]].append(float(r["speed"]))
    for k in seq:
        seq[k].sort()
    return seq, beta, {f: float(np.median(v)) for f, v in speed.items()}


def stroke_summary(s, K=2):
    h = [x[2] for x in s]
    e = next((i for i in range(len(h) - K + 1) if all(h[i:i + K])), None)
    if e is None:
        return False, np.nan
    after = [x[1] for x in s[e:] if x[2]]
    return True, float(np.min(after))


def main():
    for tag, stem in V.SPECIMENS.items():
        seq, beta, speed = load_all(stem)
        col = V.columns(beta)
        onset = {(r["folder"], int(r["trial"])): float(r["force_at_onset"]) for r in csv.DictReader(open(RAMP[tag])) if not r["error"]}
        cells = defaultdict(list)
        for k, s in seq.items():
            if k in onset:
                entered, flow = stroke_summary(s)
                cells[(k[0], col[k])].append((onset[k], entered, flow, beta[k]))
        rows = []
        for (folder, c), g in cells.items():
            g.sort()
            ent = [x[1] for x in g]
            s = next((i for i in range(len(ent) - 1) if ent[i] and ent[i + 1]), None)
            if s is None or s == 0:
                continue  # censored establishment boundary
            FE = float(np.sqrt(g[s - 1][0] * g[s][0]))
            lows = np.array([x[2] for x in g if x[1] and np.isfinite(x[2]) and x[2] > 0])
            if lows.size < 3:
                continue
            FM = float(np.quantile(lows, 0.1))
            below = float(np.mean(lows < FE))
            rows.append((speed[folder], float(np.median([x[3] for x in g])), FE, FM, below))
        R = np.array(rows)
        print(f"\n{tag}: 경계가 잡힌 열 {len(R)}개")
        for v in sorted(set(R[:, 0])):
            m = R[:, 0] == v
            print(f"  v={v:.2f}: F_E(출발 힘 기준 수립) 중앙 {np.median(R[m, 2]):.3f} N, F_M(유지 하위 10%) 중앙 {np.median(R[m, 3]):.3f} N, "
                  f"F_M/F_E 중앙 {np.median(R[m, 3] / R[m, 2]):.2f} [사분위 {np.quantile(R[m, 3] / R[m, 2], 0.25):.2f}–{np.quantile(R[m, 3] / R[m, 2], 0.75):.2f}], "
                  f"F_M<F_E 열 {np.mean(R[m, 3] < R[m, 2]):.0%}, 수립 아래에서도 헬름홀츠가 남은 긋기 비율 중앙 {np.median(R[m, 4]):.0%}")
        X = np.column_stack([np.ones(len(R)), np.log(R[:, 0]), np.log(R[:, 1])])
        for name, y in (("F_E", np.log(R[:, 2])), ("F_M", np.log(R[:, 3]))):
            coef = np.linalg.lstsq(X, y, rcond=None)[0]
            rng = np.random.default_rng(0)
            boots = []
            for _ in range(1000):
                i = rng.integers(0, len(R), len(R))
                boots.append(np.linalg.lstsq(X[i], y[i], rcond=None)[0][1])
            print(f"  {name}: 속도 지수 {coef[1]:+.2f} [{np.percentile(boots, 2.5):+.2f}, {np.percentile(boots, 97.5):+.2f}], β 지수 {coef[2]:+.2f}")


if __name__ == "__main__":
    main()
