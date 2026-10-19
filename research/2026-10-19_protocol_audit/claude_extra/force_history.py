"""IWK low-force strokes are not constant-force strokes: the bow force decays during the stroke.
(1) Quantify the decay for every A2 stroke.  (2) Natural experiment for establishment/maintenance
hysteresis in *measured* data: at the same current force F(t), beta and speed, is Helmholtz motion more
likely when the stroke started at a higher force?  (Claude, 2026-10-18)

Inputs: ramp_meta_A2.csv (force_at_onset = median force in the 10 ms before the bow starts moving),
~/violin-rne/timeres/windows_full.csv.gz (0.2 s windows: per-window mean force and label).
Model per speed, windows in the late half of the plateau, one window per stroke (the last one):
  logit P(H) = c + a1 log F(t) + a2 (log F(t))^2 + b log beta + d log beta * log F(t) + h log(F_onset / F(t))
h > 0 means history dependence (maintenance after establishment at a higher force). Stroke-level
bootstrap (the unit is the stroke; one window per stroke) for h. Confounds: F_onset/F(t) also tracks
acceleration (design) and the force-decay mechanism itself; we report h with and without log accel.
"""
import csv
import gzip
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

HERE = Path(__file__).parent
WINDOWS = Path.home() / "violin-rne/timeres/windows_full.csv.gz"


def load():
    meta = {}
    with (HERE / "ramp_meta_A2.csv").open() as fh:
        for r in csv.DictReader(fh):
            if not r["error"]:
                meta[(r["folder"].split("/")[-1], int(r["trial"]))] = r
    win = defaultdict(list)
    with gzip.open(WINDOWS, "rt") as fh:
        for r in csv.DictReader(fh):
            if not r["error"]:
                win[(r["condition"], int(r["trial"]))].append((float(r["t"]), float(r["force"]), r["label"], float(r["plateau_s"]),
                                                             float(r["beta"]), int(r["force_rank"])))
    return meta, win


def logit(X, y):
    def f(b):
        z = X @ b
        return float(np.sum(np.logaddexp(0, z) - y * z)), X.T @ (1 / (1 + np.exp(-z)) - y)
    r = minimize(f, np.zeros(X.shape[1]), jac=True, method="BFGS", options={"maxiter": 2000})
    return r.x


def main():
    meta, win = load()
    print("(1) 활 힘 감쇠: 힘 순위 구간별 중앙값 [출발 직전 → 평탄 첫 창 → 평탄 마지막 창] (N)")
    for cond in ("v1", "v2", "v3"):
        rows = []
        for (c, trial), ws in win.items():
            if c != cond or (c, trial) not in meta:
                continue
            ws.sort()
            rows.append((ws[0][5], float(meta[(c, trial)]["force_at_onset"]), ws[0][1], ws[-1][1]))
        rows = np.array(rows)
        print(f"  [{cond}]")
        for lo, hi in ((1, 10), (11, 20), (21, 30), (31, 40), (41, 50)):
            m = (rows[:, 0] >= lo) & (rows[:, 0] <= hi)
            on, first, last = np.median(rows[m, 1]), np.median(rows[m, 2]), np.median(rows[m, 3])
            frac = np.mean(rows[m, 3] < 0.5 * rows[m, 1])
            print(f"    힘 순위 {lo:2d}-{hi:2d}: {on:.3f} → {first:.3f} → {last:.3f}   (끝 힘 < 출발 힘의 절반인 긋기 {frac:.0%})")
    print("\n(2) 같은 현재 힘·β·속도에서, 더 센 힘으로 출발한 긋기가 헬름홀츠일 확률이 더 높은가 (평탄 후반 마지막 창, 긋기당 1개)")
    rng = np.random.default_rng(0)
    for cond in ("v1", "v2", "v3"):
        data = []
        for (c, trial), ws in win.items():
            if c != cond or (c, trial) not in meta:
                continue
            ws.sort()
            t, f_now, label, plateau, beta, _ = ws[-1]
            f_on = float(meta[(c, trial)]["force_at_onset"]); acc = float(meta[(c, trial)]["accel"])
            if f_now <= 0.01 or f_on <= 0.01 or t < plateau / 2 - 0.2:
                continue
            data.append((np.log(f_now), np.log(beta), np.log(f_on / f_now), np.log(acc), label == "helmholtz"))
        D = np.array(data, float)
        lf, lb, lr, la, y = D.T
        cols = [np.ones_like(lf), lf, lf ** 2, lb, lb * lf]
        for name, extra in (("이력만", [lr]), ("이력+가속", [lr, la])):
            X = np.column_stack(cols + extra)
            mu, sd = X[:, 1:].mean(0), X[:, 1:].std(0)
            Xs = X.copy(); Xs[:, 1:] = (X[:, 1:] - mu) / sd
            coef = logit(Xs, y)
            h = coef[len(cols)] / sd[len(cols) - 1]
            boots = []
            for _ in range(200):
                i = rng.integers(0, len(y), len(y))
                boots.append(logit(Xs[i], y[i])[len(cols)] / sd[len(cols) - 1])
            lo, hi = np.percentile(boots, [2.5, 97.5])
            print(f"  [{cond}] {name}: h(log F_출발/F_현재) = {h:+.2f} [{lo:+.2f}, {hi:+.2f}]  (긋기 {len(y)}, 헬름홀츠 {int(y.sum())}, log(F_출발/F_현재) 중앙 {np.median(lr):+.2f})")


if __name__ == "__main__":
    main()
