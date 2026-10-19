"""Cleaner hysteresis indicator in current-force terms (Claude, 2026-10-18; A1/A2 development data only).
Per beta column and speed, strokes on the acceleration floor only:
  entry force  = current force when the first K-window Helmholtz run completes (establishment event)
  lowest force = lowest current force that still carries a K-window Helmholtz run after entry
F_E,cur = 10 % quantile of entry forces (lowest force at which establishment was ever observed in the column)
F_M     = 10 % quantile of lowest forces
F_M < F_E,cur means Helmholtz motion is carried at forces where no stroke of that column ever established it."""
import csv
from collections import defaultdict
import numpy as np
import verify_astra_boundaries as V
from entry_exit import load_all, RAMP

K = 2
for tag, stem in V.SPECIMENS.items():
    seq, beta, speed = load_all(stem)
    col = V.columns(beta)
    meta = {(r["folder"], int(r["trial"])): r for r in csv.DictReader(open(RAMP[tag])) if not r["error"]}
    cells = defaultdict(list)
    for k, s in seq.items():
        if k not in meta:
            continue
        v = speed[k[0]]
        if float(meta[k]["accel"]) > 1.15 * v * v / (2 * 0.048):
            continue
        h = [x[2] for x in s]
        runs = [i for i in range(len(h) - K + 1) if all(h[i:i + K])]
        if not runs:
            continue
        entry = s[runs[0] + K - 1][1]
        low = min(s[i + K - 1][1] for i in runs)
        cells[(k[0], col[k])].append((entry, low))
    out = defaultdict(list)
    for (folder, c), g in cells.items():
        if len(g) < 5:
            continue
        e = np.array([x[0] for x in g]); l = np.array([x[1] for x in g])
        out[round(speed[folder], 2)].append((np.quantile(l, 0.1) / np.quantile(e, 0.1), np.median(l / e), np.mean(l < np.quantile(e, 0.1))))
    print(f"{tag}: " + " | ".join(
        f"v={v}: 열 {len(r)}, F_M/F_E,cur 중앙 {np.median([x[0] for x in r]):.2f} (<1인 열 {np.mean([x[0] < 1 for x in r]):.0%}), 긋기별 최저/진입 중앙 {np.median([x[1] for x in r]):.2f}, 어떤 진입보다도 낮게 유지된 긋기 {np.median([x[2] for x in r]):.0%}"
        for v, r in sorted(out.items())))
