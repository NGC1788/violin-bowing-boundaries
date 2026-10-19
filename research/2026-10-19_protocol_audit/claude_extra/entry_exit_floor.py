"""entry_exit.py restricted to strokes on the acceleration floor (a = v^2 / (2 * 4.8 cm), identical attack for
all strokes of a speed), with K = 2 and K = 3 (Claude, 2026-10-18; A1/A2 development data only)."""
import csv
from collections import defaultdict
import numpy as np
import verify_astra_boundaries as V
from entry_exit import load_all, RAMP

for tag, stem in V.SPECIMENS.items():
    seq, beta, speed = load_all(stem)
    col = V.columns(beta)
    meta = {(r["folder"], int(r["trial"])): r for r in csv.DictReader(open(RAMP[tag])) if not r["error"]}
    for K in (2, 3):
        cells = defaultdict(list)
        for k, s in seq.items():
            if k not in meta:
                continue
            v = speed[k[0]]
            if float(meta[k]["accel"]) > 1.15 * v * v / (2 * 0.048):
                continue  # not on the floor
            h = [x[2] for x in s]
            e = next((i for i in range(len(h) - K + 1) if all(h[i:i + K])), None)
            pairs = [s[i + K - 1][1] for i in range(e, len(h) - K + 1) if all(h[i:i + K])] if e is not None else []
            low = min(pairs) if pairs else np.nan  # lowest force that still carries a full K-window Helmholtz run
            cells[(k[0], col[k])].append((float(meta[k]["force_at_onset"]), e is not None, low))
        ratios, below, n = defaultdict(list), defaultdict(list), defaultdict(int)
        for (folder, c), g in cells.items():
            g.sort()
            ent = [x[1] for x in g]
            s = next((i for i in range(len(ent) - 1) if ent[i] and ent[i + 1]), None)
            if s is None or s == 0 or len(g) < 10:
                continue
            FE = np.sqrt(g[s - 1][0] * g[s][0])
            lows = np.array([x[2] for x in g if x[1] and x[2] > 0])
            if lows.size < 3:
                continue
            v = round(speed[folder], 2)
            ratios[v].append(np.quantile(lows, 0.1) / FE); below[v].append(np.mean(lows < FE)); n[v] += 1
        print(f"{tag} K={K} (가속 바닥 긋기만): " + " | ".join(
            f"v={v}: 열 {n[v]}, F_M/F_E 중앙 {np.median(ratios[v]):.2f}, F_M<F_E 열 {np.mean(np.array(ratios[v]) < 1):.0%}, 수립 아래 유지 긋기 {np.median(below[v]):.0%}"
            for v in sorted(n)))
