"""Do A1 and A2 agree when the boundary is expressed in the force at the attack (before the bow moves)
instead of the current window force?  (Claude, 2026-10-18; reuses verify_astra_boundaries.py)"""
import csv
import numpy as np
import verify_astra_boundaries as V

RAMP = {"A1": "ramp/ramp_meta_2024-03-25_TypeA_sample1.csv", "A2": "ramp_meta_A2.csv"}
for tag, stem in V.SPECIMENS.items():
    rows, beta, speed = V.load(stem)
    col = V.columns(beta)
    onset = {}
    for r in csv.DictReader(open(RAMP[tag])):
        if not r["error"]:
            onset[(r["folder"], int(r["trial"]))] = float(r["force_at_onset"])
    print(f"{tag}:")
    for t in V.TIMES:
        res = {}
        for name in ("현재 힘", "출발 힘"):
            R = rows if name == "현재 힘" else {k: {tt: (onset[k], h) for tt, (f, h) in d.items()} for k, d in rows.items() if k in onset}
            D = V.intervals(R, beta, col, speed, t)
            x = V.fit(D)
            rng = np.random.default_rng(0)
            b = [V.fit(D[rng.integers(0, len(D), len(D))])[1] for _ in range(100)]
            res[name] = f"{x[1]:+.3f} [{np.percentile(b, 2.5):+.2f},{np.percentile(b, 97.5):+.2f}] β{x[2]:+.2f}"
        print(f"  t={t:.1f}: 현재 힘 기준 {res['현재 힘']}   출발 힘 기준 {res['출발 힘']}")
