"""Scan direction and low-force attack for every development diagram with label-free ramp metadata (Claude, 2026-10-19,
post hoc; confirmation targets A4/D2 and the high-force parts are not read). Median attack of the 20 lowest onset-force
ranks per beta column, by speed folder."""
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np

C = Path.home() / "violin-rne/astra_20261018/archive_companion"
STEMS = {"A1": "2024-03-25_TypeA_sample1", "A2": "2024-04-19_TypeA_sample2", "C1": "2024-04-12_TypeC_sample1",
         "D1": "2024-04-15_TypeD_sample1", "A_T2": "2024-06-19_TypeA_T2_sample1", "B_T2s1": "2024-06-13_TypeB_T2_sample1",
         "C_T2": "2024-06-11_TypeC_T2_sample1", "D_T2": "2024-06-17_TypeD_T2_sample1", "A3": "2024-04-22_TypeA_sample3"}
for tag, stem in STEMS.items():
    by = defaultdict(list)
    for r in csv.DictReader(open(C / stem / f"ramp_meta_{stem}.csv")):
        if not r["error"]:
            by[r["folder"]].append((int(r["trial"]), float(r["beta"]), float(r["accel"]), float(r["force_at_onset"]), float(r["v_max"])))
    out = []
    for f in sorted(by, key=lambda f: np.median([x[4] for x in by[f]])):
        s = sorted(by[f], key=lambda x: x[1])
        cols = [s[i:i + 50] for i in range(0, len(s), 50)]
        low = [x[2] for c in cols for x in sorted(c, key=lambda x: x[3])[:20]]
        first, last = min(by[f]), max(by[f])
        scan = "beta-desc" if first[1] > last[1] else "beta-asc"
        out.append(f"v{np.median([x[4] for x in by[f]]):.2f} {scan} attack(low20) {np.median(low):.3f}")
    print(f"{tag:7s} " + " | ".join(out))
