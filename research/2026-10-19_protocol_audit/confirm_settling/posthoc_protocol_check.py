"""POST HOC (after the settling score): were COLD, WARM and A1-main(0.10 m/s) recorded with the same protocol?
Label-free: scan direction, low-force attack, and the dec5-style load decrease (median F(first 0.1-s window with >= 5 cm
travel) - F(onset 0.1-s window) over the 20 lowest onset-force strokes per beta column). Claude, 2026-10-20."""
import csv, gzip
from collections import defaultdict
from pathlib import Path
import numpy as np
H = Path.home() / "violin-rne"
SRC = {"cold/warm": (H / "astra_20261019/highforce_stream/2024-03-25_TypeA_sample1_coldwarm/features.csv.gz",
                     H / "astra_20261019/highforce_stream/2024-03-25_TypeA_sample1_coldwarm/ramp_meta.csv"),
       "A1main": (H / "timeres/features/2024-03-25_TypeA_sample1.csv.gz", H / "claude_extra/ramp/ramp_meta_2024-03-25_TypeA_sample1.csv")}
for name, (feat, ramp) in SRC.items():
    rm = defaultdict(list)
    for r in csv.DictReader(open(ramp)):
        if not r["error"] and abs(float(r["v_max"]) - 0.10) < 1e-3:
            rm[r["folder"]].append((int(r["trial"]), float(r["beta"]), float(r["force_at_onset"]), float(r["accel"])))
    F0, F5 = {}, {}
    with gzip.open(feat, "rt") as fh:
        rd = csv.reader(fh); h = next(rd); I = {k: h.index(k) for k in ("folder", "trial", "window_s", "t", "force", "speed", "error")}
        for row in rd:
            if row[I["error"]] or row[I["folder"]] not in rm or row[I["window_s"]] != "0.1":
                continue
            k = (row[I["folder"]], int(row[I["trial"]])); t = float(row[I["t"]]); f = float(row[I["force"]]); v = float(row[I["speed"]])
            if abs(t) < 1e-9:
                F0[k] = f
            elif v * t >= 0.05 and (k not in F5 or t < F5[k][0]):
                F5[k] = (t, f)
    for folder, items in sorted(rm.items()):
        first, last = min(items), max(items)
        s = sorted(items, key=lambda x: x[1]); cols = [s[i:i + 50] for i in range(0, len(s), 50)]
        low = [x for c in cols for x in sorted(c, key=lambda x: x[2])[:20]]
        dec = [F5[(folder, x[0])][1] - F0[(folder, x[0])] for x in low if (folder, x[0]) in F5 and (folder, x[0]) in F0]
        print(f"{name:9s} {folder.split('/')[-2] if name != 'A1main' else folder.split('/')[-1]:22s} scan {'beta-desc' if first[1] > last[1] else 'beta-asc'} | "
              f"low-force attack median {np.median([x[3] for x in low]):.3f} m/s2 | dec5 (N) median {np.median(dec):+.4f} (n {len(dec)}) | onset force of low strokes median {np.median([x[2] for x in low]):.3f} N")
