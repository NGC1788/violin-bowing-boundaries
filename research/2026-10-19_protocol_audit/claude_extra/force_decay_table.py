"""Force-decay table for any processed archive (Claude, 2026-10-18).
Usage: python force_decay_table.py <timeres/features/<archive>.csv.gz> <ramp_meta csv>
Nominal force rank inside a beta column = ((trial - 1) mod 50) + 1 (IWK numbering; checked on A2).
Prints median bow force [before the bow moves -> first 0.2 s plateau window -> last window] by rank bins,
and the Helmholtz-free part is left to the analysis scripts."""
import csv, gzip, sys
from collections import defaultdict
import numpy as np
feat, ramp = sys.argv[1], sys.argv[2]
onset = {}
for r in csv.DictReader(open(ramp)):
    if not r["error"]:
        onset[(r["folder"], int(r["trial"]))] = float(r["force_at_onset"])
first, last = {}, {}
for r in csv.DictReader(gzip.open(feat, "rt")):
    if r["error"] or r["window_s"] != "0.2":
        continue
    k = (r["folder"], int(r["trial"])); t = float(r["t"])
    if k not in first or t < first[k][0]:
        first[k] = (t, float(r["force"]))
    if k not in last or t > last[k][0]:
        last[k] = (t, float(r["force"]))
for folder in sorted({k[0] for k in first}):
    print(folder)
    for lo in range(1, 51, 10):
        ks = [k for k in first if k[0] == folder and lo <= (k[1] - 1) % 50 + 1 <= lo + 9 and k in onset]
        a = np.array([[onset[k], first[k][1], last[k][1]] for k in ks])
        if len(a):
            print(f"  순위 {lo:2d}-{lo+9:2d} (n {len(a)}): {np.median(a[:,0]):.3f} → {np.median(a[:,1]):.3f} → {np.median(a[:,2]):.3f} N; 끝<출발/2: {np.mean(a[:,2] < a[:,0]/2):.0%}")
