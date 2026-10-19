"""Referee 1 issue 1: is the attack acceleration indexed by recording order (trial) or by force? (Claude, 2026-10-19, post hoc)
Reads only label-free ramp metadata of A1 and A2. Within each beta column (50 strokes), Spearman correlation of the
attack acceleration with (a) trial index and (b) onset-force rank; median attack for the 20 lowest onset-force ranks."""
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

R = Path.home() / "violin-rne/claude_extra"
FILES = {"A1": R / "ramp/ramp_meta_2024-03-25_TypeA_sample1.csv", "A2": R / "ramp_meta_A2.csv"}
for tag, p in FILES.items():
    by = defaultdict(list)
    for r in csv.DictReader(open(p)):
        if r["error"]:
            continue
        by[r["folder"]].append((int(r["trial"]), float(r["beta"]), float(r["accel"]), float(r["force_at_onset"]), float(r["v_max"])))
    for f in sorted(by):
        s = sorted(by[f], key=lambda x: x[1])
        cols = [s[i:i + 50] for i in range(0, len(s), 50)]
        rt = [spearmanr([x[0] for x in c], [x[2] for x in c])[0] for c in cols]
        rf = [spearmanr([x[3] for x in c], [x[2] for x in c])[0] for c in cols]
        low = [x[2] for c in cols for x in sorted(c, key=lambda x: x[3])[:20]]
        b_first = s[0][1] if min(by[f])[0] == s[0][0] else None
        first = min(by[f]); last = max(by[f])
        print(f"{tag} {f.split('/')[-1]:14s} v {first[4]:.2f} | first trial beta {first[1]:.3f}, last trial beta {last[1]:.3f} | "
              f"within-column rho(accel, trial) median {np.nanmedian(rt):+.2f}, rho(accel, onset force) median {np.nanmedian(rf):+.2f} | "
              f"median attack, 20 lowest onset-force ranks {np.median(low):.3f} m/s2 | max {max(x[2] for x in by[f]):.1f}")


# Referee 1 issue 7: attack ratio 0.05 vs 0.20 m/s for the 20 lowest onset-force ranks, by beta tercile (A2)
by = defaultdict(list)
for r in csv.DictReader(open(FILES["A2"])):
    if not r["error"]:
        by[r["folder"]].append((float(r["beta"]), float(r["force_at_onset"]), float(r["accel"])))
edges = np.exp(np.linspace(np.log(0.02), np.log(0.2), 4))
for lo, hi in zip(edges[:-1], edges[1:]):
    med = {}
    for f in sorted(by):
        s = sorted(by[f])
        cols = [s[i:i + 50] for i in range(0, len(s), 50)]
        a = [x[2] for c in cols if lo <= np.median([y[0] for y in c]) < hi * 1.0001 for x in sorted(c, key=lambda x: x[1])[:20]]
        med[f.split("/")[-1]] = np.median(a)
    print(f"A2 beta {lo:.3f}-{hi:.3f}: median attack (20 lowest onset ranks) v1/v2/v3 = "
          + "/".join(f"{med[k]:.3f}" for k in sorted(med)) + f" m/s2; v3/v1 ratio {med['v3'] / med['v1']:.1f}")
