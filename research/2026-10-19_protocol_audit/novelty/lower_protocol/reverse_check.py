"""Reverse-causation check (Claude, 2026-10-19): is the plateau force decrease (dec5_pl) different for strokes that are
Helmholtz vs not at t = 0.2 s? Same rank range 1-20, per diagram. Output reverse_check.log."""
import csv, gzip, sys
from collections import defaultdict
from pathlib import Path
import numpy as np
sys.argv = sys.argv[:1]
sys.path.insert(0, str(Path(__file__).parent))
import descriptors as DS
D = DS.D
for tag in DS.TAGS:
    d = D.load(tag)
    m = np.isclose(d["t"], 0.2)
    H = {(DS.speed_of(d["folders"][d["folder"][i]]), int(d["trial"][i])): bool(d["H"][i]) for i in np.flatnonzero(m)}
    f0, f5 = {}, {}
    with gzip.open(D.FEAT / f"{DS.STEM[tag]}.csv.gz", "rt") as fh:
        for r in csv.DictReader(fh):
            if r["error"] or r["window_s"] != "0.1" or not DS.rep_ok(tag, r["folder"]):
                continue
            k = (DS.speed_of(r["folder"]), int(r["trial"]))
            t, v = float(r["t"]), float(r["speed"])
            if t == 0.0:
                f0[k] = float(r["force"])
            elif v * t >= 0.05 and k not in f5:
                f5[k] = float(r["force"])
    out = defaultdict(list)
    for k in f0:
        if k in f5 and k in H and DS.rank_of(tag, k[1]) <= 20:
            out[(k[0], H[k])].append((f5[k] - f0[k], f0[k]))
    line = f"{tag:7s}"
    for fo in ("v1", "v2", "v3"):
        a = np.array(out[(fo, True)]); b = np.array(out[(fo, False)])
        def s(x):
            return f"{np.median(x[:, 0]):+.3f} (F0 {np.median(x[:, 1]):.2f}, n {len(x)})" if len(x) >= 10 else "   n/a"
        line += f" | {fo} H {s(a)} nonH {s(b)}"
    # matched on starting force: strokes with F0 in 0.15-0.35 N
    allr = [(dd, f0_, h) for (fo, h), L in out.items() for dd, f0_ in L]
    A = np.array([x for x in allr if 0.15 <= x[1] <= 0.35 and x[2]]); B = np.array([x for x in allr if 0.15 <= x[1] <= 0.35 and not x[2]])
    if len(A) >= 10 and len(B) >= 10:
        line += f" || F0 0.15-0.35 N: H {np.median(A[:, 0]):+.3f} (n {len(A)}) nonH {np.median(B[:, 0]):+.3f} (n {len(B)})"
    print(line, flush=True)
