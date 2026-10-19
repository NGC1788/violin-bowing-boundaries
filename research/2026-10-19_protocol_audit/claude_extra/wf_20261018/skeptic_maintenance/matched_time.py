"""Matched-time comparison (null (b) check): within floor strokes every stroke of a speed has the same attack, so
plateau time = time since motion onset - ramp_s (constant).  Forces are normalised by the column's F_E,cur
(q10 of all entry forces, the denominator of the original statistic) and pooled over the used columns.
For each time bin: entries (first K=2 run completing in the bin), maintained strokes (K-run end in the bin after
the stroke's own entry; one value per stroke = its lowest force in the bin), never-yet-entered strokes."""
from collections import defaultdict
import numpy as np
from common import load, column_ratios, entry_low, runs_of

K = 2
BINS = [(0.0, 0.3), (0.3, 0.8), (0.8, 1.3), (1.3, 2.5)]
for tag in ("A2", "A1"):
    S, speed = load(tag)
    for f in sorted(speed):
        fl = [s for s in S if s.folder == f and s.floor]
        out = column_ratios(fl)[f]
        used = {c for c, *_ in out}
        FE = {}
        for c in used:
            e = [entry_low(s.F, s.H)[0] for s in fl if s.col == c and entry_low(s.F, s.H) is not None]
            FE[c] = np.quantile(e, 0.1)
        print(f"{tag} v={speed[f]:.2f} ({len(used)} columns; force / column F_E,cur)")
        for a, b in BINS:
            ent, mnt, nev = [], [], []
            for s in fl:
                if s.col not in used:
                    continue
                r = runs_of(s.H, K)
                ends = r + K - 1
                ie = ends[0] if r.size else 10 ** 9
                inb = (s.t >= a - 1e-9) & (s.t < b - 1e-9)
                if not inb.any():
                    continue
                Fn = s.F / FE[s.col]
                if ie < len(s.t) and inb[ie]:
                    ent.append(Fn[ie])
                m = [j for j in ends if j > ie and inb[j]]
                if m:
                    mnt.append(Fn[m].min())
                nv = np.flatnonzero(inb & (np.arange(len(s.t)) < ie))
                if nv.size:
                    nev.append(Fn[nv].max())
            ent, mnt, nev = np.array(ent), np.array(mnt), np.array(nev)
            def q(x, p):
                return np.quantile(x, p) if x.size else np.nan
            print(f"   t in [{a:.1f},{b:.1f}): entries n={ent.size:3d} q10 {q(ent, .1):.2f} min {q(ent, 0):.2f} median {q(ent, .5):.2f} | "
                  f"maintained strokes n={mnt.size:3d} q10 {q(mnt, .1):.2f} min {q(mnt, 0):.2f} | "
                  f"not-yet-entered strokes n={nev.size:3d}, share with force above maintained q10 {np.mean(nev > q(mnt, .1)) if nev.size and mnt.size else np.nan:.2f}, "
                  f"above F_E,cur {np.mean(nev > 1) if nev.size else np.nan:.2f}")
