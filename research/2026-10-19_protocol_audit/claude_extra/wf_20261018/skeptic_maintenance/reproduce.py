"""Step 1: reproduce entry_vs_low.py with the shared implementation and describe the A2 v1 floor population."""
import numpy as np
from common import load, column_ratios, summary, entry_low

for tag in ("A1", "A2"):
    S, speed = load(tag)
    fl = [s for s in S if s.floor]
    out = column_ratios(fl)
    print(tag, {round(speed[f], 3): v for f, v in summary(out).items()})

S, speed = load("A2")
v1 = [s for s in S if s.folder == 0]
fl = [s for s in v1 if s.floor]
print(f"\nA2 v1: strokes {len(v1)}, floor {len(fl)}, ramp_s floor median {np.median([s.ramp_s for s in fl]):.3f}, "
      f"n windows median {np.median([len(s.t) for s in fl])}, t max median {np.median([s.t[-1] for s in fl]):.2f}")
ranks = np.array([s.rank for s in fl])
print("floor ranks per column (min..max):")
out = column_ratios(fl)[0]
used = {c for c, *_ in out}
rows = []
for c in range(40):
    g = [s for s in fl if s.col == c]
    if not g:
        continue
    ent = [entry_low(s.F, s.H) for s in g]
    ne = sum(e is not None for e in ent)
    rr = [s.rank for s in g]
    r = next((x for x in out if x[0] == c), None)
    tE = [s.t[e[2]] for s, e in zip(g, ent) if e is not None]
    tL = [s.t[e[3]] for s, e in zip(g, ent) if e is not None and e[3] >= 0]
    print(f" col {c:2d} beta {np.median([s.beta for s in g]):.3f} floor n={len(g):2d} ranks {min(rr)}-{max(rr)} entered {ne:2d} "
          f"tEntry med {np.median(tE) if tE else np.nan:.2f} tLow med {np.median(tL) if tL else np.nan:.2f} "
          + (f"ratio {r[1]:.2f} lowfrac {r[3]:.2f}" if r else "(not used)"))
