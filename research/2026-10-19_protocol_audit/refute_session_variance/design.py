"""Per-session design descriptors (post hoc): force range, beta range, censoring fractions of the boundary columns."""
import sys; from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent)); import levels as L
print("tag      fmax_med  fmin_med  beta_min beta_max  upper: n  right-cens  lower: left-cens(F<first)  hi<0.2N")
for t in L.CACHE:
    d = L.load(t); rows = L.columns(d)
    fmax = np.median([np.exp(r["upper"][0]) if not np.isfinite(r["upper"][1]) else np.nan for r in rows if True] or [np.nan])
    key = d["folder"]*1000+d["column"]
    fm = [d["force"][key==k].max() for k in np.unique(key)]; fn = [d["force"][key==k].min() for k in np.unique(key)]
    up = [r for r in rows if np.isfinite(r["upper"][0]) or np.isfinite(r["upper"][1])]
    rc = np.mean([not np.isfinite(r["upper"][1]) for r in up])
    lc = np.mean([not np.isfinite(r["lower"][0]) for r in rows]); l02 = np.mean([r["lower"][1] < np.log(0.2) for r in rows])
    b = np.exp([r["lb"] for r in rows])
    print(f"{t:8s} {np.median(fm):7.2f}  {np.median(fn):7.3f}  {b.min():.4f}  {b.max():.4f}   {len(up):3d}  {rc:.2f}        {lc:.2f}   {l02:.2f}")
