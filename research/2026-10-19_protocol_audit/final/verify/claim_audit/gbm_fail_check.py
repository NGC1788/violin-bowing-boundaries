"""claim_audit: why D-gbm 'fails' (lower err > 2) when trained on B_T2r2 alone, even in-sample."""
import sys
sys.argv = ["x"]
sys.path.insert(0, "/Users/jeongjihoon/violin-rne/final")
import numpy as np
import compare as C
from data import load
T = 0.2
d = load("B_T2r2")
rows = C.intervals(d, T)
dm = C.data_models([d])
grid_lo = np.log(0.03)
preds = [C.band_from_prob(dm["D-gbm"], r, T) for r in rows]
fin = [(r, p) for r, p in zip(rows, preds) if np.isfinite(r["lower"][0]) and np.isfinite(r["lower"][1])]
pl = np.array([p[0] for r, p in fin]); mid = np.array([0.5 * (r["lower"][0] + r["lower"][1]) for r, p in fin])
print("finite lower cols", len(fin), "| pred lower at grid bottom (0.03 N):", np.sum(np.isclose(pl, grid_lo)), "| pred inf:", np.sum(~np.isfinite(pl)))
print("median |err|", np.median(np.where(np.isfinite(pl), np.abs(pl - mid), 3.0)))
print("median observed lower mid (N)", np.exp(np.median(mid)))
# fraction of training windows with measured force < 0.03 N labelled Helmholtz
m = d["force"] < 0.03
print("windows with force<0.03 N:", m.sum(), "H share", d["H"][m].mean() if m.any() else np.nan, "| force<=0:", (d["force"] <= 0).sum(), "H share", d["H"][d["force"] <= 0].mean())
for tag in ("A1", "A2", "B1", "C1", "D1", "B_T2r1", "B_T2r2"):
    x = load(tag); m = x["force"] < 0.05
    print(f"  {tag}: windows force<0.05 N {m.sum()} ({m.mean():.1%}), H share there {x['H'][m].mean():.2f}")
