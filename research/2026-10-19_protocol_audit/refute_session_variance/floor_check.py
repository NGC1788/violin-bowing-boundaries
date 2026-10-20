"""Post hoc: does the lower level track the force-grid floor (median lowest force per column) or the recording order? Already-seen data."""
import json, sys
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr
sys.path.insert(0, str(Path(__file__).parent)); import levels as L
LV = json.loads((Path(__file__).parent / "levels.json").read_text())
fmin = {}
for t in L.CACHE:
    d = L.load(t); key = d["folder"] * 1000 + d["column"]
    fmin[t] = float(np.median([d["force"][key == k].min() for k in np.unique(key)]))
for var in ("L_A12", "L_c02", "U_S1"):
    t = list(L.CACHE); x = [fmin[k] for k in t]; y = [LV[var][k]["level"] for k in t]
    r = spearmanr(x, y)
    a = ["A1", "A2", "A3", "A4"]
    print(f"{var}: Spearman(level, floor) all 14 runs rho {r.correlation:+.2f} p {r.pvalue:.3f}; A_T1 floors {[round(fmin[k],3) for k in a]} levels {[round(LV[var][k]['level'],3) for k in a]} rho {spearmanr([fmin[k] for k in a],[LV[var][k]['level'] for k in a]).correlation:+.2f}")
