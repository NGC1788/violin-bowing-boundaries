"""Why does the D_T2 / A_T2 / C_T2 upper SE differ between the claim (0.040) and the column-index bootstrap (0.169)? (post hoc)"""
import sys; from pathlib import Path; from collections import defaultdict
import numpy as np
sys.path.insert(0, str(Path(__file__).parent)); sys.path.insert(0, str(Path.home()/"violin-rne/novelty/thermal_upper"))
import levels as L
import thermal_law as TL
ref = L.make_ref("U_S1")
for tag in ("D_T2", "C_T2", "A_T2", "D1"):
    rows = L.columns(L.load(tag)); g = L.GRP[tag]
    rr, lo, hi = L.arrays(rows, "upper")
    print(tag, "upper columns", len(rr), "right-censored(F>top)", int(np.sum(~np.isfinite(hi))), "of which per speed",
          {round(float(np.exp(v)),2): int(np.sum((~np.isfinite(hi)) & (np.array([r['lv'] for r in rr]) == v))) for v in sorted(set(r['lv'] for r in rr))})
    by = defaultdict(list)
    for r in rows: by[r["col"]].append(r)
    keys = sorted(by); rng = np.random.default_rng(41); mine = []; nm = []
    for _ in range(200):
        kk = rng.integers(0, len(keys), len(keys)); rs = [r for k in kk for r in by[keys[k]]]
        a, l, h = L.arrays(rs, "upper"); mu = ref(a, g)
        mine.append(L.offset(mu, l, h)); nm.append(TL.offset_fit(mu, l, h)[0])
    mine, nm = np.array(mine), np.array(nm)
    print(f"   own L-BFGS-B sd {mine.std(ddof=1):.3f}  Nelder-Mead(TL.offset_fit) sd {nm.std(ddof=1):.3f}  max|diff| {np.max(np.abs(mine-nm)):.3f}  quantiles own {np.quantile(mine,[0,.05,.5,.95,1]).round(3)}")
