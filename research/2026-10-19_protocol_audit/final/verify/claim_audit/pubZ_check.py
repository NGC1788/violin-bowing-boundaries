"""claim_audit (post hoc, not part of the frozen plan): physics models with the dataset paper's published Z
(Lampis et al. POMA 58, 035013 (2025), Table 1: A_T1 1.0592, B_T1 1.0998, C_T1 1.0525, D_T1 1.0398, B_T2 1.1899 kg/s)
instead of the flyback-law Z; same train/test split and scoring as compare.py."""
import sys
T = float(sys.argv[1]) if len(sys.argv) > 1 else 0.2
sys.argv = ["x", str(T)]
sys.path.insert(0, "/Users/jeongjihoon/violin-rne/final")
import numpy as np
import compare as C
from data import load, TRAIN, TEST

pub = {"A1": 1.0592, "A2": 1.0592, "B1": 1.0998, "C1": 1.0525, "D1": 1.0398, "B_T2r1": 1.1899, "B_T2r2": 1.1899}
fly = {t: float(load(t)["Z"][0]) for t in TRAIN + TEST}
rows = {}
for t in TRAIN + TEST:
    rows[t] = C.intervals(load(t), T)
    for r in rows[t]:
        r["tag"] = t
tr = rows["A1"] + rows["A2"]
for label, z in (("flyback Z", fly), ("published Z", pub)):
    phys = C.physics_models(tr, z)
    for name in ("P-Sch", "P-Beigi", "P-free"):
        lo, up = [], []
        for t in TEST:
            sb = C.score_boundaries(rows[t], [phys[name][0](r, t) for r in rows[t]])
            lo.append(sb["lower"][0]); up.append(sb["upper"][0])
        print(f"t={T} {label:12s} {name:8s} lower mean {np.mean(lo):.3f} {np.round(lo, 3)} | upper mean {np.mean(up):.3f} {np.round(up, 3)}")
