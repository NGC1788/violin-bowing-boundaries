import sys, numpy as np
sys.path.insert(0, "/Users/jeongjihoon/violin-rne/final")
from data import load
for tag in ("A2","D1","B_T2r2"):
    d = load(tag)
    m = np.isclose(d["t"], 0.0) & (d["folder"]==1) & (d["column"]==10)
    idx = np.flatnonzero(m)
    o = idx[np.argsort(d["trial"][idx])]
    print(tag, "trial order:", d["trial"][o][:10], "...")
    print(" forces by trial:", np.round(d["force"][o],3))
    fs = np.sort(d["force"][o])
    print(" log diffs:", np.round(np.diff(np.log(np.maximum(fs,1e-3))),3))
