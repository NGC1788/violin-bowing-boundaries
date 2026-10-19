import sys, numpy as np
sys.path.insert(0, "/Users/jeongjihoon/violin-rne/final")
from data import load, STEMS
for tag in STEMS:
    d = load(tag)
    f = d["force"]
    print(tag, "n", len(f), "folders", list(d["folders"]), "Z", np.round(d["Z"],3), "f0", d["f0"])
    for t in (0.0, 0.2):
        m = np.isclose(d["t"], t)
        ff = f[m]
        print(f"  t={t} force min {ff.min():.4f} n<=1e-3 {np.sum(ff<=1e-3)} n<0.05 {np.sum(ff<0.05)} n<0.1 {np.sum(ff<0.1)} H share {d['H'][m].mean():.3f} nwin {m.sum()}")
    # trials per t
    for t in (0.0,0.1,0.2,0.3):
        m = np.isclose(d["t"], t); print("   t",t,"windows",m.sum(), end=";")
    print()
    print("  speeds per folder", {fo: np.round(np.unique(d['speed'][d['folder']==i]),4) for i,fo in enumerate(d['folders'])})
