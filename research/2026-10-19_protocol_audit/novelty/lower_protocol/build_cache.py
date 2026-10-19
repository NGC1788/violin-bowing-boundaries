"""Build labelled-window caches (final/cache) for the two T2 diagrams not cached yet. (Claude, 2026-10-19)"""
import sys, time
from pathlib import Path
sys.path.insert(0, str(Path.home() / "violin-rne/final"))
import data as D
D.STEMS["A_T2"] = "2024-06-19_TypeA_T2_sample1"
D.STEMS["B_T2s1"] = "2024-06-13_TypeB_T2_sample1"
D.STEMS["C_T2"] = "2024-06-11_TypeC_T2_sample1"
for tag in ("B_T2s1", "C_T2"):
    t0 = time.time(); d = D.load(tag)
    print(tag, len(d["t"]), list(d["folders"]), f"{time.time()-t0:.0f}s", flush=True)
