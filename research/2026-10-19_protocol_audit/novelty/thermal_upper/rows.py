"""Upper-boundary columns (t = 0.2 s, K = 2) of the 11 processed diagrams, via final/data.py + final/compare.py.
Existing caches in final/cache are read only; D_T2 is built from its features into thermal_upper/cache (no write elsewhere).
(Claude, 2026-10-19)"""
import sys
from pathlib import Path

import numpy as np

_argv = sys.argv
sys.argv = sys.argv[:1]  # final/compare.py reads sys.argv at import
sys.path.insert(0, str(Path.home() / "violin-rne/final"))
import compare as C  # noqa: E402
import data as D  # noqa: E402
sys.argv = _argv

HERE = Path(__file__).parent
T = 0.2
TAGS = ["A1", "A2", "B1", "C1", "D1", "A_T2", "B_T2s1", "B_T2r1", "B_T2r2", "C_T2", "D_T2"]
GROUP = {"A1": "A T1", "A2": "A T1", "B1": "B T1", "C1": "C T1", "D1": "D T1", "A_T2": "A T2",
         "B_T2s1": "B T2", "B_T2r1": "B T2", "B_T2r2": "B T2", "C_T2": "C T2", "D_T2": "D T2"}
# string identity for leave-one-string-out (B_T2 Part 6 rep_1/rep_2 = same physical setup; s1 = same string type,
# different session; treated as one 'string' group to be conservative)
STRING = {"A1": "A T1", "A2": "A T1", "B1": "B T1", "C1": "C T1", "D1": "D T1", "A_T2": "A T2",
          "B_T2s1": "B T2", "B_T2r1": "B T2", "B_T2r2": "B T2", "C_T2": "C T2", "D_T2": "D T2"}
# POMA 2025 Table 1 (Lampis et al., POMA 58, 035013): T (N), Z (kg/s), EI (1e-4 N m^2), d (mm)
PROP = {"A T1": (145.31, 1.0592, 3.032, 0.947), "B T1": (150.87, 1.0998, 4.397, 0.927),
        "C T1": (144.40, 1.0525, 3.142, 1.077), "D T1": (142.67, 1.0398, 5.246, 1.123),
        "A T2": (163.03, 1.1884, 3.969, 1.001), "B T2": (163.23, 1.1899, 7.295, 0.973),
        "C T2": (153.04, 1.1154, 3.969, 1.083), "D T2": (164.82, 1.2021, 5.766, 1.201)}
ZPUB = {t: PROP[GROUP[t]][1] for t in TAGS}
# A1+A2 P-free upper law without Z (t 0.2; = upper_law_t0.2 / frozen_parameters upper_empirical_reference)
BASE_A = (-0.1764277369756792, 0.6606815556587429, -1.1107153335353168)
D.STEMS.setdefault("A_T2", "2024-06-19_TypeA_T2_sample1")
D.STEMS["B_T2s1"] = "2024-06-13_TypeB_T2_sample1"
D.STEMS["C_T2"] = "2024-06-11_TypeC_T2_sample1"
D.STEMS["D_T2"] = "2024-06-17_TypeD_T2_sample1"


def load(tag):
    p = D.CACHE / f"{tag}.npz"
    if p.exists():
        return dict(np.load(p))
    mine = HERE / "cache" / f"{tag}.npz"
    if not mine.exists():
        mine.parent.mkdir(exist_ok=True)
        np.savez_compressed(mine, **D.build(tag))
    return dict(np.load(mine))


def rows_for(tag, t=T):
    rs = C.intervals(load(tag), t)
    for r in rs:
        r["tag"] = tag
    return [r for r in rs if np.isfinite(r["upper"][0]) or np.isfinite(r["upper"][1])]


def base_A(lv, lb):
    return BASE_A[0] + BASE_A[1] * np.asarray(lv) + BASE_A[2] * np.asarray(lb)


def all_rows(t=T):
    return {tag: rows_for(tag, t) for tag in TAGS}
