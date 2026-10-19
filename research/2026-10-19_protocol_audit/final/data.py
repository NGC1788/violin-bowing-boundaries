"""Shared loader for the final analysis: labelled 0.2 s windows of every processed specimen, cached as npz.
(Claude, 2026-10-18)"""
import csv
import gzip
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path.home() / "violin-rne/claude_extra"))
import verify_astra_boundaries as V  # noqa: E402  (classifier + thresholds)

FEAT = Path.home() / "violin-rne/timeres/features"
CACHE = Path(__file__).parent / "cache"
STEMS = {"A1": "2024-03-25_TypeA_sample1", "A2": "2024-04-19_TypeA_sample2", "B1": "2024-04-10_TypeB_sample1",
         "C1": "2024-04-12_TypeC_sample1", "D1": "2024-04-15_TypeD_sample1",
         "B_T2r1": ("2024-10-23_variability_setup", "rep_1"), "B_T2r2": ("2024-10-23_variability_setup", "rep_2")}
TIMES = (0.0, 0.1, 0.2, 0.3)
TRAIN = ("A1", "A2")
TEST = ("B1", "C1", "D1", "B_T2r1", "B_T2r2")


def build(tag):
    stem = STEMS[tag]
    rep = None
    if isinstance(stem, tuple):
        stem, rep = stem
    rows = []
    beta = {}
    zrows = []
    with gzip.open(FEAT / f"{stem}.csv.gz", "rt") as fh:
        for r in csv.DictReader(fh):
            if r["error"] or r["window_s"] != "0.2":
                continue
            if rep and f"/{rep}/" not in r["folder"] + "/":
                continue
            t = round(float(r["t"]), 2)
            fpp = r["flybacks_per_period"]
            lab = V.rm.classify(float(r["amplitude_std"]), float(r["periodicity"]), float(r["f0_hz"]),
                                int(float(fpp)) if fpp not in ("", "nan", "None") else None, 98.08, V.APPLIED)
            h = lab == "helmholtz"
            k = (r["folder"], int(r["trial"]))
            beta[k] = float(r["beta"])
            v = float(r["speed"])
            if h and t >= 0.1 and r["flyback_height"] not in ("", "nan"):
                zrows.append((float(r["flyback_height"]) * beta[k] / (2 * v), float(r["f0_hz"])))
            if t in TIMES:
                rows.append((k, t, float(r["force"]), v, h))
    col = V.columns(beta)
    folders = sorted({k[0] for k, *_ in rows})
    speed_of = {}
    for k, t, f, v, h in rows:
        speed_of.setdefault(k[0], []).append(v)
    speed_of = {f: float(np.median(v)) for f, v in speed_of.items()}
    arr = dict(folder=np.array([folders.index(k[0]) for k, *_ in rows]), trial=np.array([k[1] for k, *_ in rows]),
               beta=np.array([beta[k] for k, *_ in rows]), column=np.array([col[k] for k, *_ in rows]),
               speed=np.array([speed_of[k[0]] for k, *_ in rows]), t=np.array([r[1] for r in rows]),
               force=np.array([r[2] for r in rows]), H=np.array([r[4] for r in rows]),
               folders=np.array(folders))
    z = np.array(zrows)
    arr["Z"] = np.array([np.median(z[:, 0]), np.percentile(z[:, 0], 25), np.percentile(z[:, 0], 75), len(z)])
    arr["f0"] = np.array([np.median(z[:, 1])])
    return arr


def load(tag):
    CACHE.mkdir(exist_ok=True)
    path = CACHE / f"{tag}.npz"
    if not path.exists():
        np.savez_compressed(path, **build(tag))
    return dict(np.load(path))


if __name__ == "__main__":
    for tag in STEMS:
        d = load(tag)
        print(f"{tag}: windows {len(d['t'])}, folders {list(d['folders'])}, speeds {sorted(set(np.round(d['speed'], 3)))}, "
              f"Z {d['Z'][0]:.3f} [{d['Z'][1]:.3f}, {d['Z'][2]:.3f}] (n {int(d['Z'][3])}), f0 {d['f0'][0]:.2f}, H share {d['H'].mean():.2f}")
