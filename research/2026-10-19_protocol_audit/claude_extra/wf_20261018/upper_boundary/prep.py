"""Cache A1/A2 window features + labels as compact npz (upper-boundary track, Claude 2026-10-18).

Labels: repo regime_map.classify with the A2 thresholds and reference f0 98.08 Hz, exactly as in
claude_extra/verify_astra_boundaries.py. Only A1/A2 are read (B1/C1/Part 6 are held out).
Usage: python prep.py
"""
import csv
import gzip
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path.home() / "violin-bowing-boundaries"
sys.path.insert(0, str(REPO / "scripts"))
import regime_map as rm  # noqa: E402

SUMMARY = next((REPO / "reports/research_20261013/lockbox_A2_preprocess/2024-04-19_TypeA_sample2/regime_map").glob("*/summary.json"))
APPLIED = json.loads(SUMMARY.read_text())["thresholds_applied"]
FEAT = Path.home() / "violin-rne/timeres/features"
OUT = Path(__file__).resolve().parent / "cache"
SPECIMENS = {"A1": "2024-03-25_TypeA_sample1", "A2": "2024-04-19_TypeA_sample2"}
LABELS = ["helmholtz", "multiple_slip", "aperiodic", "ambiguous", "subharmonic", "no_oscillation", "error"]


def fnum(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return np.nan


def main():
    OUT.mkdir(exist_ok=True)
    for tag, stem in SPECIMENS.items():
        cols = {k: [] for k in ("folder", "trial", "beta", "window_s", "t", "plateau_s", "force", "speed", "amp",
                                "per", "f0", "fpp", "sfr", "fh", "label")}
        folders = []
        seen = set()
        dup = 0
        with gzip.open(FEAT / f"{stem}.csv.gz", "rt") as fh:
            for r in csv.DictReader(fh):
                key = (r["folder"], r["trial"], r["window_s"], r["t"])
                if key in seen:
                    dup += 1
                    continue
                seen.add(key)
                if r["folder"] not in folders:
                    folders.append(r["folder"])
                if r["error"]:
                    lab = "error"
                else:
                    fpp = r["flybacks_per_period"]
                    fpp_i = int(float(fpp)) if fpp not in ("", "nan", "None") else None
                    try:
                        lab = rm.classify(float(r["amplitude_std"]), float(r["periodicity"]), float(r["f0_hz"]),
                                          fpp_i, 98.08, APPLIED)
                    except TypeError:
                        lab = "ambiguous"
                cols["folder"].append(folders.index(r["folder"]))
                cols["trial"].append(int(r["trial"]))
                for k, src in (("beta", "beta"), ("window_s", "window_s"), ("t", "t"), ("plateau_s", "plateau_s"),
                               ("force", "force"), ("speed", "speed"), ("amp", "amplitude_std"), ("per", "periodicity"),
                               ("f0", "f0_hz"), ("fpp", "flybacks_per_period"), ("sfr", "second_flyback_ratio"),
                               ("fh", "flyback_height")):
                    cols[k].append(fnum(r[src]))
                cols["label"].append(LABELS.index(lab))
        arr = {k: np.asarray(v) for k, v in cols.items()}
        arr["folder"] = arr["folder"].astype(np.int8)
        arr["trial"] = arr["trial"].astype(np.int16)
        arr["label"] = arr["label"].astype(np.int8)
        for k in ("beta", "window_s", "t", "plateau_s", "force", "speed", "amp", "per", "f0", "fpp", "sfr", "fh"):
            arr[k] = arr[k].astype(np.float32 if k not in ("beta", "force") else np.float64)
        arr["t"] = np.round(arr["t"], 2)
        arr["window_s"] = np.round(arr["window_s"], 2)
        np.savez_compressed(OUT / f"windows_{tag}.npz", folders=np.array(folders), labels=np.array(LABELS), **arr)
        n = len(arr["trial"])
        print(f"{tag}: {n} windows, duplicates dropped {dup}, folders {folders}")
        for w in (0.1, 0.2, 0.4):
            m = arr["window_s"] == w
            print(f"  window {w}: {m.sum()} rows, label counts " +
                  ", ".join(f"{LABELS[i]} {np.sum(arr['label'][m] == i)}" for i in range(len(LABELS))))


if __name__ == "__main__":
    main()
