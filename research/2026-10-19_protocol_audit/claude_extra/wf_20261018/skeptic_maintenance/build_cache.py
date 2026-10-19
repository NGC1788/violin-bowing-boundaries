"""Parse A1/A2 window features once (all window lengths), classify every window with the repo classifier
(A2 thresholds, reference f0 98.08 Hz, exactly as verify_astra_boundaries.py) and store compact arrays.
Only A1 and A2 are read (B1/C1/Part 6 are held out and never opened).
Output: cache_<tag>.npz in this folder."""
import csv
import gzip
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(Path.home() / "violin-rne/claude_extra"))
import verify_astra_boundaries as V  # noqa: E402

LABELS = ["helmholtz", "multiple_slip", "subharmonic", "ambiguous", "aperiodic", "no_oscillation"]
RAMP = {"A1": Path.home() / "violin-rne/claude_extra/ramp/ramp_meta_2024-03-25_TypeA_sample1.csv",
        "A2": Path.home() / "violin-rne/claude_extra/ramp_meta_A2.csv"}


def main():
    for tag, stem in V.SPECIMENS.items():
        cols = {k: [] for k in ("folder", "trial", "beta", "win", "t", "plateau", "force", "speed", "amp", "per", "f0", "fpp", "lab")}
        folders = []
        with gzip.open(V.FEAT / f"{stem}.csv.gz", "rt") as fh:
            for r in csv.DictReader(fh):
                if r["error"]:
                    continue
                if r["folder"] not in folders:
                    folders.append(r["folder"])
                fpp = r["flybacks_per_period"]
                fppv = int(float(fpp)) if fpp not in ("", "nan", "None") else None
                lab = V.rm.classify(float(r["amplitude_std"]), float(r["periodicity"]), float(r["f0_hz"]), fppv, 98.08, V.APPLIED)
                cols["folder"].append(folders.index(r["folder"]))
                cols["trial"].append(int(r["trial"]))
                cols["beta"].append(float(r["beta"]))
                cols["win"].append(float(r["window_s"]))
                cols["t"].append(float(r["t"]))
                cols["plateau"].append(float(r["plateau_s"]))
                cols["force"].append(float(r["force"]))
                cols["speed"].append(float(r["speed"]))
                cols["amp"].append(float(r["amplitude_std"]))
                cols["per"].append(float(r["periodicity"]))
                cols["f0"].append(float(r["f0_hz"]))
                cols["fpp"].append(-1 if fppv is None else fppv)
                cols["lab"].append(LABELS.index(lab))
        meta = {}
        for r in csv.DictReader(open(RAMP[tag])):
            if not r["error"]:
                meta[(r["folder"], int(r["trial"]))] = (float(r["accel"]), float(r["force_at_onset"]), float(r["ramp_s"]),
                                                       float(r["v_max"]), float(r["plateau_on_s"]), float(r["onset_s"]))
        out = {k: np.array(v) for k, v in cols.items()}
        out["folders"] = np.array(folders)
        keys = sorted(meta)
        out["meta_folder"] = np.array([folders.index(k[0]) if k[0] in folders else -1 for k in keys])
        out["meta_trial"] = np.array([k[1] for k in keys])
        out["meta"] = np.array([meta[k] for k in keys])
        out["labels"] = np.array(LABELS)
        np.savez_compressed(HERE / f"cache_{tag}.npz", **out)
        print(tag, folders, len(out["t"]), "rows", len(keys), "meta")


if __name__ == "__main__":
    main()
