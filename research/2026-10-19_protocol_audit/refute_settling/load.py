"""POST HOC refutation attempt of confirm_settling (Claude, 2026-10-20). All data already seen.
Load COLD / WARM (coldwarm stream) and A1 main 0.10 m/s windows (window 0.2 s, t = 0.1/0.2/0.3) + ramp meta into a cache."""
import csv, gzip, pickle
from collections import defaultdict
from pathlib import Path
import numpy as np

H = Path.home() / "violin-rne"
OUT = Path(__file__).resolve().parent / "cache.pkl"
CW = H / "astra_20261019/highforce_stream/2024-03-25_TypeA_sample1_coldwarm"
SRC = {"cw": (CW / "features.csv.gz", CW / "ramp_meta.csv"),
       "main": (H / "timeres/features/2024-03-25_TypeA_sample1.csv.gz", H / "claude_extra/ramp/ramp_meta_2024-03-25_TypeA_sample1.csv")}
TS = (0.1, 0.2, 0.3)


def sess_of(folder):
    leaf = folder.rstrip("/").split("/")
    if leaf[-2].endswith("_cold"): return "COLD"
    if leaf[-2].endswith("_warm"): return "WARM"
    return "MAIN:" + leaf[-1]


data = {}
for src, (feat, ramp) in SRC.items():
    rm = {}
    for r in csv.DictReader(open(ramp)):
        if not r["error"] and abs(float(r["v_max"]) - 0.10) < 1e-3:
            rm[(r["folder"], int(r["trial"]))] = (float(r["force_at_onset"]), float(r["force_plateau"]), float(r["beta"]))
    rec = defaultdict(dict)
    with gzip.open(feat, "rt") as fh:
        for r in csv.DictReader(fh):
            if r["error"] or r["window_s"] != "0.2":
                continue
            t = round(float(r["t"]), 2)
            if t not in TS:
                continue
            k = (r["folder"], int(r["trial"]))
            if k not in rm:
                continue
            fpp = r["flybacks_per_period"]
            rec[k][t] = (float(r["force"]), float(r["amplitude_std"]), float(r["periodicity"]), float(r["f0_hz"]),
                         int(float(fpp)) if fpp not in ("", "nan", "None") else -1, float(r["speed"]), float(r["beta"]))
    for k, v in rec.items():
        s = sess_of(k[0])
        data.setdefault(s, {})[k[1]] = dict(t=v, onset=rm[k][0], plateau=rm[k][1], beta=rm[k][2])
for s, d in data.items():
    print(s, len(d), "all 3 t:", sum(len(x["t"]) == 3 for x in d.values()))
pickle.dump(data, open(OUT, "wb"))
