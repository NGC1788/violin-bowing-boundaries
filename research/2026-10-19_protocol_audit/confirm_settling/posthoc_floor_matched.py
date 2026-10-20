"""POST HOC sensitivity (after the settling score): COLD reached only ~0.3-0.5 N at its lowest settings, WARM ~0.1-0.2 N.
Match the force floor: in every beta column drop WARM (and A1main 0.10 m/s) strokes whose window force is below that
column's lowest COLD window force, re-extract lower intervals (K = 2) and recompute the level and the paired difference
with the same bootstrap as the frozen scorer. Claude, 2026-10-20."""
import json, sys
from collections import defaultdict
from pathlib import Path
import numpy as np
H = Path.home(); sys.path.insert(0, str(H / "violin-rne/confirm_settling"))
import score_settling as S  # noqa: E402
SV = S.SV
sys.path.insert(0, str(H / "violin-rne/claude_extra")); import verify_astra_boundaries as V  # noqa: E402
import csv, gzip


def raw_columns(path, keep_folder, v=0.10):
    win, beta, speed = {}, {}, defaultdict(list)
    with gzip.open(path, "rt") as fh:
        for r in csv.DictReader(fh):
            if r["error"] or r["window_s"] != "0.2" or round(float(r["t"]), 2) != 0.2 or not keep_folder(r["folder"]):
                continue
            fpp = r["flybacks_per_period"]
            lab = V.rm.classify(float(r["amplitude_std"]), float(r["periodicity"]), float(r["f0_hz"]), int(float(fpp)) if fpp not in ("", "nan", "None") else None, 98.08, V.APPLIED)
            k = (r["folder"], int(r["trial"])); win[k] = (float(r["force"]), lab == "helmholtz"); beta[k] = float(r["beta"]); speed[r["folder"]].append(float(r["speed"]))
    out = {}
    for folder in speed:
        if abs(np.median(speed[folder]) - v) > 1e-3:
            continue
        items = sorted((beta[k], k) for k in beta if k[0] == folder)
        out[folder] = [[(win[k][0], win[k][1], b) for b, k in items[c * 50:(c + 1) * 50]] for c in range(40)]
    return out


def rows_from(cols, floors=None):
    rows = []
    for c, g in enumerate(cols):
        g = sorted(x for x in g if floors is None or x[0] >= floors[c])
        if len(g) < 3:
            continue
        f = np.log(np.maximum([x[0] for x in g], 1e-3)); h = [x[1] for x in g]
        runs = [i for i in range(len(h) - 1) if h[i] and h[i + 1]]
        if runs:
            s, e = runs[0], runs[-1] + 1
            lower = (-np.inf, f[0]) if s == 0 else (f[s - 1], f[s]); upper = (f[-1], np.inf) if e == len(h) - 1 else (f[e], f[e + 1])
        else:
            lower, upper = (f[-1], np.inf), (np.nan, np.nan)
        rows.append(dict(lv=np.log(0.10), lb=float(np.log(np.median([x[2] for x in g]))), lower=lower, upper=upper))
    return rows


cw = raw_columns(S.STREAM / "features.csv.gz", lambda f: True)
cold = next(v for k, v in cw.items() if k.split("/")[-2].endswith("_cold")); warm = next(v for k, v in cw.items() if k.split("/")[-2].endswith("_warm"))
main = next(iter(raw_columns(H / "violin-rne/timeres/features/2024-03-25_TypeA_sample1.csv.gz", lambda f: True).values()))
main = sorted(main, key=lambda g: np.median([x[2] for x in g]))
floors = [min(x[0] for x in g) for g in cold]
res = {}
for name, (A, B) in {"warm_minus_cold": (rows_from(cold), rows_from(warm, floors)), "main_minus_warm": (rows_from(warm, floors), rows_from(main, floors))}.items():
    res[name] = {w: S.diff(A, B, w) for w in ("lower", "upper")}
res["cold_floor_N_median"] = float(np.median(floors))
res["warm_strokes_dropped"] = int(sum(sum(1 for x in g if x[0] < floors[c]) for c, g in enumerate(warm)))
Path("posthoc_floor_matched.json").write_text(json.dumps(res, indent=1, default=float))
for name in ("warm_minus_cold", "main_minus_warm"):
    for w, v in res[name].items():
        print(name, w, f"A {v['level_A']:+.3f} B {v['level_B']:+.3f} diff {v['diff']:+.3f} CI [{v['ci'][0]:+.3f}, {v['ci'][1]:+.3f}] ({v['decision']} by the frozen thresholds; POST HOC)")
print("cold floor median", res["cold_floor_N_median"], "warm strokes dropped", res["warm_strokes_dropped"])
