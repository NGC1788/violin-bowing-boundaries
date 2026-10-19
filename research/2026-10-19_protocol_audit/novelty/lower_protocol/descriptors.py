"""Session (speed-folder) protocol descriptors and lower/upper boundary levels for the 9 diagrams.
(Claude, 2026-10-19; offline, local files only; output descriptors.json)

Levels: offset of each diagram's (or folder's) boundaries from the A1+A2 P-free law without Z, t = 0.2 s, K = 2
(same as final/tension_pairs.py; interval-censored offset fit, final/confirm_A_T2.offset_fit).

Descriptors per folder (= one recording session at one speed):
  from ramp_meta_*.csv (7 diagrams; B1 and Part 6 have none):
    cal_low   median F_onset / F_nominal over force ranks 1-20  (F_nominal = 0.1*40^((rank-1)/49) N)
    cal_high  same over ranks 41-50
    g, d0     robust line F_onset = g*F_nom + d0 over all ranks (Theil-Sen)  (gain, additive offset of the elevation map)
    acc_b     median over beta columns of log10 attack acceleration of the two strokes bracketing the t=0.2 s lower boundary
  from blocks_*.npz (7 diagrams):
    dec10     median force change (N) from onset to 10 cm bow travel, ranks 1-20
    floor     median post-stop reading (N), ranks 1-5   (zero-offset proxy: true force ~0 only if contact was lost)
    floor_all 10th percentile of the post-stop reading over ranks 1-10
  from the window features (all 9 diagrams):
    cal_pl    median F(plateau start, 0.1 s window) / F_nominal, ranks 1-20
    dec5_pl   median F(plateau travel >= 5 cm) - F(plateau start), ranks 1-20 (N)
    cens      share of beta columns whose t = 0.2 s lower boundary is left-censored
    fmin      median over columns of the lowest measured window force at t = 0.2 s (N)
"""
import csv
import gzip
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import theilslopes

sys.argv = sys.argv[:1]  # compare.py reads sys.argv[1] at import
sys.path.insert(0, str(Path.home() / "violin-rne/final"))
import confirm_A_T2 as K  # noqa: E402  (also registers A_T2)
import data as D  # noqa: E402

D.STEMS["B_T2s1"] = "2024-06-13_TypeB_T2_sample1"
D.STEMS["C_T2"] = "2024-06-11_TypeC_T2_sample1"
TAGS = ["A1", "A2", "B1", "C1", "D1", "A_T2", "B_T2s1", "B_T2r1", "B_T2r2", "C_T2"]
STEM = {t: (D.STEMS[t] if isinstance(D.STEMS[t], str) else D.STEMS[t][0]) for t in TAGS}
AC = Path.home() / "violin-rne/astra_20261018/archive_companion"
CE = Path.home() / "violin-rne/claude_extra"
RAMP = {"A1": CE / "ramp/ramp_meta_2024-03-25_TypeA_sample1.csv", "A2": CE / "ramp_meta_A2.csv"}
BLOCKS = {"A1": CE / "blocks/blocks_A1.npz", "A2": CE / "blocks/blocks_A2.npz"}
for t in ("C1", "D1", "A_T2", "B_T2s1", "C_T2"):
    RAMP[t] = AC / STEM[t] / f"ramp_meta_{STEM[t]}.csv"
    BLOCKS[t] = AC / STEM[t] / f"blocks_{STEM[t]}.npz"
REVERSED = {"A1"}
T = 0.2


def speed_of(folder):
    return re.search(r"v[123]$", str(folder)).group(0)


def rank_of(tag, trial):
    r = (np.asarray(trial) - 1) % 50 + 1
    return 51 - r if tag in REVERSED else r


def fnom(rank):
    return 0.1 * 40.0 ** ((np.asarray(rank) - 1) / 49)


def med(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    return float(np.median(x)) if x.size else float("nan")


def rep_ok(tag, folder):
    s = D.STEMS[tag]
    return True if isinstance(s, str) else f"/{s[1]}/" in folder + "/"


# ---------------------------------------------------------------- levels
def rows_by_folder(tag):
    d = D.load(tag)
    rows = C_intervals(d)
    for r in rows:
        r["tag"] = tag
        r["fold"] = speed_of(d["folders"][d["folder"][r["idx"][0]]])
    return d, rows


def C_intervals(d):
    return K.C.intervals(d, T)


def boundary_strokes(d, r):
    """(folder speed token, trial) of the strokes bracketing the lower boundary of row r (K = 2)."""
    idx = r["idx"]
    h = d["H"][idx]
    runs = [i for i in range(len(h) - 1) if h[i] and h[i + 1]]
    if runs:
        s = runs[0]
        pick = [idx[s]] if s == 0 else [idx[s - 1], idx[s]]
    else:
        pick = [idx[-1]]
    return [(speed_of(d["folders"][d["folder"][i]]), int(d["trial"][i]), float(d["force"][i])) for i in pick]


def main():
    out = {"tags": TAGS, "diagram": {}, "folder": {}}
    tr_rows = K.rows_for("A1") + K.rows_for("A2")
    noz = K.C.physics_models(tr_rows, {t: 1.0 for t in TAGS})
    base_l = lambda r: noz["P-free"][0](r, "A1")[0]
    base_u = lambda r: noz["P-free"][0](r, "A1")[1]
    rng = np.random.default_rng(0)
    data, rows = {}, {}
    for tag in TAGS:
        data[tag], rows[tag] = rows_by_folder(tag)
    for tag in TAGS:
        rs = rows[tag]
        lv = K.offset_fit(rs, base_l, "lower")
        uv = K.offset_fit(rs, base_u, "upper")
        boots = [K.offset_fit([rs[i] for i in rng.integers(0, len(rs), len(rs))], base_l, "lower") for _ in range(200)]
        nleft = sum(not np.isfinite(r["lower"][0]) for r in rs)
        # high-force-only lower level (columns whose lower interval lies above 0.2 N), as in confirm_A_T2_robust
        hi_rows = [r for r in rs if np.isfinite(r["lower"][0]) and r["lower"][0] >= np.log(0.2)]
        lv_hi = K.offset_fit(hi_rows, base_l, "lower") if len(hi_rows) >= 10 else float("nan")
        out["diagram"][tag] = dict(lower=lv, lower_ci=[float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))],
                                   upper=uv, cens=nleft / len(rs), n_cols=len(rs), lower_hi02=lv_hi, n_hi02=len(hi_rows),
                                   fmin=float(np.exp(np.median([r["fmin"] for r in rs]))))
        for fo in ("v1", "v2", "v3"):
            fr = [r for r in rs if r["fold"] == fo]
            out["folder"][f"{tag}|{fo}"] = dict(tag=tag, fold=fo, lower=K.offset_fit(fr, base_l, "lower"),
                                                upper=K.offset_fit(fr, base_u, "upper"),
                                                cens=sum(not np.isfinite(r["lower"][0]) for r in fr) / len(fr),
                                                fmin=float(np.exp(np.median([r["fmin"] for r in fr]))))
        print(f"{tag:7s} lower {lv:+.3f} [{out['diagram'][tag]['lower_ci'][0]:+.3f},{out['diagram'][tag]['lower_ci'][1]:+.3f}] "
              f"upper {uv:+.3f} cens {nleft}/{len(rs)} lower(>=0.2N cols, n {len(hi_rows)}) {lv_hi:+.3f}", flush=True)

    # ------------------------------------------------------------ ramp_meta descriptors
    for tag, fn in RAMP.items():
        rm = defaultdict(dict)
        for r in csv.DictReader(open(fn)):
            if r["error"]:
                continue
            f = lambda k: float(r[k]) if r[k] not in ("", "nan") else np.nan
            rm[speed_of(r["folder"])][int(r["trial"])] = (f("force_at_onset"), f("accel"))
        d = data[tag]
        bstrokes = defaultdict(list)
        for r in rows[tag]:
            bstrokes[r["fold"]].append(boundary_strokes(d, r))
        for fo in ("v1", "v2", "v3"):
            trials = np.array(sorted(rm[fo]))
            fon = np.array([rm[fo][t][0] for t in trials]); acc = np.array([rm[fo][t][1] for t in trials])
            rk = rank_of(tag, trials); fn_ = fnom(rk)
            ok = np.isfinite(fon)
            g, d0, *_ = theilslopes(fon[ok], fn_[ok])
            accb = [med([np.log10(rm[fo][t][1]) for _, t, _f in col]) for col in bstrokes[fo]]
            # descent ratio at the boundary: log(F_onset / F_window(t=0.2)) of the bracketing strokes
            descb = [med([np.log(rm[fo][t][0] / max(fw, 1e-3)) for _, t, fw in col]) for col in bstrokes[fo]]
            rec = out["folder"][f"{tag}|{fo}"]
            rec.update(cal_low=med(fon[rk <= 20] / fn_[rk <= 20]), cal_high=med(fon[rk >= 41] / fn_[rk >= 41]),
                       g=float(g), d0=float(d0), acc_b=med(accb), desc_b=med(descb), acc_floor=float(np.log10(np.nanmin(acc))))
    # ------------------------------------------------------------ block descriptors
    for tag, fn in BLOCKS.items():
        Z = np.load(fn, allow_pickle=False)
        Dd = Z["data"].astype(float); fields = [str(x) for x in Z["fields"]]
        F = {n: Dd[:, i, :] for i, n in enumerate(fields)}
        fold = np.array([speed_of(x) for x in Z["folder"]]); trial = Z["trial"].astype(int)
        rk = rank_of(tag, trial)
        sp = np.nan_to_num(F["speed"]); vmax = sp.max(axis=1)
        moving = sp > 0.02 * vmax[:, None]
        onset = np.argmax(moving, axis=1); nb = sp.shape[1]
        last = nb - 1 - np.argmax(moving[:, ::-1], axis=1)
        fdc = F["force_dc"]; nvalid = np.sum(np.isfinite(fdc), axis=1)
        fon = np.array([np.nanmedian(fdc[i, max(onset[i] - 3, 0):max(onset[i], 1)]) for i in range(len(trial))])
        valid = np.isfinite(fon) & (onset >= 2)
        dec10 = np.full(len(trial), np.nan); fend = np.full(len(trial), np.nan)
        for i in np.flatnonzero(valid):
            x = F["position"][i] - (F["position"][i, onset[i] - 1] if onset[i] > 0 else 0.0)
            j = np.flatnonzero((x >= 0.10) & moving[i])
            if j.size:
                dec10[i] = fdc[i, j[0]] - fon[i]
            post = fdc[i, last[i] + 1:nvalid[i]]
            if np.isfinite(post).sum():
                fend[i] = np.nanmedian(post)
        for fo in ("v1", "v2", "v3"):
            m = fold == fo
            rec = out["folder"][f"{tag}|{fo}"]
            rec.update(dec10=med(dec10[m & (rk <= 20)]), floor=med(fend[m & (rk <= 5)]),
                       floor_all=float(np.nanpercentile(fend[m & (rk <= 10)], 10)),
                       floor_drift=med(fend[m & (rk <= 5) & (trial <= 500)]) - med(fend[m & (rk <= 5) & (trial > 1500)]),
                       bdc_low=med(np.nanmedian(F["bridge_dc"][m & (rk <= 10)], axis=1)))
    # ------------------------------------------------------------ feature-based proxies (all 9 diagrams)
    for stem in sorted(set(STEM.values())):
        tags = [t for t in TAGS if STEM[t] == stem]
        f0, f5 = defaultdict(dict), defaultdict(dict)
        with gzip.open(D.FEAT / f"{stem}.csv.gz", "rt") as fh:
            for r in csv.DictReader(fh):
                if r["error"] or r["window_s"] != "0.1":
                    continue
                key = (r["folder"], int(r["trial"]))
                t = float(r["t"]); v = float(r["speed"])
                if t == 0.0:
                    f0[key[0]][key[1]] = float(r["force"])
                elif v * t >= 0.05 and key[1] not in f5[key[0]]:
                    f5[key[0]][key[1]] = float(r["force"])
        for tag in tags:
            for folder in f0:
                if not rep_ok(tag, folder):
                    continue
                fo = speed_of(folder)
                trials = np.array(sorted(f0[folder]))
                rk = rank_of(tag, trials)
                a = np.array([f0[folder][t] for t in trials]); b = np.array([f5[folder].get(t, np.nan) for t in trials])
                lo = rk <= 20
                rec = out["folder"][f"{tag}|{fo}"]
                # order check: plateau-start force should rise with rank inside a column
                rho = np.corrcoef(rk[a > 0], np.log(a[a > 0]))[0, 1]
                rec.update(cal_pl=med(a[lo] / fnom(rk[lo])), dec5_pl=med(b[lo] - a[lo]), rank_check=float(rho))
    # ------------------------------------------------------------ diagram aggregates
    keys = ["cal_low", "cal_high", "g", "d0", "acc_b", "desc_b", "dec10", "floor", "floor_all", "floor_drift", "bdc_low", "cal_pl", "dec5_pl"]
    for tag in TAGS:
        recs = [out["folder"][f"{tag}|{fo}"] for fo in ("v1", "v2", "v3")]
        for k in keys:
            vals = [r.get(k, np.nan) for r in recs]
            if k in ("cal_low", "cal_high", "cal_pl", "g"):
                out["diagram"][tag][k] = float(np.exp(np.nanmean(np.log(vals)))) if np.all(np.isfinite(vals)) else float("nan")
            else:
                out["diagram"][tag][k] = float(np.nanmean(vals)) if np.all(np.isfinite(vals)) else float("nan")
    with open(Path(__file__).parent / "descriptors.json", "w") as fh:
        json.dump(out, fh, indent=1, default=float)
    hdr = ["lower", "upper", "cens", "fmin", "cal_low", "cal_high", "g", "d0", "acc_b", "desc_b", "dec10", "floor", "floor_all", "cal_pl", "dec5_pl"]
    print("\nper folder")
    print("folder        " + " ".join(f"{h:>8s}" for h in hdr) + "  rank_chk")
    for k, rec in out["folder"].items():
        print(f"{k:13s} " + " ".join(f"{rec.get(h, np.nan):+8.3f}" for h in hdr) + f"  {rec.get('rank_check', np.nan):+.2f}")
    print("\nper diagram")
    for tag in TAGS:
        rec = out["diagram"][tag]
        print(f"{tag:13s} " + " ".join(f"{rec.get(h, np.nan):+8.3f}" for h in hdr))


if __name__ == "__main__":
    main()
