"""Sustained-motion (stationarity) labels on the measured data (PLAN_stationarity.md). Claude, 2026-10-21."""
import csv, gzip, json, math, sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE))
import label_sweep as LS  # noqa: E402

TS = [round(0.2 + 0.05 * i, 2) for i in range(13)]          # 0.2 ... 0.8
BASE = LS.VARIANTS["base"]


def t_late(speed):
    return 0.3 if speed > 0.15 else 0.8


def read_series(stem, rep):
    ser = {}
    with gzip.open(LS.D.FEAT / f"{stem}.csv.gz", "rt") as fh:
        for r in csv.DictReader(fh):
            if r["error"] or r["window_s"] != "0.2":
                continue
            if rep and f"/{rep}/" not in r["folder"] + "/":
                continue
            t = round(float(r["t"]), 2)
            if t in TS:
                ser.setdefault((r["folder"], int(r["trial"])), {})[t] = (LS.fnum(r["amplitude_std"]), LS.fnum(r["periodicity"]),
                                                                       LS.fnum(r["f0_hz"]), LS.fnum(r["flybacks_per_period"]),
                                                                       LS.fnum(r["second_flyback_ratio"]))
    return ser


def variant_labels(raw, ser, speed_of):
    rows = raw[0]
    out = {"P1": [], "P2": [], "A1": []}
    for (k, f, v, a, p, f0, fpp, sfr) in rows:
        h0 = LS.helm(a, p, f0, fpp, sfr, BASE)
        tl = t_late(speed_of[k[0]]); s = ser.get(k, {})
        if tl not in s:
            for n in out:
                out[n].append(None)
            continue
        hl = LS.helm(*s[tl], BASE)
        out["P1"].append(bool(h0 and hl))
        ts = [t for t in TS if t <= tl + 1e-9]
        out["P2"].append(bool(h0 and all(t in s and LS.helm(*s[t], BASE) for t in ts)))
        out["A1"].append(bool(h0 and math.isfinite(s[tl][0]) and a > 0 and s[tl][0] >= 0.9 * a))
    return out


def subset(d, keep):
    return {k: (v[keep] if k != "folders" else v) for k, v in d.items()}


def main():
    base_res = json.loads((HERE / "results.json").read_text())["sessions"]
    rel = json.loads((HERE / "posthoc_amplitude.json").read_text())["relative"]
    res = {"t_late": {"0.05": 0.8, "0.1": 0.8, "0.2": 0.3}, "sessions": {}}
    for sess, st in LS.STEM.items():
        stem, rep = st if isinstance(st, tuple) else (st, None)
        raw = LS.read(stem, rep); ser = read_series(stem, rep)
        d0 = LS.build(raw, BASE)
        speed_of = {d0["folders"][i]: float(np.median(d0["speed"][d0["folder"] == i])) for i in range(len(d0["folders"]))}
        labs = variant_labels(raw, ser, speed_of)
        lb = np.log(d0["beta"]); terc = np.digitize(lb, np.quantile(lb, [1 / 3, 2 / 3]))
        out = {}
        for name, L in labs.items():
            avail = np.array([x is not None for x in L]); Hv = np.array([bool(x) for x in L])
            d = subset(dict(d0, H=Hv), avail)
            lost = d0["H"] & ~Hv & avail
            e = dict(dropped=int((~avail).sum()), H_share=float(Hv[avail].mean()),
                     removed_by_tercile=[float(lost[terc == i].sum() / max(1, (d0["H"] & avail)[terc == i].sum())) for i in range(3)],
                     window_b_beta=LS.fit(d)[1])
            if sess in LS.ONSET:
                e["onset_b_beta"] = LS.fit(LS.to_onset(d, sess))[1]
            out[name] = e
        res["sessions"][sess] = out
        print(f"{sess:7s} base β {base_res[sess]['base']['window_b_beta']:+.2f} | " + " ".join(
            f"{n}: β {v['window_b_beta']:+.2f} (drop {v['dropped']}, removed by β tercile {v['removed_by_tercile'][0]:.2f}/{v['removed_by_tercile'][1]:.2f}/{v['removed_by_tercile'][2]:.2f})"
            for n, v in out.items()), flush=True)
    summ = {}
    for ax, ss in (("window", list(LS.STEM)), ("onset", LS.ONSET)):
        key = f"{ax}_b_beta"; base = np.array([base_res[s]["base"][key] for s in ss])
        rows = {}
        for name in ("P1", "P2", "A1"):
            x = np.array([res["sessions"][s][name][key] for s in ss]); m, sd, ci = LS.tci(x); dm, _, dci = LS.tci(x - base)
            terc = np.array([res["sessions"][s][name]["removed_by_tercile"] for s in ss]).mean(0)
            flag = bool(abs(terc[2] - terc[0]) > 0.20)
            rows[name] = dict(mean=m, sd=sd, ci=ci, shift=dm, shift_ci=dci, removed_by_tercile_mean=terc.tolist(), beta_confounded=flag,
                              size="small" if abs(dm) < 0.1 else ("moderate" if abs(dm) < 0.3 else "large"), moves_to_schelleng=bool(m <= -1.8))
        prev = {n: base_res and np.mean([base_res[s][n][key] for s in ss]) for n in ("base", "per0.7", "per0.75", "per0.85", "per0.9", "fly0.3", "fly0.4", "fly0.6", "fly0.7")}
        prev.update({r: np.mean([rel[s][r][key] for s in ss]) for r in ("rel0.5", "rel0.7")})
        allm = list(prev.values()) + [v["mean"] for v in rows.values() if not v["beta_confounded"]]
        S = max(allm) - min(allm)
        summ[ax] = dict(rows=rows, S_unflagged=S, reading="robust (S<0.1)" if S < 0.1 else ("moderate (0.1<=S<0.3)" if S < 0.3 else "label-dependent (S>=0.3)"))
    res["summary"] = summ
    (HERE / "results_stationarity.json").write_text(json.dumps(res, indent=1, default=float))
    for ax in ("window", "onset"):
        s = summ[ax]; print(f"== {ax}: S over unflagged variants = {s['S_unflagged']:.3f} → {s['reading']}")
        for n, v in s["rows"].items():
            print(f"   {n}: mean {v['mean']:+.3f} shift {v['shift']:+.3f} [{v['shift_ci'][0]:+.3f}, {v['shift_ci'][1]:+.3f}] ({v['size']}); removed by β tercile {[round(x, 2) for x in v['removed_by_tercile_mean']]} flag {v['beta_confounded']}; to Schelleng {v['moves_to_schelleng']}")


if __name__ == "__main__":
    main()
