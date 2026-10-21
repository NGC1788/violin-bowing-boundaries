"""Label-threshold sensitivity of the measured minimum-force law (PLAN.md). Rebuilds t = 0.2 s Helmholtz labels from the
feature files with one classifier threshold changed at a time, re-extracts the minimum-force boundary (K = 2) and refits
the per-session law on the window-force and onset-force axes. Claude, 2026-10-21."""
import csv, gzip, json, math, sys
from pathlib import Path

import numpy as np

H = Path.home(); HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(H / "violin-rne/novelty/lower_law"))
import onset_axis as OA  # noqa: E402
LL, SV = OA.LL, OA.SV
C, D = SV.C, SV.D
V = D.V
OA.RAMP["B1"] = H / "violin-rne/astra_20261018/archive_companion/2024-04-10_TypeB_sample1/ramp_meta_2024-04-10_TypeB_sample1.csv"

STEM = {"A1": "2024-03-25_TypeA_sample1", "A2": "2024-04-19_TypeA_sample2", "B1": "2024-04-10_TypeB_sample1",
        "C1": "2024-04-12_TypeC_sample1", "D1": "2024-04-15_TypeD_sample1", "A3": "2024-04-22_TypeA_sample3",
        "A4": "2024-04-24_TypeA_sample4", "D2": "2024-04-26_TypeD_sample2", "A_T2": "2024-06-19_TypeA_T2_sample1",
        "B_T2s1": "2024-06-13_TypeB_T2_sample1", "C_T2": "2024-06-11_TypeC_T2_sample1", "D_T2": "2024-06-17_TypeD_T2_sample1",
        "B_T2r1": ("2024-10-23_variability_setup", "rep_1"), "B_T2r2": ("2024-10-23_variability_setup", "rep_2")}
ONSET = ["A1", "A2", "B1", "C1", "D1", "A3", "A4", "D2", "A_T2", "B_T2s1", "C_T2"]
AP = V.APPLIED; REF_F0 = 98.08
VARIANTS = {"base": (0.8, 0.5, 0)}
VARIANTS.update({f"per{p}": (p, 0.5, 0) for p in (0.7, 0.75, 0.85, 0.9)})
VARIANTS.update({f"fly{f}": (0.8, f, 0) for f in (0.3, 0.4, 0.6, 0.7)})
VARIANTS.update({"amp1x": (0.8, 0.5, 1), "amp2x": (0.8, 0.5, 2)})


def fnum(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return math.nan


def read(stem, rep):
    """t = 0.2 s windows (0.2 s long) + beta of every stroke, as in final/data.py."""
    rows, beta = [], {}
    with gzip.open(D.FEAT / f"{stem}.csv.gz", "rt") as fh:
        for r in csv.DictReader(fh):
            if r["error"] or r["window_s"] != "0.2":
                continue
            if rep and f"/{rep}/" not in r["folder"] + "/":
                continue
            k = (r["folder"], int(r["trial"]))
            beta[k] = float(r["beta"])
            if round(float(r["t"]), 2) == 0.2:
                rows.append((k, float(r["force"]), float(r["speed"]), fnum(r["amplitude_std"]), fnum(r["periodicity"]),
                             fnum(r["f0_hz"]), fnum(r["flybacks_per_period"]), fnum(r["second_flyback_ratio"])))
    return rows, beta


def helm(a, p, f0, fpp, sfr, var):
    pmin, ffrac, amp = var
    if not (a >= AP["min_amplitude"]) and not (p >= pmin):
        return False
    if not math.isfinite(p) or p < AP["max_aperiodic"] or p < pmin:
        return False
    if math.isfinite(f0) and f0 < AP["subharmonic_ratio"] * REF_F0:
        return False
    if amp and not (a >= amp * AP["min_amplitude"]):
        return False
    return bool(math.isfinite(fpp) and fpp >= 1 and math.isfinite(sfr) and sfr < ffrac)


def build(raw, var):
    rows, beta = raw
    col = V.columns(beta)
    folders = sorted({k[0] for k, *_ in rows})
    speed_of = {}
    for k, f, v, *_ in rows:
        speed_of.setdefault(k[0], []).append(v)
    speed_of = {f: float(np.median(v)) for f, v in speed_of.items()}
    Hs = np.array([helm(a, p, f0, fpp, sfr, var) for (k, f, v, a, p, f0, fpp, sfr) in rows])
    return dict(folder=np.array([folders.index(k[0]) for k, *_ in rows]), trial=np.array([k[1] for k, *_ in rows]),
                beta=np.array([beta[k] for k, *_ in rows]), column=np.array([col[k] for k, *_ in rows]),
                speed=np.array([speed_of[k[0]] for k, *_ in rows]), t=np.full(len(rows), 0.2),
                force=np.array([r[1] for r in rows]), H=Hs, folders=np.array(folders))


def to_onset(d, sess):
    rm = {}
    for r in csv.DictReader(open(OA.RAMP[sess])):
        if not r["error"] and r["force_at_onset"] not in ("", "nan"):
            rm[(r["folder"], int(r["trial"]))] = float(r["force_at_onset"])
    f = np.array([rm.get((d["folders"][fo], int(tr)), np.nan) for fo, tr in zip(d["folder"], d["trial"])])
    keep = np.isfinite(f); assert keep.mean() > 0.99, (sess, keep.mean())
    out = {k: (v[keep] if k not in ("folders",) else v) for k, v in d.items()}
    out["force"] = f[keep]
    return out


def fit(d):
    rows = [r for r in C.intervals(d, 0.2) if np.isfinite(r["lower"][0]) or np.isfinite(r["lower"][1])]
    return LL.fit(rows)


def tci(x):
    x = np.asarray(x, float); m = float(x.mean()); s = float(x.std(ddof=1)); n = len(x)
    from scipy.stats import t as tdist
    h = tdist.ppf(0.975, n - 1) * s / math.sqrt(n)
    return m, s, [m - h, m + h]


def main():
    stored_w = {k: v["measured"]["b_beta"] for k, v in json.loads((H / "violin-rne/novelty/lower_law/results.json").read_text()).items()}
    stored_o = {k: v["b_beta"] for k, v in json.loads((H / "violin-rne/novelty/lower_law/onset_axis.json").read_text())["onset"].items()}
    stored_o["B1"] = json.loads((H / "violin-rne/novelty/lower_law/onset_with_B1.json").read_text())["B1"]["b_beta"]
    res = {"variants": {k: list(v) for k, v in VARIANTS.items()}, "sessions": {}}
    for sess, st in STEM.items():
        stem, rep = st if isinstance(st, tuple) else (st, None)
        raw = read(stem, rep)
        base_H = None; out = {}
        for name, var in VARIANTS.items():
            d = build(raw, var)
            if base_H is None:
                base_H = d["H"]
            bw = fit(d)
            e = dict(window_b_beta=bw[1], window_b_v=bw[0], H_share=float(d["H"].mean()), changed=float(np.mean(d["H"] != base_H)))
            if sess in ONSET:
                bo = fit(to_onset(d, sess)); e.update(onset_b_beta=bo[1], onset_b_v=bo[0])
            out[name] = e
        res["sessions"][sess] = out
        b = out["base"]
        print(f"{sess:7s} base window β {b['window_b_beta']:+.3f} (stored {stored_w.get(sess, float('nan')):+.3f})"
              + (f" onset β {b['onset_b_beta']:+.3f} (stored {stored_o.get(sess, float('nan')):+.3f})" if sess in ONSET else "")
              + " | " + " ".join(f"{n}:{v['window_b_beta']:+.2f}" for n, v in out.items() if n != "base"), flush=True)
    gw = {s: abs(res["sessions"][s]["base"]["window_b_beta"] - stored_w[s]) for s in STEM if s in stored_w}
    go = {s: abs(res["sessions"][s]["base"]["onset_b_beta"] - stored_o[s]) for s in ONSET if s in stored_o}
    res["gate"] = dict(window_max_abs_diff=max(gw.values()), onset_max_abs_diff=max(go.values()),
                       pass_=bool(max(gw.values()) <= 0.02 and max(go.values()) <= 0.02), window=gw, onset=go)
    summ = {}
    for ax, ss in (("window", list(STEM)), ("onset", ONSET)):
        key = f"{ax}_b_beta"; base = np.array([res["sessions"][s]["base"][key] for s in ss])
        rows = {}
        for name in VARIANTS:
            x = np.array([res["sessions"][s][name][key] for s in ss])
            m, sd, ci = tci(x); dm, dsd, dci = tci(x - base) if name != "base" else (0.0, 0.0, [0.0, 0.0])
            toward = int(np.sum(np.abs(x + 2) < np.abs(base + 2) - 1e-9))
            rows[name] = dict(mean=m, sd=sd, ci=ci, shift=dm, shift_ci=dci, toward_minus2=toward,
                              changed_labels=float(np.mean([res["sessions"][s][name]["changed"] for s in ss])))
        means = [v["mean"] for v in rows.values()]
        S = max(means) - min(means)
        summ[ax] = dict(rows=rows, S=S, reading="robust (S<0.1)" if S < 0.1 else ("moderate (0.1<=S<0.3)" if S < 0.3 else "label-dependent (S>=0.3)"), n=len(ss))
    res["summary"] = summ
    (HERE / "results.json").write_text(json.dumps(res, indent=1, default=float))
    print("gate", res["gate"]["window_max_abs_diff"], res["gate"]["onset_max_abs_diff"], res["gate"]["pass_"])
    for ax in ("window", "onset"):
        s = summ[ax]; print(f"== {ax} (n {s['n']}): S = {s['S']:.3f} → {s['reading']}")
        for n, v in s["rows"].items():
            print(f"   {n:7s} mean {v['mean']:+.3f} (SD {v['sd']:.2f}) shift {v['shift']:+.3f} [{v['shift_ci'][0]:+.3f}, {v['shift_ci'][1]:+.3f}] labels changed {100 * v['changed_labels']:.1f}% toward −2: {v['toward_minus2']}")


if __name__ == "__main__":
    main()
