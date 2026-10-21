"""Independent recomputation (verifier, 2026-10-21). Does NOT import label_sweep.py.
Relabel t = 0.2 s windows from timeres/features, K = 2 lower boundary per (folder, beta column), censored fit
log F_min = a + b_v log v + b_beta log beta (final/compare.censored_fit). Also: flyback-rule equivalence on raw rows,
amp1x removal by beta tercile, column-type counts, onset-axis mapping checks."""
import csv, gzip, json, math, sys
from collections import defaultdict, Counter
from pathlib import Path
import numpy as np

H = Path.home(); R = H / "violin-rne"; OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(R / "final")); sys.path.insert(0, str(R / "claude_extra"))
sys.argv = sys.argv[:1]
import compare as C  # noqa: E402
import verify_astra_boundaries as V  # noqa: E402

AP = V.APPLIED; MINAMP = AP["min_amplitude"]; REF = 98.08
SESS = {"A1": ("2024-03-25_TypeA_sample1", None), "C1": ("2024-04-12_TypeC_sample1", None),
        "A_T2": ("2024-06-19_TypeA_T2_sample1", None), "B_T2r1": ("2024-10-23_variability_setup", "rep_1")}
AC = R / "astra_20261018/archive_companion"
RAMP = {"A1": R / "claude_extra/ramp/ramp_meta_2024-03-25_TypeA_sample1.csv",
        "C1": AC / "2024-04-12_TypeC_sample1/ramp_meta_2024-04-12_TypeC_sample1.csv",
        "A_T2": AC / "2024-06-19_TypeA_T2_sample1/ramp_meta_2024-06-19_TypeA_T2_sample1.csv"}


def num(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return math.nan


def orig_classify_H(a, p, f0, fpp_s):
    """regime_map.classify exactly as final/data.py calls it (flybacks = int(float(fpp)))."""
    fl = int(float(fpp_s)) if fpp_s not in ("", "nan", "None") else None
    return V.rm.classify(a, p, f0, fl, REF, AP) == "helmholtz"


def myH(a, p, f0, fpp, sfr, pmin=0.8, f=0.5, ampk=0.0):
    if not (a >= MINAMP) and not (p >= pmin):
        return False
    if not math.isfinite(p) or p < AP["max_aperiodic"] or p < pmin:
        return False
    if math.isfinite(f0) and f0 < AP["subharmonic_ratio"] * REF:
        return False
    if ampk and not (a >= ampk * MINAMP):
        return False
    return math.isfinite(fpp) and fpp >= 1 and math.isfinite(sfr) and sfr < f


def load(stem, rep):
    W, beta, spd = [], {}, defaultdict(list)
    eq = Counter()  # flyback-rule equivalence on all 0.2 s-long rows, all t
    with gzip.open(R / f"timeres/features/{stem}.csv.gz", "rt") as fh:
        for r in csv.DictReader(fh):
            if r["error"] or r["window_s"] != "0.2":
                continue
            if rep and f"/{rep}/" not in r["folder"] + "/":
                continue
            k = (r["folder"], int(r["trial"])); beta[k] = float(r["beta"]); t = round(float(r["t"]), 2)
            a, p, f0, fpp, sfr = (num(r[c]) for c in ("amplitude_std", "periodicity", "f0_hz", "flybacks_per_period", "second_flyback_ratio"))
            if t in (0.0, 0.1, 0.2, 0.3):
                spd[r["folder"]].append(float(r["speed"]))
            # raw rule equivalence: fpp == 1  vs  fpp >= 1 & sfr < 0.5
            r1 = math.isfinite(fpp) and int(fpp) == 1; r2 = math.isfinite(fpp) and fpp >= 1 and math.isfinite(sfr) and sfr < 0.5
            eq["rows"] += 1; eq["raw_mismatch"] += r1 != r2
            eq["nonint_fpp"] += math.isfinite(fpp) and fpp != int(fpp)
            eq["fpp1_sfr_nan"] += math.isfinite(fpp) and fpp == 1 and not math.isfinite(sfr)
            eq["fpp1_sfr_ge05"] += math.isfinite(fpp) and fpp == 1 and math.isfinite(sfr) and sfr >= 0.5
            eq["fpp2p_sfr_lt05"] += math.isfinite(fpp) and fpp >= 2 and math.isfinite(sfr) and sfr < 0.5
            ho = orig_classify_H(a, p, f0, r["flybacks_per_period"]); hm = myH(a, p, f0, fpp, sfr)
            eq["full_label_mismatch"] += ho != hm
            if t == 0.2:
                eq["t02_rows"] += 1; eq["t02_full_label_mismatch"] += ho != hm
                W.append(dict(k=k, force=float(r["force"]), a=a, p=p, f0=f0, fpp=fpp, sfr=sfr, Horig=ho))
    col = V.columns(beta)
    sp = {f: float(np.median(v)) for f, v in spd.items()}
    for w in W:
        w["col"] = col[w["k"]]; w["beta"] = beta[w["k"]]; w["v"] = sp[w["k"][0]]
    return W, dict(eq)


def intervals(W, Hl, force_key="force"):
    g = defaultdict(list)
    for i, w in enumerate(W):
        if math.isfinite(w[force_key]):
            g[(w["k"][0], w["col"])].append(i)
    rows = []
    for key, idx in sorted(g.items()):
        idx.sort(key=lambda i: W[i][force_key])
        f = np.log(np.maximum([W[i][force_key] for i in idx], 1e-3)); h = [bool(Hl[i]) for i in idx]
        s = next((i for i in range(len(h) - 1) if h[i] and h[i + 1]), None)
        if s is None:
            typ, lo, hi = "none", f[-1], np.inf
        elif s == 0:
            typ, lo, hi = "left", -np.inf, f[0]
        else:
            typ, lo, hi = "int", f[s - 1], f[s]
        rows.append(dict(lv=math.log(W[idx[0]]["v"]), lb=math.log(np.median([W[i]["beta"] for i in idx])), lower=(lo, hi), typ=typ,
                         nH=int(sum(h)), anyH=any(h)))
    return rows


def fit(rows):
    p = C.censored_fit(rows, lambda r: (np.array([1.0, r["lv"], r["lb"]]), 0.0), "lower")
    return float(p[1]), float(p[2])


def coltypes(rows):
    lbs = np.array([r["lb"] for r in rows]); q = np.quantile(lbs, [1 / 3, 2 / 3]); out = {}
    for name, sel in (("all", np.ones(len(rows), bool)), ("bT0", lbs < q[0]), ("bT2", lbs >= q[1])):
        c = Counter(r["typ"] for r, s in zip(rows, sel) if s); out[name] = dict(left=c["left"], int=c["int"], none=c["none"],
                                                                              none_but_someH=sum(1 for r, s in zip(rows, sel) if s and r["typ"] == "none" and r["anyH"]))
    return out


def main():
    stw = json.loads((R / "novelty/lower_law/results.json").read_text())
    sto = json.loads((R / "novelty/lower_law/onset_axis.json").read_text())["onset"]
    RS = json.loads((R / "novelty/label_sensitivity/results.json").read_text())["sessions"]
    PH = json.loads((R / "novelty/label_sensitivity/posthoc_amplitude.json").read_text())
    res = {}
    for s, (stem, rep) in SESS.items():
        W, eq = load(stem, rep)
        a = np.array([w["a"] for w in W]); force = np.array([w["force"] for w in W]); beta = np.array([w["beta"] for w in W])
        base = np.array([myH(w["a"], w["p"], w["f0"], w["fpp"], w["sfr"]) for w in W])
        assert np.array_equal(base, np.array([w["Horig"] for w in W]))
        L = {"base": base,
             "fly0.3": np.array([myH(w["a"], w["p"], w["f0"], w["fpp"], w["sfr"], f=0.3) for w in W]),
             "fly0.7": np.array([myH(w["a"], w["p"], w["f0"], w["fpp"], w["sfr"], f=0.7) for w in W]),
             "amp1x": np.array([myH(w["a"], w["p"], w["f0"], w["fpp"], w["sfr"], ampk=1) for w in W])}
        ck = [(w["k"][0], w["col"]) for w in W]; med = defaultdict(list)
        for kk, h, aa in zip(ck, base, a):
            if h:
                med[kk].append(aa)
        med = {kk: float(np.median(v)) for kk, v in med.items()}
        L["rel0.7"] = np.array([bool(h and aa >= 0.7 * med[kk]) for kk, h, aa in zip(ck, base, a)])
        e = {"eq": eq, "n_windows": len(W), "variants": {}}
        # amp1x removal by beta tercile (as claimed: terciles of log beta over all t=0.2 windows) + per-folder terciles
        lb = np.log(beta); terc = np.digitize(lb, np.quantile(lb, [1 / 3, 2 / 3]))
        pf = np.zeros(len(W), int)
        for fo in {w["k"][0] for w in W}:
            m = np.array([w["k"][0] == fo for w in W]); pf[m] = np.digitize(lb[m], np.quantile(lb[m], [1 / 3, 2 / 3]))
        for vn in ("amp1x", "rel0.7", "fly0.3"):
            lost = base & ~L[vn]
            e[f"{vn}_removed_by_beta_tercile"] = [float(lost[terc == i].sum() / max(1, base[terc == i].sum())) for i in range(3)]
            e[f"{vn}_removed_by_perfolder_tercile"] = [float(lost[pf == i].sum() / max(1, base[pf == i].sum())) for i in range(3)]
        lost = base & ~L["amp1x"]; lf = np.log(force)
        cm = defaultdict(list)
        for i in np.flatnonzero(base):
            cm[ck[i]].append(lf[i])
        cmean = {kk: np.mean(v) for kk, v in cm.items()}
        dm = np.array([lf[i] - cmean[ck[i]] for i in np.flatnonzero(base)])
        e["corr_lost_logF_pooled"] = float(np.corrcoef(lost[base], lf[base])[0, 1])
        e["corr_lost_logF_within_column"] = float(np.corrcoef(lost[base], dm)[0, 1])
        e["corr_logbeta_logF_baseH"] = float(np.corrcoef(lb[base], lf[base])[0, 1])
        v = np.array([w["v"] for w in W]); X = np.c_[np.ones(base.sum()), np.log(v[base]), lb[base], lf[base]]
        e["logamp_on_logv_logbeta_logF"] = np.linalg.lstsq(X, np.log(a[base]), rcond=None)[0][1:].tolist()
        e["amp_share_below_floor_baseH"] = float((a[base] < MINAMP).mean())
        for vn, Hl in L.items():
            rows = intervals(W, Hl); bv, bb = fit(rows)
            d = dict(window_b_beta=bb, window_b_v=bv, H_share=float(Hl.mean()), changed=float(np.mean(Hl != base)), coltypes=coltypes(rows),
                     n_cols=len(rows))
            if vn == "base":  # cross-check own intervals vs compare.intervals
                dd = dict(folder=np.array([sorted({w["k"][0] for w in W}).index(w["k"][0]) for w in W]), column=np.array([w["col"] for w in W]),
                          t=np.full(len(W), 0.2), force=force, H=Hl, speed=v, beta=beta)
                cr = C.intervals(dd, 0.2); d["compare_intervals_b"] = list(fit(cr))
            if s in RAMP:
                rm = {}
                for r in csv.DictReader(open(RAMP[s])):
                    if not r["error"] and r["force_at_onset"] not in ("", "nan"):
                        rm[(r["folder"], int(r["trial"]))] = (float(r["force_at_onset"]), float(r["beta"]))
                for w in W:
                    w["fon"] = rm.get(w["k"], (math.nan,))[0]
                ro = intervals(W, Hl, "fon"); d["onset_b_v"], d["onset_b_beta"] = fit(ro); d["onset_coltypes"] = coltypes(ro)
                if vn == "base":
                    hit = [w["k"] in rm for w in W]
                    db = [abs(rm[w["k"]][1] - w["beta"]) for w in W if w["k"] in rm]
                    fo = np.array([w["fon"] for w in W]); ok = np.isfinite(fo)
                    e["onset_map"] = dict(match_frac=float(np.mean(hit)), max_abs_beta_diff=float(max(db)), n_ramp=len(rm),
                                          median_ratio_onset_over_window=float(np.median(fo[ok] / force[ok])),
                                          corr_log=float(np.corrcoef(np.log(fo[ok]), np.log(force[ok]))[0, 1]))
                    # shuffle test: trial alignment off by one within folder -> beta mismatch?
                    sh = [abs(rm[(w["k"][0], w["k"][1] + 1)][1] - w["beta"]) for w in W if (w["k"][0], w["k"][1] + 1) in rm]
                    e["onset_map"]["median_abs_beta_diff_if_trial+1"] = float(np.median(sh))
            e["variants"][vn] = d
        # comparisons with claimed
        cmp = {}
        for vn in L:
            ref = PH["relative"][s]["rel0.7"] if vn == "rel0.7" else RS[s][vn]
            cmp[vn] = dict(mine_w=e["variants"][vn]["window_b_beta"], claimed_w=ref["window_b_beta"],
                           mine_o=e["variants"][vn].get("onset_b_beta"), claimed_o=ref.get("onset_b_beta"),
                           mine_changed=e["variants"][vn]["changed"], claimed_changed=ref["changed"])
        cmp["stored_window_base"] = stw[s]["measured"]["b_beta"]; cmp["stored_onset_base"] = sto.get(s, {}).get("b_beta")
        cmp["claimed_terciles"] = [PH["mechanism"][s][f"beta_tercile_{i}"] for i in range(3)]
        cmp["claimed_corr"] = PH["mechanism"][s]["corr_lost_vs_logforce_within_H"]
        e["compare"] = cmp; res[s] = e
        print(s, json.dumps({k: e[k] for k in e if k != "variants"}, default=float), flush=True)
        for vn, d in e["variants"].items():
            print(f"  {vn:7s} w {d['window_b_beta']:+.4f} o {d.get('onset_b_beta', float('nan')):+.4f} H {d['H_share']:.4f} chg {d['changed']:.4f} cols {d['coltypes']}"
                  + (f" onsetcols {d['onset_coltypes']['all']}" if 'onset_coltypes' in d else "") + (f" cmpInt {d['compare_intervals_b']}" if 'compare_intervals_b' in d else ""), flush=True)
    (OUT / "recompute.json").write_text(json.dumps(res, indent=1, default=float))


if __name__ == "__main__":
    main()
