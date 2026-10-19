"""POST HOC diagnosis of the high-force (HF) frozen failure (Claude, 2026-10-19). Plan: README.md (written first).
Q1 analysis2b 'b2' quadratic surface (fixed before HF data) -> HF local exponents and level.
Q2 sample-1 (beta <= 0.075) + HF common free shape + session offset delta; bootstrap over beta-column clusters.
Q3 measured force at the t = 0.2 s window relative to the pre-motion (onset) force, by force band; boundary strokes.
Writes results.json in this folder only. Natural log. Never opens A4, D2, B_T2 sample 2, coldwarm or A3."""
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

HOME = Path.home()
HERE = Path(__file__).resolve().parent
RNE = HOME / "violin-rne"
sys.path.insert(0, str(RNE / "novelty/thermal_upper"))
sys.path.insert(0, str(RNE / "highforce_confirm"))
import rows as R  # noqa: E402
import thermal_law as TL  # noqa: E402
import score_highforce as S  # noqa: E402

FORBIDDEN = ("2024-04-24_TypeA_sample4", "2024-04-26_TypeD_sample2", "2024-07-03_TypeB_T2_sample2", "coldwarm",
             "2024-04-22_TypeA_sample3")
DEV = R.TAGS
assert len(DEV) == 11 and not any(f in str([R.D.STEMS.get(t, "") for t in DEV]) for f in FORBIDDEN)
PAIR = {"HF_C": "C_T2", "HF_A": "A_T2", "HF_D": "D_T2", "HF_B": "B_T2s1"}
S1_STEM = {"C_T2": "2024-06-11_TypeC_T2_sample1", "A_T2": "2024-06-19_TypeA_T2_sample1",
           "D_T2": "2024-06-17_TypeD_T2_sample1", "B_T2s1": "2024-06-13_TypeB_T2_sample1"}
SCORE = json.loads((RNE / "highforce_confirm/score_result.json").read_text())
B2_SAVED = json.loads((RNE / "novelty/thermal_upper/analysis2b.json").read_text())["beta_curvature_measured"]
T = 0.2


def guard(p):
    assert not any(f in str(p) for f in FORBIDDEN), p
    return p


# ------------------------------------------------------------------ shared fitting (= analysis2b)
def cfit(X, lo, hi, x0):
    f = lambda p: TL.censored_nll(X @ p[:-1], np.exp(p[-1]), lo, hi)
    r = minimize(f, x0, method="L-BFGS-B", options={"maxiter": 5000})
    r2 = minimize(f, r.x, method="Powell", options={"maxiter": 20000, "xtol": 1e-6, "ftol": 1e-9})
    r3 = minimize(f, r2.x, method="L-BFGS-B", options={"maxiter": 5000})
    return r3.x, r3.fun


def ols_start(X, lo, hi):
    y = np.where(np.isfinite(lo) & np.isfinite(hi), 0.5 * (lo + hi), np.where(np.isfinite(lo), lo + 0.2, hi - 0.2))
    c = np.linalg.lstsq(X, y, rcond=None)[0]
    return np.r_[c, np.log(0.3)]


# ------------------------------------------------------------------ HF columns (frozen parser) with trial ids
def hf_data():
    sys.path.insert(0, str(RNE / "claude_extra"))
    import verify_astra_boundaries as V
    data = {}
    for arch in ("HighBowForceRange_1", "HighBowForceRange_2"):
        data.update(S.read_labels(arch, V.rm, V.APPLIED))
    return data


def hf_columns_with_trials(d, t=T):
    """Same grouping/sort as S.columns_from_windows, but keeps trial ids."""
    col = S.assign_columns(d["beta"])
    groups = defaultdict(list)
    for k, w in d["win"].items():
        if t in w:
            groups[(k[0], col[k])].append((w[t][0], w[t][1], d["beta"][k], k[1]))
    out = []
    for (folder, c), g in sorted(groups.items()):
        g.sort()
        f = np.log(np.maximum(np.array([x[0] for x in g]), 1e-3))
        h = np.array([x[1] for x in g], bool)
        out.append(dict(folder=folder, column=c, lv=float(np.log(d["speed"][folder])),
                        lb=float(np.log(np.median([x[2] for x in g]))), f=f, h=h, trials=[x[3] for x in g]))
    return out


def last_band_end(h, k=2):
    runs = [i for i in range(len(h) - k + 1) if all(h[i:i + k])]
    return None if not runs else runs[-1] + k - 1


# ------------------------------------------------------------------ Q1
def q1(RW, hf_rows):
    allr = [r for t in DEV for r in RW[t]]
    lv, lb, lz, lo, hi = TL.arrays(allr, R.ZPUB)
    tags = np.array([r["tag"] for r in allr])
    D = np.column_stack([(tags == t).astype(float) for t in DEV[1:]])
    X2 = np.column_stack([lv, lb, lv * lb, lb ** 2, np.ones_like(lv), D])
    p2, f2 = cfit(X2, lo, hi, ols_start(X2, lo, hi))
    a, b, g, q = p2[:4]
    repro = dict(refit=dict(a=a, b=b, g=g, q=q, nll=f2),
                 saved=dict(a=B2_SAVED["a"], b=B2_SAVED["b"], g=B2_SAVED["g"], q=B2_SAVED["q"], nll=B2_SAVED["nll"]),
                 max_abs_diff=float(max(abs(a - B2_SAVED["a"]), abs(b - B2_SAVED["b"]), abs(g - B2_SAVED["g"]),
                                        abs(q - B2_SAVED["q"]))), n_columns=int(len(lv)), n_diagrams=len(DEV))
    off = {t: p2[4] + (0.0 if i == 0 else p2[4 + i]) for i, t in enumerate(DEV)}
    # reference: linear (no g, q), diagram offsets
    XL = np.column_stack([lv, lb, np.ones_like(lv), D])
    pl, fl = cfit(XL, lo, hi, ols_start(XL, lo, hi))
    offl = {t: pl[2] + (0.0 if i == 0 else pl[2 + i]) for i, t in enumerate(DEV)}
    out = dict(reproduction=repro, linear_reference=dict(a=pl[0], b=pl[1], nll=fl, sigma=float(np.exp(pl[-1]))),
               sigma_b2=float(np.exp(p2[-1])), offsets_b2={t: off[t] for t in PAIR.values()},
               offsets_linear={t: offl[t] for t in PAIR.values()}, targets={})
    for tgt, s1 in PAIR.items():
        r = hf_rows[tgt]
        hv, hb, hlo, hhi = r["lv"], r["lb"], r["lo"], r["hi"]
        surf = a * hv + b * hb + g * hv * hb + q * hb ** 2
        Xd = np.column_stack([np.ones_like(hv), hv, hb])
        c = np.linalg.lstsq(Xd, surf, rcond=None)[0]
        lin = pl[0] * hv + pl[1] * hb
        P1 = SCORE["targets"][tgt]["t0.2"]["P1"]
        ins = lambda x, ci: bool(ci[0] <= x <= ci[1])
        le_b2, s_b2 = TL.offset_fit(surf + off[s1], hlo, hhi)
        le_l, s_l = TL.offset_fit(lin + offl[s1], hlo, hhi)
        out["targets"][tgt] = dict(
            sample1=s1, n_columns=int(len(hv)),
            b2_local=dict(e_v=float(c[1]), e_beta=float(c[2])),
            observed=dict(e_v=P1["e_v"], e_v_ci=P1["e_v_ci"], e_beta=P1["e_beta"], e_beta_ci=P1["e_beta_ci"]),
            b2_e_v_inside_obs_ci=ins(c[1], P1["e_v_ci"]), b2_e_beta_inside_obs_ci=ins(c[2], P1["e_beta_ci"]),
            linear_ref_local=dict(e_v=float(pl[0]), e_beta=float(pl[1])),
            level_error_b2=le_b2, level_sigma_b2=s_b2, level_error_linear_ref=le_l, level_sigma_linear_ref=s_l,
            level_error_S1_strict_frozen=SCORE["targets"][tgt]["t0.2"]["P2"]["S1_strict"]["level_error"],
            b2_within_0p15=bool(abs(le_b2) < 0.15))
    return out


# ------------------------------------------------------------------ Q2
def q2_fit(lv, lb, ses, lo, hi, x0=None, polish=True):
    X = np.column_stack([np.ones_like(lv), lv, lb, ses])
    f = lambda p: TL.censored_nll(X @ p[:4], np.exp(p[4]), lo, hi)
    if x0 is None:
        x0 = ols_start(X, lo, hi)
    r = minimize(f, x0, method="Nelder-Mead", options={"maxiter": 20000, "xatol": 1e-7, "fatol": 1e-9})
    if polish:
        r2 = minimize(f, r.x, method="L-BFGS-B")
        if r2.fun < r.fun:
            r = r2
    return r.x, r.fun


def q2(s1_rows, hf_rows, nb=1000, seed=7):
    out = {}
    rng = np.random.default_rng(seed)
    for tgt, s1 in PAIR.items():
        A = s1_rows[s1]; H = hf_rows[tgt]
        lv = np.r_[A["lv"], H["lv"]]; lb = np.r_[A["lb"], H["lb"]]
        lo = np.r_[A["lo"], H["lo"]]; hi = np.r_[A["hi"], H["hi"]]
        ses = np.r_[np.zeros(len(A["lv"])), np.ones(len(H["lv"]))]
        clus = np.r_[A["col"], 1000 + np.asarray(H["col"])]
        p, nll = q2_fit(lv, lb, ses, lo, hi)
        ids0 = np.unique(clus[ses == 0]); ids1 = np.unique(clus[ses == 1])
        members = {c: np.flatnonzero(clus == c) for c in np.unique(clus)}
        bs = []
        for _ in range(nb):
            pick = np.r_[rng.choice(ids0, len(ids0)), rng.choice(ids1, len(ids1))]
            ii = np.concatenate([members[c] for c in pick])
            q, _ = q2_fit(lv[ii], lb[ii], ses[ii], lo[ii], hi[ii], x0=p, polish=False)
            bs.append(q[:4])
        bs = np.array(bs)
        ci = lambda x: np.percentile(x, [2.5, 97.5]).tolist()
        dci = ci(bs[:, 3])
        if dci[0] <= 0 <= dci[1] and abs(p[3]) < 0.10:
            dec = "no session offset detected (force-range/shape or rule effect more consistent)"
        elif dci[0] > 0:
            dec = "session-level offset (HF higher) consistent with the data"
        elif dci[1] < 0:
            dec = "session-level offset (HF lower)"
        else:
            dec = "not decidable"
        # contradictions: same speed, nearest sample-1 beta column (log)
        below, above, n_below_den, n_above_den, dist = 0, 0, 0, 0, []
        ex_below, ex_above = [], []
        for j in range(len(H["lv"])):
            m = np.flatnonzero(np.isclose(A["lv"], H["lv"][j], atol=0.05))
            if not len(m):
                continue
            k = m[np.argmin(np.abs(A["lb"][m] - H["lb"][j]))]
            dist.append(abs(A["lb"][k] - H["lb"][j]))
            if np.isfinite(A["lo"][k]) and np.isfinite(H["hi"][j]):
                n_below_den += 1
                if H["hi"][j] < A["lo"][k]:
                    below += 1; ex_below.append([float(np.exp(H["lv"][j])), float(np.exp(H["lb"][j])), float(np.exp(H["hi"][j])), float(np.exp(A["lo"][k]))])
            if np.isfinite(A["hi"][k]) and np.isfinite(H["lo"][j]):
                n_above_den += 1
                if H["lo"][j] > A["hi"][k]:
                    above += 1; ex_above.append([float(np.exp(H["lv"][j])), float(np.exp(H["lb"][j])), float(np.exp(H["lo"][j])), float(np.exp(A["hi"][k]))])
        n1 = len(A["lv"])
        out[tgt] = dict(sample1=s1, n_columns_sample1=n1, n_columns_hf=int(len(H["lv"])),
                        n_beta_clusters=dict(sample1=int(len(ids0)), hf=int(len(ids1))),
                        sample1_right_censored=int(np.sum(~np.isfinite(A["hi"]))),
                        sample1_finite=int(np.sum(np.isfinite(A["hi"]) & np.isfinite(A["lo"]))),
                        sample1_left_censored=int(np.sum(~np.isfinite(A["lo"]))),
                        sample1_max_force_N=float(np.exp(np.nanmax(np.where(np.isfinite(A["lo"]), A["lo"], np.nan)))),
                        c=p[0], e_v=p[1], e_b=p[2], delta=p[3], sigma=float(np.exp(p[4])), nll=nll,
                        e_v_ci=ci(bs[:, 1]), e_b_ci=ci(bs[:, 2]), delta_ci=dci, decision=dec,
                        contradictions=dict(hf_upper_entirely_below_s1_lower=below, denom_below=n_below_den,
                                            hf_lower_above_s1_upper=above, denom_above=n_above_den,
                                            max_log_beta_distance=float(max(dist)) if dist else None,
                                            examples_below_v_beta_HFhi_S1lo=ex_below[:6],
                                            examples_above_v_beta_HFlo_S1hi=ex_above[:6]),
                        bootstrap=f"B={nb}, seed {seed}, beta-column clusters resampled within session")
    return out


# ------------------------------------------------------------------ Q3
def read_ramp(path):
    out = {}
    with open(guard(path)) as fh:
        for r in csv.DictReader(fh):
            if r["error"]:
                continue
            out[(r["folder"], int(r["trial"]))] = float(r["force_at_onset"])
    return out


def band_table(fon, fwin, f0, speed, edges):
    rows = []
    r_on = np.log(fwin / fon); r_pl = np.log(fwin / f0)
    for lo_, hi_ in zip(edges[:-1], edges[1:]):
        for v in (None, 0.05, 0.10, 0.20):
            m = (fon >= lo_) & (fon < hi_) & np.isfinite(r_on)
            if v is not None:
                m &= np.isclose(speed, v, atol=0.01)
            if m.sum() < 5:
                continue
            rows.append(dict(band_N=[lo_, hi_], speed=v if v is not None else "all", n=int(m.sum()),
                             r_on_median=float(np.median(r_on[m])), r_on_iqr=np.percentile(r_on[m], [25, 75]).tolist(),
                             r_plat_median=float(np.median(r_pl[m])),
                             dF_onset_to_window_N_median=float(np.median(fwin[m] - fon[m]))))
    return rows


def q3(hf, hf_cols, dev_loaded):
    out = dict(hf={}, sample1={}, boundary={})
    hf_edges = [3.5, 5, 7, 9, 13.5]; lo_edges = [0.0, 0.25, 0.5, 1, 2, 4.5]
    for arch in ("HighBowForceRange_1", "HighBowForceRange_2"):
        ramp = read_ramp(RNE / "astra_20261019/highforce_stream" / arch / "ramp_meta.csv")
        for tgt, (a, prefix, _) in S.TARGETS.items():
            if a != arch:
                continue
            d = hf[tgt]
            keys = [k for k in d["win"] if T in d["win"][k] and 0.0 in d["win"][k] and k in ramp]
            fon = np.array([ramp[k] for k in keys]); fw = np.array([d["win"][k][T][0] for k in keys])
            f0 = np.array([d["win"][k][0.0][0] for k in keys]); sp = np.array([d["speed"][k[0]] for k in keys])
            out["hf"][tgt] = dict(n_strokes=len(keys), n_missing_ramp=int(len(d["win"]) - len(keys)),
                                  bands=band_table(fon, fw, f0, sp, hf_edges))
            # boundary strokes (finite upper intervals, last-band rule)
            rb = []
            for c in hf_cols[tgt]:
                e = last_band_end(c["h"])
                if e is None or e == len(c["h"]) - 1:
                    continue
                for i in (e, e + 1):
                    k = (c["folder"], c["trials"][i])
                    if k in ramp:
                        rb.append(np.log(np.exp(c["f"][i]) / ramp[k]))
            out["boundary"][tgt] = dict(n_strokes=len(rb), r_on_median=float(np.median(rb)),
                                        r_on_iqr=np.percentile(rb, [25, 75]).tolist())
    for s1, stem in S1_STEM.items():
        ramp = read_ramp(RNE / "astra_20261018/archive_companion" / stem / f"ramp_meta_{stem}.csv")
        d = dev_loaded[s1]
        folders = list(d["folders"])
        idx_t = {}
        for i in range(len(d["t"])):
            idx_t[(int(d["folder"][i]), int(d["trial"][i]), round(float(d["t"][i]), 2))] = i
        keys = sorted({(int(d["folder"][i]), int(d["trial"][i])) for i in range(len(d["t"]))})
        fon, fw, f0, sp = [], [], [], []
        miss = 0
        for (fi, tr) in keys:
            k = (folders[fi], tr)
            if k not in ramp or (fi, tr, T) not in idx_t or (fi, tr, 0.0) not in idx_t:
                miss += 1; continue
            fon.append(ramp[k]); fw.append(d["force"][idx_t[(fi, tr, T)]]); f0.append(d["force"][idx_t[(fi, tr, 0.0)]])
            sp.append(d["speed"][idx_t[(fi, tr, T)]])
        fon, fw, f0, sp = map(np.array, (fon, fw, f0, sp))
        out["sample1"][s1] = dict(n_strokes=len(fon), n_missing=miss, onset_force_range_N=[float(fon.min()), float(fon.max())],
                                  bands=band_table(fon, fw, f0, sp, lo_edges))
        # boundary strokes in sample1 (t = 0.2, last-band rule, finite upper, all beta and beta <= 0.075)
        m = np.isclose(d["t"], T)
        g = defaultdict(list)
        for i in np.flatnonzero(m):
            g[(int(d["folder"][i]), int(d["column"][i]))].append(i)
        rb_all, rb_low = [], []
        for (fi, c), idx in g.items():
            idx = sorted(idx, key=lambda i: d["force"][i])
            h = d["H"][idx].astype(bool)
            e = last_band_end(h)
            if e is None or e == len(h) - 1:
                continue
            for i in (idx[e], idx[e + 1]):
                k = (folders[fi], int(d["trial"][i]))
                if k in ramp:
                    val = np.log(max(d["force"][i], 1e-3) / ramp[k])
                    rb_all.append(val)
                    if np.median(d["beta"][idx]) <= 0.075:
                        rb_low.append(val)
        out["sample1"][s1]["boundary"] = dict(n_strokes=len(rb_all), r_on_median=float(np.median(rb_all)),
                                              r_on_iqr=np.percentile(rb_all, [25, 75]).tolist(),
                                              n_strokes_beta_le_0p075=len(rb_low),
                                              r_on_median_beta_le_0p075=float(np.median(rb_low)) if rb_low else None)
    out["delta_r"] = {}
    for tgt, s1 in PAIR.items():
        dr = out["boundary"][tgt]["r_on_median"] - out["sample1"][s1]["boundary"]["r_on_median"]
        lvl = SCORE["targets"][tgt]["t0.2"]["P2"]["S1_strict"]["level_error"]
        dec = ("a force-axis shift of the needed size is possible" if dr >= 0.17 else
               "a force-reading shift of this kind is too small to explain the under-prediction" if abs(dr) < 0.10 or dr < 0 else
               "partial")
        out["delta_r"][tgt] = dict(sample1=s1, delta_r=dr, S1_level_error=lvl, residual_after_onset_axis=lvl - dr, decision=dec)
    return out


# ------------------------------------------------------------------ main
def upper_dict(rows_list, col_key):
    keep = [r for r in rows_list if np.isfinite(r["upper"][0]) or np.isfinite(r["upper"][1])]
    return dict(lv=np.array([r["lv"] for r in keep]), lb=np.array([r["lb"] for r in keep]),
                lo=np.array([r["upper"][0] for r in keep], float), hi=np.array([r["upper"][1] for r in keep], float),
                col=np.array([col_key(r) for r in keep]))


def main():
    res = dict(note="POST HOC. HF labels already seen (scored once 21:41). Natural log. t = 0.2 s, K = 2.",
               excluded_never_opened=list(FORBIDDEN))
    RW = R.all_rows()
    print("dev columns", {t: len(v) for t, v in RW.items()}, flush=True)
    hf = hf_data()
    hf_cols, hf_rows = {}, {}
    for tgt in S.TARGETS:
        cols = hf_columns_with_trials(hf[tgt])
        # identity check against the frozen parser
        ref = S.column_table(S.columns_from_windows(hf[tgt]["win"], hf[tgt]["beta"], hf[tgt]["speed"], T))
        mine = S.column_table(cols)
        assert all(np.allclose(np.nan_to_num(a["upper"], posinf=99), np.nan_to_num(b["upper"], posinf=99)) for a, b in zip(ref, mine))
        hf_cols[tgt] = cols
        hf_rows[tgt] = upper_dict(mine, lambda r: r["column"])
        n_ref = SCORE["targets"][tgt]["t0.2"]["P1"]["n_columns"]
        assert len(hf_rows[tgt]["lv"]) == n_ref, (tgt, len(hf_rows[tgt]["lv"]), n_ref)
    print("HF upper columns", {t: len(v["lv"]) for t, v in hf_rows.items()}, flush=True)
    res["Q1"] = q1(RW, hf_rows)
    print(json.dumps(res["Q1"], indent=1, default=float)[:3000], flush=True)
    dev_loaded = {s1: R.load(s1) for s1 in PAIR.values()}
    s1_rows = {}
    for s1 in PAIR.values():
        d = dev_loaded[s1]
        rs = [r for r in RW[s1] if np.exp(r["lb"]) <= 0.075]
        s1_rows[s1] = upper_dict(rs, lambda r: int(d["column"][r["idx"][0]]))
    res["Q2"] = q2(s1_rows, hf_rows)
    print(json.dumps({k: {kk: v[kk] for kk in ("delta", "delta_ci", "e_v", "e_b", "decision", "contradictions")} for k, v in res["Q2"].items()},
                     indent=1, default=float), flush=True)
    res["Q3"] = q3(hf, hf_cols, dev_loaded)
    print(json.dumps(res["Q3"]["delta_r"], indent=1, default=float), flush=True)
    mb = json.loads((RNE / "highforce_confirm/posthoc_multiband.json").read_text())
    res["Q4_summary_of_posthoc_multiband"] = dict(
        multi_share_dev={k: v["multi_share_of_banded"] for k, v in mb["dev"].items()},
        multi_share_hf={k: v["multi_share_of_banded"] for k, v in mb["hf"].items()},
        S1_level_error_by_rule={k: {r: x["S1_level_error"] for r, x in v.items()} for k, v in mb["level_rule_sensitivity"].items()},
        e_beta_by_rule={k: {r: x["e_beta"] for r, x in v.items()} for k, v in mb["level_rule_sensitivity"].items()})
    (HERE / "results.json").write_text(json.dumps(res, indent=1, default=float))
    print("wrote", HERE / "results.json")


if __name__ == "__main__":
    main()
