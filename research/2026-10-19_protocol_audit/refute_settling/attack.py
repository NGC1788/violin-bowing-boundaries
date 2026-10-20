"""POST HOC attack on the settling claim (confirm_settling). Claude, 2026-10-20. All data already seen; nothing here is a test.
Usage: python attack.py  (reads cache.pkl from load.py; writes results.json)"""
import copy, json, pickle, sys
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np
from scipy.stats import binomtest, spearmanr, mannwhitneyu

H = Path.home() / "violin-rne"; HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(H / "confirm_settling")); import score_settling as S  # noqa: E402  (frozen diff/level code, unchanged)
sys.path.insert(0, str(H / "claude_extra")); import verify_astra_boundaries as V  # noqa: E402
SV = S.SV; APPLIED = V.APPLIED; G = "A T1"
D = pickle.load(open(HERE / "cache.pkl", "rb"))
MAINK = next(k for k in D if k.startswith("MAIN"))
SESS = {"COLD": D["COLD"], "WARM": D["WARM"], "MAIN": D[MAINK]}
OUT = {}


def lab(x, ref=98.08, thr=None):
    F, a, p, f0, fpp, v, b = x
    return V.rm.classify(a, p, f0, None if fpp < 0 else fpp, ref, thr or APPLIED)


def columns(sess):
    d = SESS[sess]; items = sorted((x["beta"], tr) for tr, x in d.items()); assert len(items) == 2000
    cols = [[tr for _, tr in items[c * 50:(c + 1) * 50]] for c in range(40)]
    lbm = np.log([b for b, _ in items]).reshape(40, 50); assert np.all(lbm[1:, 0] > lbm[:-1, -1])
    return cols


COLS = {s: columns(s) for s in SESS}


def nominal(sess, tr):
    rank = (tr - 1) % 50 + 1
    if sess == "MAIN": rank = 51 - rank
    return 0.1 * 40.0 ** ((rank - 1) / 49)


def force(sess, tr, t, axis):
    x = SESS[sess][tr]
    return x["t"][t][0] if axis == "measured" else x["onset"] if axis == "onset" else nominal(sess, tr)


def rows(sess, t=0.2, K=2, axis="measured", ref=98.08, thr=None, floor=None, labels=None):
    out = []
    for c, trs in enumerate(COLS[sess]):
        g = sorted((force(sess, tr, t, axis), (labels[(sess, tr)] if labels else lab(SESS[sess][tr]["t"][t], ref, thr)) == "helmholtz", tr) for tr in trs)
        if floor is not None: g = [x for x in g if x[0] >= floor[c]]
        f = np.log(np.maximum([x[0] for x in g], 1e-3)); h = [x[1] for x in g]
        runs = [i for i in range(len(h) - K + 1) if all(h[i:i + K])]
        if runs:
            s, e = runs[0], runs[-1] + K - 1
            lower = (-np.inf, f[0]) if s == 0 else (f[s - 1], f[s]); upper = (f[-1], np.inf) if e == len(h) - 1 else (f[e], f[e + 1])
        else:
            lower, upper = (f[-1], np.inf), (np.nan, np.nan)
        out.append(dict(lv=np.log(np.median([SESS[sess][tr]["t"][t][5] for tr in trs])), lb=float(np.median(np.log([SESS[sess][tr]["beta"] for tr in trs]))),
                        lower=lower, upper=upper, col=c))
    return out


def diff_paired(rA, rB, which, nb=2000, seed=53):
    """level(B)-level(A), bootstrap pairs columns by INDEX (all 40), not by rounded log beta."""
    lA = SV.level(rA, which, G)[0]; lB = SV.level(rB, which, G)[0]; rng = np.random.default_rng(seed); bs = []
    for _ in range(nb):
        kk = rng.choice(40, 40); bs.append(SV.level([rB[k] for k in kk], which, G)[0] - SV.level([rA[k] for k in kk], which, G)[0])
    ci = np.percentile(bs, [2.5, 97.5]).tolist(); d = lB - lA; h = S.H[which]
    dec = "matters" if (ci[0] > 0 or ci[1] < 0) and abs(d) >= h else ("negligible" if -h < ci[0] and ci[1] < h else "not decided")
    return dict(A=lA, B=lB, diff=d, ci=ci, decision=dec)


def fmt(v): return f"{v['diff']:+.3f} [{v['ci'][0]:+.3f},{v['ci'][1]:+.3f}] {v['decision']}"


def nfin(rs, w="lower"): return int(sum(np.all(np.isfinite(r[w])) for r in rs))


def ncens(rs): return int(sum(r["lower"][0] == -np.inf for r in rs))


# ---------------- 0. reproduce + coding check (rounded-lb key pairing) ----------------
base = {s: rows(s) for s in SESS}
o = S.diff(base["COLD"], base["WARM"], "lower"); OUT["repro_frozen_diff"] = o
print(f"[0] reproduce frozen: lower {o['diff']:+.3f} CI {o['ci']} n_keys {o['n_keys']} (expected +0.256 [+0.003,+0.542], 29)")
kc = {round(r["lb"], 2) for r in base["COLD"]}; kw = {round(r["lb"], 2) for r in base["WARM"]}
print(f"    rounded-lb keys: cold {len(kc)} warm {len(kw)} shared {len(kc & kw)} -> bootstrap resamples only shared keys; point uses all 40")
# mean |lb_warm - lb_cold| per column
dlb = [b["lb"] - a["lb"] for a, b in zip(base["COLD"], base["WARM"])]
print(f"    per-column log-beta offset warm-cold: median {np.median(dlb):+.4f}, range [{min(dlb):+.4f},{max(dlb):+.4f}]")
OUT["coding"] = dict(keys_cold=len(kc), keys_warm=len(kw), shared=len(kc & kw), dlb_median=float(np.median(dlb)))
for w in ("lower", "upper"):
    p = diff_paired(base["COLD"], base["WARM"], w); OUT[f"paired_index_{w}"] = p
    print(f"    index-paired bootstrap ({w}): {fmt(p)}")
pm = diff_paired(base["WARM"], base["MAIN"], "lower"); OUT["paired_index_main_minus_warm_lower"] = pm
print(f"    index-paired main-warm lower: {fmt(pm)}")
print(f"    finite lower cols: cold {nfin(base['COLD'])} warm {nfin(base['WARM'])} main {nfin(base['MAIN'])}; left-censored: cold {ncens(base['COLD'])} warm {ncens(base['WARM'])} main {ncens(base['MAIN'])}")

# ---------------- 1. classifier ----------------
print("[1] classifier: windows near the lower boundary (t=0.2)")
desc = {}
for s in SESS:
    near, below, H_ = [], [], []
    for r, trs in zip(base[s], COLS[s]):
        g = sorted(trs, key=lambda tr: SESS[s][tr]["t"][0.2][0])
        F = [SESS[s][tr]["t"][0.2][0] for tr in g]
        if np.all(np.isfinite(r["lower"])):
            j = int(np.searchsorted(np.log(np.maximum(F, 1e-3)), r["lower"][1]))
            near += [SESS[s][tr]["t"][0.2] for tr in g[max(0, j - 3):j + 3]]
            below += [SESS[s][tr]["t"][0.2] for tr in g[max(0, j - 3):j]]
        H_ += [SESS[s][tr]["t"][0.2] for tr in g if lab(SESS[s][tr]["t"][0.2]) == "helmholtz"]
    q = lambda arr, i: np.round(np.percentile([x[i] for x in arr], [25, 50, 75]), 3).tolist()
    desc[s] = dict(n_near=len(near), amp=q(near, 1), per=q(near, 2), f0=q(near, 3), fpp=dict(Counter(x[4] for x in near)),
                   below_labels=dict(Counter(lab(x) for x in below)), helm_f0_median=float(np.median([x[3] for x in H_])),
                   helm_amp_median=float(np.median([x[1] for x in H_])), helm_per_median=float(np.median([x[2] for x in H_])),
                   amp_below_minamp_but_per_ok=int(sum(x[1] < APPLIED["min_amplitude"] and x[2] >= 0.8 for x in near)),
                   per_in_075_080=int(sum(0.75 <= x[2] < 0.8 for x in below)), f0_below_subharm=int(sum(x[3] < 0.75 * 98.08 for x in near)))
    print(f"    {s}: near n={len(near)} amp {desc[s]['amp']} per {desc[s]['per']} f0 {desc[s]['f0']} fpp {desc[s]['fpp']}\n"
          f"        3 strokes below boundary labels {desc[s]['below_labels']} | helmholtz f0 median {desc[s]['helm_f0_median']:.2f} amp {desc[s]['helm_amp_median']:.3f} per {desc[s]['helm_per_median']:.3f}"
          f" | near: amp<min_amp&per>=0.8 {desc[s]['amp_below_minamp_but_per_ok']}, below with per in [0.75,0.8) {desc[s]['per_in_075_080']}, f0<0.75*ref {desc[s]['f0_below_subharm']}")
OUT["classifier_desc"] = desc
# common force band, label-free distributions
print("    common force band 0.30-0.60 N (t=0.2), all windows:")
for s in SESS:
    xs = [x["t"][0.2] for x in SESS[s].values() if 0.3 <= x["t"][0.2][0] <= 0.6]
    print(f"      {s}: n {len(xs)} helmholtz frac {np.mean([lab(x) == 'helmholtz' for x in xs]):.3f} labels {dict(Counter(lab(x) for x in xs))} amp med {np.median([x[1] for x in xs]):.3f} per med {np.median([x[2] for x in xs]):.3f} f0 med {np.median([x[3] for x in xs]):.2f}")
    OUT.setdefault("band_030_060", {})[s] = dict(n=len(xs), helm=float(np.mean([lab(x) == 'helmholtz' for x in xs])), labels=dict(Counter(lab(x) for x in xs)))
# label flips by threshold variants
sens = {}
ref_own = {s: desc[s]["helm_f0_median"] for s in SESS}
amp_pool = np.median([desc[s]["helm_amp_median"] for s in SESS])
variants = {"f0_ref_own_session": dict(ref=ref_own),
            "min_amp_scaled": dict(thr={s: {**APPLIED, "min_amplitude": APPLIED["min_amplitude"] * desc[s]["helm_amp_median"] / amp_pool} for s in SESS}),
            "min_amp_x2": dict(thr={s: {**APPLIED, "min_amplitude": APPLIED["min_amplitude"] * 2} for s in SESS}),
            "min_per_0.75": dict(thr={s: {**APPLIED, "min_periodicity": 0.75} for s in SESS}),
            "min_per_0.85": dict(thr={s: {**APPLIED, "min_periodicity": 0.85} for s in SESS})}
for name, kw in variants.items():
    rr = {s: rows(s, ref=kw.get("ref", {}).get(s, 98.08), thr=kw.get("thr", {}).get(s)) for s in ("COLD", "WARM")}
    flips = {s: int(sum(lab(x["t"][0.2], kw.get("ref", {}).get(s, 98.08), kw.get("thr", {}).get(s)) != lab(x["t"][0.2]) for x in SESS[s].values())) for s in ("COLD", "WARM")}
    p = diff_paired(rr["COLD"], rr["WARM"], "lower", nb=1000); sens[name] = dict(flips=flips, **p, fin=(nfin(rr["COLD"]), nfin(rr["WARM"])))
    print(f"    sens {name:20s}: label flips {flips} | lower warm-cold {fmt(p)} | finite {sens[name]['fin']}")
OUT["classifier_sensitivity"] = sens

# ---------------- 2. force axis, 3. K and t ----------------
print("[2/3] axis, K, t (index-paired bootstrap B=1000; frozen-style rounded-lb bootstrap in brackets)")
grid = {}
for axis in ("measured", "onset", "nominal"):
    for K in (2, 3):
        for t in (0.1, 0.2, 0.3):
            rc, rw = rows("COLD", t, K, axis), rows("WARM", t, K, axis)
            p = diff_paired(rc, rw, "lower", nb=1000); u = diff_paired(rc, rw, "upper", nb=500)
            grid[f"{axis}_K{K}_t{t}"] = dict(lower=p, upper=u, fin=(nfin(rc), nfin(rw)), cens=(ncens(rc), ncens(rw)))
            print(f"    {axis:8s} K{K} t{t}: lower {fmt(p)} | upper {fmt(u)} | finite lower {nfin(rc)}/{nfin(rw)} left-cens {ncens(rc)}/{ncens(rw)}")
OUT["grid"] = grid
# floor per column on each axis
for axis in ("measured", "onset"):
    for s in SESS:
        fl = [min(force(s, tr, 0.2, axis) for tr in trs) for trs in COLS[s]]
        print(f"    floor ({axis}) {s}: median {np.median(fl):.3f} N")
# floor-matched on onset axis
fl = [min(force("COLD", tr, 0.2, "onset") for tr in trs) for trs in COLS["COLD"]]
p = diff_paired(rows("COLD", axis="onset"), rows("WARM", axis="onset", floor=fl), "lower", nb=1000); OUT["onset_floor_matched"] = p
print(f"    onset axis, floor-matched: {fmt(p)}")

# ---------------- 4. column-wise sign test ----------------
print("[4] column-wise lower shifts (midpoints of finite intervals)")
cw = {}
for key in ("measured_K2_t0.2", "onset_K2_t0.2", "nominal_K2_t0.2", "measured_K3_t0.2", "measured_K2_t0.1", "measured_K2_t0.3"):
    axis, K, t = key.split("_"); rc, rw = rows("COLD", float(t[1:]), int(K[1:]), axis), rows("WARM", float(t[1:]), int(K[1:]), axis)
    d = [np.mean(b["lower"]) - np.mean(a["lower"]) for a, b in zip(rc, rw) if np.all(np.isfinite(a["lower"])) and np.all(np.isfinite(b["lower"]))]
    pos = sum(x > 0 for x in d); neg = sum(x < 0 for x in d); bt = binomtest(pos, pos + neg) if pos + neg else None
    # censoring-aware: cold left-censored (boundary < floor_c) and warm finite with warm hi <= cold hi -> compatible both ways; count warm finite ABOVE cold's censoring point
    cens_up = sum(1 for a, b in zip(rc, rw) if a["lower"][0] == -np.inf and np.all(np.isfinite(b["lower"])) and b["lower"][0] >= a["lower"][1])
    cens_dn = sum(1 for a, b in zip(rc, rw) if a["lower"][0] == -np.inf and np.all(np.isfinite(b["lower"])) and b["lower"][1] <= a["lower"][1])
    cw[key] = dict(n=len(d), pos=pos, neg=neg, median=float(np.median(d)) if d else None, p=bt.pvalue if bt else None, cold_cens_warm_above=cens_up, cold_cens_warm_below=cens_dn)
    print(f"    {key:18s}: n {len(d)} up {pos} down {neg} median {np.median(d) if d else float('nan'):+.3f} sign p {bt.pvalue if bt else float('nan'):.3f} | cold-censored cols: warm finite above cold floor {cens_up}, below {cens_dn}")
OUT["columnwise"] = cw

# ---------------- 5. drift / stroke order ----------------
print("[5] within-session drift (column index = time order; strokes ascend in force within column)")
mu_of = lambda rs: SV.lower_mu(rs)
dr = {}
for s in SESS:
    rs = base[s]; mu = mu_of(rs); order = [r["col"] for r in rs] if s != "MAIN" else [39 - r["col"] for r in rs]  # MAIN scanned beta-descending
    res = [(o_, np.mean(r["lower"]) - m) for o_, r, m in zip(order, rs, mu) if np.all(np.isfinite(r["lower"]))]
    rho = spearmanr([a for a, _ in res], [b for _, b in res]) if len(res) > 3 else None
    flo = [(o_, min(SESS[s][tr]["t"][0.2][0] for tr in trs)) for o_, trs in zip(order, COLS[s])]
    rf = spearmanr([a for a, _ in flo], [b for _, b in flo])
    half = {}
    for hname, sel in (("first", lambda o_: o_ < 20), ("second", lambda o_: o_ >= 20)):
        half[hname] = SV.level([r for o_, r in zip(order, rs) if sel(o_)], "lower", G)[0]
    dr[s] = dict(rho_resid=rho.statistic if rho else None, p_resid=rho.pvalue if rho else None, rho_floor=rf.statistic, p_floor=rf.pvalue,
                 floor_first10=float(np.median([b for a, b in flo if a < 10])), floor_last10=float(np.median([b for a, b in flo if a >= 30])), level_first_half=half["first"], level_second_half=half["second"])
    print(f"    {s}: resid-vs-time rho {dr[s]['rho_resid']:+.2f} (p {dr[s]['p_resid']:.3f}, n {len(res)}) | floor force first10/last10 cols {dr[s]['floor_first10']:.3f}/{dr[s]['floor_last10']:.3f} N (rho {rf.statistic:+.2f}) | level 1st/2nd half (time) {half['first']:+.3f}/{half['second']:+.3f}")
OUT["drift"] = dr
# time-matched halves: cold late half vs warm early half and same halves
for a_, b_ in (("first", "first"), ("second", "second"), ("second", "first")):
    sa = [r for r in base["COLD"] if (r["col"] < 20) == (a_ == "first")]; sb = [r for r in base["WARM"] if (r["col"] < 20) == (b_ == "first")]
    print(f"    warm[{b_} half] - cold[{a_} half] lower level: {SV.level(sb, 'lower', G)[0] - SV.level(sa, 'lower', G)[0]:+.3f} (different beta columns when halves differ)")
# helmholtz fraction at lowest 3 strokes vs column (cold)
json.dump(OUT, open(HERE / "results.json", "w"), indent=1, default=float)
print("written results.json")
