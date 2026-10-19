"""robust_lower: post hoc robustness of the minimum-force association (dec5_pl, 10 development diagrams).
Plan in README.md (written before running). Claude subagent, 2026-10-19.
Reads only: final/cache/{A1,A2,B1,C1,D1,A_T2,B_T2s1,B_T2r1,B_T2r2,C_T2}.npz, label-free ramp_meta CSVs of the
7 diagrams with metadata, novelty/lower_protocol/*.json, astra_20261019/protocol_results/results.json.
Run: OPENBLAS_NUM_THREADS=1 ~/violin-bowing-boundaries/.venv-mac/bin/python robust_lower.py
"""
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats

HERE = Path(__file__).parent
ROOT = Path.home() / "violin-rne"
LP = ROOT / "novelty/lower_protocol"
ARGV = sys.argv[1:]
sys.argv = sys.argv[:1]
sys.path.insert(0, str(LP))
import descriptors as DS  # noqa: E402  (registers B_T2s1, C_T2, A_T2 stems; loads nothing at import)
import levels_variants as LV  # noqa: E402
import analysis as AN  # noqa: E402  (reads descriptors.json / levels_variants.json only)

K, D = DS.K, DS.D
TAGS = DS.TAGS
FORBIDDEN = ("TypeA_sample4", "TypeD_sample2", "TypeB_T2_sample2", "coldwarm")
assert len(TAGS) == 10 and not any(f in DS.STEM[t] for t in TAGS for f in FORBIDDEN)
desc = json.load(open(LP / "descriptors.json"))
var = json.load(open(LP / "levels_variants.json"))
an2 = json.load(open(LP / "analysis2.json"))
astra = json.load(open(ROOT / "astra_20261019/protocol_results/results.json"))
T7 = [t for t in TAGS if t in DS.RAMP]
LEVEL = {t: desc["diagram"][t]["lower"] for t in TAGS}
DEC = {t: desc["diagram"][t]["dec5_pl"] for t in TAGS}
SLOPE_REF = 7.67
out = {"meta": dict(post_hoc=True, tags=TAGS, tags_ramp=T7, log="natural", seed_L2=11, n_perm_L2=10000)}


def log(*a):
    print(*a, flush=True)


def r_of(x, y):
    return float(np.corrcoef(np.asarray(x, float), np.asarray(y, float))[0, 1])


def fisher(r, n):
    z, se = np.arctanh(r), 1 / np.sqrt(n - 3)
    return [float(np.tanh(z - 1.96 * se)), float(np.tanh(z + 1.96 * se))]


def ols(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    X = np.column_stack([np.ones(len(x)), x])
    b, *_ = np.linalg.lstsq(X, y, rcond=None)
    res = y - X @ b
    n = len(y)
    s2 = res @ res / (n - 2)
    se = np.sqrt(s2 * np.linalg.inv(X.T @ X)[1, 1])
    tq = stats.t.ppf(0.975, n - 2)
    return dict(a=float(b[0]), b=float(b[1]), b_ci=[float(b[1] - tq * se), float(b[1] + tq * se)], se=float(se))


def lodo_r2(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    press = 0.0
    for i in range(len(y)):
        m = np.arange(len(y)) != i
        f = ols(x[m], y[m])
        press += (y[i] - f["a"] - f["b"] * x[i]) ** 2
    return float(1 - press / np.sum((y - y.mean()) ** 2))


def assoc(tags, lv, dec=DEC):
    x = [dec[t] for t in tags]; y = [lv[t] for t in tags]
    r = r_of(x, y)
    return dict(n=len(tags), r=r, r_ci95=fisher(r, len(tags)), slope=ols(x, y)["b"], lodo_r2=lodo_r2(x, y))


# ----------------------------------------------------------------------------- L1 mechanical null
def ramp_onset(tag):
    rm = {}
    for r in csv.DictReader(open(DS.RAMP[tag])):
        if r["error"] or r["force_at_onset"] in ("", "nan"):
            continue
        rm[(DS.speed_of(r["folder"]), int(r["trial"]))] = float(r["force_at_onset"])
    return rm


def window_force(tag, t):
    d = D.load(tag)
    m = np.isclose(d["t"], t)
    return {(DS.speed_of(d["folders"][d["folder"][i]]), int(d["trial"][i])): float(d["force"][i]) for i in np.flatnonzero(m)}


def q_diag(tag, kind):
    fw = window_force(tag, 0.2)
    ref = ramp_onset(tag) if kind in ("onset_rank", "onset_band") else window_force(tag, 0.0)
    per = defaultdict(list)
    for (fo, tr), f in fw.items():
        f0 = ref.get((fo, tr), np.nan)
        if not np.isfinite(f0):
            continue
        rk = int(DS.rank_of(tag, tr))
        if kind == "onset_band":
            if not (0.15 <= f0 <= 0.45):
                continue
        elif rk > 20:
            continue
        if kind == "plateau0":
            f0 = max(f0, 1e-3)
        if f0 <= 0:
            continue
        per[fo].append(np.log(max(f, 1e-3) / f0))
    med = {fo: float(np.median(per[fo])) for fo in ("v1", "v2", "v3")}
    return float(np.mean(list(med.values()))), med, {fo: len(per[fo]) for fo in per}


def L1():
    res = {}
    for kind, tags in (("onset_rank", T7), ("onset_band", T7), ("plateau0", TAGS)):
        q, detail = {}, {}
        for t in tags:
            q[t], med, n = q_diag(t, kind)
            detail[t] = dict(q=q[t], folder_median=med, n_strokes=n)
        c0 = 0.5 * (q["A1"] + q["A2"])
        m = {t: q[t] - c0 for t in tags}
        L = np.array([LEVEL[t] for t in tags]); M = np.array([m[t] for t in tags])
        sst = np.sum((L - L.mean()) ** 2)
        c = float(np.mean(L - M))
        fit = ols(M, L)
        rec = dict(n=len(tags), tags=tags, mech_pred={t: float(m[t]) for t in tags}, detail=detail,
                   fit_level_on_mech=fit, r=r_of(M, L), lodo_r2=lodo_r2(M, L),
                   R2_slope1_free_c=float(1 - np.sum((L - M - c) ** 2) / sst), c_slope1=c,
                   R2_slope1_c0=float(1 - np.sum((L - M) ** 2) / sst),
                   sd_level=float(np.std(L, ddof=1)), sd_mech=float(np.std(M, ddof=1)),
                   sd_level_minus_mech=float(np.std(L - M, ddof=1)))
        dec = [DEC[t] for t in tags]
        rec["implied_mech_slope_per_N"] = ols(dec, M)
        rec["observed_slope_same_tags"] = ols(dec, L)
        rec["r_dec5_level"] = r_of(dec, L)
        rec["r_dec5_level_minus_mech"] = r_of(dec, L - M)
        rec["r_dec5_mech"] = r_of(dec, M)
        res[kind] = rec
        log(f"L1 {kind:10s} n {len(tags)}: b {fit['b']:+.2f} [{fit['b_ci'][0]:+.2f},{fit['b_ci'][1]:+.2f}] r {rec['r']:+.2f} "
            f"LODO R2 {rec['lodo_r2']:+.2f} | slope-1 R2 {rec['R2_slope1_free_c']:+.2f} (c {c:+.3f}), c=0 {rec['R2_slope1_c0']:+.2f} | "
            f"implied slope {rec['implied_mech_slope_per_N']['b']:+.2f}/N vs observed same tags {rec['observed_slope_same_tags']['b']:+.2f}/N (ref +7.67) | "
            f"r(dec5, L-m) {rec['r_dec5_level_minus_mech']:+.2f}")
        for t in tags:
            log(f"     {t:7s} level {LEVEL[t]:+.3f} mech {m[t]:+.3f} q {q[t]:+.3f} folders " +
                " ".join(f"{fo} {detail[t]['folder_median'][fo]:+.3f}(n{detail[t]['n_strokes'].get(fo, 0)})" for fo in ("v1", "v2", "v3")))
    # analytic additive context: 1/F_ref over the A1 grid
    rows = DS.rows_by_folder("A1")[1]
    _, law = LV.levels({"A1": DS.rows_by_folder("A1")[1], "A2": DS.rows_by_folder("A2")[1]})
    Fr = np.exp([law[0] + law[1] * r["lv"] + law[2] * r["lb"] for r in rows])
    res["analytic_additive"] = dict(mean_inv_Fref=float(np.mean(1 / Fr)), median_Fref=float(np.median(Fr)),
                                    law=law, n_cols=len(rows))
    log(f"L1 analytic: mean 1/F_ref over {len(rows)} A1-grid columns = {np.mean(1 / Fr):.2f} /N (median F_ref {np.median(Fr):.3f} N)")
    return res


# ----------------------------------------------------------------------------- L2 selection-adjusted permutation
def L2():
    preds = ["cal_pl", "dec5_pl", "fmin", "cens", "T", "Z", "EI", "d", "Zto"]
    sess = preds[:4]
    X = np.array([[AN.xval(t, None, p) for t in TAGS] for p in preds])
    resp3 = {k: [AN.yval(t, None, k) for t in TAGS] for k in ("base", "c=0.2", "c=0.3")}
    respA = {f"astra:{k}": [v["levels"][t] for t in TAGS] for k, v in astra["variants"].items()}
    z = lambda A: (A - A.mean(1, keepdims=True)) / A.std(1, keepdims=True)
    Xz = z(X)
    rng = np.random.default_rng(11)
    perms = np.array([rng.permutation(10) for _ in range(10000)])
    obs = abs(r_of(X[1], resp3["base"]))
    res = {"obs_abs_r_dec5_base": obs, "predictors": preds}
    for name, ps, resp in (("primary_9x3", preds, resp3), ("session_4x3", sess, resp3),
                           ("extended_9x15", preds, {**resp3, **respA})):
        idx = [preds.index(p) for p in ps]
        Y = z(np.array(list(resp.values())))
        R = Xz[idx] @ Y.T / 10  # observed
        Smax = np.empty(len(perms))
        for i, pm in enumerate(perms):
            Smax[i] = np.max(np.abs(Xz[idx] @ Y[:, pm].T / 10))
        p_adj = float((1 + np.sum(Smax >= obs - 1e-12)) / (1 + len(perms)))
        i, j = np.unravel_index(np.argmax(np.abs(R)), R.shape)
        res[name] = dict(n_pairs=int(R.size), p_adj=p_adj, obs_max_abs_r=float(np.max(np.abs(R))),
                         obs_max_pair=[ps[i], list(resp)[j]], null_q95=float(np.quantile(Smax, 0.95)),
                         null_median=float(np.median(Smax)), null_hist=np.histogram(Smax, bins=40, range=(0, 1))[0].tolist())
        log(f"L2 {name:14s}: pairs {R.size}, observed max |r| {np.max(np.abs(R)):.3f} ({ps[i]}, {list(resp)[j]}); "
            f"null max|r| median {np.median(Smax):.3f}, 95% {np.quantile(Smax, 0.95):.3f}; p_adj for |r| {obs:.3f} = {p_adj:.4f}")
    xs = Xz[1]; ys = z(np.array([resp3["base"]]))[0]
    single = np.array([abs(xs @ ys[pm] / 10) for pm in perms])
    res["p_unadjusted_dec5_base"] = float((1 + np.sum(single >= obs - 1e-12)) / (1 + len(perms)))
    log(f"L2 unadjusted permutation p (dec5_pl, base) = {res['p_unadjusted_dec5_base']:.4f}")
    return res


# ----------------------------------------------------------------------------- L3 no-band as missing
def L3():
    base_rows = {t: DS.rows_by_folder(t)[1] for t in TAGS}
    nb = {}
    miss_rows = {}
    for t, rs in base_rows.items():
        new = []
        k = 0
        for r in rs:
            r2 = dict(r)
            if np.isnan(r["upper"][0]) and np.isnan(r["upper"][1]):
                r2["lower"] = (np.nan, np.nan)
                k += 1
            new.append(r2)
        nb[t] = k
        miss_rows[t] = new
    lv_prim, law_prim = LV.levels(miss_rows)
    # secondary: original reference law (fitted with no-band right-censored), offsets on rows with no-band missing
    _, law0 = LV.levels({t: base_rows[t] for t in ("A1", "A2")})
    base = lambda r: law0[0] + law0[1] * r["lv"] + law0[2] * r["lb"]
    lv_sec = {t: float(K.offset_fit(miss_rows[t], base, "lower")) for t in TAGS}
    prim = {t: float(lv_prim[t]["all"]) for t in TAGS}
    res = dict(n_noband=nb, levels_primary=prim, levels_secondary_original_law=lv_sec, law_primary=law_prim, law_original=law0,
               assoc_primary=assoc(TAGS, prim), assoc_secondary=assoc(TAGS, lv_sec), assoc_baseline=assoc(TAGS, LEVEL))
    log("L3 no-band columns (of 120): " + " ".join(f"{t} {nb[t]}" for t in TAGS))
    for t in TAGS:
        log(f"     {t:7s} baseline {LEVEL[t]:+.3f}  no-band missing (refit law) {prim[t]:+.3f}  (original law) {lv_sec[t]:+.3f}")
    for k in ("assoc_baseline", "assoc_primary", "assoc_secondary"):
        a = res[k]
        log(f"L3 {k:16s}: r {a['r']:+.3f} [{a['r_ci95'][0]:+.2f},{a['r_ci95'][1]:+.2f}] slope {a['slope']:+.2f}/N LODO R2 {a['lodo_r2']:+.2f}")
    return res


# ----------------------------------------------------------------------------- L4 onset axis
def L4():
    ax = an2["axis"]
    res = {}
    for name, lv, tags in (("onset_7", ax["onset"], T7), ("current_same7", ax["current"], T7),
                           ("plateau0_10", ax["plateau0"], TAGS), ("current_10", ax["current"], TAGS)):
        a = assoc(tags, lv)
        a["sd_level"] = float(np.std([lv[t] for t in tags], ddof=1))
        res[name] = a
        log(f"L4 {name:14s}: n {a['n']} r {a['r']:+.3f} [{a['r_ci95'][0]:+.2f},{a['r_ci95'][1]:+.2f}] slope {a['slope']:+.2f}/N "
            f"LODO R2 {a['lodo_r2']:+.2f} level SD {a['sd_level']:.3f}")
    return res


# ----------------------------------------------------------------------------- L5 compiled table
def L5():
    keys = ["c=none", "c=0.15", "c=0.2", "c=0.25", "c=0.3", "c=0.4", "zero_corrected", "zero_corrected_c=0.2", "zero_corrected_c=0.3"]
    table = {}
    for t in TAGS:
        ncol = desc["diagram"][t]["n_cols"]
        row = dict(n_cols=ncol, cens_share_desc=desc["diagram"][t]["cens"])
        for k in keys:
            lv = var[k]["levels"].get(t)
            row[k] = None if lv is None else dict(level=lv["all"], n_left=lv["n_left"], left_share=lv["n_left"] / ncol)
        table[t] = row
    rs = {}
    for k in keys:
        tags = [t for t in TAGS if table[t][k] is not None]
        rs[k] = assoc(tags, {t: table[t][k]["level"] for t in tags})
    rs["c=none_same7"] = assoc(T7, {t: table[t]["c=none"]["level"] for t in T7})
    hdr = "diag   " + " ".join(f"{k.replace('zero_corrected', 'zc'):>14s}" for k in keys)
    log("L5 compiled levels (left-censored n/120):\n" + hdr)
    for t in TAGS:
        log(f"{t:7s}" + " ".join(f"{'—':>14s}" if table[t][k] is None else
                                 f"{table[t][k]['level']:+7.3f} ({table[t][k]['n_left']:3d})" for k in keys))
    log("r(dec5) " + " ".join(f"{rs[k]['r']:+7.3f} (n{rs[k]['n']:2d})" for k in keys) + f" | c=none on same 7: {rs['c=none_same7']['r']:+.3f}")
    return dict(table=table, assoc=rs)


# ----------------------------------------------------------------------------- L6 influence
def L6():
    res = {"all": assoc(TAGS, LEVEL)}
    for t in TAGS:
        tags = [u for u in TAGS if u != t]
        r = r_of([DEC[u] for u in tags], [LEVEL[u] for u in tags])
        res[f"drop_{t}"] = dict(n=9, r=r, r_ci95=fisher(r, 9))
    for name, drop in (("without_A1", ("A1",)), ("without_A1_A2", ("A1", "A2"))):
        tags = [u for u in TAGS if u not in drop]
        res[name] = assoc(tags, LEVEL)
    groups = defaultdict(list)
    for t in TAGS:
        groups[AN.GROUP[t]].append(t)
    gx = {g: float(np.mean([DEC[t] for t in v])) for g, v in groups.items()}
    gy = {g: float(np.mean([LEVEL[t] for t in v])) for g, v in groups.items()}
    res["condition_level"] = assoc(list(groups), gy, gx)
    res["condition_level"]["groups"] = {g: v for g, v in groups.items()}
    for k, a in res.items():
        log(f"L6 {k:16s}: n {a['n']:2d} r {a['r']:+.3f} [{a['r_ci95'][0]:+.2f},{a['r_ci95'][1]:+.2f}]"
            + (f" LODO R2 {a['lodo_r2']:+.2f} slope {a['slope']:+.2f}/N" if "lodo_r2" in a else ""))
    return res


def figure(o):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    BLUE, ORANGE, INK, MUTED = "#2a78d6", "#eb6834", "#0b0b0b", "#8a8984"
    plt.rcParams.update({"font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK})
    fig, ax = plt.subplots(2, 2, figsize=(10, 8.2))
    # (a) L1 primary
    a = ax[0, 0]; rec = o["L1"]["onset_rank"]
    m = np.array([rec["mech_pred"][t] for t in T7]); L = np.array([LEVEL[t] for t in T7])
    lim = [min(m.min(), L.min()) - 0.08, max(m.max(), L.max()) + 0.08]
    a.plot(lim, lim, color=MUTED, lw=1, ls="--", label="slope 1 (pure reading drop)")
    xx = np.linspace(*lim, 10); f = rec["fit_level_on_mech"]
    a.plot(xx, f["a"] + f["b"] * xx, color=BLUE, lw=2, label=f"primary (ranks 1-20) OLS slope {f['b']:+.2f}")
    a.scatter(m, L, s=36, color=BLUE, edgecolor="white", lw=1.5, zorder=3)
    rb = o["L1"]["onset_band"]; mb = np.array([rb["mech_pred"][t] for t in T7])
    a.scatter(mb, L, s=36, facecolor="none", edgecolor=ORANGE, lw=1.5, zorder=3,
              label=f"secondary: F_onset 0.15-0.45 N (slope-1 R² {rb['R2_slope1_free_c']:+.2f})")
    off = {"C1": (-14, -11), "C_T2": (4, 4)}
    for t, x_, y_ in zip(T7, m, L):
        a.annotate(t, (x_, y_), xytext=off.get(t, (4, 3)), textcoords="offset points", fontsize=8, color=INK)
    a.set_xlabel("mechanical prediction m (log), onset-fixed boundary,\nranks 1-20, relative to A1+A2")
    a.set_ylabel("observed lower level (log)")
    a.set_title(f"L1 mechanical null, 7 diagrams (primary slope-1 R² {rec['R2_slope1_free_c']:+.2f})", fontsize=9.5, loc="left")
    a.legend(frameon=False, fontsize=8)
    # (b) L2
    a = ax[0, 1]; l2 = o["L2"]["primary_9x3"]
    edges = np.linspace(0, 1, 41)
    a.bar(edges[:-1], l2["null_hist"], width=0.025 - 0.002, align="edge", color=MUTED)
    a.axvline(o["L2"]["obs_abs_r_dec5_base"], color=ORANGE, lw=2)
    a.text(0.0, max(l2["null_hist"]) * 1.0, f"dec5_pl |r| {o['L2']['obs_abs_r_dec5_base']:.2f} (line)\n"
           f"screen-adjusted p {l2['p_adj']:.3f}\nnull 95th pct {l2['null_q95']:.2f}", ha="left", va="top", fontsize=8.5, color=INK)
    a.set_xlabel("max |r| over the 9 predictors x 3 responses screen"); a.set_ylabel("permutations (of 10,000)")
    a.set_title("L2 max-|r| screen null (10 diagrams, 10,000 perm.)", fontsize=9.5, loc="left")
    # (c) r across variants (L3, L4, L5)
    a = ax[1, 0]
    items = [("baseline (10)", o["L6"]["all"]), ("L3 no-band missing (10)", o["L3"]["assoc_primary"]),
             ("L3 no-band missing, orig. law (10)", o["L3"]["assoc_secondary"]),
             ("L4 onset axis (7)", o["L4"]["onset_7"]), ("L4 current axis, same 7", o["L4"]["current_same7"]),
             ("L4 plateau-start axis (10)", o["L4"]["plateau0_10"])]
    for k in ("c=0.2", "c=0.3", "c=0.4", "zero_corrected", "zero_corrected_c=0.2", "zero_corrected_c=0.3"):
        a5 = o["L5"]["assoc"][k]
        items.append((f"L5 {k.replace('zero_corrected', 'zero-offset')} ({a5['n']})", a5))
    for i, (lab, it) in enumerate(items):
        y = len(items) - 1 - i
        a.plot(it["r_ci95"], [y, y], color=MUTED, lw=2, solid_capstyle="round")
        a.scatter([it["r"]], [y], s=40, color=BLUE if i else ORANGE, zorder=3, edgecolor="white", lw=1.5)
        a.text(1.02, y, f"{it['r']:+.2f}", va="center", fontsize=8, color=INK, transform=a.get_yaxis_transform())
    a.set_yticks(range(len(items))); a.set_yticklabels([l for l, _ in items][::-1], fontsize=8)
    a.axvline(0, color=MUTED, lw=0.8); a.set_xlim(-1, 1)
    a.set_xlabel("Pearson r(dec5_pl, lower level) with Fisher 95% CI")
    a.set_title("L3-L5 variants", fontsize=9.5, loc="left")
    # (d) L6 influence
    a = ax[1, 1]
    labs = [f"-{t}" for t in TAGS] + ["-A1-A2", "7 conditions"]
    recs = [o["L6"][f"drop_{t}"] for t in TAGS] + [o["L6"]["without_A1_A2"], o["L6"]["condition_level"]]
    for i, (lab, it) in enumerate(zip(labs, recs)):
        a.plot([i, i], it["r_ci95"], color=MUTED, lw=2, solid_capstyle="round")
        a.scatter([i], [it["r"]], s=40, color=BLUE, zorder=3, edgecolor="white", lw=1.5)
    a.axhline(o["L6"]["all"]["r"], color=ORANGE, lw=1.5, ls="--", label=f"all 10: r {o['L6']['all']['r']:+.2f}")
    a.axhline(0, color=MUTED, lw=0.8)
    a.set_xticks(range(len(labs))); a.set_xticklabels(labs, rotation=60, ha="right", fontsize=8)
    a.set_ylabel("r with Fisher 95% CI"); a.set_ylim(-0.2, 1)
    a.set_title("L6 influence (drop-one n 9; -A1-A2 n 8; cond. n 7)", fontsize=9.5, loc="left")
    a.legend(frameon=False, fontsize=8, loc="lower left")
    for a in ax.flat:
        for s in ("top", "right"):
            a.spines[s].set_visible(False)
    fig.suptitle("Post hoc robustness of the minimum-force association (dec5_pl), development data only", fontsize=10.5, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(HERE / "fig_robust_lower.png", dpi=170, facecolor="white")


def main():
    if "--fig" in sys.argv[1:] or "--fig" in ARGV:
        figure(json.load(open(HERE / "results.json")))
        return
    out["L5"] = L5()
    out["L6"] = L6()
    out["L4"] = L4()
    out["L2"] = L2()
    out["L1"] = L1()
    out["L3"] = L3()
    json.dump(out, open(HERE / "results.json", "w"), indent=1, default=float)
    figure(out)


if __name__ == "__main__":
    main()
