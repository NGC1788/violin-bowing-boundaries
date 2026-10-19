"""Column-level cross-validation of PACE vs baselines on the 11 exposed diagrams (post hoc development).
Splits: leave-one-diagram-out (LODO), leave-one-string-type-out (LOTO), leave-one-type x tension-out (LOTTO).
Every fold refits stage 1 (shape) and stage 2 (levels) on training diagrams only; the target contributes only
label-free controls to its prediction; labels of the target are used only for scoring."""
import json, sys, time
import numpy as np
import common as C
import model as M

PRIMARY = "full|m3|clip1"
CONFIGS = [(cs, 1, False) for cs in ("mean", "props", "propsZ", "protocol", "protocol_dec5", "full")] + \
          [(cs, 3, True) for cs in ("mean", "props", "protocol", "full")] + \
          [("mean", 3, False), ("mean", 1, True), ("full", 3, False), ("full", 1, True), ("full", 2, True), ("full", 5, True),
           ("props", 1, True), ("props", 3, False)]
DATA = {t: (C.load_exposed(t), C.GROUP[t]) for t in C.TAGS}
SCORE_ROWS = {t: C.intervals(DATA[t][0], "F_ev", "H2", "delta", 0.01) for t in C.TAGS}
SPLITS = {"LODO": {t: t for t in C.TAGS}, "LOTO": {t: t[0] for t in C.TAGS}, "LOTTO": dict(C.GROUP)}


def run(split):
    groups = SPLITS[split]
    res = {}
    for g in sorted(set(groups.values())):
        test = [t for t in C.TAGS if groups[t] == g]
        train = {t: DATA[t] for t in C.TAGS if groups[t] != g}
        t0 = time.time()
        L = M.lookup_fit(train)
        for t in test:
            res.setdefault("lookup", {})[t] = M.score(SCORE_ROWS[t], M.lookup_predict(L, DATA[t][0]))
        for cs, m, clip in CONFIGS:
            mod = M.PACE(cs, m, clip).fit(train)
            P = mod.params()
            for t in test:
                pr, info = M.predict_from_params(P, DATA[t][0], DATA[t][1])
                sc = M.score(SCORE_ROWS[t], pr)
                sc["extrapolated"] = info["extrapolated"]; sc["tau"] = P["tau"]; sc["omega"] = P["omega"]
                sc["b"] = P["b"]; sc["sigma"] = P["sigma"]
                res.setdefault(mod.name(), {})[t] = sc
        print(split, g, test, f"{time.time() - t0:.1f}s", flush=True)
    return res


def summarise(res, split):
    out = {}
    for name, per in res.items():
        tags = [t for t in C.TAGS if t in per]
        f = lambda k: [per[t][k] for t in tags]
        pooled = [e for t in tags for e in per[t]["fin_errors"]]
        typ = {}
        for t in tags:
            typ.setdefault(t[0], []).append(per[t]["col_median_abs_err"])
        out[name] = dict(col_median_err=float(np.nanmean(f("col_median_abs_err"))),
                         col_median_err_type_weighted=float(np.mean([np.nanmean(v) for v in typ.values()])),
                         pooled_col_mean_err=float(np.mean(pooled)), interval_distance=float(np.nanmean(f("interval_distance"))),
                         log_score=float(np.nanmean(f("log_score"))), hit_rate=float(np.nanmean(f("hit_rate"))),
                         level_mae=float(np.nanmean(f("level_error"))), n_level_fail=int(np.sum(np.isnan(f("level_error")))),
                         n_pred_fail=int(np.sum(f("n_pred_fail"))), n_finite=int(np.sum(f("n_finite"))),
                         per_diagram={t: dict(col_median=per[t]["col_median_abs_err"], dist=per[t]["interval_distance"],
                                              level=per[t]["level_signed"], logs=per[t]["log_score"],
                                              extrap=per[t].get("extrapolated")) for t in tags})
    # paired: primary minus each other model, per diagram (negative = primary better)
    prim = res[PRIMARY]
    for name in res:
        if name == PRIMARY:
            continue
        dd = np.array([prim[t]["col_median_abs_err"] - res[name][t]["col_median_abs_err"] for t in C.TAGS])
        di = np.array([prim[t]["interval_distance"] - res[name][t]["interval_distance"] for t in C.TAGS])
        rng = np.random.default_rng(20261019)
        bs = [np.mean(dd[rng.integers(0, 11, 11)]) for _ in range(4000)]
        out[name]["paired_vs_primary"] = dict(mean_diff_col_median=float(dd.mean()), wins_primary=int((dd < 0).sum()),
                                              losses_primary=int((dd > 0).sum()), diagram_boot95=[float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
                                              mean_diff_interval_distance=float(di.mean()))
    return out


if __name__ == "__main__":
    allres = {}
    for split in SPLITS:
        r = run(split)
        allres[split] = summarise(r, split)
        print(f"\n== {split} (mean over 11 held-out diagrams; lower is better except log score/hit) ==")
        print(f"{'model':22s} colMed  typeW  pooled  dist   logS   hit   levelMAE")
        for name, s in sorted(allres[split].items(), key=lambda kv: kv[1]["col_median_err"]):
            print(f"{name:22s} {s['col_median_err']:.3f}  {s['col_median_err_type_weighted']:.3f}  {s['pooled_col_mean_err']:.3f}  "
                  f"{s['interval_distance']:.3f}  {s['log_score']:+.3f}  {s['hit_rate']:.3f}  {s['level_mae']:.3f}"
                  + (f"  vsPRIM d={s['paired_vs_primary']['mean_diff_col_median']:+.3f} W{s['paired_vs_primary']['wins_primary']}/L{s['paired_vs_primary']['losses_primary']}"
                     if "paired_vs_primary" in s else "  <- PRIMARY"), flush=True)
    meta = dict(primary=PRIMARY, configs=[list(c) for c in CONFIGS], scoring="delta-fixed (0.01 N) legacy K=2 intervals, t=0.2 s, measured axis",
                status="POST HOC development on 11 exposed diagrams; not a confirmation",
                code_sha256={f: C.sha(C.HERE / f) for f in ("common.py", "model.py", "validate.py")})
    json.dump(dict(meta=meta, results=allres), open(C.HERE / "validation_results.json", "w"), indent=1, default=float)
