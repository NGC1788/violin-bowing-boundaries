"""Round 2 (POST HOC, written after reading validate_v1.log where the a-priori primary 'full|m3|clip1' lost to
the model-free lookup): flexible beta shapes and lookup-shape + label-free level hybrids.
Same splits, scoring and data as validate.py; all round-1 results are kept in validation_v1_results.json."""
import json, time
import numpy as np
import common as C
import model as M
import validate as V1

CONFIGS = []
for shp in ("lin", "quad", "cubic", "quad_s"):
    for cs in ("mean", "props", "protocol", "full"):
        for clip in (False, True):
            CONFIGS.append((cs, 1, clip, shp))
LOOKUPS = [((), False), ((), True), (("dec5_bar",), True), (("dec5_bar", "cal_bar"), True), (("logZ",), True),
           (("logZ", "logEI"), True), (("dec5_bar", "cal_bar", "logZ", "logEI"), True)]
REF = "lookup"   # strongest round-1 baseline; paired comparisons are against it


def run(split):
    groups = V1.SPLITS[split]
    res = {}
    for g in sorted(set(groups.values())):
        test = [t for t in C.TAGS if groups[t] == g]
        train = {t: V1.DATA[t] for t in C.TAGS if groups[t] != g}
        t0 = time.time()
        for keys, clip in LOOKUPS:
            mod = M.LookupLevel(keys, clip).fit(train)
            for t in test:
                pr, info = mod.predict(*V1.DATA[t])
                sc = M.score(V1.SCORE_ROWS[t], pr); sc["extrapolated"] = info["extrapolated"]
                res.setdefault(mod.name(), {})[t] = sc
        for cs, m, clip, shp in CONFIGS:
            mod = M.PACE(cs, m, clip, shape=shp).fit(train)
            P = mod.params()
            for t in test:
                pr, info = M.predict_from_params(P, *V1.DATA[t])
                sc = M.score(V1.SCORE_ROWS[t], pr); sc["extrapolated"] = info["extrapolated"]
                res.setdefault(mod.name(), {})[t] = sc
        print(split, g, test, f"{time.time() - t0:.1f}s", flush=True)
    return res


if __name__ == "__main__":
    V1.PRIMARY = REF
    allres = {}
    for split in V1.SPLITS:
        r = run(split)
        allres[split] = V1.summarise(r, split)
        print(f"\\n== {split} round 2 (paired d = lookup - model; positive = model better than lookup) ==")
        print(f"{'model':34s} colMed  typeW  pooled  dist   logS   hit   levelMAE")
        for name, s in sorted(allres[split].items(), key=lambda kv: kv[1]["col_median_err"]):
            pv = s.get("paired_vs_primary")
            print(f"{name:34s} {s['col_median_err']:.3f}  {s['col_median_err_type_weighted']:.3f}  {s['pooled_col_mean_err']:.3f}  "
                  f"{s['interval_distance']:.3f}  {s['log_score']:+.3f}  {s['hit_rate']:.3f}  {s['level_mae']:.3f}"
                  + (f"  d={pv['mean_diff_col_median']:+.3f} [{pv['diagram_boot95'][0]:+.3f},{pv['diagram_boot95'][1]:+.3f}] model-better {pv['losses_primary']}/11" if pv else "  <- REF"), flush=True)
    meta = dict(reference=REF, configs=[list(c) for c in CONFIGS], lookups=[[list(k), c] for k, c in LOOKUPS],
                status="POST HOC round 2 on 11 exposed diagrams (after round-1 failure of the a-priori primary)",
                code_sha256={f: C.sha(C.HERE / f) for f in ("common.py", "model.py", "validate.py", "validate2.py")})
    json.dump(dict(meta=meta, results=allres), open(C.HERE / "validation_v2_results.json", "w"), indent=1, default=float)
