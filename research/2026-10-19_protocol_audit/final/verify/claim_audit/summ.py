import json, numpy as np, sys
tests = ["B1", "C1", "D1", "B_T2r1", "B_T2r2"]
for T in ("0.0","0.2","0.3"):
    R = json.load(open(f"/Users/jeongjihoon/violin-rne/final/compare_t{T}.json"))
    print("==== t", T, "Z", {k: round(v,3) for k,v in R["Z"].items()}, "slope", R["band_slope"])
    print("params", json.dumps(R["physics_params"]))
    res = R["results"]
    for m in res["B1"]:
        zs = lambda key, sub="zero_shot": [res[t][m][sub][key] for t in tests]
        lo = zs("lower_med_abs_log"); up = zs("upper_med_abs_log"); iou = zs("iou"); ll = zs("logloss")
        lh = zs("lower_hit"); uh = zs("upper_hit")
        print(f"{m:8s} lower mean {np.mean(lo):.3f} {np.round(lo,3)} | upper mean {np.mean(up):.3f} {np.round(up,3)} | IoU {np.mean(iou):.3f} {np.round(iou,3)} | LL {np.mean(ll):.3f} | hit L {np.mean(lh):.2f} U {np.mean(uh):.2f} | nL {zs('n_lower')} nU {zs('n_upper')}")
        for k in (5,10,20,40):
            for mode in ("random","grid"):
                sub=f"k{k}_{mode}"
                print(f"     {sub:10s} lower {np.nanmean(zs('lower_med_abs_log',sub)):.3f} upper {np.nanmean(zs('upper_med_abs_log',sub)):.3f}")
