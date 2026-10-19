"""(g) Lower-boundary hit share split by censoring type (reviewer: robustness)."""
import pickle
from pathlib import Path
import numpy as np
st = pickle.load(open(Path(__file__).parent / "main_store.pkl", "rb"))
TEST = ("B1", "C1", "D1", "B_T2r1", "B_T2r2")
for t in (0.0, 0.2, 0.3):
    res = st[(2, t, "main")]
    print(f"t={t} K=2: lower hit share (mean over 5 tests): all | finite-interval cols | left-censored cols")
    for m in res["B1"]:
        a = np.mean([np.nanmean(res[g][m]["hL"]) for g in TEST])
        f = np.mean([np.nanmean(res[g][m]["hL"][res[g][m]["cL"] == "finite"]) for g in TEST])
        l = np.mean([np.nanmean(res[g][m]["hL"][res[g][m]["cL"] == "left"]) for g in TEST])
        print(f"   {m:8s} {a:.2f} | {f:.2f} | {l:.2f}")
    # 4-string mean (B_T2 repeats averaged first)
    print("  4-string mean lower |err| (B_T2 rep_1/rep_2 averaged first):", {m: round(float(np.mean([np.nanmedian(res['B1'][m]['eL']), np.nanmedian(res['C1'][m]['eL']), np.nanmedian(res['D1'][m]['eL']), 0.5*(np.nanmedian(res['B_T2r1'][m]['eL'])+np.nanmedian(res['B_T2r2'][m]['eL']))])), 3) for m in res['B1']})
