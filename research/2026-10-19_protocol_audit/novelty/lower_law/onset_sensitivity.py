"""POST HOC robustness of the onset-axis result: t = 0.1/0.3 s and K = 3, and leave-one-session-out of the RE mean.
Claude, 2026-10-20."""
import json, sys
import numpy as np
from pathlib import Path
H = Path.home(); sys.path.insert(0, str(H / "violin-rne/novelty/lower_law"))
import onset_axis as OA  # noqa: E402
C = OA.C
res = {}
sessions = ["A1", "A2", "C1", "D1", "A_T2", "B_T2s1", "C_T2", "A3", "A4", "D2"]
for label, t, K in (("t0.1", 0.1, 2), ("t0.3", 0.3, 2), ("K3", 0.2, 3)):
    C.K = K
    est, se = [], []
    for s in sessions:
        import csv
        rm = {}
        for r in csv.DictReader(open(OA.RAMP[s])):
            if not r["error"] and r["force_at_onset"] not in ("", "nan"):
                rm[(r["folder"], int(r["trial"]))] = float(r["force_at_onset"])
        d = dict(OA.SV.R.load(OA.TAG.get(s, s)))
        f = np.array([rm.get((d["folders"][fo], int(tr)), np.nan) for fo, tr in zip(d["folder"], d["trial"])]); keep = np.isfinite(f)
        for k in list(d):
            if k not in ("folders", "Z", "f0"):
                d[k] = d[k][keep] if k != "force" else f[keep]
        rows = OA.LL.with_bcol([dict(r) for r in C.intervals(d, t) if np.isfinite(r["lower"][0]) or np.isfinite(r["lower"][1])])
        bv, bb = OA.LL.fit(rows); cv, cb = OA.LL.boot(rows, nb=100)
        est.append(bb); se.append((cb[1] - cb[0]) / 3.92)
    res[label] = dict(per_session=dict(zip(sessions, est)), re=OA.re_summary(est, se))
    print(label, "RE β", round(res[label]["re"]["mu"], 2), [round(x, 2) for x in res[label]["re"]["ci_hksj"]], "SD", round(res[label]["re"]["tau"], 2), {k: round(v, 2) for k, v in res[label]["per_session"].items()})
C.K = 2
on = json.load(open("onset_axis.json"))["onset"]
names = list(on); b = np.array([on[s]["b_beta"] for s in names]); se = np.array([(on[s]["b_beta_ci"][1] - on[s]["b_beta_ci"][0]) / 3.92 for s in names])
loo = {n: OA.re_summary(np.delete(b, i), np.delete(se, i))["mu"] for i, n in enumerate(names)}
res["leave_one_out_onset_beta_mean"] = loo
print("LOO RE mean β (onset):", {k: round(v, 2) for k, v in loo.items()}, "range", round(min(loo.values()), 2), round(max(loo.values()), 2))
Path("onset_sensitivity.json").write_text(json.dumps(res, indent=1, default=float))
