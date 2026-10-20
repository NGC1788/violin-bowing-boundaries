"""POST HOC: per-session lower-boundary exponents on the ONSET-force axis (label-free force at bow onset replaces the
t = 0.2 s window force; boundaries re-extracted with K = 2 at the same windows), plus a random-effects summary of the
beta and speed exponents over sessions on both axes. Claude, 2026-10-20."""
import csv, json, sys
from collections import defaultdict
from pathlib import Path
import numpy as np
from scipy.stats import t as tdist
H = Path.home(); sys.path.insert(0, str(H / "violin-rne/novelty/lower_law"))
import lower_law as LL  # noqa: E402
SV, C, BT, V = LL.SV, LL.C, LL.BT, LL.V
AC = H / "violin-rne/astra_20261018/archive_companion"
RAMP = {"A1": H / "violin-rne/claude_extra/ramp/ramp_meta_2024-03-25_TypeA_sample1.csv", "A2": H / "violin-rne/claude_extra/ramp_meta_A2.csv"}
STEM = {"C1": "2024-04-12_TypeC_sample1", "D1": "2024-04-15_TypeD_sample1", "A_T2": "2024-06-19_TypeA_T2_sample1", "B_T2s1": "2024-06-13_TypeB_T2_sample1",
        "C_T2": "2024-06-11_TypeC_T2_sample1", "D_T2": "2024-06-17_TypeD_T2_sample1", "A3": "2024-04-22_TypeA_sample3", "A4": "2024-04-24_TypeA_sample4",
        "D2": "2024-04-26_TypeD_sample2"}
for k, s in STEM.items():
    RAMP[k] = AC / s / f"ramp_meta_{s}.csv"
TAG = {"A3": "TGT_2024-04-22_TypeA_sample3", "A4": "TGT_2024-04-24_TypeA_sample4", "D2": "TGT_2024-04-26_TypeD_sample2"}


def onset_rows(sess):
    rm = {}
    for r in csv.DictReader(open(RAMP[sess])):
        if not r["error"] and r["force_at_onset"] not in ("", "nan"):
            rm[(r["folder"], int(r["trial"]))] = float(r["force_at_onset"])
    d = dict(SV.R.load(TAG.get(sess, sess)))
    f = np.array([rm.get((d["folders"][fo], int(tr)), np.nan) for fo, tr in zip(d["folder"], d["trial"])])
    keep = np.isfinite(f)
    assert keep.mean() > 0.99, (sess, keep.mean())
    for k in list(d):
        if k in ("folders", "Z", "f0"):
            continue
        d[k] = d[k][keep] if k != "force" else f[keep]
    return C.intervals(d, 0.2)


def bt2s2_onset():
    rm = {(r["folder"], int(r["trial"])): float(r["force_at_onset"]) for r in csv.DictReader(open(H / "violin-rne/astra_20261019/highforce_stream/2024-07-03_TypeB_T2_sample2/ramp_meta.csv")) if not r["error"]}
    win, beta, speed = BT.read_labels(V.rm, V.APPLIED)
    col = BT.assign_columns(beta); g = defaultdict(list)
    for k, d in win.items():
        if 0.2 in d and min(abs(speed[k[0]] - s) for s in BT.STD_SPEEDS) < 1e-3:
            g[(k[0], col[k])].append((rm[k], d[0.2][1], beta[k]))
    rows = []
    for (folder, c), gg in sorted(g.items()):
        gg.sort(); f = np.log(np.maximum([x[0] for x in gg], 1e-3)); h = [x[1] for x in gg]
        lo_, up = BT.intervals(f, h)
        rows.append(dict(lv=float(np.log(speed[folder])), lb=float(np.log(np.median([x[2] for x in gg]))), lower=lo_, upper=up))
    return rows


def re_summary(est, se):
    est, se = np.asarray(est), np.asarray(se); w = 1 / se ** 2; m = (w * est).sum() / w.sum(); Q = (w * (est - m) ** 2).sum(); k = len(est)
    tau2 = max(0, (Q - (k - 1)) / (w.sum() - (w ** 2).sum() / w.sum())); ws = 1 / (se ** 2 + tau2); mu = (ws * est).sum() / ws.sum()
    # HKSJ
    q = (ws * (est - mu) ** 2).sum() / (k - 1); seh = np.sqrt(q / ws.sum()); tc = tdist.ppf(0.975, k - 1)
    return dict(mu=float(mu), ci_hksj=[float(mu - tc * seh), float(mu + tc * seh)], tau=float(np.sqrt(tau2)), Q=float(Q), k=k)


def main():
    res = {"onset": {}}
    sessions = ["A1", "A2", "C1", "D1", "A_T2", "B_T2s1", "C_T2", "A3", "A4", "D2"]
    for s in sessions:
        rows = LL.with_bcol([dict(r) for r in onset_rows(s) if np.isfinite(r["lower"][0]) or np.isfinite(r["lower"][1])])
        bv, bb = LL.fit(rows); cv, cb = LL.boot(rows, nb=200)
        res["onset"][s] = dict(b_v=bv, b_beta=bb, b_v_ci=cv, b_beta_ci=cb, n=len(rows))
    rows = LL.with_bcol([r for r in bt2s2_onset() if np.isfinite(r["lower"][0]) or np.isfinite(r["lower"][1])])
    bv, bb = LL.fit(rows); cv, cb = LL.boot(rows, nb=200)
    res["onset"]["B_T2s2_std3"] = dict(b_v=bv, b_beta=bb, b_v_ci=cv, b_beta_ci=cb, n=len(rows))
    meas = json.load(open("results.json"))
    # measured-axis bootstrap CIs exist only for the 4 new sessions; for dev diagrams compute them now (B 200) for the RE summary
    mse = {}
    for s in [k for k in meas if k not in ("B_T2s2_all4",)]:
        m = meas[s]["measured"]
        if m["b_beta_ci"] is None:
            rr = LL.with_bcol([dict(r) for r in SV.rows_of(s) if np.isfinite(r["lower"][0]) or np.isfinite(r["lower"][1])])
            cv, cb = LL.boot(rr, nb=200)
        else:
            cv, cb = m["b_v_ci"], m["b_beta_ci"]
        mse[s] = (m["b_v"], (cv[1] - cv[0]) / 3.92, m["b_beta"], (cb[1] - cb[0]) / 3.92)
    res["measured_se"] = mse
    res["re_measured"] = dict(beta=re_summary([v[2] for v in mse.values()], [v[3] for v in mse.values()]),
                              v=re_summary([v[0] for v in mse.values()], [v[1] for v in mse.values()]))
    on = res["onset"]
    res["re_onset"] = dict(beta=re_summary([v["b_beta"] for v in on.values()], [(v["b_beta_ci"][1] - v["b_beta_ci"][0]) / 3.92 for v in on.values()]),
                           v=re_summary([v["b_v"] for v in on.values()], [(v["b_v_ci"][1] - v["b_v_ci"][0]) / 3.92 for v in on.values()]))
    Path("onset_axis.json").write_text(json.dumps(res, indent=1, default=float))
    for s, v in on.items():
        print(f"onset {s:12s} v {v['b_v']:+.2f} β {v['b_beta']:+.2f} [{v['b_beta_ci'][0]:+.2f},{v['b_beta_ci'][1]:+.2f}] (measured β {meas[s]['measured']['b_beta']:+.2f})")
    for ax in ("re_measured", "re_onset"):
        b, vv = res[ax]["beta"], res[ax]["v"]
        print(f"{ax}: beta exponent mean {b['mu']:+.2f} HKSJ [{b['ci_hksj'][0]:+.2f},{b['ci_hksj'][1]:+.2f}] between-session SD {b['tau']:.2f} (k {b['k']}) | speed exponent {vv['mu']:+.2f} [{vv['ci_hksj'][0]:+.2f},{vv['ci_hksj'][1]:+.2f}] SD {vv['tau']:.2f}")


if __name__ == "__main__":
    main()
