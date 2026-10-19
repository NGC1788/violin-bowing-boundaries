"""Once-only scorer for FROZEN_v3 pooled hypotheses (protocol: PROTOCOL.md). Claude, 2026-10-19.
  python score_v3.py --freeze   (hashes; no boundary is computed)
  python score_v3.py --score    (claim -> compute -> result, all O_EXCL)"""
import argparse
import datetime as dt
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
TU = Path.home() / "violin-rne/novelty/thermal_upper"
sys.path.insert(0, str(TU))
STEMS = ["2024-04-24_TypeA_sample4", "2024-04-26_TypeD_sample2", "2024-07-03_TypeB_T2_sample2"]
STATE = Path.home() / "violin-rne/timeres/replicate_state.json"
CODE = [HERE / "PROTOCOL.md", HERE / "ADDENDUM_20261020.md", Path(__file__).resolve(), HERE / "test_score_v3.py", TU / "FROZEN_v3.json", TU / "FROZEN_v2.json",
        TU / "rows.py", TU / "thermal_law.py", TU / "analysis2b.py"]
KST = dt.timezone(dt.timedelta(hours=9))


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def write_excl(path, obj):
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    with os.fdopen(fd, "w") as fh:
        json.dump(obj, fh, indent=1, default=float)


def cfit(X, lo, hi, x0):
    import thermal_law as TL
    from scipy.optimize import minimize
    f = lambda p: TL.censored_nll(X @ p[:-1], np.exp(p[-1]), lo, hi)
    r = minimize(f, x0, method="L-BFGS-B", options={"maxiter": 5000})
    r2 = minimize(f, r.x, method="Powell", options={"maxiter": 20000, "xtol": 1e-6, "ftol": 1e-9})
    r3 = minimize(f, r2.x, method="L-BFGS-B", options={"maxiter": 5000})
    return r3.x


def ols_start(X, lo, hi):
    y = np.where(np.isfinite(lo) & np.isfinite(hi), 0.5 * (lo + hi), np.where(np.isfinite(lo), lo + 0.2, hi - 0.2))
    return np.r_[np.linalg.lstsq(X, y, rcond=None)[0], np.log(0.3)]


def speed_cat(lv):
    u = sorted(set(np.round(lv, 6)))
    if len(u) == 3:
        return np.array([u.index(x) for x in np.round(lv, 6)]), np.array(u), False
    nom = np.log([0.05, 0.10, 0.20])
    c = np.argmin(np.abs(lv[:, None] - nom[None, :]), axis=1)
    return c, nom, True


def designs(lv, lb, tg, tlist):
    vc, sp, nearest = speed_cat(lv)
    D = [(tg == t).astype(float) for t in tlist[1:]]
    X1 = np.column_stack([(vc == j).astype(float) for j in range(3)] + [lb] + D)
    X2 = np.column_stack([lv, lb, lv * lb, lb ** 2, np.ones_like(lv)] + D)
    return X1, X2, sp, nearest


def stats(lv, lb, lo, hi, tg, tlist, x1=None, x2=None):
    X1, X2, sp, _ = designs(lv, lb, tg, tlist)
    p1 = cfit(X1, lo, hi, ols_start(X1, lo, hi) if x1 is None else x1)
    p2 = cfit(X2, lo, hi, ols_start(X2, lo, hi) if x2 is None else x2)
    e12 = (p1[1] - p1[0]) / (sp[1] - sp[0]); e23 = (p1[2] - p1[1]) / (sp[2] - sp[1])
    return dict(e12=e12, e23=e23, curvature=e23 - e12, g=p2[2], q=p2[3]), p1, p2


def decide(x, ci, pos, neg):
    if x < 0 and ci[1] < 0:
        return neg
    if x > 0 and ci[0] > 0:
        return pos
    return f"not decided (point {'<' if x < 0 else '>'} 0, CI includes 0)"


def pooled(data, nb=200):
    tlist = [t for t in STEMS if t in data]
    lv = np.concatenate([data[t][0] for t in tlist]); lb = np.concatenate([data[t][1] for t in tlist])
    lo = np.concatenate([data[t][2] for t in tlist]); hi = np.concatenate([data[t][3] for t in tlist])
    tg = np.concatenate([np.full(len(data[t][0]), t) for t in tlist])
    est, p1, p2 = stats(lv, lb, lo, hi, tg, tlist)
    idx = {t: np.flatnonzero(tg == t) for t in tlist}
    out = {}
    for name, seed in (("cluster", 5), ("stratified", 6)):
        rng = np.random.default_rng(seed); bs = []
        for _ in range(nb):
            if name == "cluster":
                ii = np.concatenate([idx[t] for t in rng.choice(tlist, len(tlist))])
            else:
                ii = np.concatenate([rng.choice(idx[t], len(idx[t])) for t in tlist])
            b, _, _ = stats(lv[ii], lb[ii], lo[ii], hi[ii], tg[ii], tlist, p1, p2)
            bs.append([b["curvature"], b["g"]])
        bs = np.array(bs)
        out[name] = dict(curvature_ci=np.percentile(bs[:, 0], [2.5, 97.5]).tolist(), g_ci=np.percentile(bs[:, 1], [2.5, 97.5]).tolist())
    _, _, _, nearest = designs(lv, lb, tg, tlist)
    ci_c, ci_g = out["cluster"]["curvature_ci"], out["cluster"]["g_ci"]
    dec = dict(H_curv=decide(est["curvature"], ci_c, "H_curv_thermal supported", "H_curv_empirical supported"),
               H_int=decide(est["g"], ci_g, "H_int empirical supported (g > 0)", "H_int thermal supported (g < 0)"))
    per = {}
    for t in tlist:
        m = tg == t
        try:
            b, _, _ = stats(lv[m], lb[m], lo[m], hi[m], tg[m], [t])
            per[t] = {k: float(v) for k, v in b.items()} | dict(n=int(m.sum()))
        except Exception as e:  # descriptive only
            per[t] = dict(error=str(e))
    return dict(targets=tlist, n_columns=int(len(lv)), estimate={k: float(v) for k, v in est.items()}, bootstrap=out,
                decisions=dec, per_target=per, speed_nearest_assignment=bool(nearest))


def load_targets():
    import rows as R
    import thermal_law as TL
    F2 = json.loads((TU / "FROZEN_v2.json").read_text())
    st = json.loads(STATE.read_text())
    data, status = {}, {}
    for s in STEMS:
        status[s] = st.get(s + ".7z", {}).get("status")
        if status[s] != "done":
            continue
        if s == "2024-07-03_TypeB_T2_sample2":  # ADDENDUM_20261020: dedicated 40x40 parser output, standard speeds only
            bt = Path.home() / "violin-rne/confirm_B_T2s2"
            r = json.loads((bt / "score_result.json").read_text())
            assert hashlib.sha256((bt / "bt2s2_v3_rows.npz").read_bytes()).hexdigest() == r["v3_rows_sha256"]
            z = np.load(bt / "bt2s2_v3_rows.npz")
            data[s] = (z["lv"], z["lb"], z["lo"], z["hi"])
            continue
        tag = "TGT_" + s
        R.D.STEMS[tag] = s
        R.ZPUB[tag] = R.PROP[F2["targets"][s]["group"]][1]
        lv, lb, lz, lo, hi = TL.arrays(R.rows_for(tag), R.ZPUB)
        data[s] = (lv, lb, lo, hi)
    return data, status


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--freeze", action="store_true"); ap.add_argument("--score", action="store_true")
    a = ap.parse_args()
    if a.freeze:
        write_excl(HERE / "FREEZE_RECEIPT_addendum.json", dict(frozen_kst=dt.datetime.now(KST).isoformat(), sha256={str(p): sha(p) for p in CODE},
                                                      state_at_freeze={s: json.loads(STATE.read_text()).get(s + ".7z", {}).get("status") for s in STEMS}))
        print("frozen"); return
    rec = json.loads((HERE / "FREEZE_RECEIPT_addendum.json").read_text())
    for p in CODE:
        assert sha(p) == rec["sha256"][str(p)], ("changed since freeze", p)
    write_excl(HERE / "SCORE_CLAIM.json", dict(claimed_kst=dt.datetime.now(KST).isoformat(), receipt_sha256=sha(HERE / "FREEZE_RECEIPT_addendum.json")))
    data, status = load_targets()
    res = pooled(data)
    res["status"] = status
    res["complete"] = len(data) == 3
    if not res["complete"]:
        res["decisions"] = {k: "incomplete (%d of 3 targets): " % len(data) + v for k, v in res["decisions"].items()}
    res["frozen_v3_empirical_pi90_single_diagram"] = [-0.691240779819482, 0.05725511819481305]
    res["finished_kst"] = dt.datetime.now(KST).isoformat()
    write_excl(HERE / "score_result.json", res)
    print(json.dumps(res, indent=1, default=float))


if __name__ == "__main__":
    main()
