"""Post-hoc, read-only force-axis/mapping audit. No frozen scoring is called.

Descriptive MLEs deliberately reuse the legacy LL.fit; root independently audits
the likelihood. Window labels are existing classifier outputs, not independent
ground truth. Nominal force is a documented command-rank proxy, not calibration.
"""
from __future__ import annotations
import csv
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

BASE = Path.home() / "violin-rne"
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE / "novelty/lower_law"))
import onset_axis as OA
LL = OA.LL
STANDARD = ["A1", "A2", "C1", "D1", "A_T2", "B_T2s1", "C_T2", "A3", "A4", "D2"]


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def load_standard(tag):
    name = OA.TAG.get(tag, tag)
    paths = [BASE / "final/cache" / (name + ".npz"), BASE / "novelty/thermal_upper/cache" / (name + ".npz")]
    path = next(p for p in paths if p.exists())
    d = dict(np.load(path))
    m = np.isclose(d["t"], .2)
    q = {k: d[k][m] for k in ("folder", "trial", "column", "beta", "speed", "force", "H")}
    q["folder"] = d["folders"][q["folder"]]
    return q, d, path


def load_special():
    win, beta, speed = LL.BT.read_labels(LL.V.rm, LL.V.APPLIED)
    cols = LL.BT.assign_columns(beta)
    keys = [k for k in win if .2 in win[k] and min(abs(speed[k[0]] - s) for s in LL.BT.STD_SPEEDS) < 1e-3]
    q = dict(folder=np.array([k[0] for k in keys]), trial=np.array([k[1] for k in keys]),
             column=np.array([cols[k] for k in keys]), beta=np.array([beta[k] for k in keys]),
             speed=np.array([speed[k[0]] for k in keys]), force=np.array([win[k][.2][0] for k in keys]),
             H=np.array([win[k][.2][1] for k in keys]))
    return q, None, LL.BT.D / "features.csv.gz"


def parse_ramp(path):
    allrows = list(csv.DictReader(path.open()))
    counts = Counter((r["folder"], int(r["trial"])) for r in allrows)
    good = [r for r in allrows if not r["error"] and r["force_at_onset"] not in ("", "nan")]
    rm = {(r["folder"], int(r["trial"])): r for r in good}
    return rm, dict(total_rows=len(allrows), error_rows=sum(bool(r["error"]) for r in allrows),
                    duplicate_keys=[list(k) + [n] for k, n in counts.items() if n != 1], valid_rows=len(good))


def extract(q, force):
    groups = defaultdict(list)
    for i, (f, c) in enumerate(zip(q["folder"], q["column"])):
        groups[(str(f), int(c))].append(i)
    out = {}
    for key, ids in sorted(groups.items()):
        ii = np.array(sorted(ids, key=lambda i: force[i]))
        logs = np.log(np.maximum(force[ii], 1e-3))
        h = q["H"][ii]
        runs = np.flatnonzero(np.convolve(h.astype(int), np.ones(2, int), "valid") == 2)
        if len(runs):
            j = int(runs[0]); lo = -np.inf if j == 0 else logs[j-1]; hi = logs[j]
            endpoints = [None if j == 0 else int(ii[j-1]), int(ii[j])]
            reason = "left" if j == 0 else "finite"
        else:
            lo, hi = logs[-1], np.inf; endpoints = [int(ii[-1]), None]; reason = "no_H_run"
        out[key] = dict(lv=float(np.log(np.median(q["speed"][ids]))), lb=float(np.log(np.median(q["beta"][ids]))),
                        lower=(float(lo), float(hi)), endpoints=endpoints, reason=reason,
                        idx=ii.tolist(), trial_center=float(np.median(q["trial"][ids])))
    return out


def fit_info(rows):
    rr = list(rows.values())
    zero = [k for k, r in rows.items() if r["lower"][0] >= r["lower"][1]]
    kinds = dict(Counter(r["reason"] for r in rr))
    if zero:
        return dict(n=len(rr), kinds=kinds, invalid_intervals=[list(k) for k in zero], legacy_fit=None)
    bv, bb = LL.fit(rr)
    return dict(n=len(rr), kinds=kinds, invalid_intervals=[], legacy_fit=dict(b_v=bv, b_beta=bb))


def regression(rows, values):
    X = np.array([[1, r["lv"], r["lb"]] for r in rows], float)
    y = np.asarray(values, float)
    coef = np.linalg.lstsq(X, y, rcond=None)[0]
    return dict(n=len(rows), intercept=float(coef[0]), b_v=float(coef[1]), b_beta=float(coef[2]),
                median=float(np.median(y)), residual_sd=float(np.std(y-X@coef, ddof=3)))


def audit(tag):
    if tag == "B_T2s2_std3":
        q, full, source = load_special()
        ramp = LL.BT.D / "ramp_meta.csv"
    else:
        q, full, source = load_standard(tag)
        ramp = OA.RAMP[tag]
    rm, diag = parse_ramp(ramp)
    keys = list(zip(q["folder"].tolist(), q["trial"].tolist()))
    assert len(keys) == len(set(keys)), "duplicate t=.2 key"
    missing = [k for k in keys if k not in rm]
    assert not missing, (tag, missing[:3])
    onset = np.array([float(rm[k]["force_at_onset"]) for k in keys])
    b2 = np.array([float(rm[k]["beta"]) for k in keys])
    vmax = np.array([float(rm[k]["v_max"]) for k in keys])
    diag.update(t02_strokes=len(keys), missing_t02=len(missing), exact_t02_beta_max_abs=float(np.max(abs(b2-q["beta"]))),
                speed_max_abs=float(np.max(abs(vmax-q["speed"]))),
                ramp_extra_keys_not_in_std3_t02=len(set(rm)-set(keys)),
                nonpositive_onset=int(np.sum(onset<=0)), nonpositive_window=int(np.sum(q["force"]<=0)))
    if full is not None:
        allkeys = set(zip(full["folders"][full["folder"]].tolist(), full["trial"].tolist()))
        diag["all_cache_unique_strokes"] = len(allkeys)
        diag["all_cache_missing_ramp"] = len(allkeys-set(rm))
        diag["duplicate_cache_stroke_time"] = len(full["trial"])-len(set(zip(full["folder"].tolist(),full["trial"].tolist(),full["t"].tolist())))
    axes = {"window": q["force"], "onset": onset}
    if tag != "B_T2s2_std3":
        rank = (q["trial"] - 1) % 50
        if tag == "A1":
            rank = 49-rank
        axes["nominal_rank_proxy"] = .1*40**(rank/49)
    extracted = {ax: extract(q, ff) for ax, ff in axes.items()}
    out = dict(mapping=diag, source=str(source), source_sha256=sha(source), ramp=str(ramp), ramp_sha256=sha(ramp),
               fits={ax: fit_info(rr) for ax, rr in extracted.items()}, scan=[], reordering={}, shift={})
    for folder in sorted(set(q["folder"])):
        ids = np.flatnonzero(q["folder"] == folder)
        rr = [extracted["window"][k] for k in extracted["window"] if k[0] == folder]
        lb = np.array([r["lb"] for r in rr]); trial = np.array([r["trial_center"] for r in rr])
        predict = np.column_stack([np.ones(len(trial)), trial]) @ np.linalg.lstsq(np.column_stack([np.ones(len(trial)), trial]), lb, rcond=None)[0]
        r2 = 1-np.sum((lb-predict)**2)/np.sum((lb-lb.mean())**2)
        within = []
        for c in sorted(set(q["column"][ids])):
            jj = ids[q["column"][ids] == c]
            within.append(float(spearmanr(q["trial"][jj], onset[jj]).statistic))
        out["scan"].append(dict(folder=str(folder), speed=float(np.median(q["speed"][ids])),
                                beta_trial_spearman=float(spearmanr(lb,trial).statistic), log_beta_explained_by_trial_r2=float(r2),
                                onset_within_column_trial_rho_median=float(np.median(within)),
                                trial_range=[int(q["trial"][ids].min()),int(q["trial"][ids].max())]))
    w = extracted["window"]
    for ax in axes:
        if ax == "window": continue
        a = extracted[ax]
        out["reordering"][ax] = dict(columns=len(w), full_sort_changed=sum(w[k]["idx"] != a[k]["idx"] for k in w),
            lower_endpoint_pair_changed=sum(w[k]["endpoints"] != a[k]["endpoints"] for k in w),
            reason_changed=sum(w[k]["reason"] != a[k]["reason"] for k in w))
        # Same boundary strokes isolates the coordinate transform; re-extraction also changes label order.
        rr, delta, complete, dw, da, invalid_fixed = [], [], [], [], [], 0
        dec_rows, dec_coordinate, dec_reselection, dec_total = [], [], [], []
        for k, wr in w.items():
            if wr["reason"] == "finite":
                inds = wr["endpoints"]
                fw, fa = q["force"][inds], axes[ax][inds]
                if np.all(fw>0) and np.all(fa>0):
                    rr.append(wr); delta.append(float(np.mean(np.log(fa)-np.log(fw))))
                    invalid_fixed += int(fa[0] >= fa[1])
            ar = a[k]
            if np.all(np.isfinite(wr["lower"])) and np.all(np.isfinite(ar["lower"])) and wr["lower"][0]<wr["lower"][1] and ar["lower"][0]<ar["lower"][1]:
                complete.append(wr); dw.append(float(np.mean(wr["lower"]))); da.append(float(np.mean(ar["lower"])))
                inds = wr["endpoints"]
                fw, fa = q["force"][inds], axes[ax][inds]
                if np.all(fw>0) and np.all(fa>0):
                    old = float(np.mean(np.log(fw))); same = float(np.mean(np.log(fa))); new = float(np.mean(ar["lower"]))
                    dec_rows.append(wr); dec_coordinate.append(same-old); dec_reselection.append(new-same); dec_total.append(new-old)
        out["shift"][ax] = dict(fixed_window_boundary_strokes=regression(rr,delta) if len(rr)>3 else None,
                                fixed_endpoint_order_reversed_or_tied=invalid_fixed,
                                paired_finite_OLS_window=regression(complete,dw) if len(complete)>3 else None,
                                paired_finite_OLS_reextracted=regression(complete,da) if len(complete)>3 else None,
                                exact_paired_finite_decomposition={"coordinate_only":regression(dec_rows,dec_coordinate),
                                    "boundary_reselection":regression(dec_rows,dec_reselection), "total":regression(dec_rows,dec_total)} if len(dec_rows)>3 else None)
    # Per-column export permits root to join these independently by folder+column.
    export=[]
    for key in w:
        row=dict(folder=key[0],column=key[1],lv=w[key]["lv"],lb=w[key]["lb"],trial_center=w[key]["trial_center"])
        for ax in axes:
            row[ax]={k:extracted[ax][key][k] for k in ("lower","reason","endpoints")}
        export.append(row)
    (HERE/(tag+"_rows.json")).write_text(json.dumps(export,indent=1))
    return out


def main():
    result={"status":"post_hoc", "nominal_caveat":"0.1*40**(rank/49), A1 reverse, is a prior documented 50-command grid proxy; actual calibrated command forces have not been recovered. B_T2s2 nominal omitted because its 40-step command grid is not established.", "sessions":{}}
    for tag in STANDARD+["B_T2s2_std3","D_T2"]:
        result["sessions"][tag]=audit(tag)
        x=result["sessions"][tag]
        print(tag, {ax:r["legacy_fit"] for ax,r in x["fits"].items()}, x["reordering"], flush=True)
        (HERE/"axis_results.json").write_text(json.dumps(result,indent=1))


if __name__ == "__main__": main()
