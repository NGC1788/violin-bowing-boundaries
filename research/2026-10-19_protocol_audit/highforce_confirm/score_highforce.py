"""Dedicated, once-only scorer for the high-force archives (protocol: PROTOCOL.md). Claude, 2026-10-19.
  python score_highforce.py --freeze   writes FREEZE_RECEIPT.json (hashes of protocol, code, frozen files, inputs);
                                       label-free (no classifier column is parsed)
  python score_highforce.py --score    verifies every hash, creates SCORE_CLAIM.json (O_EXCL), parses labels,
                                       scores once, writes score_result.json (O_EXCL)
The analysis functions take plain arrays so test_score_highforce.py can exercise them on synthetic data."""
import argparse
import csv
import datetime as dt
import gzip
import hashlib
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

HOME = Path.home()
HERE = Path(__file__).resolve().parent
TU = HOME / "violin-rne/novelty/thermal_upper"
STREAM = HOME / "violin-rne/astra_20261019/highforce_stream"
sys.path.insert(0, str(TU))
import thermal_law as TL  # noqa: E402
import thermal_law2 as TL2  # noqa: E402

TARGETS = {  # target -> (archive, folder prefix, FROZEN group)
    "HF_C": ("HighBowForceRange_1", "HighBowForceRange_1/2024-06-22_TypeC_T2_sample2/", "C T2"),
    "HF_A": ("HighBowForceRange_1", "HighBowForceRange_1/2024-06-24_TypeA_T2_sample2/", "A T2"),
    "HF_D": ("HighBowForceRange_2", "HighBowForceRange_2/2024-06-27_TypeD_T2_sample2/", "D T2"),
    "HF_B": ("HighBowForceRange_2", "HighBowForceRange_2/2024-06-29_TypeB_T2_sample2/", "B T2"),
}
TIMES = (0.0, 0.1, 0.2, 0.3)
T_EVAL, K = 0.2, 2
N_COL, PER_COL, N_STROKES = 30, 50, 1500
STEP_HF = np.log(12.0 / 4.0) / 49
BASE_A = (-0.1764277369756792, 0.6606815556587429, -1.1107153335353168)   # = thermal_upper/rows.py BASE_A
NB_P1, NB_P3 = 500, 200
FROZEN_FILES = [TU / "FROZEN_v2.json", TU / "FROZEN_v3.json"]
CODE_FILES = [HERE / "PROTOCOL.md", Path(__file__).resolve(), HERE / "test_score_highforce.py", HERE / "controls_audit.py",
              HERE / "controls_audit.json", TU / "thermal_law.py", TU / "thermal_law2.py",
              HOME / "violin-rne/claude_extra/verify_astra_boundaries.py", HOME / "violin-bowing-boundaries/scripts/regime_map.py"]
CLAIM, RESULT, RECEIPT = HERE / "SCORE_CLAIM.json", HERE / "score_result.json", HERE / "FREEZE_RECEIPT.json"
KST = dt.timezone(dt.timedelta(hours=9))


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def now():
    return dt.datetime.now(KST).isoformat()


def write_excl(path, obj):
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    with os.fdopen(fd, "w") as fh:
        json.dump(obj, fh, indent=1, default=float)


# ---------------------------------------------------------------- parsing
def assign_columns(beta_by_stroke):
    """beta_by_stroke: {(folder, trial): beta}. Exactly 1,500 strokes / 30 columns of 50 per folder, separated."""
    col, by = {}, defaultdict(list)
    for k, b in beta_by_stroke.items():
        by[k[0]].append((b, k[1]))
    for folder, items in by.items():
        items.sort()
        assert len(items) == N_STROKES, (folder, len(items))
        lb = np.log([b for b, _ in items]).reshape(N_COL, PER_COL)
        assert np.min(lb[1:, 0] - lb[:-1, -1]) > np.max(lb[:, -1] - lb[:, 0]), folder
        for i, (_, tr) in enumerate(items):
            col[(folder, tr)] = i // PER_COL
    return col


def columns_from_windows(win, beta, speed_of_folder, t):
    """win: {(folder, trial): {t: (force, H)}} -> list of columns with sorted log forces and H at time t."""
    col = assign_columns(beta)
    groups = defaultdict(list)
    for k, d in win.items():
        if t in d:
            groups[(k[0], col[k])].append((d[t][0], d[t][1], beta[k]))
    out = []
    for (folder, c), g in sorted(groups.items()):
        g.sort()
        f = np.log(np.maximum(np.array([x[0] for x in g]), 1e-3))
        h = np.array([x[1] for x in g], bool)
        out.append(dict(folder=folder, column=c, lv=float(np.log(speed_of_folder[folder])),
                        lb=float(np.log(np.median([x[2] for x in g]))), f=f, h=h))
    return out


def intervals(f, h, k=K):
    """= final/compare.intervals for one column (f sorted ascending). Returns lower, upper, n_bands."""
    runs = [i for i in range(len(h) - k + 1) if all(h[i:i + k])]
    if not runs:
        return (f[-1], np.inf), (np.nan, np.nan), 0
    s, e = runs[0], runs[-1] + k - 1
    lower = (-np.inf, f[0]) if s == 0 else (f[s - 1], f[s])
    upper = (f[-1], np.inf) if e == len(h) - 1 else (f[e], f[e + 1])
    covered = np.zeros(len(h), bool)
    for i in runs:
        covered[i:i + k] = True
    n_bands = int(np.sum(covered[1:] & ~covered[:-1]) + covered[0])
    return lower, upper, n_bands


def column_table(cols, noband_as_left=False):
    rows = []
    for c in cols:
        lo_, up, nb = intervals(c["f"], c["h"])
        if nb == 0 and noband_as_left:
            up = (-np.inf, c["f"][0])
        rows.append(dict(c, lower=lo_, upper=up, n_bands=nb))
    return rows


def upper_arrays(rows, lz_value):
    keep = [r for r in rows if np.isfinite(r["upper"][0]) or np.isfinite(r["upper"][1])]
    lv = np.array([r["lv"] for r in keep]); lb = np.array([r["lb"] for r in keep])
    lo = np.array([r["upper"][0] for r in keep], float); hi = np.array([r["upper"][1] for r in keep], float)
    return lv, lb, np.full(len(keep), lz_value), lo, hi, keep


# ---------------------------------------------------------------- statistics (frozen forms)
def free_fit(lv, lb, lo, hi, x0=None):
    """= novelty/thermal_upper/score_target.free_fit"""
    X = np.column_stack([np.ones_like(lv), lv, lb])
    f = lambda p: TL.censored_nll(X @ p[:3], np.exp(p[3]), lo, hi)
    x0 = [0.0, 0.7, -1.0, np.log(0.3)] if x0 is None else x0
    return minimize(f, x0, method="Nelder-Mead", options={"maxiter": 40000, "xatol": 1e-8, "fatol": 1e-10}).x


def zone(x):
    return "H_thermal" if x >= -0.95 else ("H_plain" if x > -1.05 else "H_steep")


def p1(lv, lb, lo, hi, seed=2026, nb=NB_P1):
    if len(lv) < 5:
        return dict(decision="insufficient columns", n_columns=int(len(lv)))
    p = free_fit(lv, lb, lo, hi)
    rng = np.random.default_rng(seed)
    bs = np.array([free_fit(lv[ii], lb[ii], lo[ii], hi[ii], p) for ii in (rng.integers(0, len(lv), len(lv)) for _ in range(nb))])
    ci_b = np.percentile(bs[:, 2], [2.5, 97.5]).tolist()
    zones = sorted({zone(ci_b[0]), zone(ci_b[1]), zone(p[2])})
    return dict(c=float(p[0]), e_v=float(p[1]), e_v_ci=np.percentile(bs[:, 1], [2.5, 97.5]).tolist(), e_beta=float(p[2]),
                e_beta_ci=ci_b, sigma=float(np.exp(p[3])), zone=zone(p[2]), ci_zones=zones,
                decision=("supported: " + zone(p[2])) if len(zones) == 1 else "not decidable", n_columns=int(len(lv)))


def cfit(X, lo, hi, x0):
    """= analysis2b.cfit"""
    f = lambda p: TL.censored_nll(X @ p[:-1], np.exp(p[-1]), lo, hi)
    r = minimize(f, x0, method="L-BFGS-B", options={"maxiter": 5000})
    r2 = minimize(f, r.x, method="Powell", options={"maxiter": 20000, "xtol": 1e-6, "ftol": 1e-9})
    r3 = minimize(f, r2.x, method="L-BFGS-B", options={"maxiter": 5000})
    return r3.x, r3.fun


def ols_start(X, lo, hi):
    y = np.where(np.isfinite(lo) & np.isfinite(hi), 0.5 * (lo + hi), np.where(np.isfinite(lo), lo + 0.2, hi - 0.2))
    c = np.linalg.lstsq(X, y, rcond=None)[0]
    return np.r_[c, np.log(0.3)]


def curvature_design(vcat, lb, tgt, tlist):
    D = [(tgt == t).astype(float) for t in tlist[1:]]
    return np.column_stack([(vcat == j).astype(float) for j in range(3)] + [lb] + D)


def p3(vcat, lvcat, lb, lo, hi, tgt, seed=5, nb=NB_P3):
    """Pooled speed curvature e23 - e12 (analysis2b b1 design: speed-category levels, common beta slope, target offsets)."""
    tlist = sorted(set(tgt))
    X = curvature_design(vcat, lb, tgt, tlist)
    e = lambda p: ((p[1] - p[0]) / (lvcat[1] - lvcat[0]), (p[2] - p[1]) / (lvcat[2] - lvcat[1]))
    p, _ = cfit(X, lo, hi, ols_start(X, lo, hi))
    e12, e23 = e(p)
    rng = np.random.default_rng(seed)
    idx = {t: np.flatnonzero(tgt == t) for t in tlist}
    clus, strat = [], []
    for _ in range(nb):
        tt = rng.choice(tlist, len(tlist))
        ii = np.concatenate([idx[t] for t in tt])
        q, _ = cfit(X[ii], lo[ii], hi[ii], p); a, b = e(q); clus.append(b - a)
        jj = np.concatenate([rng.choice(idx[t], len(idx[t])) for t in tlist])
        q, _ = cfit(X[jj], lo[jj], hi[jj], p); a, b = e(q); strat.append(b - a)
    ci_c = np.percentile(clus, [2.5, 97.5]).tolist(); ci_s = np.percentile(strat, [2.5, 97.5]).tolist()
    cur = e23 - e12
    dec = lambda ci: ("concave, CI excludes 0" if ci[1] < 0 else "convex, CI excludes 0" if ci[0] > 0 else
                      ("concave point, CI includes 0" if cur < 0 else "convex point, CI includes 0"))
    per = {}
    for t in tlist:
        m = tgt == t
        Xt = np.column_stack([(vcat[m] == j).astype(float) for j in range(3)] + [lb[m]])
        q, _ = cfit(Xt, lo[m], hi[m], ols_start(Xt, lo[m], hi[m])); a, b = e(q)
        per[t] = dict(e12=float(a), e23=float(b), curvature=float(b - a), n=int(m.sum()))
    return dict(e12=float(e12), e23=float(e23), curvature=float(cur), ci_cluster=ci_c, ci_stratified=ci_s,
                decision_cluster=dec(ci_c), decision_stratified=dec(ci_s), per_target=per, n_columns=int(len(lb)))


def p2(F, group, lv, lb, lz, lo, hi, free):
    out = {}
    for m, f in F["fits"].items():
        if m not in TL2.MODELS:
            continue
        mu = TL2.predict(m, f["theta"], lv, lb, lz)
        fin = np.isfinite(lo) & np.isfinite(hi)
        hit = np.mean([(q >= a - STEP_HF if np.isfinite(a) else True) and (q <= b + STEP_HF if np.isfinite(b) else True)
                       for q, a, b in zip(mu, lo, hi)])
        ex = TL2.design_exponents(m, f["theta"], lv, lb, lz)
        le, ls = TL.offset_fit(mu, lo, hi)
        hf = F["high_force"][group][m]
        out[m] = dict(level_error=le, level_within_pi90=bool(abs(le) <= f["pi90"]["level"]), pi90_level=f["pi90"]["level"],
                      nll_per_col=float(TL.censored_nll(mu, f["sigma"], lo, hi) / len(lv)),
                      med_abs_finite=float(np.median(np.abs(mu[fin] - 0.5 * (lo[fin] + hi[fin])))) if fin.any() else None,
                      n_finite=int(fin.sum()), hit=float(hit), e_v_pred_own=ex["e_v"], e_beta_pred_own=ex["e_beta"],
                      e_v_pred_frozen_4_12N=hf.get("e_v_4_12N"), e_beta_pred_frozen_4_12N=hf.get("e_beta_4_12N"),
                      e_v_err=ex["e_v"] - free["e_v"], e_beta_err=ex["e_beta"] - free["e_beta"])
    verdict = {}
    for th in ("T1k_break", "T1mu", "T1k_peak"):
        d = abs(out[th]["level_error"]) - abs(out["S1_strict"]["level_error"])
        verdict[th + "_vs_S1_level"] = "not distinguishable" if abs(d) < 0.03 else (th if d < 0 else "S1_strict")
        verdict[th + "_vs_S1_nll"] = out[th]["nll_per_col"] - out["S1_strict"]["nll_per_col"]
    return out, verdict


def check_frozen_grid(F):
    """Label-free code identity: thermal_law2.predict reproduces FROZEN_v2.high_force Fmax_N grids."""
    worst = 0.0
    for g, models in F["high_force"].items():
        lz = np.log(F["z_published"][g][1])
        for m, d in models.items():
            th = F["fits"][m]["theta"]
            for vs, row in d["Fmax_N"].items():
                for bs, val in row.items():
                    pred = float(TL2.predict(m, th, np.array([np.log(float(vs))]), np.array([np.log(float(bs))]), np.array([lz]))[0])
                    worst = max(worst, abs(pred - np.log(val)))
    assert worst < 1e-6, worst
    return worst


# ---------------------------------------------------------------- I/O
def verify_inputs():
    info = {}
    for arch in ("HighBowForceRange_1", "HighBowForceRange_2"):
        rec = json.loads((STREAM / arch / "receipt.json").read_text())
        assert rec["status"] == "validated" and rec["validation"]["trials"] == 9000, arch
        h = sha(STREAM / arch / "features.csv.gz")
        assert h == rec["outputs"]["features.csv.gz"], (arch, "features hash")
        info[arch] = dict(features_sha256=h, receipt_sha256=sha(STREAM / arch / "receipt.json"))
    for p in FROZEN_FILES:
        assert sha(p) == (p.parent / (p.name + ".sha256")).read_text().split()[0], p
    return info


def read_labels(arch, rm, applied):
    """Parse one archive; returns per-target window dicts. (Reads classifier columns: only after the claim.)"""
    per = {t: dict(win=defaultdict(dict), beta={}, speed=defaultdict(list)) for t, (a, _, _) in TARGETS.items() if a == arch}
    with gzip.open(STREAM / arch / "features.csv.gz", "rt") as fh:
        for r in csv.DictReader(fh):
            if r["error"] or r["window_s"] != "0.2":
                continue
            tgt = next((t for t in per if r["folder"].startswith(TARGETS[t][1])), None)
            assert tgt is not None, r["folder"]
            t = round(float(r["t"]), 2)
            if t not in TIMES:
                continue
            fpp = r["flybacks_per_period"]
            lab = rm.classify(float(r["amplitude_std"]), float(r["periodicity"]), float(r["f0_hz"]),
                              int(float(fpp)) if fpp not in ("", "nan", "None") else None, 98.08, applied)
            k = (r["folder"], int(r["trial"]))
            per[tgt]["win"][k][t] = (float(r["force"]), lab == "helmholtz")
            per[tgt]["beta"][k] = float(r["beta"])
            per[tgt]["speed"][r["folder"]].append(float(r["speed"]))
    for t in per:
        per[t]["speed"] = {f: float(np.median(v)) for f, v in per[t]["speed"].items()}
    return per


def speed_category(folder, v):
    leaf = folder.rstrip("/").split("/")[-1]
    j = {"v1": 0, "v2": 1, "v3": 2}[leaf]
    assert abs(v - (0.05, 0.10, 0.20)[j]) < 0.005, (folder, v)
    return j


def score():
    rec = json.loads(RECEIPT.read_text())
    for p in CODE_FILES + FROZEN_FILES:
        assert sha(p) == rec["sha256"][str(p)], ("changed since freeze", p)
    inputs = verify_inputs()
    assert inputs == rec["inputs"], "input hashes changed since freeze"
    F = json.loads(FROZEN_FILES[0].read_text())
    check_frozen_grid(F)
    write_excl(CLAIM, dict(claimed_kst=now(), receipt_sha256=sha(RECEIPT), note="labels parsed only after this claim; one run"))
    sys.path.insert(0, str(HOME / "violin-rne/claude_extra"))
    import verify_astra_boundaries as V
    data = {}
    for arch in ("HighBowForceRange_1", "HighBowForceRange_2"):
        data.update(read_labels(arch, V.rm, V.APPLIED))
    res = dict(started_kst=now(), protocol_sha256=sha(HERE / "PROTOCOL.md"), frozen_v2_sha256=sha(FROZEN_FILES[0]),
               frozen_v3_sha256=sha(FROZEN_FILES[1]), inputs=inputs, targets={})
    pooled = defaultdict(list)
    for tgt, (arch, prefix, grp) in TARGETS.items():
        d = data[tgt]
        lzv = float(np.log(F["z_published"][grp][1]))
        R = dict(group=grp, Z=F["z_published"][grp][1], n_strokes=len(d["beta"]))
        for t in TIMES:
            cols = columns_from_windows(d["win"], d["beta"], d["speed"], t)
            rows = column_table(cols)
            desc = defaultdict(lambda: defaultdict(int))
            for r in rows:
                v = f"{np.exp(r['lv']):.2f}"
                u = r["upper"]
                desc[v]["upper_finite" if np.all(np.isfinite(u)) else "upper_right_censored" if np.isfinite(u[0]) else "no_band"] += 1
                desc[v]["multi_band"] += int(r["n_bands"] >= 2)
                lo_ = r["lower"]
                desc[v]["lower_finite" if np.all(np.isfinite(lo_)) else "lower_left_censored" if np.isfinite(lo_[1]) and not np.isfinite(lo_[0]) else "lower_other"] += 1
            nb_beta = [float(np.exp(r["lb"])) for r in rows if r["n_bands"] == 0]
            lv, lb, lz, lo, hi, keep = upper_arrays(rows, lzv)
            if t != T_EVAL:
                R[f"t{t}"] = dict(descriptive=desc, n_columns=len(lv), P1_point=dict(zip(("c", "e_v", "e_beta"), map(float, free_fit(lv, lb, lo, hi)[:3])))
                                  if len(lv) >= 5 else None)
                continue
            free = p1(lv, lb, lo, hi)
            m2, verdict = p2(F, grp, lv, lb, lz, lo, hi, free)
            rows_s1 = column_table(cols, noband_as_left=True)
            lv1, lb1, lz1, lo1, hi1, _ = upper_arrays(rows_s1, lzv)
            free_s1 = p1(lv1, lb1, lo1, hi1)
            R["t0.2"] = dict(descriptive=desc, no_band_beta_values=nb_beta, P1=free, P2=m2, P2_verdict=verdict,
                             level_A_observed=TL.offset_fit(BASE_A[0] + BASE_A[1] * lv + BASE_A[2] * lb, lo, hi)[0],
                             S1_P1=free_s1, S1_changes_P1_zone=(free_s1["decision"] != free["decision"]))
            keep_s1 = [r for r in rows_s1 if np.isfinite(r["upper"][0]) or np.isfinite(r["upper"][1])]
            for tag, rr, arrs in (("primary", keep, (lv, lb, lo, hi)), ("S1", keep_s1, (lv1, lb1, lo1, hi1))):
                vc = np.array([speed_category(r["folder"], np.exp(r["lv"])) for r in rr])
                pooled[tag].append((vc,) + arrs + (np.full(len(rr), tgt),))
        res["targets"][tgt] = R
    for tag, parts in pooled.items():
        vcat = np.concatenate([p[0] for p in parts]); lv = np.concatenate([p[1] for p in parts])
        lb = np.concatenate([p[2] for p in parts]); lo = np.concatenate([p[3] for p in parts]); hi = np.concatenate([p[4] for p in parts])
        tg = np.concatenate([p[5] for p in parts])
        lvcat = np.array([lv[vcat == j].mean() for j in range(3)])
        res["P3" if tag == "primary" else "S1_P3"] = p3(vcat, lvcat, lb, lo, hi, tg)
    res["finished_kst"] = now()
    write_excl(RESULT, res)
    return res


def freeze():
    inputs = verify_inputs()
    F = json.loads(FROZEN_FILES[0].read_text())
    worst = check_frozen_grid(F)
    write_excl(RECEIPT, dict(frozen_kst=now(), sha256={str(p): sha(p) for p in CODE_FILES + FROZEN_FILES}, inputs=inputs,
                             frozen_grid_identity_max_abs_log=worst,
                             note="label-free freeze: no classifier column of HighBowForceRange_1/2 parsed before this receipt"))
    print("frozen", RECEIPT, "grid identity", worst)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--freeze", action="store_true")
    ap.add_argument("--score", action="store_true")
    a = ap.parse_args()
    if a.freeze:
        freeze()
    elif a.score:
        try:
            r = score()
        except Exception:
            import traceback
            if CLAIM.exists() and not RESULT.exists():
                write_excl(HERE / "score_failure.json", dict(failed_kst=now(), traceback=traceback.format_exc()))
            raise
        print(json.dumps({t: dict(P1=x["t0.2"]["P1"], S1=x["t0.2"]["S1_P1"]["decision"], verdict=x["t0.2"]["P2_verdict"])
                          for t, x in r["targets"].items()}, indent=1, default=float))
        print(json.dumps(dict(P3=r["P3"], S1_P3=r["S1_P3"]), indent=1, default=float))
