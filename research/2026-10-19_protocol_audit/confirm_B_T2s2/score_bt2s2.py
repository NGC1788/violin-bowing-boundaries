"""Once-only scorer for B_T2 sample 2 (protocol: PROTOCOL.md). Claude, 2026-10-20.
  python score_bt2s2.py --freeze    label-free hashes
  python score_bt2s2.py --score     claim (O_EXCL) -> parse labels -> score -> score_result.json + bt2s2_v3_rows.npz"""
import argparse
import csv
import datetime as dt
import gzip
import hashlib
import io
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
D = HOME / "violin-rne/astra_20261019/highforce_stream/2024-07-03_TypeB_T2_sample2"
sys.path.insert(0, str(TU))
import thermal_law as TL  # noqa: E402
import thermal_law2 as TL2  # noqa: E402

STEM, GROUP = "2024-07-03_TypeB_T2_sample2", "B T2"
N_COL, PER_COL, N_STROKES, K, T_EVAL = 40, 40, 1600, 2, 0.2
STEP = 0.075
STD_SPEEDS = (0.05, 0.10, 0.20)
NB = 500
FROZEN = TU / "FROZEN_v2.json"
CODE = [HERE / "PROTOCOL.md", Path(__file__).resolve(), HERE / "test_score_bt2s2.py", HERE / "controls_audit.py", HERE / "controls_audit.json",
        FROZEN, TU / "thermal_law.py", TU / "thermal_law2.py", TU / "score_target.py",
        HOME / "violin-rne/claude_extra/verify_astra_boundaries.py", HOME / "violin-bowing-boundaries/scripts/regime_map.py"]
KST = dt.timezone(dt.timedelta(hours=9))


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def now():
    return dt.datetime.now(KST).isoformat()


def write_excl(path, obj):
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    with os.fdopen(fd, "w") as fh:
        json.dump(obj, fh, indent=1, default=float)


def assign_columns(beta):
    col, by = {}, defaultdict(list)
    for k, b in beta.items():
        by[k[0]].append((b, k[1]))
    for folder, items in by.items():
        items.sort()
        assert len(items) == N_STROKES, (folder, len(items))
        lb = np.log([b for b, _ in items]).reshape(N_COL, PER_COL)
        assert np.min(lb[1:, 0] - lb[:-1, -1]) > np.max(lb[:, -1] - lb[:, 0]), folder
        for i, (_, tr) in enumerate(items):
            col[(folder, tr)] = i // PER_COL
    return col


def intervals(f, h, k=K):
    """= final/compare.intervals for one force-sorted column."""
    runs = [i for i in range(len(h) - k + 1) if all(h[i:i + k])]
    if not runs:
        return (f[-1], np.inf), (np.nan, np.nan)
    s, e = runs[0], runs[-1] + k - 1
    lower = (-np.inf, f[0]) if s == 0 else (f[s - 1], f[s])
    upper = (f[-1], np.inf) if e == len(h) - 1 else (f[e], f[e + 1])
    return lower, upper


def columns(win, beta, speed, t):
    col = assign_columns(beta)
    g = defaultdict(list)
    for k, d in win.items():
        if t in d:
            g[(k[0], col[k])].append((d[t][0], d[t][1], beta[k]))
    out = []
    for (folder, c), rows in sorted(g.items()):
        rows.sort()
        f = np.log(np.maximum(np.array([x[0] for x in rows]), 1e-3)); h = [x[1] for x in rows]
        lo_, up = intervals(f, h)
        out.append(dict(folder=folder, column=c, v=speed[folder], lv=float(np.log(speed[folder])),
                        lb=float(np.log(np.median([x[2] for x in rows]))), lower=lo_, upper=up))
    return out


def upper_arrays(cols, speeds=None):
    keep = [c for c in cols if (np.isfinite(c["upper"][0]) or np.isfinite(c["upper"][1]))
            and (speeds is None or min(abs(c["v"] - s) for s in speeds) < 1e-3)]
    a = lambda k: np.array([c[k] for c in keep], float)
    return a("lv"), a("lb"), np.array([c["upper"][0] for c in keep], float), np.array([c["upper"][1] for c in keep], float), keep


def free_fit(lv, lb, lo, hi, x0=None):
    """= score_target.free_fit"""
    X = np.column_stack([np.ones_like(lv), lv, lb])
    f = lambda p: TL.censored_nll(X @ p[:3], np.exp(p[3]), lo, hi)
    x0 = [0.0, 0.7, -1.0, np.log(0.3)] if x0 is None else x0
    return minimize(f, x0, method="Nelder-Mead", options={"maxiter": 40000, "xatol": 1e-8, "fatol": 1e-10}).x


def score_block(F, lv, lb, lo, hi):
    """= score_target.main body (observed fit + per-model scores + verdict), Z of group B T2."""
    lz = np.full(len(lv), np.log(F["z_published"][GROUP][1]))
    p = free_fit(lv, lb, lo, hi)
    rng = np.random.default_rng(2026)
    bs = np.array([free_fit(lv[ii], lb[ii], lo[ii], hi[ii], p) for ii in (rng.integers(0, len(lv), len(lv)) for _ in range(NB))])
    base_A = (-0.1764277369756792, 0.6606815556587429, -1.1107153335353168)
    out = dict(n_columns=int(len(lv)), n_finite=int(np.sum(np.isfinite(lo) & np.isfinite(hi))), n_right=int(np.sum(np.isfinite(lo) & ~np.isfinite(hi))),
               observed=dict(e_v=float(p[1]), e_v_ci=np.percentile(bs[:, 1], [2.5, 97.5]).tolist(), e_beta=float(p[2]),
                             e_beta_ci=np.percentile(bs[:, 2], [2.5, 97.5]).tolist(),
                             level_A=TL.offset_fit(base_A[0] + base_A[1] * lv + base_A[2] * lb, lo, hi)[0]), models={})
    fin = np.isfinite(lo) & np.isfinite(hi)
    for m, f in F["fits"].items():
        if m not in TL2.MODELS:
            continue
        mu = TL2.predict(m, f["theta"], lv, lb, lz)
        hit = np.mean([(q >= a - STEP if np.isfinite(a) else True) and (q <= b + STEP if np.isfinite(b) else True) for q, a, b in zip(mu, lo, hi)])
        ex = TL2.design_exponents(m, f["theta"], lv, lb, lz)
        le = TL.offset_fit(mu, lo, hi)[0]
        out["models"][m] = dict(level_error=le, level_within_pi90=bool(abs(le) <= f["pi90"]["level"]), pi90_level=f["pi90"]["level"],
                                nll_per_col=float(TL.censored_nll(mu, f["sigma"], lo, hi) / len(lv)),
                                med_abs_finite=float(np.median(np.abs(mu[fin] - 0.5 * (lo[fin] + hi[fin])))) if fin.any() else None,
                                hit=float(hit), e_v_pred=ex["e_v"], e_beta_pred=ex["e_beta"],
                                e_v_err=ex["e_v"] - float(p[1]), e_beta_err=ex["e_beta"] - float(p[2]))
    ms, v = out["models"], {}
    for th in ("T1k_break", "T1mu", "T1k_peak"):
        d = abs(ms[th]["level_error"]) - abs(ms["S1_strict"]["level_error"])
        v[th + "_vs_S1_level"] = "not distinguishable" if abs(d) < 0.03 else (th if d < 0 else "S1_strict")
        v[th + "_vs_S1_nll"] = ms[th]["nll_per_col"] - ms["S1_strict"]["nll_per_col"]
    eb, ci = out["observed"]["e_beta"], out["observed"]["e_beta_ci"]
    zone = lambda x: "H_thermal" if x >= -0.95 else ("H_plain" if x > -1.05 else "H_steep")
    v["beta_exponent_zone"] = zone(eb); v["beta_exponent_ci_zones"] = sorted({zone(ci[0]), zone(ci[1]), zone(eb)})
    v["beta_decidable"] = len(v["beta_exponent_ci_zones"]) == 1
    out["verdict"] = v
    return out


def read_labels(rm, applied):
    win, beta, speed = defaultdict(dict), {}, defaultdict(list)
    with gzip.open(D / "features.csv.gz", "rt") as fh:
        for r in csv.DictReader(fh):
            if r["error"] or r["window_s"] != "0.2":
                continue
            t = round(float(r["t"]), 2)
            if t not in (0.0, 0.1, 0.2, 0.3):
                continue
            fpp = r["flybacks_per_period"]
            lab = rm.classify(float(r["amplitude_std"]), float(r["periodicity"]), float(r["f0_hz"]),
                              int(float(fpp)) if fpp not in ("", "nan", "None") else None, 98.08, applied)
            k = (r["folder"], int(r["trial"]))
            win[k][t] = (float(r["force"]), lab == "helmholtz"); beta[k] = float(r["beta"]); speed[r["folder"]].append(float(r["speed"]))
    return win, beta, {f: float(np.median(v)) for f, v in speed.items()}


def inputs():
    rec = json.loads((D / "receipt.json").read_text())
    assert rec["status"] == "validated" and sha(D / "features.csv.gz") == rec["outputs"]["features.csv.gz"]
    assert sha(FROZEN) == (TU / "FROZEN_v2.json.sha256").read_text().split()[0]
    return dict(features_sha256=rec["outputs"]["features.csv.gz"], receipt_sha256=sha(D / "receipt.json"))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--freeze", action="store_true"); ap.add_argument("--score", action="store_true")
    a = ap.parse_args()
    if a.freeze:
        write_excl(HERE / "FREEZE_RECEIPT.json", dict(frozen_kst=now(), sha256={str(p): sha(p) for p in CODE}, inputs=inputs(),
                                                      note="label-free freeze; no classifier column parsed by this code"))
        print("frozen"); return
    rec = json.loads((HERE / "FREEZE_RECEIPT.json").read_text())
    for p in CODE:
        assert sha(p) == rec["sha256"][str(p)], ("changed since freeze", p)
    assert inputs() == rec["inputs"]
    F = json.loads(FROZEN.read_text())
    write_excl(HERE / "SCORE_CLAIM.json", dict(claimed_kst=now(), receipt_sha256=sha(HERE / "FREEZE_RECEIPT.json")))
    sys.path.insert(0, str(HOME / "violin-rne/claude_extra"))
    import verify_astra_boundaries as V
    win, beta, speed = read_labels(V.rm, V.APPLIED)
    cols = columns(win, beta, speed, T_EVAL)
    res = dict(stem=STEM, group=GROUP, Z=F["z_published"][GROUP][1], frozen_v2_sha256=sha(FROZEN), contamination=
               "same-date friction session seen before FROZEN_v2 (4 upper brackets visible); partially contaminated",
               pace_star="not applicable (design mismatch), not scored", speeds=sorted(set(speed.values())))
    lv, lb, lo, hi, keep = upper_arrays(cols)
    res["primary_all_speeds"] = score_block(F, lv, lb, lo, hi)
    lv3, lb3, lo3, hi3, keep3 = upper_arrays(cols, STD_SPEEDS)
    res["secondary_standard_speeds"] = score_block(F, lv3, lb3, lo3, hi3)
    nb = defaultdict(lambda: defaultdict(int))
    for c in cols:
        u = c["upper"]
        nb[f"{c['v']:.2f}"]["upper_finite" if np.all(np.isfinite(u)) else "upper_right_censored" if np.isfinite(u[0]) else "no_band"] += 1
    res["descriptive_by_speed"] = nb
    res["finished_kst"] = now()
    buf = io.BytesIO(); np.savez(buf, lv=lv3, lb=lb3, lo=lo3, hi=hi3); data = buf.getvalue()
    fd = os.open(HERE / "bt2s2_v3_rows.npz", os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    with os.fdopen(fd, "wb") as fh:
        fh.write(data)
    res["v3_rows_sha256"] = hashlib.sha256(data).hexdigest()
    write_excl(HERE / "score_result.json", res)
    for k in ("primary_all_speeds", "secondary_standard_speeds"):
        r = res[k]
        print(k, "n", r["n_columns"], "finite", r["n_finite"], "right", r["n_right"], "obs", {x: (round(y, 3) if isinstance(y, float) else [round(z, 3) for z in y]) for x, y in r["observed"].items()})
        print("  verdict", r["verdict"])
        for m, x in r["models"].items():
            print(f"   {m:11s} lvl {x['level_error']:+.3f} (PI90 {x['pi90_level']:.3f} ok {x['level_within_pi90']}) nll {x['nll_per_col']:.3f} med {x['med_abs_finite']:.3f} hit {x['hit']:.2f}")


if __name__ == "__main__":
    main()
