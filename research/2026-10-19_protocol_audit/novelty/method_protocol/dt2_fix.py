"""POST HOC repair of the D_T2 lower primary fit (failed: invalid_or_zero_width_interval).
The pre-registered verdict stays '검증 불가'. This script only shows what the frozen lower predictions would have
scored under clearly labelled post hoc likelihood repairs. Claude, 2026-10-19."""
import json
import numpy as np
from scipy.optimize import minimize

import common as C

FROZEN = json.loads((C.BASE / "astra_20261019/d_t2_freeze/frozen_parameters.json").read_text())
CTRL = json.loads((C.BASE / "astra_20261019/D_T2_controls_frozen.json").read_text())
RES = json.loads((C.BASE / "astra_20261019/D_T2_confirm_result.json").read_text())
c, a, b, _ = FROZEN["lower_reference_shape_c_v_beta_logsigma"]
PRED = CTRL["predictions"]["lower_levels"]          # frozen before scoring (protocol, mean, log_Z, log_EI, log_Z_EI)
TOL = FROZEN["thresholds_log"]["lower_tolerance"]   # 0.23


def nm_single(rows, mu0):
    """Replica of the frozen executor: one Nelder-Mead start [0, log .5], scipy defaults."""
    lo, hi, ok, bad = C.check_rows(rows)
    if bad.any():
        return dict(success=False, reason="invalid_or_zero_width_interval")
    lo, hi, m = lo[ok], hi[ok], mu0[ok]
    f = lambda p: -C.log_int_prob(lo, hi, m + p[0], np.exp(np.clip(p[1], -50, 50))).sum()
    r = minimize(f, [0.0, np.log(.5)], method="Nelder-Mead")
    return dict(success=bool(r.success and np.isfinite(r.fun)), level=float(r.x[0]), sigma=float(np.exp(r.x[1])))


def robust(rows, mu0, pi=0.02):
    """Mixture likelihood: (1-pi) interval-normal + pi x (share of log-force range [log .005, log 5] covered)."""
    lo, hi, ok, bad = C.check_rows(rows)
    lo, hi, m = lo[ok], hi[ok], mu0[ok]
    L, U = np.log(.005), np.log(5.0)
    cover = (np.minimum(hi, U) - np.maximum(lo, L)).clip(0) / (U - L)
    def f(p):
        with np.errstate(divide="ignore"):
            lp = C.log_int_prob(lo, hi, m + p[0], np.exp(p[1]))
        return -np.log((1 - pi) * np.exp(lp) + pi * cover + 1e-300).sum()
    r = min((minimize(f, [0.0, np.log(s)], method="Nelder-Mead") for s in (.15, .5, 1.5)), key=lambda r: r.fun)
    return dict(success=bool(r.success), level=float(r.x[0]), sigma=float(np.exp(r.x[1])))


def boot(rows, mu0, fitter, n=2000, seed=20261019):
    rng = np.random.default_rng(seed)
    by = {}
    for i, r in enumerate(rows):
        by.setdefault(r["column"], []).append(i)
    cols = sorted(by)
    vals, fails = [], 0
    for _ in range(n):
        idx = [i for cc in rng.integers(0, len(cols), len(cols)) for i in by[cols[cc]]]
        out = fitter([rows[i] for i in idx], mu0[idx])
        if out.get("success"):
            vals.append(out["level"])
        else:
            fails += 1
    return dict(successful=len(vals), failed=fails,
                p95=[float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))] if vals else None)


def main():
    d = C.load_exposed("D_T2")
    legacy = C.intervals(d, "F_ev", "H2", "none")
    # independent-loader check against the scored executor's observed lower intervals
    obs = RES["observed_boundaries"]
    tofl = lambda x: float(x) if not isinstance(x, str) else float(x.replace("+inf", "inf").replace("-inf", "-inf"))
    diff = max(abs(tofl(o["lower"][k]) - r[("lo", "hi")[k]]) if np.isfinite(r[("lo", "hi")[k]]) else
               (0.0 if not np.isfinite(tofl(o["lower"][k])) else np.inf) for o, r in zip(obs, legacy) for k in (0, 1))
    bad = [(r["folder"], r["column"], r["lo"], r["hi"], r["s"]) for r in legacy if not r["lo"] < r["hi"]]
    st = d["F_ev"][legacy[[i for i, r in enumerate(legacy) if not r["lo"] < r["hi"]][0]]["strokes"]][:4]
    mu = lambda rows: np.array([c + a * r["lv"] + b * r["lb"] for r in rows])
    out = dict(status="POST HOC; pre-registered D_T2 lower verdict remains '검증 불가' (invalid_or_zero_width_interval)",
               loader_max_abs_diff_vs_executor=float(diff), zero_width_columns=bad,
               zero_width_column_lowest_measured_forces_N=st.tolist(), frozen_predictions=PRED, tolerance=TOL, fixes={})
    variants = {"legacy (as executed)": ("none", None), "drop zero-width column": ("drop", None),
                "floor-censor (<=1 mN -> -inf)": ("floor", None), "delta 0.005 N": ("delta", .005),
                "delta 0.01 N (PACE default)": ("delta", .01), "delta 0.02 N": ("delta", .02)}
    for name, (fix, dl) in variants.items():
        rows = C.intervals(d, "F_ev", "H2", fix, dl or 0.01)
        m0 = mu(rows)
        f1 = nm_single(rows, m0)
        f2 = C.fit_offset(rows, m0)
        rec = dict(nm_single_start=f1, lbfgs_multistart={k: v for k, v in f2.items()})
        if f1.get("success"):
            lev = f1["level"]
            err = {k: abs(lev - v) for k, v in PRED.items()}
            rec["abs_error_vs_frozen_prediction"] = err
            rec["within_0.23"] = {k: bool(e <= TOL) for k, e in err.items()}
            rec["protocol_minus_mean_gap"] = abs(err["protocol"] - err["mean"])
            rec["protocol_minus_logZ_EI_gap"] = abs(err["protocol"] - err["log_Z_EI"])
            rec["distinguishable_at_0.23"] = bool(rec["protocol_minus_mean_gap"] >= TOL)
        if fix in ("drop", "delta") and (dl in (None, .01)):
            rec["bootstrap_nm_single"] = boot(rows, m0, nm_single)
        out["fixes"][name] = rec
        print(name, json.dumps({k: v for k, v in rec.items() if k != "lbfgs_multistart"}, default=float)[:600], flush=True)
    rows = C.intervals(d, "F_ev", "H2", "delta", .01)
    out["fixes"]["delta 0.01 N + robust mixture pi=.02"] = robust(rows, mu(rows))
    print("robust", out["fixes"]["delta 0.01 N + robust mixture pi=.02"])
    out["code_sha256"] = {f: C.sha(C.HERE / f) for f in ("common.py", "dt2_fix.py")}
    json.dump(out, open(C.HERE / "dt2_fix_results.json", "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
