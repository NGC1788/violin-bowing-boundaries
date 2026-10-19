"""Freeze upper-boundary predictions of the thermal Schelleng law (and comparators) for UNSEEN archives.
(Claude, 2026-10-19). Uses only the 11 processed diagrams. Never reads features of A3, A4, D2, B_T2 sample2,
HighBowForceRange_1/2 or coldwarm. Usage: python freeze.py <version_tag>
Writes FROZEN_<tag>.json (+ .sha256) with code hashes, fitted parameters and the scoring rule."""
import datetime as dt
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

import rows as R
import thermal_law as TL

HERE = Path(__file__).parent
VER = sys.argv[1] if len(sys.argv) > 1 else "v1"
TARGETS = {  # archive stem -> (string group for Z, design donors = processed diagrams of the same design/type)
    "2024-04-22_TypeA_sample3": ("A T1", ["A1", "A2"]),
    "2024-04-24_TypeA_sample4": ("A T1", ["A1", "A2"]),
    "2024-04-26_TypeD_sample2": ("D T1", ["D1"]),
    "2024-07-03_TypeB_T2_sample2": ("B T2", ["B_T2s1"]),
}
HF_BETA = [0.02, 0.03, 0.04, 0.05, 0.06, 0.07]
HF_V = [0.05, 0.10, 0.15, 0.20]
HF_Z = {"B T2": 1.1899, "A T2": 1.1884, "D T2": 1.2021, "A T1": 1.0592}


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def free_fit(rows):
    lv, lb, lz, lo, hi = TL.arrays(rows, R.ZPUB)
    from scipy.optimize import minimize
    X = np.column_stack([np.ones_like(lv), lv, lb])
    f = lambda p: TL.censored_nll(X @ p[:3], np.exp(p[3]), lo, hi)
    r = min((minimize(f, x0, method="Nelder-Mead", options={"maxiter": 40000, "xatol": 1e-8, "fatol": 1e-10})
             for x0 in ([0.0, 0.7, -1.0, np.log(0.3)], [0.5, 0.5, -1.2, np.log(0.3)])), key=lambda r: r.fun)
    return dict(c=float(r.x[0]), e_v=float(r.x[1]), e_beta=float(r.x[2]), sigma=float(np.exp(r.x[3])))


def level_A(rows):
    lv, lb, lz, lo, hi = TL.arrays(rows, R.ZPUB)
    return TL.offset_fit(R.base_A(lv, lb), lo, hi)[0]


def main():
    RW = R.all_rows()
    allr = [r for t in R.TAGS for r in RW[t]]
    obs = {t: dict(level_A=level_A(RW[t]), **free_fit(RW[t])) for t in R.TAGS}
    fits = {}
    for name in TL.MODELS:
        th, ls, nll = TL.fit(name, allr, R.ZPUB)
        # in-sample level residual per diagram (model's level error) and its between-diagram RMS
        res = {}
        for t in R.TAGS:
            lv, lb, lz, lo, hi = TL.arrays(RW[t], R.ZPUB)
            res[t] = TL.offset_fit(TL.predict(name, th, lv, lb, lz), lo, hi)[0]
        rms = float(np.sqrt(np.mean(np.square(list(res.values())))))
        fits[name] = dict(theta=[float(x) for x in th], sigma=float(np.exp(ls)), nll=nll, n_free=TL.MODELS[name][0],
                          mu_s=TL.mu_s_of(name, th), level_residual_in_sample=res, level_rms_in_sample=rms)
        print(f"{name:14s} theta {np.round(th, 4)} sigma {np.exp(ls):.3f} nll {nll:.1f} | level RMS (in-sample) {rms:.3f}", flush=True)

    out = dict(version=VER, created_kst=dt.datetime.now(dt.timezone(dt.timedelta(hours=9))).isoformat(),
               status="UNSEEN targets: no feature/label of any target was opened before this freeze.",
               t_eval=R.T, K=2, training_tags=R.TAGS, z_published=R.PROP, base_A=R.BASE_A,
               friction_law=dict(A_MU=TL.A_MU, A_EXP=TL.A_EXP, MU_PEAK=TL.MU_PEAK, MU_BREAK=TL.MU_BREAK, DMU_STEADY=TL.DMU_STEADY),
               observed_training=obs, fits=fits, targets={}, high_force={})
    for stem, (grp, donors) in TARGETS.items():
        z = R.PROP[grp][1]
        des = [r for d in donors for r in RW[d]]
        lv = np.array([r["lv"] for r in des]); lb = np.array([r["lb"] for r in des]); lz = np.full(len(des), np.log(z))
        tm = dict(level_A=float(np.mean([obs[d]["level_A"] for d in donors])),
                  e_v=float(np.mean([obs[d]["e_v"] for d in donors])), e_beta=float(np.mean([obs[d]["e_beta"] for d in donors])))
        pm = {}
        for name, f in fits.items():
            y = TL.predict(name, f["theta"], lv, lb, lz)
            ex = TL.design_exponents(name, f["theta"], lv, lb, lz)
            sp = {f"{np.exp(s):.2f}": float(np.mean(y[np.isclose(lv, s)])) for s in sorted(set(np.round(lv, 6)))}
            ref = float(TL.predict(name, f["theta"], np.log([0.1]), np.log([0.1]), np.log([z]))[0])
            pi90 = 1.645 * f["level_rms_in_sample"]
            pm[name] = {"level_A": float(np.mean(y - R.base_A(lv, lb))), "level_A_pi90": pi90, "e_v": ex["e_v"],
                        "e_beta": ex["e_beta"], "mean_logF_by_speed": sp, "logF_v0.1_b0.1": ref}
        out["targets"][stem] = dict(group=grp, Z=z, design_donors=donors, type_mean_baseline=tm, models=pm)
    for grp, z in HF_Z.items():
        tab = {}
        for name, f in fits.items():
            g = {}
            for v in HF_V:
                lb = np.log(HF_BETA); lv = np.full(len(HF_BETA), np.log(v)); lz = np.full(len(HF_BETA), np.log(z))
                y = TL.predict(name, f["theta"], lv, lb, lz)
                g[f"{v:.2f}"] = {f"{b:.2f}": float(np.exp(yy)) for b, yy in zip(HF_BETA, y)}
            LV, LB = np.meshgrid(np.log(HF_V), np.log(HF_BETA))
            ex = TL.design_exponents(name, f["theta"], LV.ravel(), LB.ravel(), np.full(LV.size, np.log(z)))
            tab[name] = dict(Fmax_N=g, e_v=ex["e_v"], e_beta=ex["e_beta"])
        out["high_force"][grp] = tab
    out["scoring_rule"] = (
        "For each target diagram (t 0.2 s, K 2, measured window force, same intervals() as final/compare.py): "
        "(1) model level error = interval-censored single-offset fit of the target's upper columns with the frozen model "
        "as base (thermal_law.offset_fit), 0 = perfect; primary comparison T1k_break / T1k_peak / T1mu vs S1_strict "
        "(|error| smaller wins; differences < 0.03 log = 'not distinguishable', the A1-A2/B_T2 rep-rep yardstick). "
        "(2) level in the established statistic (offset vs A1+A2 no-Z law, base_A) compared to frozen level_A and its 90 % PI. "
        "(3) speed and beta exponents: the target's free censored fit (free_fit) vs the frozen model exponents over the "
        "target's own design (design_exponents evaluated at the target's (v, beta) columns; the frozen numbers here use the "
        "donor design). High-force archives: the frozen Fmax_N grid is scored only where the measured upper boundary is "
        "bracketed (finite interval) between 4 and 12 N; columns right-censored at the ceiling count as hits if the "
        "prediction lies above the ceiling. Versioning: a target is scored against the latest freeze whose created_kst "
        "precedes the target's processing start in timeres/replicate.log; A3 (processing started 10:10:53) is scored "
        "against the first freeze written after 10:10:53 only if that freeze precedes the A3 'done' line; otherwise A3 "
        "is reported as NOT pre-registered.")
    p = HERE / f"FROZEN_{VER}.json"
    out["code_sha256"] = {f: sha(HERE / f) for f in ("thermal_law.py", "rows.py", "freeze.py")}
    txt = json.dumps(out, indent=1, ensure_ascii=False, default=float)
    p.write_text(txt)
    h = hashlib.sha256(txt.encode()).hexdigest()
    (HERE / f"FROZEN_{VER}.json.sha256").write_text(f"{h}  FROZEN_{VER}.json\n" + "\n".join(f"{v}  {k}" for k, v in out["code_sha256"].items()) + "\n")
    print("wrote", p, h, out["created_kst"])
    for stem, d in out["targets"].items():
        print(stem, "type-mean", {k: round(v, 3) for k, v in d["type_mean_baseline"].items()})
        for name in ("S1_strict", "T1k_peak", "T1k_break", "T1mu", "T2_mu_k", "T0_break", "E3_free_Z"):
            m = d["models"][name]
            print(f"   {name:12s} level_A {m['level_A']:+.3f} ±{m['level_A_pi90']:.3f}  e_v {m['e_v']:+.2f} e_b {m['e_beta']:+.2f}")


if __name__ == "__main__":
    main()
