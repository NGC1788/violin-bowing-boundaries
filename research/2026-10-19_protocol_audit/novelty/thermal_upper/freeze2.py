"""Freeze v2 (Claude, 2026-10-19): adds post hoc models of thermal_law2.py, LODO/LOSO-based 90 % prediction intervals
(from analysis.json, computed on the 11 processed diagrams only), a corrected secondary level statistic, and explicit
high-force hypotheses. Never opens features of A3, A4, D2, B_T2 sample2, HighBowForceRange_1/2, coldwarm.
A3 stays governed by FROZEN_v1 (its processing had started). Usage: python freeze2.py v2"""
import datetime as dt
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

import rows as R
import thermal_law as TL
import thermal_law2 as TL2

HERE = Path(__file__).parent
VER = sys.argv[1] if len(sys.argv) > 1 else "v2"
TARGETS = {
    "2024-04-22_TypeA_sample3": ("A T1", ["A1", "A2"], "governed by FROZEN_v1 (processing started 10:10:53); v2 values informational only"),
    "2024-04-24_TypeA_sample4": ("A T1", ["A1", "A2"], "pre-registered by this freeze if created before its processing start"),
    "2024-04-26_TypeD_sample2": ("D T1", ["D1"], "pre-registered by this freeze if created before its processing start"),
    "2024-07-03_TypeB_T2_sample2": ("B T2", ["B_T2s1"], "CAUTION: friction-reconstruction strokes (session 030724 = 2024-07-03, Helicore T2) "
                                    "were seen and the friction law was fitted on them; likely same string/day -> partially contaminated"),
}
MODELS = ["S1_strict", "T1k_peak", "T1k_break", "T1mu", "T2_mu_k", "T0_break", "E3_free_Z", "T2km_peak", "T2km_break", "S2m"]
HF_BETA = [0.02, 0.025, 0.03, 0.04, 0.05, 0.06, 0.07]
HF_V = [0.05, 0.10, 0.15, 0.20]
HF_Z = {"B T2": 1.1899, "A T2": 1.1884, "C T2": 1.1154, "D T2": 1.2021, "A T1": 1.0592, "B T1": 1.0998}
# friction-reconstruction brackets (B T2 Helicore, session 030724; steady windows; ALREADY SEEN, friction_offline.log [D])
SEEN_BRACKETS = [(0.0405, 0.05, 5.26, 6.70), (0.1044, 0.05, 1.35, 2.50), (0.1044, 0.10, 2.66, 3.33), (0.1175, 0.10, 1.52, 2.64)]
SEEN_LOWER_BOUNDS = [(0.0405, 0.10, 6.72), (0.0484, 0.10, 2.92), (0.0484, 0.20, 3.05), (0.0613, 0.10, 4.70), (0.0777, 0.10, 4.17)]


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def fit(name, rows):
    n, init, f = TL2.MODELS[name]
    lv, lb, lz, lo, hi = TL.arrays(rows, R.ZPUB)
    nll = lambda p: TL.censored_nll(f(p[:n], lv, lb, lz), np.exp(p[n]), lo, hi)
    best = None
    for d in (0.0, 0.3, -0.3):
        x0 = np.r_[np.array(init, float) + d, np.log(0.3)] if n else np.r_[np.log(0.3)]
        r = minimize(nll, x0, method="Nelder-Mead", options={"maxiter": 20000, "xatol": 1e-7, "fatol": 1e-9})
        best = r if best is None or r.fun < best.fun else best
        if n == 0:
            break
    return best.x[:n], float(np.exp(best.x[n])), float(best.fun)


def main():
    A = json.loads((HERE / "analysis.json").read_text())
    RW = R.all_rows()
    allr = [r for t in R.TAGS for r in RW[t]]
    lvlA = A["level_A"]
    obs = A["per_diagram"]
    fits = {}
    for m in MODELS:
        th, s, nll = fit(m, allr)
        res = {}
        for t in R.TAGS:
            lv, lb, lz, lo, hi = TL.arrays(RW[t], R.ZPUB)
            res[t] = TL.offset_fit(TL2.predict(m, th, lv, lb, lz), lo, hi)[0]
        rms_in = float(np.sqrt(np.mean(np.square(list(res.values())))))
        if m in A["LOSO_summary"]:
            pil = 1.645 * A["LOSO_summary"][m]["rms_level"]
            piv = 1.645 * A["LODO_summary"][m]["rms_err_e_v"]
            pib = 1.645 * A["LODO_summary"][m]["rms_err_e_beta"]
            src = "LOSO RMS level error / LODO RMS exponent errors (analysis.json)"
        else:  # post hoc models not cross-validated at freeze time: in-sample RMS x 1.15, E3 exponent errors
            pil = 1.645 * 1.15 * rms_in
            piv = 1.645 * A["LODO_summary"]["E3_free_Z"]["rms_err_e_v"]
            pib = 1.645 * A["LODO_summary"]["E3_free_Z"]["rms_err_e_beta"]
            src = "in-sample RMS x1.15 (no CV yet); exponent PI borrowed from E3_free_Z LODO"
        fits[m] = dict(theta=[float(x) for x in th], sigma=s, nll=nll, n_free=TL2.MODELS[m][0], mu_s=TL2.mu_s_of(m, th),
                       level_residual_in_sample=res, level_rms_in_sample=rms_in, pi90=dict(level=pil, e_v=piv, e_beta=pib, source=src))
        print(f"{m:12s} theta {np.round(th, 4)} sigma {s:.3f} nll {nll:.1f} | in-sample level RMS {rms_in:.3f} | PI90 level ±{pil:.3f} e_v ±{piv:.2f} e_b ±{pib:.2f}", flush=True)

    out = dict(version=VER, created_kst=dt.datetime.now(dt.timezone(dt.timedelta(hours=9))).isoformat(),
               status="UNSEEN targets: no feature/label of A3, A4, D2, B_T2 sample2, HighBowForceRange_1/2 or coldwarm was opened.",
               supersedes="FROZEN_v1 for targets whose processing starts after this freeze; v1 secondary 'level_A' (plain mean of model-base "
                          "over donor columns) is NOT comparable to the censored level statistic and is withdrawn from scoring (never scored).",
               t_eval=R.T, K=2, training_tags=R.TAGS, z_published=R.PROP, base_A=R.BASE_A,
               friction_law=dict(A_MU=TL.A_MU, A_EXP=TL.A_EXP, MU_PEAK=TL.MU_PEAK, MU_BREAK=TL.MU_BREAK),
               fits=fits, targets={}, high_force={}, seen_friction_session_check={})
    for stem, (grp, donors, note) in TARGETS.items():
        z = R.PROP[grp][1]
        des = [r for d in donors for r in RW[d]]
        lv = np.array([r["lv"] for r in des]); lb = np.array([r["lb"] for r in des])
        zd = np.array([np.log(R.ZPUB[r["tag"]]) for r in des]); zt = np.full(len(des), np.log(z))
        donor_level = float(np.mean([lvlA[d] for d in donors]))
        tm = dict(level_A=donor_level, e_v=float(np.mean([obs[d]["e_v"] for d in donors])), e_beta=float(np.mean([obs[d]["e_beta"] for d in donors])))
        pm = {}
        for m, f in fits.items():
            ex = TL2.design_exponents(m, f["theta"], lv, lb, zt)
            shift = float(np.mean(TL2.predict(m, f["theta"], lv, lb, zt) - TL2.predict(m, f["theta"], lv, lb, zd)))
            donor_err = float(np.mean([f["level_residual_in_sample"][d] for d in donors]))
            pm[m] = {"model_level_error_expected": 0.0, "model_level_error_pi90": f["pi90"]["level"],
                     "donor_in_sample_level_error": donor_err,
                     "level_A_pred": donor_level + shift, "e_v": ex["e_v"], "e_v_pi90": f["pi90"]["e_v"],
                     "e_beta": ex["e_beta"], "e_beta_pi90": f["pi90"]["e_beta"],
                     "logF_v0.1_b0.1": float(TL2.predict(m, f["theta"], np.log([0.1]), np.log([0.1]), np.log([z]))[0])}
        out["targets"][stem] = dict(group=grp, Z=z, design_donors=donors, note=note, type_mean_baseline=tm, models=pm)
    for grp, z in HF_Z.items():
        tab = {}
        LV, LB = np.meshgrid(np.log(HF_V), np.log(HF_BETA))
        for m, f in fits.items():
            y = TL2.predict(m, f["theta"], LV.ravel(), LB.ravel(), np.full(LV.size, np.log(z))).reshape(LV.shape)
            g = {f"{v:.2f}": {f"{b:.3f}": float(np.exp(y[j, i])) for j, b in enumerate(HF_BETA)} for i, v in enumerate(HF_V)}
            ex = TL2.design_exponents(m, f["theta"], LV.ravel(), LB.ravel(), np.full(LV.size, np.log(z)))
            # exponents restricted to the grid points whose prediction lies in the 4-12 N window
            ins = (np.exp(y.ravel()) >= 4) & (np.exp(y.ravel()) <= 12)
            exw = TL2.design_exponents(m, f["theta"], LV.ravel()[ins], LB.ravel()[ins], np.full(ins.sum(), np.log(z))) if ins.sum() >= 4 else None
            tab[m] = dict(Fmax_N=g, e_v_grid=ex["e_v"], e_beta_grid=ex["e_beta"], n_grid_in_4_12N=int(ins.sum()),
                          e_v_4_12N=None if exw is None else exw["e_v"], e_beta_4_12N=None if exw is None else exw["e_beta"])
        out["high_force"][grp] = tab
    # already-seen same-string brackets (consistency only, NOT a test)
    z = HF_Z["B T2"]
    for m, f in fits.items():
        rr = []
        for b, v, lo_, hi_ in SEEN_BRACKETS:
            p = float(np.exp(TL2.predict(m, f["theta"], np.log([v]), np.log([b]), np.log([z]))[0]))
            rr.append(dict(beta=b, v=v, bracket=[lo_, hi_], pred=p, inside=bool(lo_ <= p <= hi_), log_miss=float(0 if lo_ <= p <= hi_ else (np.log(lo_ / p) if p < lo_ else np.log(hi_ / p)))))
        lb_ok = []
        for b, v, lo_ in SEEN_LOWER_BOUNDS:
            p = float(np.exp(TL2.predict(m, f["theta"], np.log([v]), np.log([b]), np.log([z]))[0]))
            lb_ok.append(dict(beta=b, v=v, lower_bound=lo_, pred=p, consistent=bool(p >= lo_)))
        out["seen_friction_session_check"][m] = dict(brackets=rr, lower_bounds=lb_ok,
                                                     n_inside=sum(x["inside"] for x in rr), n_lb_consistent=sum(x["consistent"] for x in lb_ok))
    out["hypotheses_high_force"] = {
        "statistic": "free censored fit log F = c + e_v log v + e_beta log beta over the high-force archive's upper-boundary columns "
                     "(t 0.2, K 2, measured window force); level = censored single offset vs each frozen model (B T2 Z 1.1899 unless the "
                     "archive metadata names another string; then that string's published Z).",
        "H_thermal": "e_beta >= -0.95 (T1k_break predicts about -0.92 on the high-force grid; T1k_peak about -0.85)",
        "H_plain": "-1.05 < e_beta < -0.95 (S1_strict -1.00)",
        "H_steep": "e_beta <= -1.05 (E3_free_Z -1.06, T2km/S2m with fitted m)",
        "speed": "thermal T1k_break predicts e_v about +0.87-0.9 on the high-force grid (rises with F v); S1 +1.00; E3 +0.70",
        "decision": "the hypothesis whose interval contains the point estimate is 'supported'; bootstrap 95 % CI reported; "
                    "if the CI spans two hypotheses -> 'not decidable'",
    }
    out["scoring_rule"] = (
        "Primary per target: model level error = thermal_law.offset_fit(target upper columns, base = frozen model); "
        "|error| within model_level_error_pi90 = 'within PI'. Thermal (T1k_break, T1mu) vs plain S1_strict: smaller |error| wins only if the "
        "difference is >= 0.03 log, else 'not distinguishable'. Also held-out mean censored NLL per column with the frozen sigma "
        "(analysis.score). Secondary: level_A (censored offset vs base_A) vs level_A_pred +- model_level_error_pi90; exponents (free_fit) vs "
        "e_v, e_beta (model evaluated over the target's own columns at scoring; values here use the donor design) +- PI90. "
        "Versioning as in FROZEN_v1: a target is scored against the latest freeze created before its processing start in replicate.log.")
    out["code_sha256"] = {f: sha(HERE / f) for f in ("thermal_law.py", "thermal_law2.py", "rows.py", "freeze2.py", "analysis.py", "analysis.json")}
    txt = json.dumps(out, indent=1, ensure_ascii=False, default=float)
    p = HERE / f"FROZEN_{VER}.json"
    p.write_text(txt)
    h = hashlib.sha256(txt.encode()).hexdigest()
    (HERE / f"FROZEN_{VER}.json.sha256").write_text(f"{h}  FROZEN_{VER}.json\n" + "\n".join(f"{v}  {k}" for k, v in out["code_sha256"].items()) + "\n")
    print("wrote", p, h, out["created_kst"])
    for stem, d in out["targets"].items():
        print(stem, "| type-mean", {k: round(v, 3) for k, v in d["type_mean_baseline"].items()})
        for m in MODELS:
            x = d["models"][m]
            print(f"   {m:12s} level_A {x['level_A_pred']:+.3f} ±{x['model_level_error_pi90']:.3f} (donor err {x['donor_in_sample_level_error']:+.3f}) "
                  f"e_v {x['e_v']:+.2f}±{x['e_v_pi90']:.2f} e_b {x['e_beta']:+.2f}±{x['e_beta_pi90']:.2f}")
    print("\nhigh force, B T2 (Fmax N at v 0.05/0.10/0.20; beta 0.02 0.03 0.05 0.07) and exponents")
    for m in MODELS:
        t = out["high_force"]["B T2"][m]
        cells = " | ".join(f"v{v}: " + " ".join(f"{t['Fmax_N'][v][b]:.1f}" for b in ("0.020", "0.030", "0.050", "0.070")) for v in ("0.05", "0.10", "0.20"))
        print(f"  {m:12s} {cells} | grid e_v {t['e_v_grid']:+.2f} e_b {t['e_beta_grid']:+.2f} | 4-12N n {t['n_grid_in_4_12N']} e_v {t['e_v_4_12N']} e_b {t['e_beta_4_12N']}")
    print("\nseen friction-session brackets (B T2 string, consistency only):")
    for m in MODELS:
        c = out["seen_friction_session_check"][m]
        print(f"  {m:12s} inside {c['n_inside']}/4 brackets " + " ".join(f"{x['pred']:.2f}[{x['bracket'][0]}-{x['bracket'][1]}]" for x in c["brackets"]) +
              f" | lower bounds consistent {c['n_lb_consistent']}/5")


if __name__ == "__main__":
    main()
