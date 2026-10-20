"""Session vs string-type variance of boundary levels (post hoc; plan in README.md). Claude, 2026-10-20."""
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.optimize import minimize, minimize_scalar
from scipy.stats import chi2

HOME = Path.home()
TU = HOME / "violin-rne/novelty/thermal_upper"
sys.path.insert(0, str(TU))
import rows as R  # noqa: E402  (imports final/compare.py and final/data.py)
import thermal_law as TL  # noqa: E402
import thermal_law2 as TL2  # noqa: E402

C, D = R.C, R.D
HERE = Path(__file__).resolve().parent
F2 = json.loads((TU / "FROZEN_v2.json").read_text())
EXTRA = {"A3": ("2024-04-22_TypeA_sample3", "A T1"), "A4": ("2024-04-24_TypeA_sample4", "A T1"), "D2": ("2024-04-26_TypeD_sample2", "D T1")}
GROUP = dict(R.GROUP)
for tag, (stem, g) in EXTRA.items():
    t = "TGT_" + stem
    D.STEMS[t] = stem; GROUP[t] = g
SESSIONS = {"A1": ["A1"], "A2": ["A2"], "A3": ["TGT_2024-04-22_TypeA_sample3"], "A4": ["TGT_2024-04-24_TypeA_sample4"], "B1": ["B1"],
            "C1": ["C1"], "D1": ["D1"], "D2": ["TGT_2024-04-26_TypeD_sample2"], "A_T2": ["A_T2"], "B_T2s1": ["B_T2s1"],
            "B_T2p6": ["B_T2r1", "B_T2r2"], "C_T2": ["C_T2"], "D_T2": ["D_T2"]}
SGROUP = {s: GROUP[tags[0]] for s, tags in SESSIONS.items()}


def widen(lo, hi):
    lo, hi = np.array(lo, float), np.array(hi, float)
    z = np.isfinite(lo) & np.isfinite(hi) & (hi - lo < 1e-9)
    F = np.exp(hi[z])
    lo[z] = np.where(F > 0.011, np.log(np.maximum(F - 0.01, 1e-12)), -np.inf); hi[z] = np.log(F + 0.01)
    return lo, hi, int(z.sum())


def rows_of(tag):
    rs = C.intervals(R.load(tag), 0.2)
    for r in rs:
        r["tag"] = tag
    return rs


A12 = rows_of("A1") + rows_of("A2")
LOWER_BASE = C.physics_models(A12, {"A1": 1.0, "A2": 1.0})["P-free"][1][0]   # lower coefficients (c, a, b), sigma last


def lower_mu(rs):
    p = LOWER_BASE
    return np.array([p[0] + p[1] * r["lv"] + p[2] * r["lb"] for r in rs])


def upper_mu(rs, group):
    th = F2["fits"]["S1_strict"]["theta"]; lz = np.log(F2["z_published"][group][1])
    return TL2.predict("S1_strict", th, np.array([r["lv"] for r in rs]), np.array([r["lb"] for r in rs]), np.full(len(rs), lz))


def level(rs, which, group):
    keep = [r for r in rs if np.isfinite(r[which][0]) or np.isfinite(r[which][1])]
    lo, hi, nz = widen([r[which][0] for r in keep], [r[which][1] for r in keep])
    mu = lower_mu(keep) if which == "lower" else upper_mu(keep, group)
    return TL.offset_fit(mu, lo, hi)[0], nz, len(keep)


def session_level(tags, which, group, nb=200, seed=41):
    per = []
    for t in tags:
        rs = rows_of(t)
        y, nz, n = level(rs, which, group)
        clus = defaultdict(list)
        for r in rs:
            clus[round(r["lb"], 2)].append(r)
        keys = sorted(clus); rng = np.random.default_rng(seed); bs = []
        for _ in range(nb):
            kk = rng.choice(len(keys), len(keys))
            bs.append(level([r for k in kk for r in clus[keys[k]]], which, group)[0])
        per.append((y, float(np.std(bs, ddof=1)), nz, n))
    y = float(np.mean([p[0] for p in per])); se = float(np.sqrt(np.sum([p[1] ** 2 for p in per])) / len(per))
    return dict(level=y, se=se, parts=[dict(level=p[0], se=p[1], zero_width_widened=p[2], n_cols=p[3]) for p in per])


def reml(y, se, g):
    y, se = np.asarray(y), np.asarray(se); groups = sorted(set(g)); Zm = np.array([[gi == gg for gg in groups] for gi in g], float)
    X = np.ones((len(y), 1))
    def ll(tau, om):
        V = tau ** 2 * Zm @ Zm.T + np.diag(om ** 2 + se ** 2)
        Vi = np.linalg.inv(V); XtVi = X.T @ Vi; b = np.linalg.solve(XtVi @ X, XtVi @ y); r = y - X @ b
        return -0.5 * (np.linalg.slogdet(V)[1] + np.linalg.slogdet(XtVi @ X)[1] + r @ Vi @ r), float(b[0])
    f = lambda p: -ll(np.exp(p[0]), np.exp(p[1]))[0]
    best = min((minimize(f, x0, method="Nelder-Mead", options={"xatol": 1e-6, "fatol": 1e-9, "maxiter": 4000})
                for x0 in ([np.log(.08), np.log(.08)], [np.log(.02), np.log(.15)], [np.log(.15), np.log(.02)])), key=lambda r: r.fun)
    tau, om = np.exp(best.x); L, mu = ll(tau, om)
    prof_om = lambda o: -minimize_scalar(lambda lt: -ll(np.exp(lt), o)[0], bounds=(np.log(1e-4), np.log(2)), method="bounded").fun
    prof_tau = lambda t_: -minimize_scalar(lambda lo_: -ll(t_, np.exp(lo_))[0], bounds=(np.log(1e-4), np.log(2)), method="bounded").fun
    def ci(prof, est):
        grid = np.linspace(0, 1.0, 401); v = np.array([prof(x) for x in grid]); ok = grid[2 * (L - v) <= 3.841]
        return [float(ok.min()), float(ok.max())] if len(ok) else None
    lr_om = max(0.0, 2 * (L - prof_om(1e-6))); lr_tau = max(0.0, 2 * (L - prof_tau(1e-6)))
    return dict(mu=mu, tau=float(tau), omega=float(om), tau_ci=ci(prof_tau, tau), omega_ci=ci(prof_om, om),
                p_omega0=float(0.5 * chi2.sf(lr_om, 1)), p_tau0=float(0.5 * chi2.sf(lr_tau, 1)), n_sessions=len(y), n_groups=len(groups))


def main():
    out = {}
    for which in ("upper", "lower"):
        lev = {s: session_level(tags, which, SGROUP[s]) for s, tags in SESSIONS.items()}
        y = [lev[s]["level"] for s in SESSIONS]; se = [lev[s]["se"] for s in SESSIONS]; g = [SGROUP[s] for s in SESSIONS]
        res = dict(levels=lev, model=reml(y, se, g))
        p6 = lev["B_T2p6"]["parts"]; res["same_day_diff_r1_r2"] = p6[0]["level"] - p6[1]["level"]
        res["same_day_sd_equiv"] = abs(res["same_day_diff_r1_r2"]) / np.sqrt(2)
        # sensitivity: B_T2 sample 2 truncated at <= 4 N (from novelty/range_effect, upper only, standard speeds)
        out[which] = res
    re = json.loads((HOME / "violin-rne/novelty/range_effect/results.json").read_text())
    up = out["upper"]; s = list(SESSIONS); y = [up["levels"][k]["level"] for k in s] + [re["B_T2s2"]["4"]["std3"]["S1_strict"]]
    se = [up["levels"][k]["se"] for k in s] + [0.05]; g = [SGROUP[k] for k in s] + ["B T2"]
    out["upper"]["with_B_T2s2_c4"] = dict(model=reml(y, se, g), note="B_T2 s2 level from range_effect c = 4 N (standard speeds); SE set to 0.05 (not bootstrapped)")
    (HERE / "results.json").write_text(json.dumps(out, indent=1, default=float))
    for which in ("upper", "lower"):
        r = out[which]
        print(f"== {which}")
        for k, v in r["levels"].items():
            print(f"   {k:7s} {SGROUP[k]:5s} level {v['level']:+.3f} ± {v['se']:.3f}" + (f" (widened {sum(p['zero_width_widened'] for p in v['parts'])})" if any(p['zero_width_widened'] for p in v['parts']) else ""))
        m = r["model"]
        print(f"   REML: string type x tension tau {m['tau']:.3f} {m['tau_ci']} (p tau=0 {m['p_tau0']:.3f}) | session omega {m['omega']:.3f} {m['omega_ci']} (p omega=0 {m['p_omega0']:.3f}) | same-day |r1-r2|/sqrt2 {r['same_day_sd_equiv']:.3f}")
    m = out["upper"]["with_B_T2s2_c4"]["model"]
    print(f"   upper + B_T2s2(c4): tau {m['tau']:.3f} {m['tau_ci']} omega {m['omega']:.3f} {m['omega_ci']} p_omega0 {m['p_omega0']:.3f}")


if __name__ == "__main__":
    main()
