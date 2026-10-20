"""Variance components tau (type x tension) vs omega (session within group): REML (own code), method of moments,
Bayesian grid posterior, parametric-bootstrap behaviour of the decision rule, influence, alternative references/SEs.
Post hoc on already-seen data (refutation pass of novelty/session_variance; Claude 2026-10-20). Natural log."""
import json
from pathlib import Path
import numpy as np
from scipy.optimize import minimize
from scipy.stats import chi2

HERE = Path(__file__).parent
LV = json.loads((HERE / "levels.json").read_text())
F2 = json.loads((Path.home() / "violin-rne/novelty/thermal_upper/FROZEN_v2.json").read_text())
GRP = {"A1": "A T1", "A2": "A T1", "A3": "A T1", "A4": "A T1", "B1": "B T1", "C1": "C T1", "D1": "D T1", "D2": "D T1",
       "A_T2": "A T2", "B_T2s1": "B T2", "B_T2p6": "B T2", "C_T2": "C T2", "D_T2": "D T2"}


def sessions(var, sekey="se", p6="mean"):
    v = LV[var]; out = {}
    for s in GRP:
        if s == "B_T2p6":
            if "B_T2r1" not in v:
                continue
            a, b = v["B_T2r1"], v["B_T2r2"]
            if p6 == "mean":
                out[s] = ((a["level"] + b["level"]) / 2, np.hypot(a[sekey], b[sekey]) / 2)
            else:
                out[s] = (v[p6]["level"], v[p6][sekey])
        elif s in v:
            out[s] = (v[s]["level"], v[s][sekey])
    return out


def ll(y, se, Zm, tau, om):
    V = tau ** 2 * Zm @ Zm.T + np.diag(om ** 2 + se ** 2); Vi = np.linalg.inv(V); one = np.ones(len(y))
    a = one @ Vi @ one; mu = (one @ Vi @ y) / a; r = y - mu
    return -0.5 * (np.linalg.slogdet(V)[1] + np.log(a) + r @ Vi @ r)


def reml(y, se, g, ci=True):
    y, se = np.asarray(y, float), np.asarray(se, float); G = sorted(set(g)); Zm = np.array([[gi == k for k in G] for gi in g], float)
    f = lambda p: -ll(y, se, Zm, abs(p[0]), abs(p[1]))
    best = min((minimize(f, x0, method="Nelder-Mead", options={"xatol": 1e-7, "fatol": 1e-10}) for x0 in
                ([.05, .05], [.01, .15], [.15, .01], [.2, .2], [0.0, .1], [.1, 0.0])), key=lambda r: r.fun)
    tau, om = abs(best.x[0]), abs(best.x[1]); L = -best.fun
    prof = lambda which, val: max(-minimize(lambda q: -ll(y, se, Zm, *((val, abs(q[0])) if which == "tau" else (abs(q[0]), val))),
                                             [x0], method="Nelder-Mead").fun for x0 in (0.01, 0.1, 0.3))
    out = dict(tau=tau, omega=om, lr_om0=max(0, 2 * (L - prof("omega", 0.0))), lr_tau0=max(0, 2 * (L - prof("tau", 0.0))))
    out["p_omega0"] = 0.5 * chi2.sf(out["lr_om0"], 1); out["p_tau0"] = 0.5 * chi2.sf(out["lr_tau0"], 1)
    if ci:
        grid = np.linspace(0, 0.6, 121)
        for k in ("tau", "omega"):
            v = np.array([prof(k, x) for x in grid]); ok = grid[2 * (L - v) <= 3.841]
            out[k + "_ci"] = [float(ok.min()), float(ok.max())]
    return out


def mom(y, se, g):
    """unweighted nested ANOVA moments with known sampling variances subtracted (truncated at 0)."""
    y, se = np.asarray(y), np.asarray(se); G = sorted(set(g)); idx = {k: [i for i, gi in enumerate(g) if gi == k] for k in G}
    num = sum(np.sum((y[i] - y[i].mean()) ** 2) for i in idx.values() if len(i) > 1)
    den = sum(len(i) - 1 for i in idx.values())
    om2 = (num - sum(np.sum(se[i] ** 2) * (1 - 1 / len(i)) for i in idx.values() if len(i) > 1)) / den
    om2 = max(om2, 0.0)
    gm = np.array([y[i].mean() for i in idx.values()]); n = np.array([len(i) for i in idx.values()])
    ev = np.array([(om2 + np.mean(se[i] ** 2)) / len(i) for i in idx.values()])
    tau2 = max(np.var(gm, ddof=1) - ev.mean(), 0.0)
    return dict(tau=float(np.sqrt(tau2)), omega=float(np.sqrt(om2)), within_df=den)


def bayes(y, se, g, scale=0.25, prior="halfnormal"):
    y, se = np.asarray(y), np.asarray(se); G = sorted(set(g)); Zm = np.array([[gi == k for k in G] for gi in g], float)
    gr = np.linspace(0.0025, 0.6, 120)
    LL = np.array([[ll(y, se, Zm, t, o) for o in gr] for t in gr])
    lp = (-0.5 * (gr / scale) ** 2) if prior == "halfnormal" else np.zeros_like(gr)
    P = np.exp(LL - LL.max() + lp[:, None] + lp[None, :]); P /= P.sum()
    pt, po = P.sum(1), P.sum(0)
    q = lambda p, x: [float(np.interp(x, np.cumsum(p), gr)) for x in (0.025, 0.5, 0.975)]
    T, O = np.meshgrid(gr, gr, indexing="ij")
    return dict(tau=q(pt, None) if False else q(pt, 0), omega=q(po, 0), P_omega_ge_tau=float(P[O >= T].sum()))


def fix_q(d):
    return d


def summarize(name, S, extra=""):
    s = list(S); y = [S[k][0] for k in s]; se = [S[k][1] for k in s]; g = [GRP[k] for k in s]
    r = reml(y, se, g); m = mom(y, se, g)
    print(f"{name:34s} n{len(s):2d} REML tau {r['tau']:.3f} [{r['tau_ci'][0]:.3f},{r['tau_ci'][1]:.3f}] p{r['p_tau0']:.3f} | "
          f"omega {r['omega']:.3f} [{r['omega_ci'][0]:.3f},{r['omega_ci'][1]:.3f}] p{r['p_omega0']:.4f} | MoM tau {m['tau']:.3f} omega {m['omega']:.3f} {extra}")
    return dict(reml=r, mom=m, sessions=s)


def main():
    out = {}
    base = {}
    print("== Claim reproduction and alternative references / SEs")
    specs = [("U_S1 (claim)", "U_S1", "se", 0.0), ("U_S1 block5 SE", "U_S1", "se_block5", 0.0), ("U_S1 no Z (raw type diff)", "U_S1", "se", 1.0),
             ("U_E3_free_Z", "U_E3", "se", 0.0), ("L_A12 (claim)", "L_A12", "se", 0.0), ("L_A12 block5 SE", "L_A12", "se_block5", 0.0),
             ("L_c02 (censor <0.2 N)", "L_c02", "se", 0.0), ("L_onset axis (11 sessions)", "L_onset", "se", 0.0)]
    for name, var, sek, zshift in specs:
        S = sessions(var, sek)
        if zshift:
            S = {k: (v[0] + np.log(F2["z_published"][GRP[k]][1]), v[1]) for k, v in S.items()}
        base[name] = S
        out[name] = summarize(name, S)
    print("\n== Bayesian grid posterior (REML likelihood x half-normal(0,0.25) priors; and flat priors on [0,0.6])")
    for name in ("U_S1 (claim)", "L_A12 (claim)", "U_S1 block5 SE", "L_A12 block5 SE", "L_onset axis (11 sessions)"):
        S = base[name]; y = [v[0] for v in S.values()]; se = [v[1] for v in S.values()]; g = [GRP[k] for k in S]
        for pr in ("halfnormal", "flat"):
            b = bayes(y, se, g, prior=pr)
            out[name][f"bayes_{pr}"] = b
            print(f"   {name:28s} {pr:10s} tau median {b['tau'][1]:.3f} [{b['tau'][0]:.3f},{b['tau'][2]:.3f}]  omega {b['omega'][1]:.3f} [{b['omega'][0]:.3f},{b['omega'][2]:.3f}]  P(omega>=tau) {b['P_omega_ge_tau']:.2f}")
    print("\n== Influence (claim references, column SEs)")
    for name in ("U_S1 (claim)", "L_A12 (claim)", "U_S1 block5 SE", "L_A12 block5 SE"):
        S = base[name]; rows = {}
        print(f" -- {name}")
        drops = [[k] for k in S] + [["A1", "A4"], ["D2"], ["D1", "D2"]]
        for dr in drops:
            T = {k: v for k, v in S.items() if k not in dr}
            y = [v[0] for v in T.values()]; se = [v[1] for v in T.values()]; g = [GRP[k] for k in T]; r = reml(y, se, g, ci=False)
            rows["-" + "+".join(dr)] = r
            print(f"   drop {'+'.join(dr):8s} tau {r['tau']:.3f} omega {r['omega']:.3f} p_om0 {r['p_omega0']:.4f} omega>=tau {r['omega'] >= r['tau']}")
        for grp in sorted(set(GRP.values())):
            T = {k: v for k, v in S.items() if GRP[k] != grp}
            y = [v[0] for v in T.values()]; se = [v[1] for v in T.values()]; g = [GRP[k] for k in T]; r = reml(y, se, g, ci=False)
            rows["-group " + grp] = r
            print(f"   drop group {grp}: tau {r['tau']:.3f} omega {r['omega']:.3f} p_om0 {r['p_omega0']:.4f} omega>=tau {r['omega'] >= r['tau']}")
        # B_T2 Part 6 handled as r1 only / r2 only
        for p6 in ("B_T2r1", "B_T2r2"):
            var = "U_S1" if name.startswith("U") else "L_A12"; sek = "se_block5" if "block5" in name else "se"
            T = sessions(var, sek, p6=p6); y = [v[0] for v in T.values()]; se = [v[1] for v in T.values()]; g = [GRP[k] for k in T]
            r = reml(y, se, g, ci=False); rows["p6=" + p6] = r
            print(f"   B_T2 Part6 = {p6} only: tau {r['tau']:.3f} omega {r['omega']:.3f} p_om0 {r['p_omega0']:.4f}")
        out[name]["influence"] = rows
    print("\n== Where omega comes from: within-group pairs (level diff, z of diff using column SEs)")
    for name in ("U_S1 (claim)", "L_A12 (claim)"):
        S = base[name]
        for grp in ("A T1", "D T1", "B T2"):
            ks = [k for k in S if GRP[k] == grp]; y = np.array([S[k][0] for k in ks]); se = np.array([S[k][1] for k in ks])
            Q = np.sum((y - y.mean()) ** 2); EQ = np.sum(se ** 2) * (1 - 1 / len(ks))
            om_g = np.sqrt(max(Q - EQ, 0) / (len(ks) - 1))
            print(f"   {name:14s} {grp}: levels {np.round(y, 3)}  within-SS {Q:.4f} vs expected from SEs {EQ:.4f}  -> group-only omega {om_g:.3f}")
    print("\n== Same-day repeat B_T2 Part 6 r1 vs r2")
    for var in ("U_S1", "L_A12", "L_c02"):
        a, b = LV[var]["B_T2r1"], LV[var]["B_T2r2"]
        for sek in ("se", "se_block5"):
            if a.get(sek) is None:
                continue
            d = a["level"] - b["level"]; s = np.hypot(a[sek], b[sek])
            print(f"   {var} {sek:9s}: r1-r2 {d:+.3f}, SD of diff from SEs alone {s:.3f} (z {d / s:+.2f}); |d|/sqrt2 {abs(d) / np.sqrt(2):.3f}; 95% CI for a 1-df SD: [{abs(d)/np.sqrt(2)*np.sqrt(1/chi2.ppf(.975,1)):.3f}, {abs(d)/np.sqrt(2)*np.sqrt(1/chi2.ppf(.025,1)):.2f}]")
    (HERE / "stats.json").write_text(json.dumps(out, indent=1, default=float))


if __name__ == "__main__":
    main()
