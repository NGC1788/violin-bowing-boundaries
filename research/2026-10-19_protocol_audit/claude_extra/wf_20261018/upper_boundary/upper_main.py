"""Main upper-boundary analysis (A1, A2; 0.2 s windows; matched plateau times 0-0.3 s; current window force).

Prints and saves (results_main.json):
  1. per specimen x t: censoring counts, interval-censored fit log Fmax = c + a log v + b log beta,
     stratified column bootstrap (1000) for a and b, LR tests vs Schelleng (a=1, b=-1) and a=0
  2. treatment of columns with no Helmholtz run ('none': excluded vs right-censored at the top force)
  3. per-speed beta slopes and per-speed levels (at beta = 0.1) with bootstrap intervals
  4. A1 vs A2: exponent differences (bootstrap), per-speed level ratios A1/A2 for the UPPER and LOWER boundary
"""
import json
import sys

import numpy as np
from scipy.stats import chi2

import ub

NB = int(sys.argv[1]) if len(sys.argv) > 1 else 1000
LB_REF = np.log(0.1)
out = {}


def fit_report(D, nb=NB, seed=0):
    full, nll, z = ub.fit(D)
    _, bs = ub.boot(D, nboot=nb, seed=seed, stat=lambda c: c[:3])
    _, n_s, _ = ub.fit(D, fixed={1: 1.0, 2: -1.0})
    _, n_a1, _ = ub.fit(D, fixed={1: 1.0})
    _, n_b1, _ = ub.fit(D, fixed={2: -1.0})
    _, n_a0, _ = ub.fit(D, fixed={1: 0.0})
    return {"c": full[0], "a": full[1], "b": full[2], "sigma": float(np.exp(full[3])), "nll": nll,
            "a_ci": list(ub.ci(bs[:, 1])), "b_ci": list(ub.ci(bs[:, 2])),
            "p_schelleng": float(chi2.sf(2 * (n_s - nll), 2)), "p_a1": float(chi2.sf(2 * (n_a1 - nll), 1)),
            "p_b-1": float(chi2.sf(2 * (n_b1 - nll), 1)), "p_a0": float(chi2.sf(2 * (n_a0 - nll), 1)),
            "boot": bs}


def per_speed(D, nb=NB, seed=0):
    """log F = c_v + b_v log beta per speed folder (separate fits), with column bootstrap within speed."""
    res = {}
    rng = np.random.default_rng(seed)
    for f in range(3):
        Df = D[D[:, 0] == f]
        X = np.column_stack([np.ones(len(Df)), Df[:, 3]])
        c, nll, z = ub.fit(Df, X=X)
        bs = []
        for _ in range(nb):
            idx = rng.integers(0, len(Df), len(Df))
            Xb = X[idx]
            cb, _, _ = ub.fit(Df[idx], X=Xb, x0=z)
            bs.append([cb[1], cb[0] + cb[1] * LB_REF])
        bs = np.array(bs)
        res[f] = {"b": c[1], "b_ci": list(ub.ci(bs[:, 0])), "level_at_0.1": float(np.exp(c[0] + c[1] * LB_REF)),
                  "level_ci": [float(np.exp(x)) for x in ub.ci(bs[:, 1])], "n": int(len(Df)),
                  "n_right": int(np.sum(~np.isfinite(Df[:, 5]))), "n_left": int(np.sum(~np.isfinite(Df[:, 4]))),
                  "boot_level": bs[:, 1]}
    return res


def per_speed_common_slope(D, nb=NB, seed=0):
    """log F = c_v + b log beta (speed-specific levels, common beta slope); levels at beta 0.1; stratified boot."""
    def X_fn(Dx):
        return np.column_stack([(Dx[:, 0] == f).astype(float) for f in range(3)] + [Dx[:, 3]])
    c, nll, z = ub.fit(D, X=X_fn(D))
    rng = np.random.default_rng(seed)
    bs = []
    for _ in range(nb):
        idx = np.concatenate([rng.choice(np.flatnonzero(D[:, 0] == f), np.sum(D[:, 0] == f)) for f in range(3)])
        cb, _, _ = ub.fit(D[idx], X=X_fn(D[idx]), x0=z)
        bs.append([cb[f] + cb[3] * LB_REF for f in range(3)])
    lev = np.array([c[f] + c[3] * LB_REF for f in range(3)])
    return lev, np.array(bs), c[3]


def main():
    tabs = {}
    for tag in ("A1", "A2"):
        sp = ub.Spec(tag)
        for t in ub.TIMES:
            groups = ub.column_groups(sp, t)
            up = ub.table(sp, t, "upper", groups=groups)
            low = ub.table(sp, t, "lower", groups=groups)
            tabs[(tag, t)] = (up, low, groups, sp)

    print("== 1. Upper boundary, K=2, current window force, 0.2 s windows; 'none' columns excluded")
    print("spec   t   cols int/right/none   a [95% col-boot]        b [95%]                 sigma  p(Schelleng a=1,b=-1)  p(a=1)  p(b=-1)  p(a=0)")
    res_up, res_low = {}, {}
    for (tag, t), (up, low, groups, sp) in tabs.items():
        D = ub.as_array(up)
        st = [r[6] for r in up]
        r = fit_report(D, seed=int(t * 10))
        res_up[(tag, t)] = r
        print(f"{tag}  {t:.1f}  {len(D):3d} {st.count('interval'):3d}/{st.count('right'):3d}/{st.count('none'):2d}   "
              f"{r['a']:+.3f} [{r['a_ci'][0]:+.2f},{r['a_ci'][1]:+.2f}]   {r['b']:+.3f} [{r['b_ci'][0]:+.2f},{r['b_ci'][1]:+.2f}]   "
              f"{r['sigma']:.2f}   {r['p_schelleng']:.1e}   {r['p_a1']:.1e}  {r['p_b-1']:.2f}  {r['p_a0']:.1e}")
        Dl = ub.as_array(low, drop_none=False)
        rl = fit_report(Dl, seed=int(t * 10))
        res_low[(tag, t)] = rl
    print("\n   (lower boundary, same code, for reference)")
    for (tag, t), rl in res_low.items():
        st = [r[6] for r in tabs[(tag, t)][1]]
        print(f"{tag}  {t:.1f}  lower: int/left/right {st.count('interval')}/{st.count('left')}/{st.count('right')}  "
              f"a {rl['a']:+.3f} [{rl['a_ci'][0]:+.2f},{rl['a_ci'][1]:+.2f}]  b {rl['b']:+.3f} [{rl['b_ci'][0]:+.2f},{rl['b_ci'][1]:+.2f}]  sigma {rl['sigma']:.2f}")

    print("\n== 2. 'none' columns (no K=2 Helmholtz run) treated as right-censored at the column's top force")
    for (tag, t), (up, low, groups, sp) in tabs.items():
        rows = []
        for r_ in up:
            if r_[6] == "none":
                g = groups[(r_[0], r_[1])]
                rows.append(r_[:4] + (np.log(g[-1][0]), np.inf, "right"))
            else:
                rows.append(r_)
        D = ub.as_array(rows)
        r = fit_report(D, nb=300, seed=int(t * 10))
        nn = [(int(x[0]), int(x[1])) for x in up if x[6] == "none"]
        print(f"{tag} {t:.1f}: a {r['a']:+.3f} [{r['a_ci'][0]:+.2f},{r['a_ci'][1]:+.2f}]  b {r['b']:+.3f}  "
              f"(none columns: {nn})")

    print("\n== 3a. Per-speed separate fits log Fmax = c_v + b_v log beta (column bootstrap within speed)")
    ps_all = {}
    for (tag, t), (up, low, groups, sp) in tabs.items():
        if t not in (0.0, 0.2):
            continue
        D = ub.as_array(up)
        ps = per_speed(D, nb=NB // 2, seed=1)
        ps_all[(tag, t)] = ps
        print(f"{tag} t={t:.1f}: " + "  ".join(
            f"v{f+1}: b {ps[f]['b']:+.2f} [{ps[f]['b_ci'][0]:+.2f},{ps[f]['b_ci'][1]:+.2f}] (n {ps[f]['n']}, right {ps[f]['n_right']}), "
            f"F(0.1) {ps[f]['level_at_0.1']:.2f} N" for f in range(3)))

    print("\n== 3b. Speed-specific levels at beta 0.1 with a common beta slope; pairwise speed exponents")
    levels = {}
    for kind in ("upper", "lower"):
        for (tag, t), (up, low, groups, sp) in tabs.items():
            D = ub.as_array(up) if kind == "upper" else ub.as_array(low, drop_none=False)
            lev, bs, bslope = per_speed_common_slope(D, nb=NB // 2, seed=2)
            levels[(kind, tag, t)] = (lev, bs)
            lv = np.log([sp.speed[f] for f in range(3)])
            e12 = (lev[1] - lev[0]) / (lv[1] - lv[0]); e23 = (lev[2] - lev[1]) / (lv[2] - lv[1])
            b12 = (bs[:, 1] - bs[:, 0]) / (lv[1] - lv[0]); b23 = (bs[:, 2] - bs[:, 1]) / (lv[2] - lv[1])
            print(f"{kind:5s} {tag} t={t:.1f}: F(beta 0.1) = " + ", ".join(
                f"{np.exp(lev[f]):.3f} [{np.exp(np.percentile(bs[:, f], 2.5)):.3f},{np.exp(np.percentile(bs[:, f], 97.5)):.3f}]" for f in range(3)) +
                f" N; beta slope {bslope:+.2f}; exponent v1->v2 {e12:+.2f} [{np.percentile(b12, 2.5):+.2f},{np.percentile(b12, 97.5):+.2f}], "
                f"v2->v3 {e23:+.2f} [{np.percentile(b23, 2.5):+.2f},{np.percentile(b23, 97.5):+.2f}]")

    print("\n== 4. A1 vs A2: exponent difference (A1 - A2) and level ratio A1/A2 per speed (beta 0.1)")
    for kind, R in (("upper", res_up), ("lower", res_low)):
        for t in ub.TIMES:
            r1, r2 = R[("A1", t)], R[("A2", t)]
            da = r1["boot"][:, 1] - r2["boot"][:, 1]
            db = r1["boot"][:, 2] - r2["boot"][:, 2]
            l1, b1 = levels[(kind, "A1", t)]; l2, b2 = levels[(kind, "A2", t)]
            ratio = np.exp(l1 - l2)
            rb = np.exp(b1 - b2)
            print(f"{kind:5s} t={t:.1f}: delta a {r1['a'] - r2['a']:+.3f} [{np.percentile(da, 2.5):+.2f},{np.percentile(da, 97.5):+.2f}]  "
                  f"delta b {r1['b'] - r2['b']:+.3f} [{np.percentile(db, 2.5):+.2f},{np.percentile(db, 97.5):+.2f}]  "
                  f"A1/A2 level: " + ", ".join(f"v{f+1} {ratio[f]:.2f} [{np.percentile(rb[:, f], 2.5):.2f},{np.percentile(rb[:, f], 97.5):.2f}]" for f in range(3)) +
                  f"  | mean |log ratio| {np.mean(np.abs(np.log(ratio))):.3f}")

    # joint LR test: shared vs specimen-specific (a, b) with specimen-specific intercepts, common sigma
    print("\n== 4b. Joint likelihood-ratio test: do A1 and A2 share the speed and beta exponents? (specimen intercepts free)")
    for kind in ("upper", "lower"):
        for t in ub.TIMES:
            D1 = ub.as_array(tabs[("A1", t)][0]) if kind == "upper" else ub.as_array(tabs[("A1", t)][1], drop_none=False)
            D2 = ub.as_array(tabs[("A2", t)][0]) if kind == "upper" else ub.as_array(tabs[("A2", t)][1], drop_none=False)
            D = np.vstack([D1, D2]); s = np.r_[np.zeros(len(D1)), np.ones(len(D2))]
            X0 = np.column_stack([np.ones(len(D)), s, D[:, 2], D[:, 3]])
            X1 = np.column_stack([np.ones(len(D)), s, D[:, 2], D[:, 2] * s, D[:, 3], D[:, 3] * s])
            _, n0, _ = ub.fit(D, X=X0); c1, n1, _ = ub.fit(D, X=X1)
            Xs = np.column_stack([np.ones(len(D)), D[:, 2], D[:, 3]])
            _, ns, _ = ub.fit(D, X=Xs)
            print(f"{kind:5s} t={t:.1f}: LR shared exponents p = {chi2.sf(2 * (n0 - n1), 2):.2e} (2 df); "
                  f"fully shared incl. intercept p = {chi2.sf(2 * (ns - n1), 3):.2e} (3 df)")

    # save
    def clean(r):
        return {k: v for k, v in r.items() if k != "boot"}
    out["upper"] = {f"{k[0]}_t{k[1]}": clean(v) for k, v in res_up.items()}
    out["lower"] = {f"{k[0]}_t{k[1]}": clean(v) for k, v in res_low.items()}
    out["levels"] = {f"{k[0]}_{k[1]}_t{k[2]}": [float(np.exp(x)) for x in v[0]] for k, v in levels.items()}
    out["per_speed"] = {f"{k[0]}_t{k[1]}": {str(f): {kk: vv for kk, vv in d.items() if kk != "boot_level"} for f, d in v.items()}
                        for k, v in ps_all.items()}
    with open(ub.HERE / "results_main.json", "w") as fh:
        json.dump(out, fh, indent=1, default=float)
    # boundary tables for figures
    rows = []
    for (tag, t), (up, low, groups, sp) in tabs.items():
        for r_u, r_l in zip(up, low):
            rows.append([tag, t, r_u[0], r_u[1], float(np.exp(r_u[3])), sp.speed[r_u[0]], r_u[4], r_u[5], r_u[6], r_l[4], r_l[5], r_l[6]])
    import csv
    with open(ub.HERE / "boundary_table.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["spec", "t", "folder", "col", "beta", "speed", "up_lo_logF", "up_hi_logF", "up_status", "low_lo_logF", "low_hi_logF", "low_status"])
        w.writerows(rows)


if __name__ == "__main__":
    main()
