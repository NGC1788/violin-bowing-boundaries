"""How much does the upper-boundary speed exponent rest on right-censored columns and on the parametric model?
 1. restrict to beta columns whose upper boundary is interval-censored (observed) at ALL three speeds
    (fit + per-column paired midpoint slopes)
 2. speed x beta interaction (does the speed exponent change with beta?  Schoonderwaldt 2008: upper-limit slopes
    steeper at higher speed <=> negative interaction)
 3. heteroscedastic check: speed-specific sigma
"""
import numpy as np
from scipy.stats import chi2

import ub

NB = 500


def main():
    for tag in ("A1", "A2"):
        sp = ub.Spec(tag)
        lv = np.log([sp.speed[f] for f in range(3)])
        for t in ub.TIMES:
            up = ub.table(sp, t, "upper")
            D = ub.as_array(up)
            st = {(int(r[0]), int(r[1])): r[6] for r in up}
            cols_all = [c for c in range(40) if all(st.get((f, c)) == "interval" for f in range(3))]
            Ds = D[np.isin(D[:, 1], cols_all)]
            line = f"{tag} t={t:.1f}: columns observed at all 3 speeds: {len(cols_all)}"
            if len(cols_all) >= 4:
                full, bs = ub.boot(Ds, nboot=NB, seed=1, stat=lambda c: c[:3])
                betas = sorted({round(float(np.exp(x)), 3) for x in Ds[:, 3]})
                line += (f" (beta {betas[0]:.3f}-{betas[-1]:.3f}); fit a {full[1]:+.2f} [{np.percentile(bs[:, 1], 2.5):+.2f},{np.percentile(bs[:, 1], 97.5):+.2f}]"
                         f" b {full[2]:+.2f}")
                mid = {}
                for row in Ds:
                    mid[(int(row[0]), int(row[1]))] = 0.5 * (row[4] + row[5])
                s12 = [(mid[(1, c)] - mid[(0, c)]) / (lv[1] - lv[0]) for c in cols_all]
                s23 = [(mid[(2, c)] - mid[(1, c)]) / (lv[2] - lv[1]) for c in cols_all]
                s13 = [(mid[(2, c)] - mid[(0, c)]) / (lv[2] - lv[0]) for c in cols_all]
                line += (f"; paired midpoint slopes median v1->v2 {np.median(s12):+.2f}, v2->v3 {np.median(s23):+.2f}, "
                         f"v1->v3 {np.median(s13):+.2f}")
            print(line)
            # interaction
            X0 = ub.design(D)
            lbc = D[:, 3] - np.log(0.07)
            X1 = np.column_stack([X0, (D[:, 2] - np.log(0.1)) * lbc])
            c0, n0, _ = ub.fit(D, X=X0)
            c1, n1, z1 = ub.fit(D, X=X1)
            rng = np.random.default_rng(2)
            gs = []
            for _ in range(NB):
                idx = np.concatenate([rng.choice(np.flatnonzero(D[:, 0] == f), np.sum(D[:, 0] == f)) for f in range(3)])
                gs.append(ub.fit(D[idx], X=X1[idx], x0=z1)[0][3])
            print(f"   interaction (change of speed exponent per unit ln beta): {c1[3]:+.2f} [{np.percentile(gs, 2.5):+.2f},{np.percentile(gs, 97.5):+.2f}], "
                  f"LR p {chi2.sf(2 * (n0 - n1), 1):.3f}; speed exponent at beta 0.03 / 0.07 / 0.15: "
                  + " / ".join(f"{c1[1] + c1[3] * (np.log(b) - np.log(0.07)):+.2f}" for b in (0.03, 0.07, 0.15)))


if __name__ == "__main__":
    main()
