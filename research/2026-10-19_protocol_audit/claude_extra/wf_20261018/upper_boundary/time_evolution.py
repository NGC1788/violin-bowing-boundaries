"""Does the upper boundary move with plateau time? (A1, A2; 0.2 s windows; current window force)
Per speed folder and plateau time t (0.1 s steps while the high-force side of a column is complete):
  per-speed fit log Fmax = c + b log beta (interval-censored), level at beta 0.1 with column bootstrap.
Also the lower boundary at the same times (same code) for contrast, and label persistence of high-force strokes.
"""
from collections import defaultdict

import numpy as np

import ub

NB = 300
LB = np.log(0.1)


def fit_speed(rows, seed):
    D = ub.as_array(rows)
    if len(D) < 8:
        return None
    X = np.column_stack([np.ones(len(D)), D[:, 3]])
    c, _, z = ub.fit(D, X=X)
    rng = np.random.default_rng(seed)
    lev = []
    for _ in range(NB):
        idx = rng.integers(0, len(D), len(D))
        cb, _, _ = ub.fit(D[idx], X=X[idx], x0=z)
        lev.append(cb[0] + cb[1] * LB)
    return np.exp(c[0] + c[1] * LB), np.exp(np.percentile(lev, [2.5, 97.5])), c[1], len(D), int(np.sum(~np.isfinite(D[:, 5])))


def main():
    for tag in ("A1", "A2"):
        sp = ub.Spec(tag)
        meta = ub.meta_by_sid(sp)
        # top-10 strokes (by onset force) of each column: must all be present for the upper boundary
        bycol = defaultdict(list)
        for s in sp.stroke_beta:
            bycol[(s // 10000, sp.col[s])].append((meta[s]["force_at_onset"], s))
        top10 = {k: {s for _, s in sorted(v)[-10:]} for k, v in bycol.items()}
        bot10 = {k: {s for _, s in sorted(v)[:10]} for k, v in bycol.items()}
        tmax = defaultdict(float)
        for i in range(len(sp.sid)):
            s = int(sp.sid[i])
            tmax[s] = max(tmax[s], float(sp.w["t"][i]))
        print(f"=== {tag}")
        for f in range(3):
            pl = [tmax[s] for s in sp.stroke_beta if s // 10000 == f]
            print(f"  v{f+1}: last window start per stroke: median {np.median(pl):.2f} s, 5-95% {np.percentile(pl, 5):.2f}-{np.percentile(pl, 95):.2f} s")
            for t in np.round(np.arange(0, 4.0, 0.1), 2):
                groups = ub.column_groups(sp, float(t))
                groups = {k: g for k, g in groups.items() if k[0] == f}
                if not groups:
                    break
                full_up = {k: g for k, g in groups.items() if top10[k] <= {x[4] for x in g}}
                full_low = {k: g for k, g in groups.items() if bot10[k] <= {x[4] for x in g}}
                up = ub.table(sp, float(t), "upper", groups=full_up) if full_up else []
                low = ub.table(sp, float(t), "lower", groups=full_low) if full_low else []
                ru = fit_speed(up, 1) if len(up) >= 8 else None
                rl = fit_speed(low, 2) if len(low) >= 8 else None
                if ru is None and rl is None:
                    break
                txt_u = (f"upper F(0.1) {ru[0]:.2f} [{ru[1][0]:.2f},{ru[1][1]:.2f}] N, b {ru[2]:+.2f}, cols {ru[3]} (right {ru[4]})"
                         if ru else "upper n/a")
                txt_l = (f"lower F(0.1) {rl[0]:.3f} [{rl[1][0]:.3f},{rl[1][1]:.3f}] N, b {rl[2]:+.2f}, cols {rl[3]}" if rl else "lower n/a")
                if t <= 1.0 or abs(t * 10 % 5) < 1e-6:
                    print(f"    t={t:.1f}: {txt_u} | {txt_l}", flush=True)
        # label persistence for strokes in the top region: H fraction vs time for strokes whose column's
        # upper boundary (t=0) lies below them (non-H at t=0) and strokes in the 5 ranks below the boundary
        print("  persistence: strokes above the t=0 upper boundary (non-H at t=0) -> fraction Helmholtz later; "
              "strokes in the 3 positions just below -> fraction still Helmholtz")
        g0 = ub.column_groups(sp, 0.0)
        above, below = defaultdict(list), defaultdict(list)
        for k, g in g0.items():
            f_ = [x[0] for x in g]; h_ = [x[1] for x in g]
            lo, hi, st = ub.upper_interval(f_, h_)
            if st != "interval":
                continue
            e = int(np.searchsorted(np.log(np.maximum(f_, 1e-4)), hi))
            for x in g[e:]:
                if not x[1]:
                    above[k[0]].append(x[4])
            for x in g[max(0, e - 3):e]:
                if x[1]:
                    below[k[0]].append(x[4])
        lab = defaultdict(dict)
        for i in range(len(sp.sid)):
            lab[int(sp.sid[i])][round(float(sp.w["t"][i]), 2)] = int(sp.w["label"][i]) == ub.H
        for f in range(3):
            line = []
            for t in (0.1, 0.2, 0.3, 0.5, 0.8, 1.0, 1.5, 2.0, 3.0):
                a = [lab[s][t] for s in above[f] if t in lab[s]]
                b = [lab[s][t] for s in below[f] if t in lab[s]]
                if len(a) < 10:
                    continue
                line.append(f"t {t}: above {np.mean(a):.2f} (n {len(a)}), below {np.mean(b):.2f} (n {len(b)})")
            print(f"    v{f+1}: " + "; ".join(line))


if __name__ == "__main__":
    main()
