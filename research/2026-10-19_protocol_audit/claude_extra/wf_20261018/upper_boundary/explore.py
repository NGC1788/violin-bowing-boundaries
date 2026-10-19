"""Descriptive look at the upper end of the Helmholtz region in A1/A2 (0.2 s windows, matched plateau times).
Also re-checks the lower-boundary fit against the published reproduction (sanity check of ub.fit)."""
import numpy as np

import ub

for tag in ("A1", "A2"):
    sp = ub.Spec(tag)
    print(f"=== {tag}: speeds {[round(sp.speed[f], 4) for f in range(3)]}")
    for t in ub.TIMES:
        groups = ub.column_groups(sp, t)
        up = ub.table(sp, t, "upper", groups=groups)
        lo = ub.table(sp, t, "lower", groups=groups)
        # sanity: lower exponent
        Dl = ub.as_array(lo, drop_none=False)
        cl, _, _ = ub.fit(Dl)
        Du = ub.as_array(up)
        cu, _, _ = ub.fit(Du)
        line = [f"t={t:.1f} lower a={cl[1]:+.3f} b={cl[2]:+.3f} | upper a={cu[1]:+.3f} b={cu[2]:+.3f} sigma={np.exp(cu[3]):.2f}"]
        for f in range(3):
            st = [r[6] for r in up if r[0] == f]
            n = [len(g) for (ff, c), g in groups.items() if ff == f]
            fmax = [g[-1][0] for (ff, c), g in groups.items() if ff == f]
            tops = [g[-1][1] for (ff, c), g in groups.items() if ff == f]
            line.append(f"   v{f+1}: cols {len(st)} strokes/col {min(n)}-{max(n)}  upper: interval {st.count('interval')} "
                        f"right {st.count('right')} none {st.count('none')}; top stroke H in {sum(tops)}/40; "
                        f"max force per col median {np.median(fmax):.2f} N (range {min(fmax):.2f}-{max(fmax):.2f})")
        print("\n".join(line))
    # label composition above the upper boundary (non-censored columns), t = 0 and 0.3
    for t in (0.0, 0.3):
        groups = ub.column_groups(sp, t)
        for f in range(3):
            comp = np.zeros(len(ub.LABELS), int)
            comp_below = np.zeros(len(ub.LABELS), int)
            for (ff, c), g in groups.items():
                if ff != f:
                    continue
                fs = [x[0] for x in g]; hs = [x[1] for x in g]
                lo_, hi_, st = ub.upper_interval(fs, hs)
                if st != "interval":
                    continue
                e = int(np.searchsorted(np.log(np.maximum(fs, 1e-4)), hi_))
                for x in g[e:]:
                    comp[x[2]] += 1
                for x in g[max(0, e - 5):e]:
                    comp_below[x[2]] += 1
            tot = comp.sum()
            print(f"  t={t:.1f} v{f+1} above upper boundary: {tot} windows: " +
                  ", ".join(f"{ub.LABELS[i]} {comp[i]}" for i in range(len(ub.LABELS)) if comp[i]) +
                  f" | 5 strokes below: " + ", ".join(f"{ub.LABELS[i]} {comp_below[i]}" for i in range(len(ub.LABELS)) if comp_below[i]))
