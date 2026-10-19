"""Attack acceleration and the upper boundary (A1, A2).

Key design fact (checked here): the acceleration schedule is tied to the trial number in the same way in both
specimens, but the force order over trials is reversed in A1.  Hence at large beta, A2's high-force strokes get
the steepest attacks (several m/s^2) while A1's high-force strokes get gentle attacks (0.1-0.4 m/s^2); at small
beta both sit on the same 4.8 cm-ramp floor.  This is a natural experiment for the attack effect on Fmax.

 1. acceleration of the strokes bracketing the upper boundary (t = 0.2) per specimen x speed x beta band
 2. A1/A2 upper-boundary level ratio in columns where both specimens share the floor attack (small beta) vs columns
    where A2's attack is >= 5x steeper (large beta): joint censored fit with specimen x speed x group intercepts and a
    common beta slope per specimen; bootstrap over columns
 3. same contrast for the lower boundary (where A1 has the steeper attacks at large beta)
 4. column-level covariate model: log Fmax = c_spec,speed + b_spec log beta + g log(accel at boundary)
"""
from collections import defaultdict

import numpy as np

import ub

NB = 500
TIMES = (0.0, 0.2)


def accel_at(g, lo, hi, st, meta, which="upper"):
    """geometric mean acceleration of the two strokes bracketing the boundary (or the top/bottom stroke if censored)."""
    lf = np.log(np.maximum([x[0] for x in g], 1e-4))
    if st == "interval":
        e = int(np.searchsorted(lf, hi))
        pair = [g[e - 1][4], g[e][4]]
    elif st == "right":
        pair = [g[-1][4]] if which == "upper" else [g[-1][4]]
    elif st == "left":
        pair = [g[0][4]]
    else:
        return np.nan
    return float(np.exp(np.mean([np.log(meta[s]["accel"]) for s in pair])))


def build(t, kind):
    rows = []
    for si, tag in enumerate(("A1", "A2")):
        sp = ub.Spec(tag)
        meta = ub.meta_by_sid(sp)
        groups = ub.column_groups(sp, t)
        for (f, c), g in sorted(groups.items()):
            fs = [x[0] for x in g]; hs = [x[1] for x in g]
            lo, hi, st = ub.upper_interval(fs, hs) if kind == "upper" else ub.lower_interval(fs, hs)
            if st == "none":
                continue
            acc = accel_at(g, lo, hi, st, meta, kind)
            # column-level attack summary independent of the boundary: accel of the top-5 / bottom-5 strokes
            top = np.exp(np.mean([np.log(meta[x[4]]["accel"]) for x in g[-5:]]))
            bot = np.exp(np.mean([np.log(meta[x[4]]["accel"]) for x in g[:5]]))
            rows.append(dict(spec=si, f=f, c=c, lv=np.log(sp.speed[f]), lb=np.log(np.median([x[3] for x in g])),
                             lo=lo, hi=hi, st=st, acc=acc, top=top, bot=bot))
    return rows


def to_D(rows):
    return np.array([[r["spec"] * 3 + r["f"], r["c"], r["lv"], r["lb"], r["lo"], r["hi"]] for r in rows], float)


def main():
    for kind in ("upper", "lower"):
        for t in TIMES:
            rows = build(t, kind)
            print(f"\n=== {kind} boundary, t = {t:.1f}")
            # 1. acceleration at the boundary
            for si, tag in enumerate(("A1", "A2")):
                for f in range(3):
                    rr = [r for r in rows if r["spec"] == si and r["f"] == f]
                    bands = [(0.019, 0.05), (0.05, 0.08), (0.08, 0.21)]
                    txt = []
                    for lo_, hi_ in bands:
                        a = [r["acc"] for r in rr if lo_ <= np.exp(r["lb"]) < hi_]
                        if a:
                            txt.append(f"beta {lo_}-{hi_}: {np.median(a):.3f} m/s2 (n {len(a)})")
                    print(f"  {tag} v{f+1}: accel at boundary: " + "; ".join(txt))
            # 2. A1/A2 contrast by attack group (per column: group by A2/A1 accel ratio of the relevant strokes)
            key = "top" if kind == "upper" else "bot"
            ratio = {}
            for r in rows:
                ratio.setdefault((r["f"], r["c"]), {})[r["spec"]] = r[key]
            grp = {}
            for k, d in ratio.items():
                if 0 in d and 1 in d:
                    q = d[1] / d[0]
                    grp[k] = 0 if 0.8 < q < 1.25 else (1 if q >= 5 else (2 if q <= 0.2 else 3))
            names = {0: "same attack (ratio A2/A1 0.8-1.25)", 1: "A2 attack >= 5x steeper", 2: "A1 attack >= 5x steeper", 3: "other"}
            cnt = defaultdict(int)
            for k, gk in grp.items():
                cnt[gk] += 1
            print("  column groups (A2/A1 accel of the " + ("top" if kind == "upper" else "bottom") + "-5 strokes): " +
                  ", ".join(f"{names[k]}: {v}" for k, v in sorted(cnt.items())))
            rr = [r for r in rows if (r["f"], r["c"]) in grp and grp[(r["f"], r["c"])] in (0, 1, 2)]
            if len({grp[(r['f'], r['c'])] for r in rr}) < 2:
                print("  fewer than two attack groups -> contrast not estimable")
                continue
            D = to_D(rr)
            G = np.array([grp[(r["f"], r["c"])] for r in rr])
            S = np.array([r["spec"] for r in rr])
            F = np.array([r["f"] for r in rr])
            groups_present = sorted(set(G.tolist()))

            def X_fn(idx):
                cols = []
                for si in (0, 1):
                    for f in range(3):
                        for gk in groups_present:
                            m = (S[idx] == si) & (F[idx] == f) & (G[idx] == gk)
                            if m.any():
                                cols.append(m.astype(float))
                cols.append(D[idx, 3] * (S[idx] == 0))
                cols.append(D[idx, 3] * (S[idx] == 1))
                return np.column_stack(cols)

            def contrast(c, idx):
                """mean over speeds of [log(A1/A2) in group gk] for each group."""
                out = {}
                j = 0
                coef = {}
                for si in (0, 1):
                    for f in range(3):
                        for gk in groups_present:
                            m = (S[idx] == si) & (F[idx] == f) & (G[idx] == gk)
                            if m.any():
                                coef[(si, f, gk)] = c[j]
                                j += 1
                bs = {0: c[j], 1: c[j + 1]}
                for gk in groups_present:
                    vals = []
                    for f in range(3):
                        if (0, f, gk) in coef and (1, f, gk) in coef:
                            # evaluate at the group's median log beta
                            m = (G[idx] == gk) & (F[idx] == f)
                            lbm = np.median(D[idx][m, 3])
                            vals.append((coef[(0, f, gk)] + bs[0] * lbm) - (coef[(1, f, gk)] + bs[1] * lbm))
                    out[gk] = np.mean(vals) if vals else np.nan
                return out

            idx0 = np.arange(len(D))
            c, _, z = ub.fit(D, X=X_fn(idx0))
            est = contrast(c, idx0)
            rng = np.random.default_rng(3)
            boots = defaultdict(list)
            for _ in range(NB):
                # resample (speed, column) pairs, keeping both specimens of a column together
                keys = sorted({(r["f"], r["c"]) for r in rr})
                pick = [keys[i] for i in rng.integers(0, len(keys), len(keys))]
                pos = defaultdict(list)
                for i, r in enumerate(rr):
                    pos[(r["f"], r["c"])].append(i)
                idx = np.array([i for k in pick for i in pos[k]])
                Xb = X_fn(idx)
                if Xb.shape[1] != X_fn(idx0).shape[1]:
                    continue
                cb, _, _ = ub.fit(D[idx], X=Xb, x0=z)
                e = contrast(cb, idx)
                for gk, v in e.items():
                    boots[gk].append(v)
            for gk in groups_present:
                b = np.array(boots[gk])
                print(f"  {names[gk]:36s}: A1/A2 {kind} level ratio {np.exp(est[gk]):.2f} [{np.exp(np.nanpercentile(b, 2.5)):.2f},{np.exp(np.nanpercentile(b, 97.5)):.2f}]"
                      f"  (columns {int(np.sum(G == gk) / 2)}, valid boots {np.sum(np.isfinite(b))})")
            if 0 in groups_present and (1 in groups_present or 2 in groups_present):
                other = 1 if 1 in groups_present else 2
                d = np.array(boots[other]) - np.array(boots[0])
                print(f"  difference ({names[other]} minus same attack) in log ratio: {est[other] - est[0]:+.3f} "
                      f"[{np.nanpercentile(d, 2.5):+.3f},{np.nanpercentile(d, 97.5):+.3f}]")
            # 4. covariate model
            Dall = to_D(rows)
            Sall = np.array([r["spec"] for r in rows]); Fall = np.array([r["f"] for r in rows])
            la = np.log([r["acc"] for r in rows])

            def Xc(idx, with_acc=True):
                cols = [((Sall[idx] == si) & (Fall[idx] == f)).astype(float) for si in (0, 1) for f in range(3)]
                cols += [Dall[idx, 3] * (Sall[idx] == 0), Dall[idx, 3] * (Sall[idx] == 1)]
                if with_acc:
                    cols.append(la[idx])
                return np.column_stack(cols)
            ia = np.arange(len(Dall))
            c1, n1, z1 = ub.fit(Dall, X=Xc(ia))
            c0, n0, _ = ub.fit(Dall, X=Xc(ia, False))
            gs = []
            for _ in range(NB):
                idx = rng.integers(0, len(Dall), len(Dall))
                gs.append(ub.fit(Dall[idx], X=Xc(idx), x0=z1)[0][8])
            r_ = np.corrcoef(la, Dall[:, 3])[0, 1]
            print(f"  covariate model: g (log Fmax per log accel at boundary) = {c1[8]:+.3f} [{np.percentile(gs, 2.5):+.3f},{np.percentile(gs, 97.5):+.3f}]; "
                  f"LR chi2 {2 * (n0 - n1):.2f}; corr(log accel, log beta) {r_:+.2f}")


if __name__ == "__main__":
    main()
