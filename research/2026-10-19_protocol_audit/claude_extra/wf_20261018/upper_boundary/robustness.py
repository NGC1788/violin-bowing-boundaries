"""Robustness of the upper-boundary exponents (A1, A2) to boundary definition, labelling thresholds,
window length and the force variable.  Column bootstrap (stratified by speed), NB draws per setting.
Usage: python robustness.py [NB]
"""
import sys
from collections import defaultdict

import numpy as np

import ub

NB = int(sys.argv[1]) if len(sys.argv) > 1 else 200


def inside_out(f, h, K=2):
    """Start inside the Helmholtz region (first K-run from below), scan upward for the first K consecutive
    non-Helmholtz strokes; boundary between the last Helmholtz-side stroke and that run."""
    n = len(h)
    lf = np.log(np.maximum(np.asarray(f), 1e-4))
    s = next((i for i in range(n - K + 1) if all(h[i:i + K])), None)
    if s is None:
        return None, None, "none"
    j = next((i for i in range(s + K, n - K + 1) if not any(h[i:i + K])), None)
    if j is None:
        return lf[-1], np.inf, "right"
    return lf[j - 1], lf[j], "interval"


def table_custom(sp, t, fn, groups):
    rows = []
    for (folder, c), g in sorted(groups.items()):
        f = [x[0] for x in g]; h = [x[1] for x in g]
        lo, hi, st = fn(f, h)
        rows.append((folder, c, np.log(sp.speed[folder]), np.log(np.median([x[3] for x in g])),
                     lo if lo is not None else np.nan, hi if hi is not None else np.nan, st))
    return rows


def summarize(rows, seed):
    D = ub.as_array(rows)
    st = [r[6] for r in rows]
    full, bs = ub.boot(D, nboot=NB, seed=seed, stat=lambda c: c[:3])
    a_lo, a_hi = ub.ci(bs[:, 1]); b_lo, b_hi = ub.ci(bs[:, 2])
    return (f"a {full[1]:+.2f} [{a_lo:+.2f},{a_hi:+.2f}]  b {full[2]:+.2f} [{b_lo:+.2f},{b_hi:+.2f}]  "
            f"(int {st.count('interval')}, right {st.count('right')}, none {st.count('none')})"), full


def main():
    results = defaultdict(dict)
    for tag in ("A1", "A2"):
        sp = ub.Spec(tag)
        meta = ub.meta_by_sid(sp)
        onset = {s: m["force_at_onset"] for s, m in meta.items()}
        plateau = {s: m["force_plateau"] for s, m in meta.items()}
        # rank-median onset force per folder (a proxy of the nominal grid)
        by = defaultdict(list)
        for s, m in meta.items():
            by[(s // 10000, sp.rank[s])].append(m["force_at_onset"])
        rankmed = {s: float(np.median(by[(s // 10000, sp.rank[s])])) for s in meta}
        variants = [
            ("base: K=2 from top, current force", dict(), lambda f, h: ub.upper_interval(f, h, 2)),
            ("K=1", dict(), lambda f, h: ub.upper_interval(f, h, 1)),
            ("K=3", dict(), lambda f, h: ub.upper_interval(f, h, 3)),
            ("widest bridged run (repo, max_hole 2)", dict(), ub.widest_upper),
            ("inside-out: first 2 non-H above region", dict(), inside_out),
            ("flyback frac 0.6", dict(label_override=ub.relabel_fn(sp, frac=0.6)), lambda f, h: ub.upper_interval(f, h, 2)),
            ("flyback frac 0.7", dict(label_override=ub.relabel_fn(sp, frac=0.7)), lambda f, h: ub.upper_interval(f, h, 2)),
            ("min periodicity 0.7", dict(label_override=ub.relabel_fn(sp, min_periodicity=0.7)), lambda f, h: ub.upper_interval(f, h, 2)),
            ("min periodicity 0.9", dict(label_override=ub.relabel_fn(sp, min_periodicity=0.9)), lambda f, h: ub.upper_interval(f, h, 2)),
            ("force at onset (before bow moves)", dict(force_override=onset), lambda f, h: ub.upper_interval(f, h, 2)),
            ("whole-plateau mean force", dict(force_override=plateau), lambda f, h: ub.upper_interval(f, h, 2)),
            ("rank-median onset force (nominal proxy)", dict(force_override=rankmed), lambda f, h: ub.upper_interval(f, h, 2)),
        ]
        for name, kw, fn in variants:
            for t in ub.TIMES:
                groups = ub.column_groups(sp, t, **kw)
                rows = table_custom(sp, t, fn, groups)
                txt, full = summarize(rows, seed=int(10 * t))
                results[name][(tag, t)] = txt
                print(f"{tag} t={t:.1f} {name:42s} {txt}", flush=True)
        for w in (0.1, 0.4):
            spw = ub.Spec(tag, window_s=w)
            for t in ub.TIMES:
                groups = ub.column_groups(spw, t)
                n_f = {f: sum(1 for k in groups if k[0] == f) for f in range(3)}
                rows = table_custom(spw, t, lambda f, h: ub.upper_interval(f, h, 2), groups)
                # drop columns with too few strokes (window does not fit in the plateau)
                ok = {k for k, g in groups.items() if len(g) >= 40}
                rows = [r for r in rows if (r[0], r[1]) in ok]
                if len({r[0] for r in rows}) < 3:
                    print(f"{tag} t={t:.1f} window {w} s: fewer than 3 speeds with full columns -> skipped ({n_f})", flush=True)
                    continue
                txt, full = summarize(rows, seed=int(10 * t))
                print(f"{tag} t={t:.1f} {'window ' + str(w) + ' s':42s} {txt}", flush=True)


if __name__ == "__main__":
    main()
