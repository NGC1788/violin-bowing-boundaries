"""Descriptive attack on the A2 v1 maintenance statistic: where in time are entries and lows, how persistent are
labels, how often do strokes switch H<->non-H, and what happens at MATCHED plateau time."""
from collections import defaultdict
import numpy as np
from common import load, entry_low, runs_of, column_ratios

K = 2
for tag in ("A2", "A1"):
    S, speed = load(tag)
    for f in sorted(speed):
        fl = [s for s in S if s.folder == f and s.floor]
        out = column_ratios(fl)[f]
        used = {c for c, *_ in out}
        g = [s for s in fl if s.col in used]
        el = [(s, entry_low(s.F, s.H, K)) for s in g]
        ent = [(s, e) for s, e in el if e is not None]
        t_first = min(s.t[1] for s in g)
        te = np.array([s.t[e[2]] for s, e in ent]); tl = np.array([s.t[e[3]] for s, e in ent])
        n_win = np.median([len(s.t) for s in g])
        # switching: count H->nonH and nonH->H transitions over all windows of used-column floor strokes
        up = down = 0; nH = nN = 0
        for s in g:
            h = s.H.astype(int)
            d = np.diff(h)
            up += np.sum(d == 1); down += np.sum(d == -1); nH += np.sum(h[:-1] == 1); nN += np.sum(h[:-1] == 0)
        # strokes that are H in the first window pair and H in the last window pair
        allH = np.mean([s.H[:2].all() and s.H[-2:].all() for s, e in ent])
        ex = np.mean([not s.H[-2:].all() for s, e in ent])  # entered but not H at the end
        ratio_force = np.median([s.F[-1] / s.F[0] for s, e in ent])
        print(f"{tag} v={speed[f]:.2f}: cols {len(used)} floor strokes {len(g)} entered {len(ent)}; windows/stroke {n_win:.0f}; "
              f"entries at first possible window {np.mean(te <= t_first + 1e-9):.0%}, entries t>=0.5 s {np.mean(te >= 0.5):.0%}; "
              f"low in last 0.25 s {np.mean(tl >= np.array([s.t[-1] for s, e in ent]) - 0.25):.0%}; "
              f"P(up | nonH) {up / max(nN, 1):.4f} P(down | H) {down / max(nH, 1):.4f}; entered & not H at end {ex:.0%}; "
              f"median F_end/F_start (entered) {ratio_force:.2f}")

# A2 v1 matched-time comparison
S, speed = load("A2")
fl = [s for s in S if s.folder == 0 and s.floor]
out = column_ratios(fl)[0]
used = {c for c, *_ in out}
print("\nA2 v1 matched-time comparison (per column, then median over columns)")
for lo_t, hi_t in ((0.0, 0.3), (0.5, 1.0), (1.0, 1.5), (1.4, 2.1)):
    rat, nlate, rat2 = [], [], []
    for c in used:
        g = [s for s in fl if s.col == c]
        ent_f, maint_f, never_f = [], [], []
        for s in g:
            r = runs_of(s.H, K)
            ie = r[0] + K - 1 if r.size else None
            ends = r + K - 1
            for j in range(len(s.t)):
                if not (lo_t <= s.t[j] < hi_t):
                    continue
                if ie is not None and j == ie:
                    ent_f.append(s.F[j])
                elif ie is not None and j > ie and j in set(ends):
                    maint_f.append(s.F[j])
                elif ie is None or j < ie:
                    never_f.append(s.F[j])
        nlate.append(len(ent_f))
        if len(ent_f) >= 3 and len(maint_f) >= 3:
            rat.append(np.quantile(maint_f, 0.1) / np.quantile(ent_f, 0.1))
        if len(maint_f) >= 3 and len(never_f) >= 3:
            # how much of the not-yet-entered population sits above the low maintained forces
            rat2.append(np.mean(np.array(never_f) > np.quantile(maint_f, 0.1)))
    print(f"  t in [{lo_t},{hi_t}): entries in window per column median {np.median(nlate):.0f} (sum {sum(nlate)}); "
          f"q10(maintained H)/q10(entries in same window) median {np.median(rat) if rat else np.nan:.2f} over {len(rat)} cols; "
          f"share of not-yet-entered windows above q10(maintained) {np.median(rat2) if rat2 else np.nan:.2f}")
