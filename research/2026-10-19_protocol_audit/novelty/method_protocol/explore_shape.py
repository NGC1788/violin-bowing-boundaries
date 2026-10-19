"""Exploration (post hoc): per-diagram beta slope and speed intercepts on the measured axis (delta fix)."""
import numpy as np, json
import common as C
D = {t: C.load_exposed(t) for t in C.TAGS}
out = {}
for t in C.TAGS:
    rows = C.intervals(D[t], "F_ev", "H2", "delta")
    s = C.fit_sessions({t: rows})
    lo, hi, ok, bad = C.check_rows(rows)
    fin = np.isfinite(lo) & np.isfinite(hi)
    lcens = ~np.isfinite(lo); rcens = ~np.isfinite(hi)
    out[t] = dict(b=s["b"], alpha=np.round(s["alpha"], 3).tolist(), sigma=s["sigma"], finite=int(fin.sum()),
                  left=int(lcens.sum()), right=int(rcens.sum()),
                  left_by_speed=[int(lcens[[r['folder'] == f for r in rows]].sum()) for f in range(3)])
    print(t, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in out[t].items()})
json.dump(out, open(C.HERE / "explore_shape.json", "w"), indent=1)
