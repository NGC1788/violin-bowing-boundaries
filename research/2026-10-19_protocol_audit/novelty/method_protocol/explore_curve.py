"""Exploration (post hoc): lower-boundary interval ends vs beta for each diagram (v=0.10), and the
measured-force floor of the grid (median F_ev of nominal ranks 1-3) per column bin."""
import numpy as np
import common as C
D = {t: C.load_exposed(t) for t in C.TAGS}
print("tag   | 8 beta bins (5 cols each): finite-mid / L=left-censored count | grid floor log F_ev(rank<=3)")
for t in C.TAGS:
    d = D[t]
    rows = [r for r in C.intervals(d, "F_ev", "H2", "delta") if r["folder"] == 1]
    s = []
    fl = []
    for b in range(8):
        rr = rows[b * 5:(b + 1) * 5]
        mids = [0.5 * (r["lo"] + r["hi"]) for r in rr if np.isfinite(r["lo"]) and np.isfinite(r["hi"])]
        L = sum(1 for r in rr if not np.isfinite(r["lo"]))
        s.append(f"{np.median(mids) if mids else np.nan:+.2f}/{L}")
        m = (d["folder"] == 1) & (d["column"] >= b * 5) & (d["column"] < (b + 1) * 5) & (d["rank"] <= 3)
        fl.append(f"{np.log(max(np.median(d['F_ev'][m]), 1e-3)):+.2f}")
    print(f"{t:7s}", " ".join(s), "|", " ".join(fl), "| lb", f"{rows[0]['lb']:.2f}..{rows[-1]['lb']:.2f}")
