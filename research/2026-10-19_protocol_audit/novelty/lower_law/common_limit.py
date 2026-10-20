"""POST HOC test (decided after floor_check): impose one common detection limit c on every session (all readings below c
treated as left-censored at c), refit the per-session lower exponents, and see whether the between-session spread of
the beta exponent shrinks. c = 0.35 N (above every session's floor) and 0.25 N; also complete-case (finite intervals only).
Claude, 2026-10-20."""
import json, sys
import numpy as np
from pathlib import Path
H = Path.home(); sys.path.insert(0, str(H / "violin-rne/novelty/lower_law"))
import lower_law as LL  # noqa: E402


def censor_at(rows, c):
    lc = np.log(c); out = []
    for r in rows:
        lo, hi = r["lower"]
        if np.isfinite(hi) and hi <= lc:
            lo, hi = -np.inf, lc
        elif np.isfinite(lo) and lo < lc < hi:
            lo = -np.inf
        out.append(dict(r, lower=(lo, hi)))
    return out


sets = {t: LL.SV.rows_of(t) for t in LL.DEV}
sets.update({k: LL.SV.rows_of(v) for k, v in LL.NEW.items()})
sets["B_T2s2_std3"] = LL.bt2s2_rows(LL.BT.STD_SPEEDS)
res = {}
for name, rows in sets.items():
    rows = [dict(r) for r in rows if np.isfinite(r["lower"][0]) or np.isfinite(r["lower"][1])]
    out = {}
    for c in (0.25, 0.35):
        out[f"c{c}"] = LL.fit(censor_at(rows, c))
    cc = [r for r in rows if np.all(np.isfinite(r["lower"]))]
    out["complete_case"] = LL.fit(cc) + (len(cc),)
    res[name] = out
Path("common_limit.json").write_text(json.dumps(res, indent=1, default=float))
ref = json.load(open("results.json"))
print(f"{'session':12s} {'measured β':>10s} {'c0.25 β':>8s} {'c0.35 β':>8s} {'complete β (n)':>16s}")
for k, v in res.items():
    mb = ref[k]["measured"]["b_beta"]
    print(f"{k:12s} {mb:+10.2f} {v['c0.25'][1]:+8.2f} {v['c0.35'][1]:+8.2f} {v['complete_case'][1]:+10.2f} ({v['complete_case'][2]})")
for key in ("c0.25", "c0.35", "complete_case"):
    b = [v[key][1] for v in res.values()]
    print(f"spread of beta exponent, {key}: SD {np.std(b, ddof=1):.2f}, range {min(b):+.2f}..{max(b):+.2f}")
b = [ref[k]["measured"]["b_beta"] for k in res]; print(f"spread, measured: SD {np.std(b, ddof=1):.2f}, range {min(b):+.2f}..{max(b):+.2f}")
