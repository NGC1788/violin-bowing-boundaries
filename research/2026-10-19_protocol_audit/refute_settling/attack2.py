"""POST HOC follow-ups (Claude, 2026-10-20): complete-case, interval-difference bounds, additive force-offset sensitivity,
helmholtz-specific label flips, negative forces. Data already seen."""
import json, sys
from pathlib import Path
import numpy as np
sys.argv = ["x"]
HERE = Path(__file__).resolve().parent
import importlib.util
src = (HERE / "attack.py").read_text().split("# ---------------- 0.")[0]  # reuse definitions only
ns = {"__file__": str(HERE / "attack.py")}; exec(compile(src, "attack_defs", "exec"), ns)
rows, diff_paired, SV, G, SESS, COLS, lab, APPLIED, fmt = (ns[k] for k in ("rows", "diff_paired", "SV", "G", "SESS", "COLS", "lab", "APPLIED", "fmt"))
OUT = {}
rc, rw, rm_ = rows("COLD"), rows("WARM"), rows("MAIN")
fin = lambda r: np.all(np.isfinite(r["lower"]))
# (a) complete case
both = [i for i in range(40) if fin(rc[i]) and fin(rw[i])]
a = SV.level([rc[i] for i in both], "lower", G)[0]; b = SV.level([rw[i] for i in both], "lower", G)[0]
rng = np.random.default_rng(53); bs = []
for _ in range(1000):
    kk = rng.choice(both, len(both)); bs.append(SV.level([rw[k] for k in kk], "lower", G)[0] - SV.level([rc[k] for k in kk], "lower", G)[0])
OUT["complete_case"] = dict(n=len(both), diff=b - a, ci=np.percentile(bs, [2.5, 97.5]).tolist())
print(f"(a) complete-case ({len(both)} cols finite in both) warm-cold lower level {b - a:+.3f} CI {np.round(np.percentile(bs, [2.5, 97.5]), 3).tolist()}")
# (b) per-column difference intervals
lo = [rw[i]["lower"][0] - rc[i]["lower"][1] for i in range(40)]; hi = [rw[i]["lower"][1] - rc[i]["lower"][0] for i in range(40)]
pos = sum(l > 0 for l in lo); neg = sum(h < 0 for h in hi)
lo_f = np.where(np.isfinite(lo), lo, -np.inf); hi_f = np.where(np.isfinite(hi), hi, np.inf)
OUT["diff_intervals"] = dict(surely_up=int(pos), surely_down=int(neg), straddle=int(40 - pos - neg), median_bounds=[float(np.median(lo_f)), float(np.median(hi_f))])
print(f"(b) per-column warm-cold intervals: surely up {pos}, surely down {neg}, straddle 0 {40 - pos - neg}; median shift bounded in [{np.median(lo_f):+.3f}, {np.median(hi_f):+.3f}]")
# (c) additive force offset in WARM (N): what offset removes the shift?
print("(c) add constant c (N) to every WARM window force (sensor zero offset sensitivity), measured K2 t0.2")
neg_w = sum(x["t"][0.2][0] < 0 for x in SESS["WARM"].values()); neg_c = sum(x["t"][0.2][0] < 0 for x in SESS["COLD"].values()); neg_m = sum(x["t"][0.2][0] < 0 for x in SESS["MAIN"].values())
print(f"    windows with negative measured force at t=0.2: cold {neg_c} warm {neg_w} main {neg_m}")
OUT["neg_force"] = dict(cold=neg_c, warm=neg_w, main=neg_m)
orig = {tr: x["t"][0.2] for tr, x in SESS["WARM"].items()}
OUT["offset"] = {}
for c in (-0.05, 0.0, 0.03, 0.05, 0.08, 0.10, 0.15):
    for tr, x in SESS["WARM"].items():
        x["t"][0.2] = (orig[tr][0] + c,) + tuple(orig[tr][1:])
    p = diff_paired(rc, rows("WARM"), "lower", nb=500); OUT["offset"][c] = p
    print(f"    c={c:+.2f}: {fmt(p)}")
for tr, x in SESS["WARM"].items():
    x["t"][0.2] = orig[tr]
# where are warm/cold lower boundaries in N (finite)
for s, rr in (("COLD", rc), ("WARM", rw), ("MAIN", rm_)):
    ub = [np.exp(r["lower"][1]) for r in rr if fin(r)]
    print(f"    {s} finite lower-boundary upper edge (N): median {np.median(ub):.3f} IQR {np.round(np.percentile(ub, [25, 75]), 3).tolist()}")
# (d) helmholtz-specific flips under classifier variants
v = {"min_amp_x2": {**APPLIED, "min_amplitude": APPLIED["min_amplitude"] * 2}, "min_amp_x0.5": {**APPLIED, "min_amplitude": APPLIED["min_amplitude"] / 2},
     "min_per_0.85": {**APPLIED, "min_periodicity": 0.85}, "min_per_0.9": {**APPLIED, "min_periodicity": 0.9}}
OUT["helm_flips"] = {}
for name, thr in v.items():
    fl = {s: int(sum((lab(x["t"][0.2], 98.08, thr) == "helmholtz") != (lab(x["t"][0.2]) == "helmholtz") for x in SESS[s].values())) for s in ("COLD", "WARM")}
    p = diff_paired(rows("COLD", thr=thr), rows("WARM", thr=thr), "lower", nb=500); OUT["helm_flips"][name] = dict(flips=fl, **p)
    print(f"(d) {name:12s} helmholtz-label flips {fl} -> {fmt(p)}")
for ref in (98.08 * 0.9, 98.08 * 1.1, 130.0):
    fl = {s: int(sum((lab(x["t"][0.2], ref) == "helmholtz") != (lab(x["t"][0.2]) == "helmholtz") for x in SESS[s].values())) for s in ("COLD", "WARM")}
    print(f"(d) f0 ref {ref:.1f}: helmholtz flips {fl} (subharmonic gate = f0 < 0.75*ref)")
json.dump(OUT, open(HERE / "results2.json", "w"), indent=1, default=float)
