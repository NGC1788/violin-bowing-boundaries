"""More attacks on the observed statistic (A2 v1 unless stated):
 1. observed-establishment-only: drop left-censored strokes (already H at plateau start)
 2. force-validity floor: windows with F < Fv treated as non-H (low-force labels may be classifier artefacts)
 3. classifier threshold perturbations (re-classified from cached raw features)
 4. exits: where Helmholtz is lost, relative to the column's F_E,cur and F_M
 5. late entries (established from a non-H state during the plateau): where they lie relative to F_E,cur, F_M
"""
from collections import defaultdict
import copy
import math

import numpy as np

import common
from common import load, column_ratios, entry_low, runs_of

import sys
from pathlib import Path
sys.path.insert(0, str(Path.home() / "violin-rne/claude_extra"))
import verify_astra_boundaries as V  # noqa: E402


def med(out, f):
    r = out.get(f, [])
    return (len(r), float(np.median([x[1] for x in r])) if r else np.nan, float(np.mean([x[1] < 1 for x in r])) if r else np.nan)


def fmt(t):
    return f"cols {t[0]:2d} ratio {t[1]:.2f} (<1 {t[2]:.0%})"


S2, sp2 = load("A2")
S1, sp1 = load("A1")
fl2 = [s for s in S2 if s.floor]
fl1 = [s for s in S1 if s.floor]

print("== 1. observed establishment only (entry run starts at t >= t0; strokes already H at plateau start dropped) ==")
for tag, fl, sp in (("A2", fl2, sp2), ("A1", fl1, sp1)):
    for t0 in (0.0, 0.1, 0.3):
        sel = lambda s, t0=t0: (lambda r: r.size > 0 and s.t[r[0]] >= t0 - 1e-9)(runs_of(s.H, 2))
        o = column_ratios(fl, select=sel, min_n=5)
        o3 = column_ratios(fl, select=sel, min_n=3)
        print(f"{tag} first-run start >= {t0}: " + " | ".join(f"v={sp[f]:.2f} {fmt(med(o, f))} [min_n=3: {fmt(med(o3, f))}]" for f in sorted(sp)))

print("\n== 2. force-validity floor: windows with F < Fv set to non-H ==")
for tag, fl, sp in (("A2", fl2, sp2), ("A1", fl1, sp1)):
    for Fv in (0.0, 0.02, 0.03, 0.05, 0.08):
        labels = {id(s): s.H & (s.F >= Fv) for s in fl}
        o = column_ratios(fl, labels=labels)
        print(f"{tag} Fv={Fv:.2f} N: " + " | ".join(f"v={sp[f]:.2f} {fmt(med(o, f))}" for f in sorted(sp)))
    # how many H windows sit below small forces at all
    for f in sorted(sp):
        g = [s for s in fl if s.folder == f]
        Fw = np.concatenate([s.F for s in g]); Hw = np.concatenate([s.H for s in g])
        print(f"   {tag} v={sp[f]:.2f}: H windows with F<0.03 N: {np.sum(Hw & (Fw < 0.03))} of {np.sum(Hw)}; "
              f"share H among windows F<0.03: {np.mean(Hw[Fw < 0.03]) if np.any(Fw < 0.03) else np.nan:.2f}")

print("\n== 3. classifier perturbations (A2 thresholds varied one at a time) ==")
z = {tag: np.load(common.HERE / f"cache_{tag}.npz") for tag in ("A1", "A2")}


def reclassify(tag, thr):
    """returns dict (folder, trial, round(t,3)) -> bool helmholtz, for 0.2 s windows"""
    d = z[tag]
    m = d["win"] == 0.2
    amp, per, f0, fpp = d["amp"][m], d["per"][m], d["f0"][m], d["fpp"][m]
    lab = np.empty(len(amp), dtype=object)
    out = {}
    H = np.zeros(len(amp), bool)
    for i in range(len(amp)):
        fp = None if fpp[i] < 0 else int(fpp[i])
        H[i] = V.rm.classify(float(amp[i]), float(per[i]), float(f0[i]), fp, 98.08, thr) == "helmholtz"
    keys = zip(d["folder"][m], d["trial"][m], np.round(d["t"][m], 3))
    return {k: h for k, h in zip(keys, H)}


base = dict(V.APPLIED)
variants = [("baseline", {})]
for k, vals in (("min_periodicity", (0.75, 0.85)), ("max_aperiodic", (0.4, 0.6)),
                ("min_amplitude", (base["min_amplitude"] * 0.5, base["min_amplitude"] * 2)),
                ("subharmonic_ratio", (0.7, 0.8))):
    for v in vals:
        variants.append((f"{k}={v:.3g}", {k: v}))
for tag, fl, sp in (("A2", fl2, sp2), ("A1", fl1, sp1)):
    for name, ch in variants:
        thr = dict(base); thr.update(ch)
        hmap = reclassify(tag, thr)
        labels = {id(s): np.array([hmap[(s.folder, s.trial, round(float(t), 3))] for t in s.t]) for s in fl}
        o = column_ratios(fl, labels=labels)
        print(f"{tag} {name:28s}: " + " | ".join(f"v={sp[f]:.2f} {fmt(med(o, f))}" for f in sorted(sp)), flush=True)

print("\n== 4/5. exits and late entries relative to the column's F_E,cur (q10 entries) and F_M (q10 lows), K=2 ==")
for tag, fl, sp in (("A2", fl2, sp2), ("A1", fl1, sp1)):
    for f in sorted(sp):
        out = column_ratios([s for s in fl if s.folder == f])[f]
        used = {c for c, *_ in out}
        n_exit = n_entered = 0
        ex_rel_E, ex_rel_M, late_rel_E, late_rel_M, n_late, n_early = [], [], [], [], 0, 0
        for c in used:
            g = [s for s in fl if s.folder == f and s.col == c]
            el = [(s, entry_low(s.F, s.H)) for s in g]
            el = [(s, e) for s, e in el if e is not None]
            E = np.array([e[0] for s, e in el]); L = np.array([e[1] for s, e in el])
            FE, FM = np.quantile(E, 0.1), np.quantile(L, 0.1)
            for s, e in el:
                n_entered += 1
                te = s.t[e[2]]
                if te >= 0.3:
                    n_late += 1; late_rel_E.append(e[0] / FE); late_rel_M.append(e[0] / FM)
                else:
                    n_early += 1
                # exits: H (end of a 2-run) followed by >= 2 non-H windows, after entry
                h = s.H
                for j in range(e[2] + 1, len(h) - 2):
                    if h[j] and not h[j + 1] and not h[j + 2]:
                        n_exit += 1; ex_rel_E.append(s.F[j] / FE); ex_rel_M.append(s.F[j] / FM)
                        break
        late_rel_E, late_rel_M = np.array(late_rel_E), np.array(late_rel_M)
        print(f"{tag} v={sp[f]:.2f}: entered {n_entered} (late t>=0.3 s: {n_late}); late entries below F_E,cur {np.sum(late_rel_E < 1)}"
              f" ({np.mean(late_rel_E < 1) if n_late else np.nan:.0%}), below F_M {np.sum(late_rel_M < 1)} ({np.mean(late_rel_M < 1) if n_late else np.nan:.0%}),"
              f" median late F/F_E {np.median(late_rel_E) if n_late else np.nan:.2f}; strokes with an exit {n_exit}"
              + (f", exit force / F_E median {np.median(ex_rel_E):.2f}, / F_M median {np.median(ex_rel_M):.2f}, exits below F_E {np.mean(np.array(ex_rel_E) < 1):.0%}" if n_exit else ""))
