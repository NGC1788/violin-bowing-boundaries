"""Bow-force profile of high-force strokes along the stroke (50 ms blocks; claude_extra/blocks/*.npz) and its
relation to the upper boundary.
 1. force_dc / force before motion vs bow travel (cm) for nominal ranks 41-50 (high force), per speed and specimen:
    is there a mid-stroke dip, how deep, where?
 2. travel at the matched analysis windows (t = 0, 0.3 s) for strokes near the upper boundary
 3. regime -> force coupling: within (column, rank) cells of high-force strokes, do non-Helmholtz windows show a
    different force change (current / onset) than Helmholtz windows? (paired across speeds is impossible, so compare
    within specimen x speed, strokes above vs below the upper boundary, after regressing on beta)
"""
from collections import defaultdict

import numpy as np

import ub

BLK = {"A1": "/Users/jeongjihoon/violin-rne/claude_extra/blocks/blocks_A1.npz",
       "A2": "/Users/jeongjihoon/violin-rne/claude_extra/blocks/blocks_A2.npz"}


def main():
    for tag in ("A1", "A2"):
        sp = ub.Spec(tag)
        meta = ub.meta_by_sid(sp)
        z = np.load(BLK[tag])
        fields = list(z["fields"])
        data = z["data"]
        fold_names = [f.split("/")[-1] for f in sp.folders]
        sid = np.array([fold_names.index(f) * 10000 + t for f, t in zip(z["folder"], z["trial"])])
        iF, iP, iV = fields.index("force_dc"), fields.index("position"), fields.index("speed")
        block_s = float(z["block_s"])
        print(f"=== {tag} (block {block_s*1000:.0f} ms)")
        for f in range(3):
            prof = defaultdict(list)
            for k, s in enumerate(sid):
                if s // 10000 != f or sp.rank[s] < 41:
                    continue
                pos = data[k, iP] * 100.0  # m -> cm
                fdc = data[k, iF]
                spd = data[k, iV]
                ok = np.isfinite(pos) & np.isfinite(fdc)
                moving = ok & (spd > 0.2 * sp.speed[f])
                if moving.sum() < 4:
                    continue
                f0 = meta[s]["force_at_onset"]
                pm = pos[moving]; fm = fdc[moving] / f0
                for x in (0.5, 2, 4, 6, 8, 10, 12, 14, 16):
                    j = np.flatnonzero(pm >= x)
                    if j.size:
                        prof[x].append(fm[j[0]])
                prof["min"].append(np.min(fm)); prof["argmin_cm"].append(pm[np.argmin(fm)]); prof["end"].append(fm[-1])
                prof["travel"].append(pm[-1])
            txt = ", ".join(f"{x} cm {np.median(prof[x]):.3f}" for x in (0.5, 2, 4, 6, 8, 10, 12, 14, 16) if len(prof[x]) > 20)
            print(f"  v{f+1} ranks 41-50 (n {len(prof['min'])}): force/onset vs travel: {txt}; end of motion {np.median(prof['end']):.3f}; "
                  f"min {np.median(prof['min']):.3f} at {np.median(prof['argmin_cm']):.1f} cm; total travel {np.median(prof['travel']):.1f} cm")
        # travel at analysis windows near the upper boundary
        for t in (0.0, 0.3):
            groups = ub.column_groups(sp, t)
            for f in range(3):
                tr, ratio = [], []
                for k, g in groups.items():
                    if k[0] != f:
                        continue
                    fs = [x[0] for x in g]; hs = [x[1] for x in g]
                    lo, hi, st = ub.upper_interval(fs, hs)
                    if st != "interval":
                        continue
                    e = int(np.searchsorted(np.log(np.maximum(fs, 1e-4)), hi))
                    for x in g[max(0, e - 1):e + 1]:
                        m = meta[x[4]]
                        ramp_dist = 0.5 * sp.speed[f] * m["ramp_s"]
                        tr.append(100 * (ramp_dist + sp.speed[f] * (t + 0.1)))  # window centre
                        ratio.append(x[0] / m["force_at_onset"])
                print(f"  t={t:.1f} v{f+1}: strokes bracketing the upper boundary: travel at window centre median {np.median(tr):.1f} cm "
                      f"(5-95% {np.percentile(tr, 5):.1f}-{np.percentile(tr, 95):.1f}); current/onset force median {np.median(ratio):.3f} "
                      f"(5-95% {np.percentile(ratio, 5):.3f}-{np.percentile(ratio, 95):.3f})")
        # regime -> force coupling at high force: log(current/onset) for H vs non-H windows among strokes with onset > 1 N
        print("  regime vs force change (onset > 1 N, t = 0.2): median log(current/onset) by label, and within-column difference")
        groups = ub.column_groups(sp, 0.2)
        for f in range(3):
            dH, dN, diffs = [], [], []
            for k, g in groups.items():
                if k[0] != f:
                    continue
                hh = [np.log(x[0] / meta[x[4]]["force_at_onset"]) for x in g if meta[x[4]]["force_at_onset"] > 1 and x[1]]
                nn = [np.log(x[0] / meta[x[4]]["force_at_onset"]) for x in g if meta[x[4]]["force_at_onset"] > 1 and not x[1]]
                dH += hh; dN += nn
                if len(hh) >= 2 and len(nn) >= 2:
                    diffs.append(np.median(nn) - np.median(hh))
            if dN:
                print(f"    v{f+1}: H {np.median(dH):+.4f} (n {len(dH)}), non-H {np.median(dN):+.4f} (n {len(dN)}); "
                      f"within-column non-H minus H: median {np.median(diffs) if diffs else np.nan:+.4f} over {len(diffs)} columns")


if __name__ == "__main__":
    main()
