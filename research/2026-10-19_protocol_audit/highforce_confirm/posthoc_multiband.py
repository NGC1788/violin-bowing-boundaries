"""POST HOC diagnostics after the once-only high-force scoring (Claude, 2026-10-19). Does not alter score_result.json.
(1) Multi-band prevalence (>= 2 separated K=2 H bands per column, t = 0.2 s) in the 11 low-force development diagrams
    vs the 4 high-force targets, and H-band structure (number, lengths, gap lengths).
(2) Sensitivity of the frozen S1_strict level error to the upper-boundary rule: end of LAST band (frozen), of the
    FIRST band, of the LONGEST band. Labelled post hoc; no decision is changed."""
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
import score_highforce as S  # noqa: E402

DEV = ["A1", "A2", "B1", "C1", "D1", "A_T2", "B_T2s1", "B_T2r1", "B_T2r2", "C_T2", "D_T2"]
CACHE = {t: Path.home() / "violin-rne/final/cache" / f"{t}.npz" for t in DEV}
CACHE["D_T2"] = Path.home() / "violin-rne/novelty/thermal_upper/cache/D_T2.npz"


def bands(h, k=2):
    runs = [i for i in range(len(h) - k + 1) if all(h[i:i + k])]
    cov = np.zeros(len(h), bool)
    for i in runs:
        cov[i:i + k] = True
    segs, i = [], 0
    while i < len(h):
        if cov[i]:
            j = i
            while j < len(h) and cov[j]:
                j += 1
            segs.append((i, j - 1))
            i = j
        else:
            i += 1
    return segs


def dev_columns(tag, t=0.2):
    d = dict(np.load(CACHE[tag]))
    m = np.isclose(d["t"], t)
    g = defaultdict(list)
    for i in np.flatnonzero(m):
        g[(int(d["folder"][i]), int(d["column"][i]))].append(i)
    out = []
    for (f, c), idx in g.items():
        idx = sorted(idx, key=lambda i: d["force"][i])
        out.append(dict(f=np.log(np.maximum(d["force"][idx], 1e-3)), h=d["H"][idx].astype(bool), lv=np.log(d["speed"][idx[0]]),
                        lb=np.log(np.median(d["beta"][idx]))))
    return out


def describe(cols):
    nb = [len(bands(c["h"])) for c in cols]
    withb = [c for c, n in zip(cols, nb) if n >= 1]
    multi = [c for c, n in zip(cols, nb) if n >= 2]
    gaps, extra_len, main_len = [], [], []
    for c in multi:
        s = bands(c["h"])
        L = [b - a + 1 for a, b in s]
        main_len.append(max(L)); extra_len += sorted(L)[:-1]
        gaps += [s[i + 1][0] - s[i][1] - 1 for i in range(len(s) - 1)]
    return dict(n_columns=len(cols), with_band=len(withb), multi_band=len(multi),
                multi_share_of_banded=len(multi) / max(len(withb), 1),
                median_main_band_len=float(np.median(main_len)) if main_len else None,
                median_extra_band_len=float(np.median(extra_len)) if extra_len else None,
                median_gap_len=float(np.median(gaps)) if gaps else None)


def upper_rule(c, rule):
    s = bands(c["h"])
    if not s:
        return (np.nan, np.nan)
    a, b = {"last": s[-1], "first": s[0], "longest": max(s, key=lambda x: x[1] - x[0])}[rule]
    f = c["f"]
    return (f[-1], np.inf) if b == len(f) - 1 else (f[b], f[b + 1])


def main():
    out = dict(note="POST HOC; frozen score_result.json unchanged", dev={}, hf={}, level_rule_sensitivity={})
    for tag in DEV:
        out["dev"][tag] = describe(dev_columns(tag))
    data = {}
    sys.path.insert(0, str(Path.home() / "violin-rne/claude_extra"))
    import verify_astra_boundaries as V
    for arch in ("HighBowForceRange_1", "HighBowForceRange_2"):
        data.update(S.read_labels(arch, V.rm, V.APPLIED))
    F = json.loads(S.FROZEN_FILES[0].read_text())
    th = F["fits"]["S1_strict"]["theta"]
    import thermal_law2 as TL2
    import thermal_law as TL
    for tgt, (_, _, grp) in S.TARGETS.items():
        d = data[tgt]
        cols = S.columns_from_windows(d["win"], d["beta"], d["speed"], 0.2)
        out["hf"][tgt] = describe(cols)
        lz = np.log(F["z_published"][grp][1])
        res = {}
        for rule in ("last", "first", "longest"):
            up = [upper_rule(c, rule) for c in cols]
            keep = [i for i, u in enumerate(up) if np.isfinite(u[0]) or np.isfinite(u[1])]
            lv = np.array([cols[i]["lv"] for i in keep]); lb = np.array([cols[i]["lb"] for i in keep])
            lo = np.array([up[i][0] for i in keep]); hi = np.array([up[i][1] for i in keep])
            mu = TL2.predict("S1_strict", th, lv, lb, np.full(len(lv), lz))
            p = S.free_fit(lv, lb, lo, hi)
            res[rule] = dict(S1_level_error=TL.offset_fit(mu, lo, hi)[0], e_v=float(p[1]), e_beta=float(p[2]),
                             n=len(keep), n_right=int(np.sum(~np.isfinite(hi))))
        out["level_rule_sensitivity"][tgt] = res
    (HERE / "posthoc_multiband.json").write_text(json.dumps(out, indent=1, default=float))
    print("multi-band share of columns with any band (t=0.2, K=2):")
    for k, v in list(out["dev"].items()) + list(out["hf"].items()):
        print(f"  {k:7s} {v['multi_band']:3d}/{v['with_band']:3d} = {v['multi_share_of_banded']:.2f} | main band len {v['median_main_band_len']} extra {v['median_extra_band_len']} gap {v['median_gap_len']}")
    print("S1_strict level error and free exponents by upper rule (post hoc):")
    for k, v in out["level_rule_sensitivity"].items():
        print("  " + k + "  " + " | ".join(f"{r}: lvl {x['S1_level_error']:+.3f} e_v {x['e_v']:+.2f} e_b {x['e_beta']:+.2f} (n {x['n']}, right {x['n_right']})" for r, x in v.items()))


if __name__ == "__main__":
    main()
