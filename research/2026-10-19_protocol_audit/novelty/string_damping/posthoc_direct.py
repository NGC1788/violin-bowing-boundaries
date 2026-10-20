"""POST HOC (added after damping_test.py results, not in the plan): model-light check of T1.
For every (speed, beta) column, the lower end of the boundary interval is a force at which the stroke was observed
NOT to be Helmholtz (K = 2 rule). If that force is many times the predicted Schelleng minimum F_S(v, beta) of the same
string type, Helmholtz motion failed at a force where the pluck-loss maintenance theory says it must succeed.
Reports, per axis, the share of columns whose observed non-H force exceeds 1, 2, 4, 8 x F_S. Claude, 2026-10-20."""
import json, sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import damping_test as DT  # noqa: E402


def main():
    res = json.loads((HERE / "results.json").read_text())
    out = {}
    for ax, ss in (("onset", ["A1", "A2", "A3", "A4", "B1", "C1", "D1", "D2", "A_T2", "B_T2s1", "B_T2s2", "C_T2"]),
                   ("window", ["A1", "A2", "A3", "A4", "B1", "C1", "D1", "D2", "A_T2", "B_T2s1", "B_T2s2", "B_T2r1", "B_T2r2", "C_T2", "D_T2"])):
        ratios, per = [], {}
        for s in ss:
            rows = DT.onset_rows(s) if ax == "onset" else DT.window_rows(s)
            t = DT.TYPE.get(s, "B T2")
            S10 = res["damping"][t]["S10"]
            r_s, n_cols, n_left = [], 0, 0
            for r in rows:
                lo, hi = r["lower"]
                if not (np.isfinite(lo) or np.isfinite(hi)):
                    continue
                n_cols += 1
                if not np.isfinite(lo):
                    n_left += 1
                    continue
                fs = DT.F_S(DT.Z[t], S10, v=float(np.exp(r["lv"])), b=float(np.exp(r["lb"])))
                r_s.append(float(np.exp(lo) / fs))
            ratios += r_s
            per[s] = dict(n_cols=n_cols, n_left_censored=n_left, median_ratio=float(np.median(r_s)) if r_s else None,
                          share_gt8=float(np.mean(np.array(r_s) > 8)) if r_s else None)
        ra = np.array(ratios)
        out[ax] = dict(n_cols_with_nonH=len(ra), share_gt={k: float(np.mean(ra > k)) for k in (1, 2, 4, 8)},
                       median=float(np.median(ra)), q10=float(np.percentile(ra, 10)), sessions=per)
        print(f"{ax}: {len(ra)} columns with an observed non-H stroke; F_nonH / F_S median {np.median(ra):.1f} (10th pct {np.percentile(ra, 10):.1f}); "
              + ", ".join(f">{k}x {100 * np.mean(ra > k):.0f}%" for k in (1, 2, 4, 8)))
        for s, p in per.items():
            print(f"   {s:7s} cols {p['n_cols']:3d} left-censored {p['n_left_censored']:3d} median ratio {p['median_ratio']:.1f} share>8x {p['share_gt8']:.2f}")
    (HERE / "posthoc_direct.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
