"""Per-session lower-boundary exponents (plan: README.md). Claude, 2026-10-20 (post hoc)."""
import json, sys
from collections import defaultdict
from pathlib import Path
import numpy as np
H = Path.home(); HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(H / "violin-rne/novelty/session_variance")); sys.path.insert(0, str(H / "violin-rne/confirm_B_T2s2")); sys.path.insert(0, str(H / "violin-rne/claude_extra"))
import session_variance as SV  # noqa: E402
import score_bt2s2 as BT  # noqa: E402
import verify_astra_boundaries as V  # noqa: E402
C = SV.C
DEV = ["A1", "A2", "B1", "C1", "D1", "A_T2", "B_T2s1", "B_T2r1", "B_T2r2", "C_T2"]
NEW = {"A3": "TGT_2024-04-22_TypeA_sample3", "A4": "TGT_2024-04-24_TypeA_sample4", "D2": "TGT_2024-04-26_TypeD_sample2"}
LOG02 = np.log(0.2)


def detection(rows):
    out = []
    for r in rows:
        lo, hi = r["lower"]
        if np.isfinite(hi) and hi < LOG02:        # whole interval below 0.2 N -> left-censored at 0.2 N
            lo, hi = -np.inf, LOG02
        elif np.isfinite(lo) and lo < LOG02 <= hi:  # straddles 0.2 N -> upper end kept, lower end unknown
            lo = -np.inf
        out.append(dict(r, lower=(lo, hi)))
    return out


def fit(rows):
    p = C.censored_fit(rows, lambda r: (np.array([1.0, r["lv"], r["lb"]]), 0.0), "lower")
    return float(p[1]), float(p[2])


def boot(rows, nb=500, seed=61):
    by = defaultdict(list)
    for r in rows:
        by[r["bcol"]].append(r)
    keys = sorted(by); rng = np.random.default_rng(seed); bs = []
    for _ in range(nb):
        kk = rng.choice(len(keys), len(keys))
        bs.append(fit([r for k in kk for r in by[keys[k]]]))
    bs = np.array(bs)
    return np.percentile(bs[:, 0], [2.5, 97.5]).tolist(), np.percentile(bs[:, 1], [2.5, 97.5]).tolist()


def with_bcol(rows):
    # beta-column index within each speed: rank of lb inside the speed
    by = defaultdict(list)
    for r in rows:
        by[round(r["lv"], 6)].append(r)
    for v, rs in by.items():
        for i, r in enumerate(sorted(rs, key=lambda r: r["lb"])):
            r["bcol"] = i
    return rows


def bt2s2_rows(speeds=None):
    win, beta, speed = BT.read_labels(V.rm, V.APPLIED)
    col = BT.assign_columns(beta); g = defaultdict(list)
    for k, d in win.items():
        if 0.2 in d:
            g[(k[0], col[k])].append((d[0.2][0], d[0.2][1], beta[k]))
    rows = []
    for (folder, c), gg in sorted(g.items()):
        if speeds is not None and min(abs(speed[folder] - s) for s in speeds) > 1e-3:
            continue
        gg.sort(); f = np.log(np.maximum([x[0] for x in gg], 1e-3)); h = [x[1] for x in gg]
        lo_, up = BT.intervals(f, h)
        rows.append(dict(lv=float(np.log(speed[folder])), lb=float(np.log(np.median([x[2] for x in gg]))), lower=lo_, upper=up))
    return rows


def classify(bv, bb):
    hs = 0.9 <= bv <= 1.5 and -2.4 <= bb <= -1.5
    a = "A-type specific" if bb > -1.3 else ("A1/A2 atypical" if bb < -1.5 else "in between")
    return hs, a


def main():
    res = {}
    sets = {t: SV.rows_of(t) for t in DEV}
    sets.update({k: SV.rows_of(v) for k, v in NEW.items()})
    sets["B_T2s2_all4"] = bt2s2_rows(); sets["B_T2s2_std3"] = bt2s2_rows(BT.STD_SPEEDS)
    for name, rows in sets.items():
        rows = with_bcol([dict(r) for r in rows if np.isfinite(r["lower"][0]) or np.isfinite(r["lower"][1])])
        out = {}
        for variant, rr in (("measured", rows), ("detect_0.2N", with_bcol(detection(rows)))):
            bv, bb = fit(rr); cv, cb = boot(rr) if name not in DEV else (None, None)
            hs, a = classify(bv, bb)
            n_left = sum(1 for r in rr if not np.isfinite(r["lower"][0]))
            out[variant] = dict(b_v=bv, b_v_ci=cv, b_beta=bb, b_beta_ci=cb, H_S=hs, A_reading=a if name in ("A3", "A4") else None, n=len(rr), n_left_censored=n_left)
        res[name] = out
    (HERE / "results.json").write_text(json.dumps(res, indent=1, default=float))
    for name, out in res.items():
        m, d = out["measured"], out["detect_0.2N"]
        ci = lambda c: f"[{c[0]:+.2f},{c[1]:+.2f}]" if c else ""
        print(f"{name:12s} v {m['b_v']:+.2f}{ci(m['b_v_ci'])} β {m['b_beta']:+.2f}{ci(m['b_beta_ci'])} H_S {m['H_S']} {m['A_reading'] or ''} (n {m['n']}, left {m['n_left_censored']}) | det0.2N v {d['b_v']:+.2f} β {d['b_beta']:+.2f}{ci(d['b_beta_ci'])} H_S {d['H_S']}")


if __name__ == "__main__":
    main()
