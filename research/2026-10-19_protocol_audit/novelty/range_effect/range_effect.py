"""Scan-range (ceiling) truncation analysis — post hoc, plan in README.md (Claude, 2026-10-20)."""
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

HOME = Path.home()
sys.path.insert(0, str(HOME / "violin-rne/confirm_B_T2s2"))
sys.path.insert(0, str(HOME / "violin-rne/highforce_confirm"))
sys.path.insert(0, str(HOME / "violin-rne/claude_extra"))
import score_bt2s2 as BT  # noqa: E402
import score_highforce as HF  # noqa: E402
import thermal_law as TL  # noqa: E402
import thermal_law2 as TL2  # noqa: E402
import verify_astra_boundaries as V  # noqa: E402

F2 = json.loads((HOME / "violin-rne/novelty/thermal_upper/FROZEN_v2.json").read_text())
HERE = Path(__file__).resolve().parent


def columns_truncated(win, beta, speed, assign, c, t=0.2):
    col = assign(beta)
    g = defaultdict(list)
    for k, d in win.items():
        if t in d and d[t][0] <= c:
            g[(k[0], col[k])].append((d[t][0], d[t][1], beta[k]))
    out = []
    for (folder, cc), rows in sorted(g.items()):
        if len(rows) < 2:
            continue
        rows.sort()
        f = np.log(np.maximum(np.array([x[0] for x in rows]), 1e-3)); h = [x[1] for x in rows]
        lo_, up = BT.intervals(f, h)
        out.append(dict(v=speed[folder], lv=float(np.log(speed[folder])), lb=float(np.log(np.median([x[2] for x in rows]))), upper=up))
    return out


def stats(cols, group, speeds=None):
    keep = [c for c in cols if (np.isfinite(c["upper"][0]) or np.isfinite(c["upper"][1]))
            and (speeds is None or min(abs(c["v"] - s) for s in speeds) < 1e-3)]
    nb = sum(1 for c in cols if not (np.isfinite(c["upper"][0]) or np.isfinite(c["upper"][1])) and (speeds is None or min(abs(c["v"] - s) for s in speeds) < 1e-3))
    lv = np.array([c["lv"] for c in keep]); lb = np.array([c["lb"] for c in keep])
    lo = np.array([c["upper"][0] for c in keep]); hi = np.array([c["upper"][1] for c in keep])
    lz = np.full(len(lv), np.log(F2["z_published"][group][1]))
    out = dict(n=len(keep), finite=int(np.sum(np.isfinite(lo) & np.isfinite(hi))), right=int(np.sum(~np.isfinite(hi))), no_band=nb)
    if len(keep) < 5:
        return out, None
    for m in ("S1_strict", "E3_free_Z"):
        mu = TL2.predict(m, F2["fits"][m]["theta"], lv, lb, lz)
        out[m] = TL.offset_fit(mu, lo, hi)[0]
    p = BT.free_fit(lv, lb, lo, hi)
    out["e_v"], out["e_beta"] = float(p[1]), float(p[2])
    return out, (lv, lb, lo, hi, lz)


def main():
    res = {}
    # B_T2 sample 2
    win, beta, speed = BT.read_labels(V.rm, V.APPLIED)
    res["B_T2s2"] = {}
    for c in (2, 3, 4, 5, 6, 99):
        cols = columns_truncated(win, beta, speed, BT.assign_columns, c)
        res["B_T2s2"][str(c)] = dict(std3=stats(cols, "B T2", BT.STD_SPEEDS)[0], all4=stats(cols, "B T2")[0])
    # paired bootstrap of the c=4 -> full change (standard speeds), beta-column clusters paired across speeds
    full = columns_truncated(win, beta, speed, BT.assign_columns, 99); c4 = columns_truncated(win, beta, speed, BT.assign_columns, 4)
    def by_beta(cols):
        d = defaultdict(list)
        for c in cols:
            if min(abs(c["v"] - s) for s in BT.STD_SPEEDS) < 1e-3:
                d[round(c["lb"], 2)].append(c)
        return d
    bf, b4 = by_beta(full), by_beta(c4)
    keys = sorted(set(bf) & set(b4)); rng = np.random.default_rng(31); diffs = []
    for _ in range(1000):
        kk = rng.choice(keys, len(keys))
        a, _ = stats([c for k in kk for c in bf[k]], "B T2"); b, _ = stats([c for k in kk for c in b4[k]], "B T2")
        if a.get("S1_strict") is not None and b.get("S1_strict") is not None:
            diffs.append(a["S1_strict"] - b["S1_strict"])
    res["B_T2s2_change_full_minus_c4"] = dict(point=res["B_T2s2"]["99"]["std3"]["S1_strict"] - res["B_T2s2"]["4"]["std3"]["S1_strict"],
                                             ci=np.percentile(diffs, [2.5, 97.5]).tolist(), n_boot=len(diffs))
    # high-force sessions
    for arch in ("HighBowForceRange_1", "HighBowForceRange_2"):
        data = HF.read_labels(arch, V.rm, V.APPLIED)
        for tgt, d in data.items():
            grp = HF.TARGETS[tgt][2]
            res[tgt] = {}
            for c in (7, 9, 99):
                cols = columns_truncated(d["win"], d["beta"], d["speed"], HF.assign_columns, c)
                res[tgt][str(c)] = stats(cols, grp)[0]
    (HERE / "results.json").write_text(json.dumps(res, indent=1, default=float))
    print("B_T2s2 (standard speeds | all four)")
    for c, x in res["B_T2s2"].items():
        s, a = x["std3"], x["all4"]
        print(f"  c={c:>2} N: S1 {s.get('S1_strict', float('nan')):+.3f} E3 {s.get('E3_free_Z', float('nan')):+.3f} e_v {s.get('e_v', float('nan')):+.2f} e_b {s.get('e_beta', float('nan')):+.2f} "
              f"(n {s['n']}, fin {s['finite']}, right {s['right']}, noband {s['no_band']}) | all4 S1 {a.get('S1_strict', float('nan')):+.3f}")
    print("  change full - c4 (S1, std3):", res["B_T2s2_change_full_minus_c4"])
    for tgt in ("HF_C", "HF_A", "HF_D", "HF_B"):
        print(tgt, " | ".join(f"c={c}: S1 {x.get('S1_strict', float('nan')):+.3f} E3 {x.get('E3_free_Z', float('nan')):+.3f} e_b {x.get('e_beta', float('nan')):+.2f} (n {x['n']}, right {x['right']}, noband {x['no_band']})" for c, x in res[tgt].items()))


if __name__ == "__main__":
    main()
