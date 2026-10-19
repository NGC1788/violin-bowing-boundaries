"""Absolute lower/upper boundary of every processed string (t = 0.2 s windows, K = 2), from the project's own
cached labelled windows (read-only import of ~/violin-rne/final/{data,compare}.py; run with python -B so that
nothing is written there). Fits a free law per string: log F = a + b log v + c log beta (interval-censored),
and prints the law and the predicted boundary at the (beta, v) points of the friction-reconstruction strokes.
Output: results/boundary_laws.json"""
import json
import sys
from pathlib import Path

import numpy as np

FINAL = Path.home() / "violin-rne/final"
sys.path.insert(0, str(FINAL))
_ARGV = sys.argv[:]; sys.argv = sys.argv[:1]  # compare.py parses argv at import
import compare as C  # noqa: E402
import data as D  # noqa: E402

D.STEMS["A_T2"] = "2024-06-19_TypeA_T2_sample1"
T = 0.2
OUT = Path(_ARGV[1]) if len(_ARGV) > 1 else Path("results/boundary_laws.json")


def design(r):
    return np.array([1.0, r["lv"], r["lb"]]), 0.0


def main():
    res = {}
    pts = [(0.0405, 0.05), (0.0405, 0.10), (0.1044, 0.05), (0.1044, 0.10), (0.1175, 0.05), (0.1175, 0.20),
           (0.0484, 0.05), (0.0484, 0.20), (0.1777, 0.10)]
    for tag in ["A1", "A2", "B1", "C1", "D1", "A_T2", "B_T2r1", "B_T2r2"]:
        rows = C.intervals(D.load(tag), T)
        out = {}
        for which in ("lower", "upper"):
            th = C.censored_fit(rows, design, which)
            a, b, c, ls = th
            out[which] = dict(a=float(a), b_speed=float(b), c_beta=float(c), sigma=float(np.exp(ls)),
                              at={f"{bb}_{vv}": float(np.exp(a + b * np.log(vv) + c * np.log(bb))) for bb, vv in pts})
        # columns near beta 0.104 and 0.0405 : raw intervals (N) per speed
        near = {}
        for bb in (0.0405, 0.1044, 0.1777):
            for r in rows:
                if abs(np.exp(r["lb"]) - bb) / bb < 0.06:
                    key = f"{bb}_{np.exp(r['lv']):.3f}"
                    near.setdefault(key, []).append(dict(beta=float(np.exp(r['lb'])), lower=[float(np.exp(x)) if np.isfinite(x) else None for x in r['lower']],
                                                         upper=[float(np.exp(x)) if np.isfinite(x) else None for x in r['upper']]))
        out["near_columns"] = near
        res[tag] = out
        lo, up = out["lower"], out["upper"]
        print(f"{tag:7s} lower: v^{lo['b_speed']:+.2f} beta^{lo['c_beta']:+.2f} (sd {lo['sigma']:.2f}) | upper: v^{up['b_speed']:+.2f} beta^{up['c_beta']:+.2f} (sd {up['sigma']:.2f})")
        print("        " + "  ".join(f"[{k}] {lo['at'][k]:.2f}/{up['at'][k]:.2f}" for k in lo['at']))
    OUT.write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
