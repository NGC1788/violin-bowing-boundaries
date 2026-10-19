"""Freeze v3 = ADDENDUM to FROZEN_v2 (Claude, 2026-10-19): curvature predictions (speed curvature e(v2->v3) - e(v1->v2)
and the lv*lb interaction sign), plus hashes of score_target.py / analysis2b.py. Does not change any v2 number.
Built from the 11 processed diagrams only; no target feature opened. Usage: python freeze3.py v3"""
import datetime as dt
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

import rows as R
import thermal_law2 as TL2

HERE = Path(__file__).parent
VER = sys.argv[1] if len(sys.argv) > 1 else "v3"


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    A2 = json.loads((HERE / "analysis2.json").read_text())
    A2b = json.loads((HERE / "analysis2b.json").read_text())
    V2 = json.loads((HERE / "FROZEN_v2.json").read_text())
    RW = R.all_rows()
    curv = np.array([A2["speed_curvature"][t]["curvature"] for t in R.TAGS])
    sd = float(np.std(curv, ddof=1))
    pooled = A2b["speed_curvature_pooled"]["curvature"]
    out = dict(version=VER, addendum_to="FROZEN_v2.json", v2_sha256=sha(HERE / "FROZEN_v2.json"),
               created_kst=dt.datetime.now(dt.timezone(dt.timedelta(hours=9))).isoformat(),
               status="UNSEEN: A4, D2, B_T2 sample2, HighBowForceRange_1/2, coldwarm not processed; A3 in processing (v1 governs A3).",
               empirical=dict(per_diagram_curvature=dict(zip(R.TAGS, curv.tolist())), between_diagram_sd=sd, pooled_curvature=pooled,
                              pooled_ci=A2b["speed_curvature_pooled"]["curvature_ci"], interaction_g=A2b["beta_curvature_measured"]["g"],
                              interaction_g_ci=A2b["beta_curvature_measured"]["g_ci"]),
               targets={})
    for stem, d in V2["targets"].items():
        donors = d["design_donors"]
        des = [r for t in donors for r in RW[t]]
        lv = np.array([r["lv"] for r in des]); lb = np.array([r["lb"] for r in des]); lz = np.full(len(des), np.log(d["Z"]))
        sp = sorted(set(np.round(lv, 6)))
        X = np.column_stack([np.isclose(lv, s).astype(float) for s in sp] + [lb])
        pm = {}
        for m in ("S1_strict", "T1k_break", "T1k_peak", "T1mu", "T2km_break", "E3_free_Z"):
            y = TL2.predict(m, V2["fits"][m]["theta"], lv, lb, lz)
            c = np.linalg.lstsq(X, y, rcond=None)[0]
            e12 = (c[1] - c[0]) / (sp[1] - sp[0]); e23 = (c[2] - c[1]) / (sp[2] - sp[1])
            pm[m] = dict(e12=float(e12), e23=float(e23), curvature=float(e23 - e12))
        out["targets"][stem] = dict(models=pm, empirical_curvature=pooled, empirical_curvature_pi90=[pooled - 1.645 * sd, pooled + 1.645 * sd],
                                    governed_by=("FROZEN_v1 (no curvature prediction was frozen for A3 before its processing)" if "sample3" in stem else VER))
    out["hypotheses"] = {
        "H_curv_empirical": "speed curvature e(v2->v3) - e(v1->v2) < 0 (concave in log v), pooled over the new T1/T2 targets A4, D2, B_T2 sample2 "
                            f"(predicted mean {pooled:+.2f}; single-diagram 90 % PI {pooled - 1.645 * sd:+.2f}..{pooled + 1.645 * sd:+.2f})",
        "H_curv_thermal": "curvature > 0 (thermal T1k_break about +0.04, T1k_peak about +0.11: more power -> smaller contrast elasticity)",
        "H_int": "interaction g(lv*lb) in the pooled-over-new-targets censored fit (analysis2b form, target offsets): empirical > 0 (11-diagram value "
                 f"{A2b['beta_curvature_measured']['g']:+.2f}); thermal < 0 (-0.03..-0.13)",
        "high_force": "if a high-force archive has >= 3 speeds: same curvature test with the ceiling lifted; ceiling censoring at 0.2 m/s is the "
                      "main alternative explanation of the concave pattern in the 11 diagrams, so a concave result with the ceiling lifted "
                      "would remove that alternative; a convex result would favour a ceiling artefact plus thermal contrast.",
        "decision": "per hypothesis: sign of the point estimate and whether the 95 % bootstrap CI excludes 0",
    }
    out["code_sha256"] = {f: sha(HERE / f) for f in ("freeze3.py", "score_target.py", "analysis2.py", "analysis2b.py", "analysis2.json",
                                                     "analysis2b.json", "thermal_law2.py", "rows.py")}
    txt = json.dumps(out, indent=1, ensure_ascii=False, default=float)
    (HERE / f"FROZEN_{VER}.json").write_text(txt)
    h = hashlib.sha256(txt.encode()).hexdigest()
    (HERE / f"FROZEN_{VER}.json.sha256").write_text(f"{h}  FROZEN_{VER}.json\n" + "\n".join(f"{v}  {k}" for k, v in out["code_sha256"].items()) + "\n")
    print("wrote", VER, h, out["created_kst"], "| between-diagram SD", round(sd, 3))
    for stem, d in out["targets"].items():
        print(stem, {m: round(x["curvature"], 3) for m, x in d["models"].items()}, "empirical", round(pooled, 3), np.round(d["empirical_curvature_pi90"], 2))


if __name__ == "__main__":
    main()
