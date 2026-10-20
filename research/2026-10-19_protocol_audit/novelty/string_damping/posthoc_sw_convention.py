"""POST HOC: T1 ratio Q in the convention of Schoonderwaldt et al. 2008 (R = 2 Z f1 tau1, i.e. Z/R = pi*zeta_1, fundamental only)
instead of the plan's energy-weighted S10. Q_SW = Q * S10 / (pi^2 zeta_1 / 6). Also within- vs between-type spread. Claude, 2026-10-21."""
import json, math, statistics as st
from pathlib import Path
HERE = Path(__file__).resolve().parent
r = json.loads((HERE / "results.json").read_text()); out = {}
TYPE = {"A1": "A T1", "A2": "A T1", "A3": "A T1", "A4": "A T1", "B1": "B T1", "C1": "C T1", "D1": "D T1", "D2": "D T1", "A_T2": "A T2",
        "B_T2s1": "B T2", "B_T2s2": "B T2", "B_T2p6": "B T2", "C_T2": "C T2", "D_T2": "D T2"}
for ax in ("onset", "window"):
    for t, x in r[ax]["T1"].items():
        d = r["damping"][t]; out.setdefault(ax, {})[t] = x["Q"] * d["S10"] / (math.pi ** 2 * d["zeta1"] / 6)
    T = {}
    for s, v in r[ax]["sessions"].items():
        T.setdefault(TYPE[s], []).append(v["level"])
    within = [x - st.mean(v) for v in T.values() if len(v) > 1 for x in v]; dfw = sum(len(v) - 1 for v in T.values() if len(v) > 1)
    out[ax + "_spread"] = dict(within_sd=math.sqrt(sum(w * w for w in within) / dfw), df=dfw, between_sd=st.stdev([st.mean(v) for v in T.values()]))
(HERE / "posthoc_sw_convention.json").write_text(json.dumps(out, indent=1)); print(json.dumps(out, indent=1))
