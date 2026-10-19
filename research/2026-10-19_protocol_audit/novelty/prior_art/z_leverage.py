"""POST HOC leverage check of the D_T2 upper-boundary confirmation (prior-art / reviewer audit).

Question: how strongly does the D_T2 hit (strict cZv/beta level error 0.0018 log) constrain the Z exponent,
and can a Z law be told apart from a "T2 batch offset" with the existing diagrams?

Inputs (read only):
  novelty/upper_law/upper_law_t0.2.json  -> strict v/beta law, ln c calibrated on pooled T1 columns, per-diagram residual levels
  astra_20261019/D_T2_confirm_result.json -> scored D_T2 level (already scored once; this script does not rescore anything)
Published Z: Lampis, Chatziioannou & Scavone, POMA 58, 035013 (2025) Table 1 (copied from novelty/upper_law/upper_law.py).

Everything here is post hoc: D_T2 is already spent. No future archive is opened.
"""
import json, math
from pathlib import Path

ROOT = Path("/Users/jeongjihoon/violin-rne")
OUT = Path(__file__).parent / "z_leverage.json"

Z = {"A T1": 1.0592, "B T1": 1.0998, "C T1": 1.0525, "D T1": 1.0398,
     "A T2": 1.1884, "B T2": 1.1899, "C T2": 1.1154, "D T2": 1.2021}
GROUP = {"A1": "A T1", "A2": "A T1", "B1": "B T1", "C1": "C T1", "D1": "D T1", "A_T2": "A T2",
         "B_T2s1": "B T2", "B_T2r1": "B T2", "B_T2r2": "B T2", "C_T2": "C T2", "D_T2": "D T2"}
T1 = ["A1", "A2", "B1", "C1", "D1"]
T2_old = ["A_T2", "B_T2s1", "B_T2r1", "B_T2r2", "C_T2"]
TOL = 0.12  # frozen practical tolerance for the upper level (CONFIRM_PLAN_D_T2)

up = json.loads((ROOT / "novelty/upper_law/upper_law_t0.2.json").read_text())["schelleng_global"]["strict v/beta"]
res = dict(up["resid_level"])  # obs - pred, strict law with published Z
dt2 = json.loads((ROOT / "astra_20261019/D_T2_confirm_result.json").read_text())
res["D_T2"] = float(dt2["level_fits"]["upper_strict"]["level"])
boot = dt2["bootstrap"]["upper_strict"].get("percentile_95_conditional_successes")

lnZ = {t: math.log(Z[GROUP[t]]) for t in res}
# ln c was calibrated on pooled T1 columns (equal column count per diagram; A T1 appears twice)
mean_lnZ_T1 = sum(lnZ[t] for t in T1) / len(T1)

def err_k(t, k):
    """level error (obs - pred) of a Z^k law whose constant is recalibrated on the same T1 pool.
    With k = 1 this is the frozen strict prediction; k = 0 is a fair no-Z baseline (constant recalibrated without Z)."""
    return res[t] - (k - 1.0) * (lnZ[t] - mean_lnZ_T1)

out = {"status": "post hoc; D_T2 already scored once; no rescoring, no new archive",
       "mean_lnZ_T1": mean_lnZ_T1, "lnZ": lnZ,
       "lnZ_range_all_strings": [min(lnZ.values()), max(lnZ.values())],
       "T1_between_diagram_SD_of_strict_residual": up["sd_T1_resid"],
       "per_diagram": {}}
for t in T1 + T2_old + ["D_T2"]:
    out["per_diagram"][t] = {"strict_Z1_error": err_k(t, 1.0), "fair_noZ_error": err_k(t, 0.0),
                             "Z2_error": err_k(t, 2.0), "leverage_lnZ_minus_T1mean": lnZ[t] - mean_lnZ_T1}
lev = lnZ["D_T2"] - mean_lnZ_T1
e1 = res["D_T2"]
# |e1 - (k-1)*lev| < TOL  ->  k in (1 + (e1 - TOL)/lev, 1 + (e1 + TOL)/lev)
out["D_T2"] = {"strict_error": e1, "frozen_noZ_comparator_error": abs(e1 + lnZ["D_T2"]),
               "frozen_noZ_comparator_note": "confirm_d_t2.py drops log Z from the strict prediction without recalibrating c; it is not a fair no-Z model",
               "fair_noZ_error": err_k("D_T2", 0.0), "leverage": lev,
               "Z_exponent_interval_passing_tolerance": [1 + (e1 - TOL) / lev, 1 + (e1 + TOL) / lev],
               "conditional_bootstrap_95_level": boot,
               "Z_exponent_interval_from_bootstrap_level_interval": None if not boot else
                   sorted([1 + (boot[0]) / lev, 1 + (boot[1]) / lev])}
# alternative: one constant T2 batch offset learned from the earlier T2 diagrams (no Z); B averaged over its 3 diagrams first
noz = {t: err_k(t, 0.0) for t in T2_old}
strings = {"A": [noz["A_T2"]], "B": [noz["B_T2s1"], noz["B_T2r1"], noz["B_T2r2"]], "C": [noz["C_T2"]]}
delta = sum(sum(v) / len(v) for v in strings.values()) / len(strings)
out["batch_offset_alternative"] = {
    "delta_T2_from_A_B_C": delta,
    "D_T2_error_batch_model": err_k("D_T2", 0.0) - delta,
    "D_T2_error_Z1_model": e1,
    "note": "post hoc: a T2-session offset without Z would also have hit D_T2 within tolerance; Z and T1/T2 batch are not separable in these data"}
# within-T2 leverage: does Z explain level differences among T2 strings?
t2s = {"A": "A_T2", "B": None, "C": "C_T2", "D": "D_T2"}
lvl_noZ = {"A": noz["A_T2"], "B": sum(strings["B"]) / 3, "C": noz["C_T2"], "D": err_k("D_T2", 0.0)}
lz = {"A": lnZ["A_T2"], "B": lnZ["B_T2s1"], "C": lnZ["C_T2"], "D": lnZ["D_T2"]}
mz = sum(lz.values()) / 4; ml = sum(lvl_noZ.values()) / 4
sxx = sum((lz[s] - mz) ** 2 for s in lz); sxy = sum((lz[s] - mz) * (lvl_noZ[s] - ml) for s in lz)
out["within_T2_slope_of_noZ_level_on_lnZ"] = {"slope": sxy / sxx, "n_strings": 4, "lnZ_SD": math.sqrt(sxx / 3),
                                              "note": "4 strings, lnZ SD ~0.03; slope is not interpretable as an exponent estimate"}
OUT.write_text(json.dumps(out, indent=2, ensure_ascii=False))
print(json.dumps(out, indent=2, ensure_ascii=False))
