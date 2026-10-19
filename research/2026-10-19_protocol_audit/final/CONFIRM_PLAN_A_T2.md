# Confirmatory test on the reserved specimen A_T2 — frozen before A_T2 is scored
Written 2026-10-18 21:38:36 KST by Claude, while A_T2 (2024-06-19_TypeA_T2_sample1) was still being extracted (replicate.log: v2 extraction started 21:35); nobody has looked at any A_T2 boundary or label.

Why A_T2: same string type as the training strings (A, tension T1) but higher tension. Published impedance (Lampis et al., POMA 58, 035013 (2025), Table 1): A_T1 Z = 1.0592, A_T2 Z = 1.1884 kg/s (ratio 1.122, log 0.115).
Hypotheses raised post hoc by the verification workflow on B1/C1/D1/B_T2 (wf_c40aee19-a2c): (i) a model-free lookup of the type-A boundary is as good as any model; (ii) with published Z the 3-constant physics law beats boosting and the lookup.

Setting (identical to ANALYSIS_PLAN_FINAL.md): 0.2 s windows starting 0.2 s after the plateau start; K = 2; measured window force; fixed classifier; training data A1 + A2 only; no refitting on A_T2.
Models: Triv-5 (for each test column: median boundary of the A1+A2 finite-interval midpoints over the 5 nearest beta columns at the same speed, pooled over A1 and A2); P-free with published Z (lower offset 2 log Z, upper offset log Z); P-Beigi with published Z; P-Sch with published Z; P-free without Z (offset 0); D-gbm (no Z).
Primary endpoint: median |log(F_pred / F_obs_mid)| over finite-interval columns, lower and upper separately, plus the number of finite / left / right-censored columns.
Primary hypothesis H1: P-free(published Z) < Triv-5 on both boundaries. Report also P-free(published Z) vs P-free(no Z) — the Z-scaling test.
Secondary (Z exponent): the observed mean log level shift of A_T2 boundaries relative to the A1+A2 P-free fit without Z (interval-censored fit of a single offset with the A1+A2 exponents fixed), divided by log(1.122): Schelleng predicts 2 (lower) and 1 (upper), Beigi predicts 1 (lower). Uncertainty: column bootstrap plus the A1-vs-A2 level difference as a specimen-variability yardstick.
One scoring run; any change afterwards is post hoc.
