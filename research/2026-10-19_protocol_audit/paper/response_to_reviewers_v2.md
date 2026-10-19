# Response to the internal reviews of draft v1 → draft v2 (Claude, 2026-10-19 22:00 KST)

Sources: `paper/referee_reviews_v1.md` (Referees 1 and 2, workflow wf_34724f06-87c) and
`astra_20261019/paper_review_root.md` (root review, items 1–7). Numbers quoted in the reviews were re-checked against
the named files before use; where I recomputed, the file is named. "Not done" items are listed honestly.

## Root review (paper_review_root.md)

| # | Point | Change in v2 |
|---|---|---|
| 1 | D_T2 secondary column medians must not use the 0.23 level margin | §3.3 D_T2 now uses the recommended wording ("the predeclared level-scale margin does not define a decision for this secondary metric"); primary stays untestable |
| 2 | Table 1 denominators, conditional errors, point-hit ± grid step | Table 1 caption rewritten (finite intersection, counts differ by model, ±log40/49 point-hit ≠ interval coverage, pre-correction GBM, Beigi=Schelleng upper) |
| 3 | A4 result | New §3.3 "A4 (frozen)": H1/H1b met vs lookup_clip, H2 failed, H3 nothing added, mean_only better on all three metrics, denominators 117 vs 120 |
| 4 | no-band as lower-right-censored is an assumption | §2 Boundaries states the non-identifiability; §4 recommendation; §5 limitation |
| 5 | High-force data are different sessions, not an extension | §2 High-force sessions; §3.2 scored per string with its own Z, no merging with sample1 |
| 6 | Synthetic band model is a diagnostic, not a new method result | §3.5: one frozen test–retest check on development data → no consistent improvement → direction ended; prior art [11] cited |
| 7 | No claims on independent string counts, session variance, violin | §2 counts diagrams; §5 scope bullet (no violin measurement) |

## Referee 1

| # | Severity | Change |
|---|---|---|
| 1 | critical — attack follows trial order; A1 reverse scan | Verified (`paper/scripts/attack_scan_order.py`, `attack_scan_all.py`; logs in `paper/repro/`): within-column ρ(accel, trial) +0.98…+1.00; A1 low-force attack 0.563 vs A2 0.038/0.104/0.417 m/s². New §3.1 paragraph; A1 named as competing explanation and as the leverage point in §3.3; A1 is the only reverse scan among 8 diagrams with metadata |
| 2 | critical — force-axis coupling; 8-diagram correlations | §3.3: onset-axis SD 0.125, nominal-axis r, 8-diagram EI +0.86 vs dec5_pl +0.69; recommendation changed to "report both axes"; "Two boundaries, two behaviours" framing removed |
| 3 | critical — A_T2 −0.10 truncation | Withdrawn; replaced by detection-limit censoring −0.42 (0.2 N), −0.29 (0.3 N), onset axis −0.19 (`novelty/lower_protocol/levels_variants.log`, `analysis2.log`) |
| 4 | critical — Z claim overstated | Abstract and §3.2 rewritten: χ²(g=1) 14.2/3 df, Z ∝ T (T/Z = 137.2 m/s), EI r 0.80 vs Z 0.71, C_T2 Z uncertainty. Torsion-corrected impedance attempted (`robust_upper/` U4): not computable in absolute terms; ratio-only sensitivity leaves A and D unidentified. Flyback-Z discussion still not added |
| 5 | major — ceiling censoring | §3.2 censoring shares by speed (25–42/50–68/56–83 %); D_T2 reported with column medians and per-speed residuals. 0.20 m/s exclusion run (`robust_upper/` U1): pooled v^0.86 β^−1.19; A_T2/D_T2 conclusions unchanged |
| 6 | major — A1 drift direction | Corrected (+4 → +158 mN, pose-dependent offset) |
| 7 | major — 16× only at small β | Ratios by β tercile 16×/11×/1.4× (verified) |
| 8 | major — window-placement confound | Renamed; per-diagram matched-time exponents added; the five pooled diagrams named |
| 9 | major — level fragility | Detection-limit r, Fisher CI, zero-offset table compiled (`robust_lower/` L5), D_T2 near the A1 end |
| 10 | major — losses omitted | Limitation added; "string properties" restricted to Z, EI, T, d, Zto |
| 11 | major — A3 | §2 states A3 was processed, not analysed, and why it was excluded from frozen tests; string-count sentence corrected |
| 12 | major — thermal paragraph | Cut to a short hedged paragraph; Δμ vs μ_d distinguished; thermal laws lost in all four frozen high-force sessions |
| 13 | major — pre-registration details | Evidence-level table in §2 (A_T2 no hash, "before any label or boundary", D_T2 "before any outcome column was parsed"); FROZEN_v2/v3 are byte-identical in GitHub commit d6b4081 (12:02) before the high-force archives were processed. Original development plan (flyback Z, few-shot k = 10) and its post hoc change disclosed in §2 Models |
| 14 | major — Table 1 code mix | Footnoted (pre-correction GBM; Beigi=Schelleng upper; legacy lookup 0.266) |
| 15 | major — same-session descriptor | §2 and abstract say "same-session" |
| 16 | minor — data details | diagram defined vs POMA; six T2 with D_T2 held out; log-spaced β; 50 kHz from the record description; K = 3 upper sensitivity run (`robust_upper/` U2), little change |
| 17 | minor — log base | "log = natural log" in §1 |
| 18 | minor — shape source | bootstrap units named |
| 19 | minor — A1 travel collapse | A1 SDs added; POMA "almost constant" quoted |
| 20 | minor — references | [4] marked preprint/simulation; [6] Woodhouse 2003 no longer cited for establishment/maintenance; [14] split to a data citation; [5], [7], [9] still need checking |
| 21 | minor — two A CIs; abstract length | Second CI labelled post hoc; abstract ~150 words, limit still to be checked |

## Referee 2

| # | Severity | Change |
|---|---|---|
| 1 | critical — g = 1 rejected; D pair | χ² 14.2/3 df reported; D pair added g = 0.23 [−0.25, 0.70] (recomputed from `novelty/upper_final/levels.json`); D_T2 described as cross-type calibration success |
| 2 | critical — fixed-effect pooled CI | Random-effects (DerSimonian–Laird, recomputed) +1.37 [−0.49, 3.23], t(2) [−2.7, 5.5]; τ ≈ 0.12 = T1 residual SD; C pair not identified given published Z uncertainty |
| 3 | major — D_T2 0.002 misread as precision | Per-speed residuals +0.07/−0.01/−0.55 (recomputed from the D_T2 result/controls files), σ 0.396, Z-exponent interval 0.37–1.74 |
| 4 | major — A_T2 −0.10 | As Referee 1 #3 |
| 5 | major — detection-limit sensitivity; mechanical null | r under c = 0.2/0.3/0.4 N; mechanical null run (`robust_lower/` L1): not ruled out (§3.6) |
| 6 | major — selection outside CV | Max-|r| permutation over the screen: p_adj 0.010 (27 pairs) / 0.028 (135 pairs), stated as optimistic (§3.6) |
| 7 | major — drift direction | Corrected |
| 8 | major — A_T2 yardsticks | Added (lower 0.233, upper 0.005; −0.65 vs A1, −0.42 vs A2) |
| 9 | major — D_T2 deferred secondaries, bootstrap failures | Disclosed (K3, t = 0.1/0.3, nominal axis not run; 1272/2000 failures) |
| 10 | moderate — within-condition comparison | ANOVA numbers and B_T2 repeat interval overlap added |
| 11 | moderate — repeat uncertainty | Upper repeat SEs not yet added to the text (in upper_law_t0.2.log) |
| 12 | moderate — mixed metrics | A_T2 column median 0.121 vs D_T2 0.124; A_T2 level error 0.082 vs D_T2 0.002; A_T2 marked secondary |
| 13 | moderate — reference-shape mis-specification | Per-diagram exponents and fixed-reference-point r (0.90 on 10 diagrams, 0.60 on 8; `novelty/lower_protocol/analysis2.log` §4) added |
| 14 | moderate — sub-linear speed exponent and ceiling | High-force e_v 0.57–0.91 with ceiling lifted (frozen-scored, descriptive) |
| 15 | moderate — like-for-like exponents | Pooled set named; matched-time range for all diagrams added |
| 16 | moderate — bootstrap settings | Canonical β-cluster bootstrap B = 2000 for upper quantities (`robust_upper/` U3) |
| 17 | moderate — no uncertainty on model differences | Bold removed; paired example and LOTO added |
| 18 | moderate — freeze evidence; contamination | Evidence table; B_T2 friction-session contamination disclosed |
| 19 | minor — D_T2 at A1 edge; negative H forces | Added |
| 20 | minor — no-band sensitivity; ML pseudo-replication | No-band-as-missing: r 0.90 (§3.6); ML caveat in limitations |
| 21 | minor — terminology | Development set named; A1 travel SDs; attack shares cite `paper/repro/attack_design_A2.log` |

## New since v1 (not requested by the reviews)

- High-force frozen confirmation (4 strings): failed level transfer; Table 2, Fig. 4, `highforce_confirm/`.
- A4 frozen result.
- Band-model method candidate: frozen test–retest, no improvement, direction ended.

## Added 2026-10-19 23h

Post hoc robustness (§3.6): three tracks run by separate analysts and re-run by independent verifiers (workflow wf_80e89ee6-df4); verifier corrections adopted verbatim in wording. D2 frozen results added (§3.2, §3.3).
