# robust_lower — robustness of the post hoc minimum-force association (dec5_pl, r = 0.88)

Claude subagent, 2026-10-19. **Everything here is POST HOC** on development data that have already been
analysed (10 diagrams A1, A2, B1, C1, D1, A_T2, B_T2s1, B_T2r1, B_T2r2, C_T2; D_T2 is not part of the
10-diagram association and is not used). No reserved data (A4, D2, B_T2 sample 2, coldwarm) are read.
Natural log throughout. "Level" = interval-censored offset of a diagram's t = 0.2 s, K = 2 lower boundary
from the A1+A2 P-free reference law without Z (novelty/lower_protocol/descriptors.py). dec5_pl = median
(F at plateau travel >= 5 cm − F at plateau start), nominal force ranks 1–20, mean of the 3 speed folders (N).
Reference values: r(dec5_pl, level) = +0.880, slope +7.67 log/N (analysis2.log §5), n = 10 diagrams.

## PLAN (written 2026-10-19 22:36, before any computation in this folder)

Only existing logs/json were read before this plan (descriptors.log, analysis.log, analysis2.log,
analysis3.log, levels_variants.log, reverse_check.log, final_checks.log, astra_20261019/protocol_results/results.json).

### L1 Mechanical null (referee 2 #5)
Question: if each diagram's lower boundary were fixed in onset-force terms (force before the bow moves) and
the window reading simply dropped by that session's measured decrease, which level shift would each diagram show,
and how much of the between-diagram level variance does that alone explain?
- Per stroke q = log(max(F_w, 1e-3 N) / F_onset), F_w = cached 0.2 s window force at t = 0.2 s
  (final/cache, same reading as the boundary), F_onset = `force_at_onset` from label-free ramp metadata.
- **Primary stroke set**: nominal force ranks 1–20 in every speed folder (label-free, independent of the reading,
  same strokes as dec5_pl). Folder median of q; diagram q_d = mean of the 3 folder medians.
- Mechanical predicted level m_d = q_d − mean(q_A1, q_A2) (the reference law is fitted on A1+A2 pooled).
- Denominator: the 7 diagrams with ramp metadata (A1, A2, C1, D1, A_T2, B_T2s1, C_T2); B1 and B_T2r1/r2 have none.
- Statistics: (a) OLS level = a + b·m, b with t-based 95% CI (5 df), r, LODO-CV R² (as analysis.py);
  (b) share of between-diagram variance explained by the mechanical null alone with slope fixed at 1:
  R²_1 = 1 − Σ(L − m − c)² / Σ(L − L̄)², c = mean(L − m) (one free constant), and R²_1,0 with c = 0;
  (c) implied mechanical slope in log/N = OLS slope of m_d on dec5_pl_d (7 diagrams), compared with the
  fitted +7.67 log/N (10 diagrams) and with the observed slope of level on dec5_pl over the same 7;
  (d) partial check: r(dec5_pl, L − m) and r(dec5_pl, L) on the 7 diagrams.
- Secondary a: same, strokes selected by F_onset in [0.15, 0.45] N (selection on the pre-motion force, not on the reading).
- Secondary b (cache only, all 10): q = log(max(F_w(0.2), 1e-3) / max(F_w(0), 1e-3)), ranks 1–20
  (boundary fixed in plateau-start-force terms instead of onset terms).
- Context (analytic additive version): if the reading drops by an additive Δ = dec5_pl at a boundary force F_b,
  the log shift is log(1 + Δ/F_b), slope 1/F_b per N; report mean over the 120 A1-grid columns of 1/F_ref
  (F_ref = reference-law lower boundary) as the implied additive slope.
- Decision wording (primary): b's 95% CI contains 1 and R²_1 ≥ 0.5 → "the between-diagram level differences are
  compatible with a pure reading-drop shift of a boundary fixed in onset-force terms; the association does not
  require a protocol effect on the boundary itself". R²_1 < 0.2 or CI excludes 1 → "a pure reading-drop shift
  does not account for the level differences". Otherwise → "partially accounts". Association only, either way.

### L2 Selection-adjusted permutation (referee 2 #6)
- Screen re-run on the 10 diagrams: predictors computable for all 10 = session descriptors cal_pl, dec5_pl, fmin,
  cens (analysis.py PROTO10) + string properties T, Z, EI, d, Zto (log); responses base, c = 0.2, c = 0.3
  (existing levels, descriptors.json / levels_variants.json). Statistic S = max |Pearson r| over 9 × 3 = 27 pairs.
- 10,000 permutations of the diagram labels of the response (one permutation applied jointly to all 3 response
  vectors), numpy default_rng(11). p_adj = (1 + #{S_perm ≥ |r_obs(dec5_pl, base)|}) / 10,001. Also report the
  unadjusted permutation p of dec5_pl/base from the same permutations.
- Secondary a: session descriptors only (4 × 3 = 12). Secondary b: 9 predictors × (3 + Astra's 12 K/t/axis
  variants in astra_20261019/protocol_results/results.json) = 135 pairs.
- Not covered (stated as limitation): the 7-diagram descriptors (desc_b, dec10, acc_b, cal_low, d0, floor) and
  the construction choices inside dec5_pl (5 cm, ranks 1–20, 0.1 s window) — so p_adj is still optimistic.
- Decision wording: p_adj < 0.05 → "survives the screen-wide max-|r| correction"; otherwise "does not survive".

### L3 No-band as missing (referee 2 #20)
- Columns with no K = 2 Helmholtz run are lower-right-censored (f_max, ∞) in the baseline; here they are dropped
  (missing). Primary: reference law refitted on A1+A2 with the same rule; levels of all 10; r and LODO-CV R² with
  dec5_pl. Secondary: original reference law. Report the number of no-band columns per diagram (of 120).
- Decision wording: |Δr| < 0.05 → "insensitive"; otherwise report the change.

### L4 Onset-force axis
- Onset-axis levels from novelty/lower_protocol/analysis2.json (7 diagrams; label still at t = 0.2 s, law refitted
  on the onset axis). dec5_pl r, slope, LODO-CV R², Fisher 95% CI; same on the current axis restricted to the same
  7 for comparison; plateau-start axis (10, analysis2.json) also reported. No new level computation.

### L5 Zero-offset table (referee 1 #9)
- Compile (no recomputation) from levels_variants.json: levels for c = none/0.15/0.2/0.25/0.3/0.4, zero-offset
  corrected (7 diagrams with blocks) and zero-corrected c = 0.2/0.3, each with the left-censored count/120.
- Derived only by arithmetic on compiled numbers: r(dec5_pl, level) per variant (n = 10 or 7) — all reported.

### L6 Influence
- r on all 10; drop-one r for each diagram; r without A1 (9); without A1 + A2 (8); condition-level r
  (mean level and mean dec5_pl over replicates of the same string/tension: 7 conditions); Fisher z 95% CI for each
  (SE 1/√(n−3)). No bootstrap.

General: OPENBLAS_NUM_THREADS=1, no pools, single process. Script `robust_lower.py`, output `results.json`,
`run.log`, figure `fig_robust_lower.png`.

## RESULTS (run 2026-10-19 ~22:45; `run.log`, `results.json`, `fig_robust_lower.png`)

All numbers post hoc on development data. Denominators: 10 diagrams × 120 β–speed columns (L2, L3, L6),
7 diagrams with ramp metadata (L1 primary/secondary a, L4 onset axis, L5 zero-offset), 7 string/tension
conditions (L6 condition level). No bootstrap was run in this folder (Fisher z CIs only); the only random
procedure is the L2 permutation (numpy default_rng(11), 10,000 permutations of the 10 diagram labels).

### L1 Mechanical null — "a pure reading-drop shift does not account for the level differences" (primary)
| variant | n | OLS slope of level on m [95% CI] | r | LODO R² | R² with slope fixed at 1 (free c / c = 0) | implied slope of m on dec5_pl (log/N) |
|---|---|---|---|---|---|---|
| primary: ranks 1–20, onset vs t = 0.2 s reading | 7 | +0.27 [−0.68, +1.21] | +0.31 | −1.21 | −0.62 / −1.59 | +4.80 [−5.4, +15.0] |
| secondary a: F_onset 0.15–0.45 N | 7 | +0.66 [−0.27, +1.59] | +0.63 | −0.28 | +0.30 / +0.02 | +6.63 [+0.8, +12.5] |
| secondary b: plateau start vs t = 0.2 s (cache only) | 10 | +0.29 [−1.23, +1.80] | +0.15 | −0.48 | −0.12 / −2.73 | +0.78 [−2.9, +4.5] |

Observed slope of level on dec5_pl: +8.26 [+5.0, +11.5] log/N on the same 7, +7.67 [+4.3, +11.1] on all 10.
Analytic additive version (Δ = dec5_pl at the reference-law boundary force): mean 1/F_ref over the 120 A1-grid
columns = 2.22 /N (median F_ref 0.63 N) — about 3.5× smaller than the fitted 7.67.
- Per-diagram mechanical predictions (primary, relative to A1+A2): A1 +0.38, A2 −0.38, C1 −0.04, D1 +0.19,
  A_T2 −0.23, B_T2s1 −0.44, C_T2 −0.01 vs observed +0.12, −0.12, −0.56, −0.25, −0.54, −0.26, −0.56. The null
  over-predicts the A1–A2 gap (0.76 vs 0.23) and does not predict the low C1/C_T2/A_T2 levels.
- After subtracting m, r(dec5_pl, level − m) stays positive: +0.31 (primary), +0.22 (secondary a), +0.75 (secondary b).
- Decision (pre-specified rule): primary R²_1 = −0.62 < 0.2 → "does not account"; secondary a R²_1 = +0.30 →
  "partially accounts"; secondary b → "does not account". The implied mechanical slopes (+4.8 / +6.6 log/N) are
  of the same order as the fitted +7.67 but with very wide CIs; i.e. the size of the dec5_pl slope alone does not
  rule a reading-shift explanation in or out, while the per-diagram pattern does not follow the reading drop.
- Caveat: the primary rank-1–20 set contains near-zero t = 0.2 s readings (clamped at 1 mN, as in the boundary
  extraction), so q is sensitive to the low-force tail (e.g. A2 v3 and B_T2s1 v3 folder medians ≈ −1.9). The onset
  force itself carries the same zero-offset problem as the reading. This is an association check, not causal.

### L2 Selection-adjusted permutation — "survives the screen-wide max-|r| correction"
| screen | pairs | observed max |r| (pair) | null max-|r| median / 95th pct | p_adj for |r| = 0.880 |
|---|---|---|---|---|
| primary: 9 predictors × 3 responses | 27 | 0.880 (dec5_pl, base) | 0.570 / 0.806 | **0.0103** |
| secondary a: 4 session descriptors × 3 | 12 | 0.880 (dec5_pl, base) | 0.437 / 0.752 | 0.0056 |
| secondary b: 9 × (3 + Astra's 12 K/t/axis variants) | 135 | 0.977 (cens, nominal_rank_K2_t0.1) | 0.653 / 0.854 | 0.0277 |
Unadjusted permutation p for dec5_pl/base: 0.0009 (same permutations). Not covered: the 7-diagram-only
descriptors and dec5_pl's own construction choices (5 cm, ranks 1–20, 0.1 s), so p_adj is still optimistic. In
the extended screen another pair (cens vs Astra's nominal-axis K2 t = 0.1 level, |r| 0.977) exceeds dec5_pl; cens is
itself label-derived, so this is a reminder that high |r| arises easily with n = 10, not a competing finding.

### L3 No-band columns as missing — "insensitive" (|Δr| < 0.05)
No-band columns per diagram (of 120): A1 8, A2 9, B1 8, C1 7, D1 9, A_T2 6, B_T2s1 10, B_T2r1 11, B_T2r2 11, C_T2 5.
Levels (baseline → missing, law refitted / original law): A1 +0.115 → +0.134 / +0.082; A2 −0.118 → −0.136 / −0.186;
B1 −0.357 → −0.380 / −0.434; C1 −0.561 → −0.566 / −0.627; D1 −0.248 → −0.265 / −0.322; A_T2 −0.539 → −0.537 / −0.605;
B_T2s1 −0.260 → −0.310 / −0.356; B_T2r1 −0.318 → −0.368 / −0.425; B_T2r2 −0.139 → −0.173 / −0.221; C_T2 −0.556 → −0.524 / −0.601.
r(dec5_pl): baseline +0.880 [0.56, 0.97] → +0.903 [0.63, 0.98] (refit law; slope +7.79, LODO R² 0.76) / +0.899 (original law).

### L4 Onset-force axis (levels from analysis2.json; no new level computation)
| axis | n | r [Fisher 95% CI] | slope (log/N) | LODO R² | level SD |
|---|---|---|---|---|---|
| onset force | 7 | +0.818 [+0.17, +0.97] | +3.46 | +0.44 | 0.125 |
| current reading, same 7 | 7 | +0.946 [+0.67, +0.99] | +8.26 | +0.83 | 0.258 |
| plateau start | 10 | +0.788 [+0.31, +0.95] | +5.88 | +0.50 | 0.187 |
| current reading | 10 | +0.880 [+0.56, +0.97] | +7.67 | +0.71 | 0.218 |
On the onset axis the association remains positive, but the slope and the level spread are about halved; B1,
B_T2r1, B_T2r2 have no onset metadata (denominator 7).

### L5 Detection-limit and zero-offset variants (compiled from levels_variants.json; left-censored columns of 120)
| diagram | c none | c 0.15 | c 0.2 | c 0.25 | c 0.3 | c 0.4 | zero-offset | zero-offset c 0.2 | zero-offset c 0.3 |
|---|---|---|---|---|---|---|---|---|---|
| A1 | +0.115 (0) | +0.096 (1) | +0.095 (4) | +0.092 (8) | +0.079 (17) | +0.060 (39) | +0.058 (0) | +0.047 (8) | +0.027 (26) |
| A2 | −0.118 (1) | −0.102 (11) | −0.106 (21) | −0.104 (28) | −0.089 (35) | −0.065 (51) | −0.059 (1) | −0.053 (20) | −0.031 (35) |
| B1 | −0.357 (21) | −0.335 (34) | −0.284 (41) | −0.236 (47) | −0.172 (50) | −0.063 (56) | — | — | — |
| C1 | −0.561 (27) | −0.553 (36) | −0.544 (50) | −0.498 (58) | −0.424 (62) | −0.304 (68) | −0.541 (27) | −0.509 (51) | −0.382 (63) |
| D1 | −0.248 (33) | −0.264 (33) | −0.280 (44) | −0.227 (46) | −0.190 (54) | −0.055 (58) | −0.928 (33) | −0.474 (55) | −0.284 (61) |
| A_T2 | −0.539 (32) | −0.475 (45) | −0.420 (50) | −0.354 (53) | −0.288 (56) | −0.175 (62) | −0.446 (32) | −0.360 (50) | −0.227 (56) |
| B_T2s1 | −0.260 (19) | −0.256 (40) | −0.151 (41) | −0.025 (42) | −0.016 (45) | +0.087 (53) | −0.153 (19) | −0.088 (41) | +0.057 (44) |
| B_T2r1 | −0.318 (46) | −0.338 (52) | −0.289 (53) | −0.238 (55) | −0.177 (56) | −0.038 (62) | — | — | — |
| B_T2r2 | −0.139 (51) | −0.158 (51) | −0.155 (51) | −0.137 (52) | −0.109 (55) | +0.020 (60) | — | — | — |
| C_T2 | −0.556 (43) | −0.572 (44) | −0.577 (54) | −0.522 (60) | −0.452 (64) | −0.324 (71) | −0.708 (43) | −0.607 (58) | −0.446 (65) |
| r(dec5_pl) | +0.880 (10) | +0.859 (10) | +0.795 (10) | +0.729 (10) | +0.701 (10) | +0.530 (10) | +0.561 (7) | +0.708 (7) | +0.607 (7) |
Uncorrected c = none on the same 7 diagrams: r +0.946 → after zero-offset correction +0.561 [Fisher −0.33, +0.93].
Left-censored shares rise to 0.33–0.59 at c = 0.4 N; levels at c ≥ 0.3 N rest on ≥ 45 % censored columns in 8/10 diagrams.

### L6 Influence (Fisher z 95% CI)
All 10: r +0.880 [+0.56, +0.97], LODO R² 0.71. Drop-one (n 9): A1 0.772 [0.22, 0.95], A2 0.868, B1 0.887, C1 0.918,
D1 0.880, A_T2 0.857, B_T2s1 0.886, B_T2r1 0.880, B_T2r2 0.929, C_T2 0.875 (range 0.77–0.93). Without A1: 0.772
(LODO R² 0.46); without A1 + A2 (n 8): 0.688 [−0.03, +0.94] (LODO R² 0.23, slope +8.39); 7 conditions (replicates
averaged): 0.884 [+0.39, +0.98] (LODO R² 0.68). A1 is the single most influential diagram.

### Overall (post hoc, association only, seen data)
The dec5_pl association survives the screen-wide max-|r| correction (p_adj 0.010 on 27 pairs) and treating no-band
columns as missing (r 0.90). It weakens under detection-limit censoring (0.80 / 0.70 / 0.53 at 0.2 / 0.3 / 0.4 N),
on the onset axis (0.82, slope halved, n 7), after zero-offset correction (0.56 on 7 vs 0.95 uncorrected on the same 7)
and without A1+A2 (0.69, CI touches 0). A pure mechanical reading-drop null does not reproduce the per-diagram levels
(primary R² with slope 1 = −0.62; secondary +0.30), so the association is not explained by that simple null alone,
but neither is a protocol effect on the boundary established: the CIs are wide, the descriptor's construction was
chosen post hoc, and the only held-out check (D_T2) was not better than the mean model.


## Verifier corrections (workflow wf_80e89ee6-df4, independent re-run; reproduced = True, forbidden data used = False)

**Corrected headline (use this one):** These are post hoc checks on 10 development diagrams the team has already seen. The dec5_pl association (r 0.88, n 10) survives a max-|r| correction over the 27-pair 10-diagram screen (p_adj 0.010; 0.028 over 135 pairs). That screen leaves out the 7-diagram descriptors and dec5_pl's own construction choices, so p_adj is optimistic. Treating no-band columns as missing barely changes it (r 0.90). It weakens under detection-limit censoring (0.80/0.70/0.53 at 0.2/0.3/0.4 N), on the onset-force axis (0.82, slope 3.46 vs 8.26, n 7), after zero-offset correction (0.56 vs 0.95 on the same 7) and without A1+A2 (0.69, Fisher CI [-0.03, 0.94] includes 0). A pure reading-drop null does not account for the per-diagram levels in the primary stroke set (R2 with slope 1 = -0.62, n 7) and only partially accounts for them in the onset-band set (+0.30). Its implied slope (+4.8 to +6.6 log/N, wide CIs) is compatible with the fitted +7.67, so a mechanical explanation is not ruled out. This is an association only; no protocol effect on the boundary is established.

- [moderate] Headline L2 wording — The headline says the association 'still holds after correcting for the whole descriptor screen'. The 27-pair screen leaves out the 7-diagram descriptors (PROTO7), the folder-level and within-string analyses in analysis.log, and the construction choices inside dec5_pl (5 cm, ranks 1-20, 0.1 s). The README admits this, so p_adj 0.010 is optimistic. In the extended 135-pair screen p_adj is 0.028, and the largest pair there is not dec5_pl (cens vs nominal_rank_K2_t0.1, |r| 0.977). The permutation also treats the three near-duplicate B_T2 diagrams as exchangeable.
  Correction: Say 'survives a max-|r| correction over the 27-pair 10-diagram screen (p_adj 0.010; 0.028 over 135 pairs). The screen is incomplete, so p_adj is optimistic.'
- [moderate] Headline L1 wording — 'The pure mechanical reading-drop null does not reproduce the per-diagram levels' depends on which stroke set is used. The primary set (ranks 1-20, which includes near-zero readings) gives R2_1 -0.62. The onset-band secondary set gives +0.30, which the pre-specified rule calls 'partially accounts'. The pre-specified slope comparison also does not reject the null: the implied mechanical slopes are +4.80 [-5.4, +15.0] and +6.63 [+0.8, +12.5] log/N, and both CIs include the observed +7.67/+8.26. The denominator is 7 of 10 diagrams. The headline mentions none of this.
  Correction: Say: 'By the pre-specified rule, a pure reading-drop null does not account for the per-diagram levels in the primary stroke set (R2 with slope 1 = -0.62, n 7) and partially accounts for them in the onset-band set (+0.30). Its implied slope (+4.8 to +6.6 log/N, wide CIs) is compatible with the fitted +7.67, so the mechanical explanation is not ruled out.'
- [minor] L6 wording 'CI reaches 0' — Without A1+A2 (n 8) the Fisher CI is [-0.03, +0.94], so it includes 0. 'Reaches' or 'touches' understates this. A1 alone moves r from 0.88 to 0.77, and the three B_T2 near-replicates make the effective n smaller than 10.
  Correction: Say 'without A1+A2: r 0.69 [-0.03, 0.94], CI includes 0, LODO R2 0.23'.
- [minor] L1 primary stroke set — The primary q takes folder medians over ranks 1-20, and these include readings near zero (clamped at 1 mN). The A2 v3 and B_T2s1 v3 medians are about -1.9 against about -0.2 to -0.5 elsewhere, so this low-force tail largely drives the primary verdict and the 0.76 A1-A2 gap. The README states this caveat, but the headline treats the primary result as decisive.
  Correction: Report primary and secondary a side by side as equally fragile. Do not call the null rejected.
- [info] Reproduction / process checks (no defect) — I re-ran robust_lower.py from a copy in verify_scratch. run.log came out byte-identical and every value in results.json matched within 1e-9. I checked a Fisher CI by hand (r 0.688, n 8 gives [-0.03, 0.94]) and the analytic ratio (7.67/2.22 = 3.45). Grepping the script, descriptors.py, levels_variants.py, analysis.py and final/data.py found no forbidden stems; the script loads only the 10 TAGS, it asserts that none is forbidden, and it reads ramp CSVs only for the 7 diagrams that have them. In README.md the PLAN section comes before RESULTS, with every one of L1-L6 and its variants planned and then reported. The mechanical-prediction sign convention is correct. I could not verify chronology beyond file order: README mtime 22:39, results.json 22:38.
  Correction: None needed.
