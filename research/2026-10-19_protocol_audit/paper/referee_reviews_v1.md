# Referee reviews of draft_v1 (workflow wf_34724f06-87c, 2026-10-19 ~12:30; extracted by Claude 21:45)

## Referee 1

**verdict**: MAJOR REVISION. Most numbers in /Users/jeongjihoon/violin-rne/paper/draft_v1.md trace to their source files. I checked every number in Table 1 against compare_fixed_results/README.md and final/confirm_A_T2.log; every D_T2 value against astra_20261019/D_T2_confirm_result.json and D_T2_controls_frozen.json; levels, r and SD against novelty/lower_protocol/descriptors.json and astra_20261019/protocol_results/results.json; leverage against novelty/prior_art/z_leverage.json; pair exponents against novelty/upper_law/upper_law_t0.2.log; drift and attack values against paper/repro/*.log; nested values against astra_20261019/audit_tables.md. The labelling of pre-registered vs post hoc results is mostly honest.

However, the draft contains one factual error: the direction of the A1 zero-offset drift is reversed. Two robustness numbers are computed in a biased or over-general way: the A_T2 −0.10 shift and the claim that attacks are "16× gentler near the lower boundary". Above all, the draft omits a protocol confound that undercuts its central framing. A1 was scanned in reverse order, and the attack acceleration follows the trial index, not the force. In both A1 and A2 the within-column correlation with trial order is +0.98 to +1.00. As a result, A1's low-force strokes had about 15× harsher attacks at 0.05 m/s than A2's. That is exactly where the A1–A2 minimum-force difference sits, and A1 is also the most influential point in the r = 0.88 correlation.

The upper-boundary claim "consistent with Schelleng's impedance scaling" is overstated, for four reasons:
- The post hoc pairs reject a common Z¹ exponent (χ² = 14.2 on 3 df).
- Z is exactly proportional to T in this dataset.
- EI predicts the upper levels at least as well as Z.
- The 4-N ceiling right-censors 56–83% of the 0.20 m/s columns, and the dataset authors themselves declined to compute the upper limit at that speed.

The lower-boundary story depends on the force axis and on the two reference diagrams: on the onset-force axis the between-diagram SD falls from 0.218 to 0.125, and without A1 and A2, EI gives r = +0.86 versus r = 0.69 for the descriptor.

All numbers below come from the named files, or from read-only computations I ran on /Users/jeongjihoon/violin-rne/claude_extra/ramp/ramp_meta_2024-03-25_TypeA_sample1.csv and /Users/jeongjihoon/violin-rne/claude_extra/ramp_meta_A2.csv. I wrote no files and opened no unseen archives.

### Issue 1
- **severity**: critical
- **where**: Sec. 3.1 'Attack' and 'Sessions'; Sec. 3.3; Abstract ('attack acceleration is a design function of force and bow position')
- **problem**: The attack schedule is indexed by trial order, not by force. In both A1 and A2, acceleration rises monotonically with trial order within a β column: median within-column Spearman +1.00/+1.00/+0.98 in both diagrams, with the same maxima of 17.0, 18.9 and 19.7 m/s². A1 was recorded in reverse order: β descending and force descending (docstring of /Users/jeongjihoon/violin-rne/claude_extra/wf_20261018/skeptic_force/t3_order.py; A1 col = 39 − (trial−1)//50 in common.py). The dataset paper (/Users/jeongjihoon/violin-rne/papers/Lampis2025_POMA_58_035013.txt line 178) says the scan was 'in ascending order'. As a result, in A1 the acceleration FALLS with force (ρ = −1.00/−1.00/−0.98). At β≈0.13 and 0.05 m/s it goes from 10.2 m/s² at rank 1 to 0.15 m/s² at rank 48, whereas A2 goes from 0.12 to 7.9 m/s². The median attack for force ranks 1–20 is 0.563 m/s² in A1 at every speed, versus 0.038/0.104/0.417 in A2: ratios of 15×, 5.4× and 1.35×. The A1–A2 lower-level difference has the same speed profile. From /Users/jeongjihoon/violin-rne/paper/repro/confirm_A_T2_robust.log, A1 vs A2 is +0.243 vs −0.246 at 0.05 m/s, +0.042 vs −0.046 at 0.10 m/s and +0.053 vs −0.038 at 0.20 m/s. Under Guettler's criterion, harsher attacks need more force to establish Helmholtz motion. A1 is also the unique outlier in /Users/jeongjihoon/violin-rne/novelty/lower_protocol/analysis3.log §1 (attack vs level r = +0.73, driven by A1 alone). It is the extreme point of the dec5_pl correlation too: r is 0.88 with all ten, 0.77 without A1 (my recomputation from frozen_parameters.json values) and 0.69 without A1 and A2 (analysis2.log §4). The draft's power law a ∝ F^1.00–1.09 β^2.04–2.65 and 'rises monotonically with force rank' are artefacts of A2's ascending scan.
- **fix**: Report A1's reversed scan and the trial-order-indexed attack schedule in Sec. 3.1, and restate the attack design as 'a fixed sequence over trial index; correlated with force only when the scan is ascending'. In Sec. 3.3 and the Discussion, name attack acceleration (scan direction) as a competing explanation for the A1–A2 difference and for the leverage of A1 in the r = 0.88 association. Report r without A1 (0.77) and without A1 and A2 (0.69, p ≈ 0.06). Add a script under paper/scripts that reproduces these A1/A2 acceleration statistics from the two ramp_meta CSVs.

### Issue 2
- **severity**: critical
- **where**: Sec. 3.3 'Association with the session descriptor'; Sec. 4 'Recommendations' (use measured force) and 'Two boundaries, two behaviours'; Abstract
- **problem**: The lower-level spread is largely tied to the force axis on which boundaries are read. From /Users/jeongjihoon/violin-rne/novelty/lower_protocol/analysis2.log §2: on the onset-force axis the between-diagram SD falls from 0.218 to 0.125 log (plateau-start axis 0.187). The A_T2 lower shift falls from −0.539 to −0.189, and A1–A2 from 0.233 to 0.097. A label-free history measure, median log(F_onset/F(0.2 s)), correlates with level at r = −0.88 (analysis2.log §1, 7 diagrams). The descriptor and the boundary are therefore partly arithmetically coupled through the same decaying channel, and the draft never quantifies this. In the 8 diagrams other than the reference pair, EI correlates with the lower level at r = +0.86 (p 0.01) and torsional stiffness (Zto) at +0.85, against dec5_pl +0.69 (p 0.06) (analysis2.log §4). The draft's 'published string properties correlated more weakly (|r| ≤ 0.52)' holds only when A1 and A2 are included, and those two define the reference shape. Recommending the measured window force contradicts the draft's own statement that the cause of the decrease (real force change vs pose-dependent reading) is unresolved.
- **fix**: Report the lower levels and the A_T2 shift on the onset-force axis next to the window-force axis. Report the 8-diagram correlations: EI +0.86 vs dec5_pl +0.69. State that part of the r = 0.88 association is expected from reading boundaries on a decaying force. Replace the 'use measured force' recommendation with 'report boundaries on both onset-force and window-force axes until the cause of the decrease is resolved'. Drop or retitle 'Two boundaries, two behaviours'; it restates the framing the draft's own wording constraints (internal note C) forbid. Also note that upper levels correlate with dec5_pl at r = −0.60 (p 0.06), and dec5_pl correlates with Z at −0.52 (analysis3.log §2; analysis.log), so the asymmetric attribution is not identified.

### Issue 3
- **severity**: critical
- **where**: Sec. 3.3 'A_T2 (frozen secondary)', last sentence ('reduced the shift to −0.10 log')
- **problem**: The −0.10 comes from /Users/jeongjihoon/violin-rne/final/confirm_A_T2_robust.py. It drops A_T2 columns whose lower interval lies below 0.2 N, which is truncation on the outcome, while keeping the reference law fitted on all A1+A2 columns. The restriction is therefore applied asymmetrically. Applied to A1 and A2, the same restriction gives +0.128 and +0.063 (/Users/jeongjihoon/violin-rne/novelty/lower_protocol/descriptors.log, 'lower(>=0.2N cols)'), so the like-for-like A_T2 shift is about −0.20. The principled alternative treats readings below c as left-censored and refits the reference on A1+A2 with the same rule (/Users/jeongjihoon/violin-rne/novelty/lower_protocol/levels_variants.log). It gives A_T2 −0.48, −0.42, −0.35 and −0.29 at c = 0.15, 0.20, 0.25 and 0.30 N, and still −0.17 at 0.40 N.
- **fix**: Replace the sentence with the symmetric detection-limit result: '−0.42 at c = 0.2 N, −0.29 at 0.3 N, post hoc'. Optionally add the onset-axis value (−0.19). Also report the plan-specified specimen yardstick: A_T2 is −0.65 relative to A1 and −0.42 relative to A2 (analysis.log, tension-pair section), against an A1–A2 difference of 0.233.

### Issue 4
- **severity**: critical
- **where**: Abstract ('consistent with Schelleng's impedance scaling'); Sec. 3.2 'A_T2', 'D_T2', 'Pooled tension pairs'; Sec. 4 'What is and is not new'
- **problem**: The impedance claim is weaker than written, for six reasons.
(i) /Users/jeongjihoon/violin-rne/novelty/upper_law/upper_law_t0.2.log line 24 also gives 'chi2 for g=1 14.2 on 3 dof' (p ≈ 0.003; 12.7/3 even with τ = 0.012). Z¹ for all three pairs is rejected, and the draft reports only χ²(common) = 13.4/2.
(ii) Z is exactly proportional to T at fixed pitch and length. In POMA Table 1, T/Z = 137.2 for every string, i.e. Z = T/(2Lf0). 'Z scaling' is therefore indistinguishable from tension or linear-density scaling.
(iii) POMA states that 'the geometry of the strings had to be altered' for T2 and that EI is 32% larger in T2. Within pairs, ΔlnEI = 0.27/0.51/0.23 vs ΔlnZ = 0.115/0.079/0.058 (upper_law_t0.2.log).
(iv) EI explains the upper levels at least as well as Z: cross-string lnEI R² 0.62 vs Z¹ 0.36; permutation p is 0.10 for lnZ and 0.06 for lnEI; best-of-10 p = 0.235 (upper_law_t0.2.log). In analysis3.log §2, r = 0.80 for EI vs 0.71 for Z.
(v) The impedance derived in situ from the flyback amplitude is 0.89–0.97 kg/s, 10–25% below the published values, and mis-orders the strings (ASTRA_HANDOFF_PROMPT.md line 340). For A_T2 it is 1.028 vs a published 1.1884 (final/confirm_A_T2.log).
(vi) C_T2's published Z is 1.1154 ± 0.1023 and T is 153.04 ± 14.04 N (POMA Table 1). That uncertainty exceeds the C-pair ΔlnZ of 0.058, so the C exponent +3.25 is not identifiable.
Torsion is not discussed, although Schoonderwaldt et al. 2008 found Z scaling held only with torsional impedance (novelty/prior_art/prior_art.json, claim 1), and POMA reports torsional stiffness about 15% lower in T2.
- **fix**: Abstract: 'did not contradict Z¹ at the level of two frozen single-diagram tests; post hoc tension pairs reject a common Z¹ exponent (χ² = 14.2, 3 df); Z, T and the T2 batch are collinear and EI predicts the upper levels as well'. Report the g = 1 χ², the EI association, the flyback-Z discrepancy and the published Z uncertainties, and propagate the C_T2 Z uncertainty into the pair exponent. Add a torsion-corrected effective impedance as a sensitivity analysis, or state why it is omitted. For D_T2, report the Z-exponent interval implied by the bootstrap, 0.37–1.74 (z_leverage.json), rather than only the pass/fail.

### Issue 5
- **severity**: major
- **where**: Sec. 3.2 'Shape', 'D_T2 (frozen)'; Sec. 4 ('errors of 0.121 log (A_T2 columns) and 0.002 log (D_T2 level)'); Limitations
- **problem**: The upper boundary is heavily right-censored at the 4-N ceiling. Shares of available columns are 25–40% at 0.05 m/s, 50–68% at 0.10 m/s and 56–83% at 0.20 m/s (/Users/jeongjihoon/violin-rne/novelty/upper_final/levels.json 'census'). POMA (papers/Lampis2025_POMA_58_035013.txt lines 318–320) did not compute the upper limit at 0.2 m/s 'because the borders were not clearly identifiable'. The pooled v^0.70 and every level fit therefore depend on parametric extrapolation into censored columns. The strict cZv/β law is misspecified in speed (the exponent is 0.70, not 1): the D_T2 strict-level σ is 0.396 log, against 0.328 for the empirical shape (D_T2_confirm_result.json). A 'within 0.002 log' fit is therefore a censoring-weighted average and implies far more precision than the column-level error of 0.124. The Discussion also puts a column median (0.121) beside a level error (0.002). The strict law's column errors on the other T2 diagrams were 0.113–0.223 (astra_20261019/report_audit.md line 22).
- **fix**: Report the censoring shares by speed. Add a sensitivity analysis excluding 0.20 m/s, or restricted to columns finite at all speeds. Report D_T2 primarily as the column median (0.124 vs 0.113 for the empirical shape and 0.226 without Z), with the level error as secondary. Mention the 0.113–0.223 strict-law column errors on the other T2 diagrams. In the Discussion, compare like with like: A_T2 0.121 vs D_T2 0.124 column medians.

### Issue 6
- **severity**: major
- **where**: Sec. 3.1 'Low-force reliability' (A1 folder drift sentence)
- **problem**: The direction is reversed and the confound is omitted. /Users/jeongjihoon/violin-rne/claude_extra/wf_20261018/skeptic_force/t3_order.log line 77, for A1 v3, reads: corr with trial +0.95, corr with β −0.88, 'first/last 10 columns +158.0/+3.6 mN'. Columns are β-ordered, and in A1 (reverse scan) col < 10 are the LAST-recorded strokes. The reading therefore ROSE from about +4 to about +158 mN over the recording order; it did not drift from +158 to +4. Trial order and β are almost collinear in every folder. A2 v1/v2 show corr with β of −0.74/−0.79 under an ascending scan, and A1 v3 shows −0.88 under a descending scan. Across opposite scan directions, the offset falls consistently with increasing β. That points to a pose (β)-dependent offset rather than a time drift.
- **fix**: Rewrite as: 'In one A1 folder the post-contact reading varied from ≈ +4 mN (large β, recorded first) to ≈ +158 mN (small β, recorded last). Because β and recording order are confounded, and three folders with opposite scan directions all show the reading falling with β, the pattern is more consistent with a pose-dependent offset than with drift.' Cite t3_order.log.

### Issue 7
- **severity**: major
- **where**: Sec. 3.1 'Attack' ('Near the lower boundary, the attack at 0.05 m/s is therefore 16 times gentler than at 0.20 m/s')
- **problem**: The 16× ratio holds only for strokes on the constant-ramp floor, i.e. small β. For A2 force ranks 1–20 I split β into log-spaced terciles, using /Users/jeongjihoon/violin-rne/claude_extra/ramp_meta_A2.csv. At β < 0.043, 100% of strokes are on the floor at all speeds (16×). In the middle tercile the median attack is 0.038 vs 0.417 m/s² (about 11×). At β > 0.093, the on-floor share is 0%/4%/64% and the median attack is 0.332/0.330/0.417 m/s², a ratio of about 1.3. The draft's own off-floor fit also puts lower-boundary strokes at β ≈ 0.1 off the floor.
- **fix**: State the ratio by β range: '16× for β ≲ 0.04, ≈ 11× at mid β, ≈ 1.3× for β ≳ 0.09 (A2)', and add the A1 values from the critical issue on attack scheduling.

### Issue 8
- **severity**: major
- **where**: Sec. 3.1 'Settling-time confound'; Sec. 4 recommendation 'Align analysis windows by time'
- **problem**: The draft attributes the +1.77 to settling time. However, the force decrease collapses on bow travel (A2 between-speed SD 0.0016 N on travel vs 0.040 N on time, paper/repro/drift_tests_A2.log). The published windows (1.55/0.35/0.08 s) and the matched-time windows both differ in travel between speeds, and so in how much force has been lost when the boundary is read. At equal time, the 0.20 m/s strokes have travelled 4× further. The mechanism is therefore unidentified: it could be settling, or a force-axis artefact. The '(A, B, C, D, A2 pooled)' set is ambiguous: is 'A' the same as A1? The draft also reports matched-time speed exponents only for A1 and A2. The per-diagram fits for the other 8 diagrams give lower-boundary exponents v 1.09–1.42 and β −1.61 to −2.18 (/Users/jeongjihoon/violin-rne/novelty/lower_protocol/analysis2.log §3), close to Schelleng's v/β². The A1+A2 reference shape (β −1.03) is atypical. The minimum-force speed question raised in the Introduction is left unanswered.
- **fix**: Rename it a 'window-placement confound' and state that both settling time and travel-dependent force loss change with window placement. Add a travel-matched sensitivity analysis or justify time alignment. Name the five pooled diagrams and the source script for +1.77. Add a short table of per-diagram lower-boundary v and β exponents (with the speed–session caveat), and note that the A1+A2 reference shape differs from the other eight.

### Issue 9
- **severity**: major
- **where**: Sec. 3.3 (levels, descriptor, D_T2); Limitations
- **problem**: The lower levels are fragile in ways the draft does not show. Zero-offset correction moves D1 from −0.248 to −0.928 and C_T2 from −0.556 to −0.708 (/Users/jeongjihoon/violin-rne/novelty/lower_protocol/levels_variants.log). Left-censored shares reach 0.36–0.43 per diagram and 0.65–0.68 at 0.05 m/s for B_T2r1/r2 (descriptors.log 'cens'), so those levels are extrapolations from the Gaussian censored likelihood. Under the c = 0.2/0.3 N detection limit, the descriptor r falls to 0.80/0.70 (analysis.log, A' blocks). The D_T2 zero-width column came from negative readings of −23 to −27 mN (novelty/method_protocol/dt2_fix_results.json). D_T2 was the first unseen point near the A1 end of the descriptor range (−0.038 N). There the descriptor model predicted +0.074 but the repaired level was −0.09 to −0.13, an error of 0.17–0.20, about twice the cross-validated level MAE. Adding the repaired D_T2 point lowers r to 0.85–0.87 (my recomputation).
- **fix**: Add a short robustness table: detection limit c, zero-offset correction, onset axis, and the left-censored share per diagram. State that the D_T2 failure occurred near the high-leverage end. Report a CI for r (Fisher, n = 10: roughly 0.56–0.97, wider with the three B_T2 near-duplicates). Note the selection among about 10 descriptors × 3 response variants.

### Issue 10
- **severity**: major
- **where**: Sec. 1 (Schelleng F_min ∝ Z²v/(Rβ²)); Sec. 3.3 'string properties'; Sec. 3.4 property models
- **problem**: The physics predictors omit losses. On a monochord with terminations designed to have very low mobility (van Walstijn 2026, papers/vanWalstijn2026_ActaAcustica_10_47.txt, around line 915), the R in Schelleng's F_min is not a meaningful bridge resistance; F_min is controlled by the string's internal damping. POMA reports damping factors per string that differ by 20–30% (Fig. 1 and text around line 205). Yet all property models use only Z, EI, T, d and Zto. The claim that 'string properties do not predict F_min' therefore leaves out the property theory says matters most.
- **fix**: Add a loss predictor (for example the POMA damping factors, mode-averaged or for the modes that dominate the Helmholtz corner), or state its omission explicitly as a limitation and soften 'string properties' to 'Z, EI, T, d'.

### Issue 11
- **severity**: major
- **where**: Sec. 2 'Data'; Limitations; internal note on further diagrams
- **problem**: The A3 archive (2024-04-22_TypeA_sample3) is marked 'done' in /Users/jeongjihoon/violin-rne/timeres/replicate_state.json. COLLAB_LOG says it was excluded from the frozen predictions because it was 'processed before freeze'. As a third A_T1 sample, it bears directly on the within-condition variability that drives the minimum-force argument. The draft never mentions it, which a referee will read as selective reporting. The draft also says the number of distinct strings is 'not documented'. POMA documents that each string type was mounted new and that three additional A_T1 samples exist (lines 163, 505).
- **fix**: Either include A3 as a clearly labelled post hoc development diagram, or state why it is excluded. Correct the string-count sentence to what POMA does document.

### Issue 12
- **severity**: major
- **where**: Sec. 4 thermal-friction paragraph
- **problem**: The paragraph conflates μ_d with Δμ = μ_peak − μ_d, which is the quantity in Schelleng's F_max. In /Users/jeongjihoon/violin-rne/novelty/friction_contrast/results/analysis.log, Δμ has R² 0.77 and depends on β (+0.548 ± 0.130), and it does not collapse on F·v (d = −0.505 ± 0.190). The 'no residual dependence on v/β' applies to μ_d only. The implied speed exponents are 0.48 (global), 0.45 (low F), 0.48 (high F) and 0.94 (top F, n = 13), so the range is 0.45–0.94, not 0.48–0.94. The absolute Schelleng F_max computed with the measured Δμ overshoots the measured upper limit by 1.24–3.5× (one case 0.93; analysis.log §[3]). 20 of the 31 windows are at 0.05 m/s. The friction string's identity is not established: its folder is '2024-08-13_Helicore_T2_complete' and its properties are closest to B_T2 (friction_contrast/offline/results/coverage.log).
- **fix**: Cut to one hedged sentence or move to supplementary material: '0.45–0.94; absolute Schelleng prediction with measured Δμ off by 1.2–3.5×; one string of uncertain identity; post hoc'. If kept, state which coefficient is used and why. The one-parameter identity test assumes Δμ = f(Fv), which the same friction data already contradict.

### Issue 13
- **severity**: major
- **where**: Sec. 2 'Frozen tests'; Table 1 caption; internal note B
- **problem**: Several pre-registration details are incomplete.
(a) A_T2 plan (/Users/jeongjihoon/violin-rne/final/CONFIRM_PLAN_A_T2.md): it pre-specified 'the A1-vs-A2 level difference as a specimen-variability yardstick' as part of the uncertainty. That yardstick (0.233 lower; 0.005 upper, in confirm_A_T2.log) is not reported. The plan was also written at 21:38 while A_T2 extraction had already begun at 21:35, so 'before the target data were processed' is inaccurate. The correct wording is 'before any boundary or label was inspected'.
(b) The development plan (final/ANALYSIS_PLAN_FINAL.md) pre-registered flyback-Z physics models and few-shot k = 10 as the primary analysis. The few-shot analysis failed, and the switch to published Z was post hoc. Neither is mentioned.
(c) D_T2: the README says stored H columns may exist in the features file and were not read. 'Before any label was computed' should be 'before any outcome column was parsed'.
(d) The local mtimes are not tamper-evident, and the GitHub snapshot (commit d6b4081, 12:10 on 10-19) post-dates both tests.
- **fix**: Report the plan's yardstick and disclose the original pre-registered development analysis and its outcome in one sentence. Fix the timing wording. Before scoring A4 (now downloading), D2, B_T2s2 and the high-force parts, push the frozen-prediction hashes to an external timestamp such as a Git commit or an OSF registration, and say so in the paper.

### Issue 14
- **severity**: major
- **where**: Table 1 and Sec. 3.4
- **problem**: Table 1 mixes corrected and uncorrected evaluation code. The old code assigned an arbitrary 3.0 log error and automatic hits to non-finite predictions (/Users/jeongjihoon/violin-rne/astra_20261019/compare_fixed_results/README.md). This can change the A_T2 gradient-boosting entries (0.271/0.176) but not the physics or lookup entries. The Beigi upper entries equal the Schelleng ones (0.195/0.215) because Beigi has no upper law; this is not footnoted. The lookup 'tie' (0.269 vs 0.268) depends on the lookup definition: the legacy-edge lookup gives 0.266/0.176 (same README).
- **fix**: Add an A_T2 re-score of the D models with the corrected code as a labelled post hoc row, or footnote that the A_T2 GBM entries used the buggy code. Footnote 'Beigi upper = Schelleng upper'. State the lookup sensitivity.

### Issue 15
- **severity**: major
- **where**: Sec. 2 'Load-decrease descriptor'; Abstract; Sec. 3.3
- **problem**: dec5_pl is measured on the target session itself. At 0.05 m/s it uses a window that starts about 1.05 s into the plateau (D_T2_controls_frozen.json, selected_start_range_s), i.e. after the 0.2–0.4 s boundary window. It is a same-session covariate, not an a priori predictor of an unseen string. The abstract and Sec. 3.3 do not say this, while the D_T2 plan (CONFIRM_PLAN_D_T2.md §1) does.
- **fix**: State this in Sec. 2 and in the abstract sentence: 'a label-free descriptor measured on the same session'. Note that the window used for it at 0.05 m/s lies later than the boundary window.

### Issue 16
- **severity**: minor
- **where**: Sec. 2 'Data'
- **problem**: Several data details are imprecise:
- 'Five are at T2 ... D_T2 was held out' leaves the reader to infer that there are six T2 diagrams.
- POMA calls each 2000-stroke single-speed set a 'Schelleng diagram', whereas the draft's 'diagram' is the 6000-stroke three-speed set.
- β is log-spaced (POMA line 173).
- The 50 kHz rate is sourced from Lampis 2024, a different experiment (papers/Lampis2024_ActaAcustica_playability.txt line 215).
- POMA's own boundary rule is K = 3, while the draft uses K = 2. No upper-boundary K = 3 sensitivity is shown.
- **fix**: Say 'six T2 diagrams, D_T2 held out'. Define 'diagram' explicitly. Add 'log-spaced' for β. Cite the Zenodo record description for the sampling rate. Report a K = 3 sensitivity analysis for the upper law.

### Issue 17
- **severity**: minor
- **where**: Throughout ('log', 'ln Z'); Fig. 1b
- **problem**: 'log' is used for natural-log offsets (e.g. 0.539 → 42%; tolerances of 0.12/0.23), while Fig. 1b uses log₁₀. The base is never defined.
- **fix**: Define log = natural log at first use (e.g. 'log units, natural'), and label log₁₀ explicitly in Fig. 1b.

### Issue 18
- **severity**: minor
- **where**: Sec. 3.2 'Shape'
- **problem**: v^0.70 [0.63, 0.76] β^−1.07 [−1.14, −1.01] matches novelty/thermal_upper/results_summary.json (diagram-cluster bootstrap). The newer novelty/upper_final/levels.json gives 0.697 [0.644, 0.757] and −1.063 [−1.138, −0.996] with a β-position cluster bootstrap. In that version the β CI includes −1. The draft names neither source.
- **fix**: Name the source file and the bootstrap unit. Note that the β exponent's CI borders −1 depending on the clustering.

### Issue 19
- **severity**: minor
- **where**: Fig. 1 caption; Sec. 3.1 'Size'
- **problem**: 'The curves collapse on travel, not on time' holds strongly for A2 (SD 0.0016 vs 0.0397 N) but only weakly for A1 (0.0077 vs 0.0135 N; paper/repro/drift_tests_A1.log). In A1 v1 the decrease saturates by 5 cm. The dataset paper states Fb 'remained almost constant during the strokes' (POMA line 176). Citing that sentence would sharpen the audit's contribution, but the draft does not.
- **fix**: Qualify the caption as 'A2'. Add the A1 SDs. Quote the POMA 'almost constant' statement as the claim being audited. Optionally test a position-controlled-indentation (non-parallel bow path) model in which ΔF ∝ k_eff(β) × travel. The T4 β trend in drift_tests_A2.log is consistent with it.

### Issue 20
- **severity**: minor
- **where**: References and Sec. 1
- **problem**: Some references are incomplete or rest on weak support:
- [4] has no title and is an unreviewed preprint (simulation).
- [5] has no page range.
- [3] and [13] are checked at abstract level only.
- [7] (Galluzzo thesis) was not read in full text.
- [6] (Woodhouse 2003) supports sensitivity to initial conditions and thermal-model transients, but not directly 'establishment and maintenance differ' (texts/Woodhouse2003_thermal.txt).
- [11]: part-1 Zenodo ID 17749110 (pipeline) vs 17749111 (Korean report).
- [14] merges a data record with a conference abstract the authors did not read.
- **fix**: Complete [4] and [5] and mark [4] '(preprint, simulation)'. Cite a specific section or figure for the establishment/maintenance claim, or replace [6]. Resolve the Zenodo ID on the record page. Split [14] into a data citation and a separate abstract citation, or drop the abstract.

### Issue 21
- **severity**: minor
- **where**: Sec. 3.2 A_T2 vs pooled pairs; Abstract length
- **problem**: The A pair Z exponent appears with two different CIs: [+0.92, +2.49] (frozen, confirm_A_T2.log) and [0.86, 2.47] (post hoc pair analysis, upper_law_t0.2.log) without explanation. The abstract (about 150 words) may exceed JASA-EL's limit; I could not verify the limit offline.
- **fix**: State that the second CI comes from a separate post hoc bootstrap. Check the abstract against the official JASA-EL author guidelines and shorten it if needed.

## Referee 2

**verdict**: MAJOR REVISION. Most numbers in /Users/jeongjihoon/violin-rne/paper/draft_v1.md match their source files. Checked and matching:
- Sec. 3.1 (re-run in my scratchpad): drift_tests on A2 and A1; paper/repro/attack_design_A2.log; onset_vs_nominal.log; skeptic_force/t3_order.log.
- A_T2: final/confirm_A_T2.log, confirm_A_T2_robust.log.
- D_T2: astra_20261019/D_T2_confirm_result.json, D_T2_controls_frozen.json, d_t2_freeze/frozen_parameters.json.
- Z leverage, fair no-Z model, batch offset: novelty/prior_art/z_leverage.json.
- Pair exponents and chi-square values: novelty/upper_law/upper_law_t0.2.log and .json.
- Thermal exponents: novelty/thermal_upper/analysis.log, friction_contrast/offline/results/friction_offline.log.
- Descriptor screen, r = 0.88 and property R²: novelty/lower_protocol/descriptors.log, analysis.log; astra_20261019/protocol_falsification.log.
- Nested refits and Table 1 development values: astra_20261019/audit_tables.md.
- 11-diagram leave-one-out: novelty/method_protocol/validate_v2.log.
- D_T2 repairs: dt2_fix_results.json.

I recomputed r = 0.8798 (p = 0.0008, n = 10) and the attack floor shares 44.8/51.2/66.4%. One statement is factually wrong: the direction of the A1 zero-offset drift. Several inferential claims are stronger than the statistics allow:
1. The abstract and discussion call the result "consistent with Schelleng's Z scaling", but the within-type pairs reject exponent g = 1 (χ² 14.2 on 3 df, p 0.003). They are also heterogeneous. The post hoc D1→D_T2 pair gives g ≈ 0.23 [−0.25, 0.70].
2. The pooled +1.28 [0.61, 1.85] is a fixed-effect estimate whose interval ignores significant heterogeneity. A random-effects estimate is 1.32 [−0.39, 3.04].
3. The D_T2 level error of 0.002 log averages over speeds that the strict v/β shape does not fit: per-speed residuals are +0.07 (v1), −0.01 (v2) and −0.55 (v3).
4. The A_T2 "−0.10 log above 0.2 N" figure comes from dropping columns based on the outcome. The repository's own censoring version gives −0.42.
5. The descriptor association is selected post hoc and its cross-validation does not repeat the selection inside each fold. It weakens under the detection-limit censoring that the paper itself motivates: r 0.80 at 0.2 N, 0.70 at 0.3 N.

Several pre-specified items are missing:
- The A1−A2 yardstick required by the A_T2 plan.
- The deferred D_T2 secondary analyses: K=3, t=0.1/0.3 s and the nominal-force axis.
- The D_T2 empirical-shape secondary result.
- The D_T2 lower-boundary bootstrap failure count (1272 of 2000).

Pre-registered and post hoc results are mostly labelled correctly. The issues listed below are all fixable.

### Issue 1
- **severity**: critical
- **where**: Abstract; Sec. 3.2 'Pooled tension pairs'; Sec. 4 'What is and is not new'
- **problem**: The draft says the maximum-force transfer is 'consistent with Schelleng's impedance scaling', but the within-type pairs reject exponent g = 1. upper_law_t0.2.json gives chi2_schelleng_g1 = 14.18 on 3 df (p = 0.0026), and the draft reports only the common-slope χ² 13.4 (p = 0.0012). I computed the post hoc D pair from novelty/upper_final/levels.json (E shape): D1 +0.172 → D_T2 +0.205, Δlevel +0.033 (SE 0.035) for ΔlnZ 0.145, so g = 0.23 [−0.25, 0.70], which excludes 1. The strict-shape version gives 0.34 [−0.32, 1.00]. The four pair exponents are therefore A 1.71, B −0.51, C 3.25 and D 0.23. D_T2 passed because D1 sat +0.094 above the strict T1 line (z_leverage.json) while D_T2 lay on it. That is a cross-type calibration success, not a within-type Z¹ shift.
- **fix**: Report the g = 1 test and its p-value. Add the D pair as a post hoc fourth pair. Reword the abstract and Sec. 4 to say: 'a single Z¹-scaled constant calibrated on the T1 group predicted T2 group levels; within-type T1→T2 shifts were heterogeneous and jointly inconsistent with g = 1 (χ² 14.2, 3 df).' Drop 'consistent with Schelleng's impedance scaling' as an unqualified claim.

### Issue 2
- **severity**: critical
- **where**: Sec. 3.2 'Pooled +1.28 [0.61, 1.85]'; Fig. 2a
- **problem**: The pooled exponent is a fixed-effect inverse-variance fit through the origin, and its bootstrap interval (upper_law.py lines 175-188) uses only within-session variances, even though heterogeneity is significant. I took per-pair SEs from the CIs in upper_law_t0.2.json and ran a DerSimonian-Laird random-effects fit: τ ≈ 0.12 log, g = 1.32, 95% CI [−0.39, 3.04] (normal) or about [−2.4, 5.1] (t with 2 df). The heterogeneity τ of 0.12 log equals the T1 strict-law residual SD of 0.118. In addition, POMA Table 1 gives C_T2 Z = 1.1154 ± 0.1023 kg/s (papers/Lampis2025_POMA_58_035013.txt). That SD (about 0.09 in ln Z) exceeds the C pair's ΔlnZ of 0.058, but published Z is treated as exact.
- **fix**: Report a random-effects estimate with a t- or HKSJ-based interval, or remove the pooled value and present pair-level results only. Propagate the published Z uncertainty (errors-in-variables or simulation over the Z SD). State that the C pair carries almost no information about the exponent.

### Issue 3
- **severity**: major
- **where**: Sec. 3.2 'D_T2 (frozen) ... within 0.002 log'; Sec. 4 'Two boundaries'
- **problem**: The level error of 0.002 is a censoring-weighted average over speeds of a mis-specified v¹ shape. D_T2's own speed exponent is 0.61 [0.43, 0.82] (thermal_upper/analysis.log). I computed signed residuals of the 48 finite upper columns against the frozen strict predictions (D_T2_confirm_result.json observed_boundaries vs D_T2_controls_frozen.json predictions.upper_strict). Medians are +0.07 (v1, n 23), −0.01 (v2, n 15) and −0.55 (v3, n 10); the mean is −0.13. Most v3 columns are right-censored, which down-weights the speed with the largest misfit. The fitted censored-Gaussian sigma is 0.396 log, against a level tolerance of 0.12.
- **fix**: Report per-speed residuals or levels and say that the strict v/β shape misfits the speed dependence inside D_T2. Do not present 0.002 as a measure of precision. Give the relevant predictive uncertainty: the between-diagram residual SD of 0.118 and the bootstrap-implied Z-exponent interval [0.37, 1.74] (z_leverage.json). Primary endpoint and verdict stay unchanged.

### Issue 4
- **severity**: major
- **where**: Sec. 3.3 'A_T2 (frozen secondary)': 'Restricting ... above 0.2 N reduced the shift to −0.10 log'
- **problem**: final/confirm_A_T2_robust.py keeps only columns whose lower interval starts at or above 0.2 N, which also drops the left-censored columns: 70 of 120 remain. It then compares them to an A1+A2 law fitted on all columns. This selects on the outcome (truncation), so the shift is biased toward zero. The statistically appropriate version already exists in novelty/lower_protocol/levels_variants.json: readings below c are treated as left-censored and the reference law is refit the same way. It gives A_T2 −0.42 (c = 0.2), −0.29 (c = 0.3), −0.175 (c = 0.4), and −0.446 after zero-offset correction.
- **fix**: Replace the −0.10 sentence with the censoring-based values. Say that the A_T2 shift shrinks but persists (−0.42 at 0.2 N). If the truncated number is kept at all, mention it only as a biased bound.

### Issue 5
- **severity**: major
- **where**: Abstract 'r = 0.88'; Sec. 3.3 'Association ... (post hoc)'
- **problem**: The detection-limit sensitivity is omitted, although the paper itself states that forces below 0.2 N are unreliable. In lower_protocol/analysis.log section A', with the c = 0.2 response, dec5_pl gives r 0.80 (LODO R² 0.51); with c = 0.3 it gives r 0.70 (0.32). Only the K/t variants (0.815-0.880) and the nominal axis are reported. The descriptor is also computed from the same lowest-20-rank strokes that define the lower boundary, not merely the same channel. Its slope (7.67 log per N over a 0.09 N range) is of the size expected if the reading simply shifts.
- **fix**: Report r and the cross-validated metrics under c = 0.2 and 0.3 N censoring as co-primary sensitivity checks. Add the onset-force axis. Test a mechanical null that predicts the measured-force boundary shift if the boundary is fixed in onset-force terms and the reading drops by dec5_pl, and compare its slope with the fitted one.

### Issue 6
- **severity**: major
- **where**: Sec. 2 'screened among about ten'; Sec. 3.3 'Predictive value'; Fig. 3
- **problem**: The screen covered 8 session descriptors on a 7-diagram subset (desc_b, dec10, acc_b, cal_low, cens, d0, fmin, floor) plus dec5_pl, cal_pl, cens and fmin on all 10 diagrams, crossed with 3 response definitions (descriptors.log, analysis.log). dec5_pl's parameters (5 cm, ranks 1-20, 0.1-s windows) were also chosen post hoc. The nested LODO/LOTO refits the reference shape inside each fold, but the descriptor choice and its construction stay fixed, so 0.213→0.101 is optimistic. The only held-out check is D_T2, scored once with a post hoc repair (dt2_fix_results.json). There the descriptor's level error was 0.165-0.203, against 0.169-0.207 for the mean model, about twice the cross-validated error. No multiplicity-aware evidence is given. I computed nominal p = 0.0008 (Bonferroni over 10: 0.008). Influence: drop-one r ranges 0.77-0.93; without A1 r = 0.772; without A1+A2 r = 0.688 (final report §5.1); collapsing to 7 conditions gives r = 0.884.
- **fix**: Add a selection-adjusted permutation test that reruns the whole screen (max |r|) on permuted levels, report the influence diagnostics and the condition-level r, and state that descriptor selection is outside the cross-validation loop. Contrast the D_T2 held-out error with the cross-validated error explicitly.

### Issue 7
- **severity**: major
- **where**: Sec. 3.1 'Low-force reliability': 'drifted from +158 to +4 mN over the recording order'
- **problem**: The direction is reversed. In claude_extra/wf_20261018/skeptic_force/common.py, the A1 trial index is the recording order and col = 39 − (trial−1)//50 is β-ascending, so A1 was recorded from high β to low β. t3_order.log (A1 v3) gives 'corr with trial +0.95' and 'first/last 10 columns +158.0/+3.6 mN', where col < 10 means low β, recorded last. The reading therefore rose from about +4 to about +158 mN during recording. Recording order is also completely confounded with β (r with β −0.88).
- **fix**: Correct to 'rose from about +4 to +158 mN over the recording (r with trial index +0.95)' and add that recording order and β cannot be separated in this folder. The same reversal appears in ASTRA_HANDOFF_PROMPT.md §10.16.

### Issue 8
- **severity**: major
- **where**: Sec. 3.2 and 3.3, A_T2 frozen secondary results
- **problem**: The frozen A_T2 plan (final/CONFIRM_PLAN_A_T2.md) required reporting 'the A1-vs-A2 level difference as a specimen-variability yardstick' next to the column-bootstrap interval. It is not reported: lower 0.233 log, which is ±2.0 in Z-exponent units; upper 0.005 (confirm_A_T2.log). The '42% lower' figure uses the A1+A2 average as reference. Against A2 alone the shift is −0.421 (34%); against A1 alone it is −0.654 (48%) (lower_protocol/analysis.log, tension pairs).
- **fix**: Report the yardsticks next to the Z-exponent intervals, as the plan pre-specified. Give the shift as a range against A1 and A2 (34-48% lower).

### Issue 9
- **severity**: major
- **where**: Sec. 3.2 and 3.3, D_T2 (frozen)
- **problem**: Pre-specified elements are missing or extended:
(a) The plan's secondary analyses (K=3, t = 0.1/0.3 s, nominal axis) were never run. D_T2_confirm_result.json lists them as secondary_variants_deferred, and this is not disclosed.
(b) The frozen secondary upper result (empirical shape + Z: level error 0.012, column median 0.113, 48 columns) is in the internal notes only, not in the results.
(c) The lower-boundary bootstrap failed in 1272 of 2000 resamples for the same zero-width reason; this is not reported.
(d) 'All differences were below the frozen 0.23 margin' applies the level-comparison margin to column medians. The plan sets no decision rule for the column metric, and the result JSON has no column-level verdict.
- **fix**: Run and report the deferred secondary analyses, labelled as pre-specified secondaries, or state explicitly that they were not run and why. Add the empirical-shape secondary and the bootstrap failure counts. Describe the column-median comparison as descriptive.

### Issue 10
- **severity**: moderate
- **where**: Sec. 4 'This pattern may also explain ...: within-condition differences (0.18-0.23 log) were as large as the between-diagram SD (0.218 log)'
- **problem**: This compares pairwise differences with an SD; a difference of two draws has SD √2σ. The proper decomposition is in lower_protocol/analysis.log section D: within-repeat SD 0.121 (3 df) against SD of condition means 0.211, F = 4.40, permutation p = 0.096. In addition, the B_T2 same-day difference of 0.18 is not resolved: the intervals [−0.485, −0.183] and [−0.269, −0.024] overlap (descriptors.log). So 'varied even within a string condition' is supported for A1 vs A2 but not for the B_T2 repeats.
- **fix**: Use the ANOVA numbers, report the uncertainty of the B_T2 repeat difference, and qualify the within-condition claim.

### Issue 11
- **severity**: moderate
- **where**: Sec. 3.2 'Repeatability'; Sec. 4 '|Δ| ≤ 0.028 log'
- **problem**: The upper-level repeat differences are given without uncertainty. upper_law_t0.2.log gives bootstrap SEs of 0.032 (A1−A2), 0.055 (r1−r2) and 0.044 (s1 vs Part 6). The final report §6 gives about [−0.112, 0.106] for r1−r2. The data rule out differences larger than about 0.1 log, not about 0.03.
- **fix**: Report the intervals and rephrase as 'no difference detectable at about ±0.1 log resolution'.

### Issue 12
- **severity**: moderate
- **where**: Sec. 4 'errors of 0.121 log (A_T2 columns) and 0.002 log (D_T2 level)'
- **problem**: This mixes metrics: a column-median error versus a level error. Like-for-like values are as follows. Column medians: A_T2 0.121 (P-free + Z), D_T2 0.124 (strict). Level errors: A_T2 0.082 for the frozen P-free + Z model (+0.197 observed − 0.115 predicted), D_T2 0.002. The abstract's 'transferred to two held-out strings' also mixes a frozen secondary endpoint (A_T2) with a frozen primary one (D_T2).
- **fix**: Report both metrics for both strings and label A_T2's level transfer as a frozen secondary endpoint.

### Issue 13
- **severity**: moderate
- **where**: Sec. 2 'Boundaries and levels'; Sec. 3.3 level definitions
- **problem**: Δ_d is an offset from the A1+A2 lower shape (β^−1.03), but the per-diagram lower β exponents are −1.6 to −2.2 for every non-A diagram and −0.97/−1.09 for A1/A2 (friction_contrast/offline/results/levels.log). With a mis-specified shape, the level depends on which columns are finite or censored: A_T2 has 32 left-censored lower columns, A1 has 0. Level comparisons and the descriptor correlation may partly reflect shape × censoring pattern.
- **fix**: Add a sensitivity check that uses levels at a fixed reference point (for example v = 0.1 m/s, β = 0.1) from per-diagram free fits, or a hierarchical shape model, and report r on that scale.

### Issue 14
- **severity**: moderate
- **where**: Sec. 3.2 'Shape ... v^0.70'; Sec. 4 thermal paragraph
- **problem**: The sub-linear upper speed exponent may be partly a censoring artifact. Local exponents are v1→v2 0.64-1.00 but v2→v3 0.16-0.84 (upper_law_t0.2.log), and v3 is the most right-censored at the ~3.8 N ceiling (for example B_T2s1 v3: 5 finite, 25 right-censored). Speed is also confounded with session. The thermal comparison sets two point estimates (0.48, 0.94) against a range of per-diagram estimates; the friction-side CIs are [0.39, 0.61] and [0.70, 1.19].
- **fix**: Run a sensitivity check restricted to v1-v2 or to columns well below the ceiling, flag the dependence on the ceiling, and give CIs on both sides of the thermal comparison. The text says 'ten diagrams' but the pooled fit uses 11; the 0.49-0.86 range is unchanged with D_T2 (0.61).

### Issue 15
- **severity**: moderate
- **where**: Sec. 3.1 'Settling-time confound'; Sec. 4 recommendations
- **problem**: The comparison is not like-for-like. +1.77 is a pooled fit of A, B, C, D and A2 on 0.4-s published windows (handoff §2), but the matched-time values quoted are for A1 and A2 only. Per-diagram matched-time lower exponents at t = 0.2 s are +0.67 (A1), 0.94 (A2), 1.09-1.13 (B1, C1, D1) and 1.16-1.42 (T2 diagrams) (friction_contrast/offline/results/levels.log). The B_T2 repeats are +1.25 to +1.32 (handoff §10.16). Quoting only 0.57-1.04 understates the spread.
- **fix**: Report published-window and matched-time exponents for the same diagrams, and give the matched-time range for all diagrams.

### Issue 16
- **severity**: moderate
- **where**: Sec. 2 'intervals are 95% percentile bootstraps'; all CIs
- **problem**: Bootstrap settings vary. A_T2 uses 200 unpaired row resamples over 120 speed-column rows (confirm_A_T2.py lines 74-78) instead of β positions paired across speeds. Other analyses use 100-400 resamples. As a result the same quantity has different intervals: A_T2 upper Z exponent [0.92, 2.49] vs [0.86, 2.47]; A2 lower [−0.227, 0.001] vs [−0.272, −0.013]; A_T2 lower [−0.722, −0.382] vs [−0.679, −0.377]. The A_T2 lower bound of 0.92, which decides whether 1 is inside the interval, is within Monte Carlo error of 1.
- **fix**: Use one canonical bootstrap for every reported interval: β-position clusters paired across speeds, B ≥ 2000, stated seeds. Keep the frozen A_T2 interval as executed, and add the canonical version as a labelled sensitivity check.

### Issue 17
- **severity**: moderate
- **where**: Table 1; Sec. 3.4; Abstract 'Neither ... beat a model-free lookup'
- **problem**: There is no uncertainty on model differences. 0.268 is bolded over 0.269 although the two are tied, and on the development set the free-exponent law literally beat the lookup by 0.001. The A_T2 columns come from the pre-correction evaluation code while the development columns come from the corrected code. Beigi and Schelleng share the same upper form, so their upper values are identical (0.195/0.215) by construction. Paired intervals do exist in validate_v2.log, for example protocol|quad_s d = −0.026 [−0.084, +0.030]. LOTO also supports the lookup (0.283, not beaten) but is not reported.
- **fix**: Add paired column-bootstrap CIs for each model−lookup difference and remove bold where differences are within noise. Add a labelled post hoc A_T2 rescore with compare_fixed.py, keeping the frozen numbers. Note the shared upper form. Reword the abstract as 'no model had a meaningfully lower error; paired differences included zero', and cite LOTO.

### Issue 18
- **severity**: moderate
- **where**: Sec. 2 'Frozen tests'; Sec. 5 last paragraph
- **problem**: The freeze evidence is weaker than stated.
(a) The A_T2 plan has no hash seal, only file text and mtime. It was written while A_T2 extraction was in progress (v2 started at 21:35), so 'before the target data were processed' overstates; 'before any label or boundary was computed' is accurate.
(b) The stored future predictions exist in three freeze versions (thermal_upper FROZEN_v1/v2/v3). A3 was frozen during processing. For B_T2 sample2 and the high-force parts, the same-string friction session (2024-07-03) had already been seen and the friction law was fitted on it (FREEZE_RECEIPT_v2.txt).
- **fix**: State the evidence level for each freeze, name the governing version for each future target, and disclose the B_T2 sample2 / high-force contamination before promising a 'scored once' test.

### Issue 19
- **severity**: minor
- **where**: Sec. 3.3 'D_T2 (frozen) ... descriptor lay inside the training range'
- **problem**: D_T2's dec5_pl (−0.0384 N) lies next to A1 (−0.032), the most influential training point; the next nearest is A2 (−0.063). The +0.074 prediction is effectively an extrapolation from A1. Also, the zero-width column's strokes had measured forces of −0.023 to −0.027 N while inside an H run (dt2_fix_results.json), which is direct evidence of an offset of at least 25 mN in D_T2. This belongs in the 'Low-force reliability' bullet.
- **fix**: Note the edge position and the influence of A1. Cite the negative-force H strokes as offset evidence.

### Issue 20
- **severity**: minor
- **where**: Sec. 2 'Labels'; Methods 'Models'
- **problem**: The single-band convention treats no-band columns as right-censored for the lower boundary; D_T2 has 9 such columns, v3 at the smallest β. Its effect on the levels is not tested. The ML baselines are trained on several windows per stroke and use random-row internal validation (final report §7). Neither limitation is stated.
- **fix**: Add a sensitivity check that treats no-band columns as missing. State the ML pseudo-replication and validation-split caveat in Limitations.

### Issue 21
- **severity**: minor
- **where**: Sec. 2 'Data'; Sec. 3.1 'Size'; terminology
- **problem**: POMA defines a 'Schelleng diagram' as 2000 strokes at one speed (44 diagrams), whereas the draft uses 6000 strokes across three speeds. POMA also documents 'three additional samples of the string labeled A T1' and 'repeated measurements for the same strings', so the number of distinct strings is partly documented. The 50-kHz rate comes from Lampis 2024 [18], not [9]. 'Development diagrams' refers to the 5-diagram set in one place and the 10-diagram set in another. A1's travel collapse is weak (between-speed SD 0.0077 N on travel vs 0.0135 N on time; drift_tests re-run), but only A2 is reported. The attack shares 45/51/67% are correct (my recomputation from ramp_meta_A2.csv: 44.8/51.2/66.4%), while the handoff says 45/52/68%.
- **fix**: Define 'diagram' against POMA's usage, correct the string-count statement and the citation, give the 5-set and 10-set different names, add A1's travel-vs-time spread, and cite paper/repro/attack_design_A2.log for the attack numbers.

