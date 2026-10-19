# posthoc_hf — why did the frozen upper laws under-predict the four high-force sessions?

Claude, 2026-10-19 (plan written 22:37 KST, before any computation in this folder).

**Status of everything here: POST HOC.** The high-force (HF) labels were scored once at 21:41
(`highforce_confirm/score_result.json`) and are now seen. Nothing in this folder changes that frozen verdict
(P2 failed 4/4). Natural log throughout. Window t = 0.2 s (0.2 s long, starting 0.2 s after the speed plateau
begins), K = 2, measured window force, as in the frozen scorer.

Data used (all allowed): the 11 development diagrams A1, A2, B1, C1, D1, A_T2, B_T2s1, B_T2r1, B_T2r2, C_T2, D_T2
(caches via `novelty/thermal_upper/rows.py`), the four HF sessions (parser `highforce_confirm/score_highforce.py`:
`read_labels`, `columns_from_windows`, `column_table`), label-free ramp metadata (`astra_20261018/archive_companion/*/ramp_meta_*.csv`
for the four T2 sample-1 sessions only, `astra_20261019/highforce_stream/HighBowForceRange_{1,2}/ramp_meta.csv`),
and existing json. NOT used: A4, D2, B_T2 sample 2 (2024-07-03), coldwarm, A3 — no file of theirs is opened.

## PLAN (fixed before running)

### Q1 — pre-existing quadratic surface (analysis2b "b2")
- Refit exactly `analysis2b.py` b2: interval-censored normal model of log F_max on the upper-boundary columns of the
  11 development diagrams, design `[lv, lb, lv*lb, lb^2, 1, diagram dummies (A1 = reference)]`, `cfit` (L-BFGS-B → Powell → L-BFGS-B)
  from the OLS start. Deterministic point fit only (no bootstrap). Check: |Δ| of a, b, g, q vs `analysis2b.json`
  (saved 10:38, before HF labels existed, 21:41); report the max |Δ|.
- Local exponents: for each HF string, take its own primary upper-boundary columns (the 77/77/80/81 columns of the
  frozen scorer), evaluate the surface y = a lv + b lb + g lv lb + q lb² there and OLS y on [1, lv, lb]
  (= `thermal_law.design_exponents`). Compare e_v, e_β with the observed free fit (`score_result.json`
  targets.*.t0.2.P1, with its 95 % column-bootstrap CI). Report "inside / outside observed CI" per string and exponent.
- Level: predicted HF log F_max = surface + intercept + the fitted offset of the same string type's sample-1 diagram
  (HF_C←C_T2, HF_A←A_T2, HF_D←D_T2, HF_B←B_T2s1). Level error = observed − predicted via `thermal_law.offset_fit`
  on the HF upper intervals. Compare with the frozen S1_strict level errors (+0.364/+0.224/+0.165/+0.357).
- Pre-declared reference (one, reported whatever it shows): the same model without g and q (linear, diagram offsets),
  same fit and level procedure, to show what the curvature terms contribute.
- Wording: "post hoc evaluation of a model fixed before the HF data, not a pre-declared test". If |level error| of b2
  is < 0.15 (the frozen PI90 width) for a string, say "the pre-existing surface would have placed this string's level
  within ±0.15"; otherwise "also under/over-predicts". No claim that b2 is "the" explanation.

### Q2 — session offset vs force range
- Per string type: sample-1 columns (C_T2 / A_T2 / D_T2 / B_T2s1) with β ≤ 0.075 and an upper interval, plus the HF
  primary upper columns. Model: log F_max = c + e_v lv + e_b lb + δ·[HF] (+ normal σ), interval-censored likelihood
  (`thermal_law.censored_nll`; Nelder–Mead from OLS start, then L-BFGS-B polish).
- Bootstrap δ: B = 1000, seed 7, unit = β column cluster (all speeds of one β column of one session), resampled
  with replacement within each session (stratified). 95 % percentile CI.
- Contradictions (same speed, nearest sample-1 β column in log β; report the max log-β distance):
  (i) HF upper interval entirely below sample-1's lower bound of F_max (HF hi < s1 upper lo) — evidence HF lower;
  (ii) HF lower bound of F_max above sample-1's upper bound (HF upper lo > s1 upper hi) — evidence HF higher.
  Denominator = HF columns with a matched sample-1 column that has the relevant finite bound.
- Also report, as context, the fraction of sample-1 β ≤ 0.075 columns that are right-censored.
- Interpretation rule: CI of δ includes 0 and |δ| < 0.10 → "no session offset detected; the frozen failure is more
  consistent with a force-range/shape effect (or extraction rule)"; CI excludes 0 with δ > 0 → "session-level offset
  (HF session higher) is consistent with the data"; otherwise "not decidable". Caveat always stated: in this region
  sample 1 is mostly right-censored near 4 N, so δ is identified mainly by the censoring bounds and the common-shape
  assumption; session and force range remain confounded by design.

### Q3 — force reading at high force
- Label-free. Per stroke: r_on = ln(F_window(t=0.2) / F_onset), F_onset = `force_at_onset` (median force in the 10 ms
  before bow-motion onset), F_window = mean force of the 0.2 s window at t = 0.2 s (features.csv.gz / caches). Also
  r_plat = ln(F_window(0.2)/F_window(0)) (drift within the plateau).
- HF: by onset-force band [3.5–5, 5–7, 7–9, 9–13.5 N] × speed, per string. Low force (comparison): the four T2 sample-1
  sessions by onset-force band [<0.25, 0.25–0.5, 0.5–1, 1–2, 2–4.5 N] (medians, IQR, n).
- Boundary-relevant shift (uses labels; HF and dev both allowed/seen): for the two strokes that bracket each upper
  boundary (f[e], f[e+1] in the frozen last-band rule), median r_on, pooled per session. Δr = median r_on(HF boundary
  strokes) − median r_on(sample-1 boundary strokes) per string type. If the physically relevant force were the onset
  force, the frozen level errors would change by about −Δr (window-force boundary = onset boundary + r_on).
  Decision wording: |Δr| ≥ 0.17 and of the right sign → "a force-axis shift of the needed size is possible";
  |Δr| < 0.10 → "a force-reading shift of this kind is too small to explain +0.17…+0.36"; in between → "partial".
  A2 reference (0.17 N fall over 14 cm at ranks 1–20) quoted from the earlier analysis, not recomputed.

### Q4 — descriptive only
- Summarise `highforce_confirm/posthoc_multiband.json` (multi-band prevalence dev vs HF; S1_strict level error under
  last/first/longest band rules). No recomputation.

### Outputs
`posthoc_hf.py` (all of Q1–Q3), `results.json`, `fig_posthoc_hf.png` (English), RESULTS appended below.
CPU: single process, OPENBLAS_NUM_THREADS=1.

### Additions decided AFTER seeing the planned Q1–Q3 numbers (22:41 KST; labelled as such everywhere)
Reason: Q2's δ rests on only 8–13 finite sample-1 columns, and the nearest-β match distance reached 0.29–0.53 log.
- A1 (contradiction robustness): for each "HF higher" contradiction, how many sample-1 strokes lie above the sample-1
  upper bound (1 = only the top stroke is non-H, i.e. possibly a single-stroke dropout); and the counts with the
  nearest-β distance capped at 0.06 log (≈ one sample-1 β step).
- A2 (δ sensitivity, one variant only): sample-1 finite upper intervals whose last band ends one stroke below the top
  (only the highest stroke non-H) re-coded as right-censored at the band end; refit δ (same bootstrap: B 1000, seed 7).
- A3 (model-free overlap): strokes whose t = 0.2 s window force lies in the force range shared by both sessions
  (HF minimum to sample-1 maximum, ≈ 3.8–4.5 N), β ≤ 0.07: Helmholtz fraction (K-free, single-window label) per
  session and speed, with counts. Same force, same nominal v and β region, different session → a difference here
  cannot be a force-range extrapolation effect.

## RESULTS (computed 22:38–22:50 KST; `run.log`, `run_additions.log`, `run_a3split.log`, `results.json`, `fig_posthoc_hf.png`)

Denominators: development fit = 1,227 upper-boundary columns of 11 diagrams; HF = 77/77/80/81 upper columns
(C/A/D/B, no-band excluded, identical to the frozen scorer — asserted in the script). Sample-1 β ≤ 0.075 columns with
an upper interval: C_T2 64, A_T2 63, D_T2 60, B_T2s1 59 (23 β clusters each).

### Q1 — the pre-HF quadratic surface does NOT explain the failure (post hoc evaluation of a model fixed before the data)
- Reproduction: a, b, g, q = 1.7901, −1.5356, +0.4807, −0.3438, NLL 1581.60 — identical to `analysis2b.json`
  (max |Δ| = 0.0; saved 10:38, HF scored 21:41).
- Local exponents on each HF design (b2 surface → observed free fit [95 % CI]):
  e_β C −0.25 → −0.75 [−1.27, −0.44]; A −0.25 → −0.82 [−1.30, −0.47]; D −0.27 → −0.48 [−0.79, −0.24];
  B −0.32 → −0.39 [−0.68, −0.16] (inside the CI for D and B only).
  e_v C 0.13 → 0.79 [0.59, 1.11]; A 0.13 → 0.91; D 0.14 → 0.79; B 0.17 → 0.57 [0.45, 0.77] (outside all 4).
  The surface's lv·lb interaction (g > 0) extrapolates to an almost speed-independent F_max at β 0.02–0.07; the data
  keep e_v ≈ 0.6–0.9.
- Level error (observed − predicted, sample-1 offset of the same string type): b2 **+0.68 / +0.62 / +0.57 / +0.53**
  (C/A/D/B), i.e. worse than frozen S1_strict (+0.36/+0.22/+0.17/+0.36). Linear reference (same fit, no g, q):
  +0.36/+0.28/+0.20/+0.25. None within ±0.15. → "also under-predicts"; the curvature terms make it worse.

### Q2 — planned parametric result says "session offset", but its identification is weak (see additions)
- δ (HF − sample 1, common free shape, β-cluster bootstrap B 1000 seed 7): C **+0.45 [0.14, 0.62]**,
  A **+0.51 [0.28, 0.65]**, D **+0.39 [0.15, 0.55]**, B **+0.38 [0.26, 0.48]**. Shape: e_v 0.57–0.68, e_b −0.48…−0.52.
  Planned wording: "session-level offset (HF higher) consistent with the data" for 4/4.
- Caveat (planned): sample-1 columns here are right-censored 55/64, 50/63, 52/60, 51/59; only 9/13/8/8 are finite,
  maximum sample-1 force 3.9–4.5 N. δ is carried by those few finite intervals plus the linear common-shape assumption.
- Contradictions (nearest β at same speed; max log-β distance 0.30/0.29/0.53/0.51):
  HF entirely below sample 1: **0/36, 0/38, 0/43, 0/39**. HF lower bound above sample-1 finite upper bound:
  **11/11, 7/7, 3/3, 3/3** (C, A, D at v = 0.20 except one column each for C and A at v = 0.10; B at v = 0.05).

### Additions after results (post hoc on post hoc; README section above, 22:41)
- A1: with nearest-β distance ≤ 0.06: 5/5, 7/7, 3/3, 3/3 still contradict. Number of sample-1 strokes above the
  sample-1 band end in those columns is 1–3 (C: eight have 1; A: all 2–3; D: 1,1,3; B: 1,1,2). Excluding the
  single-top-stroke cases leaves 3, 7, 1, 1.
- A2 (one variant: sample-1 top-stroke dropouts → right-censored; 1/1/2/1 columns recoded):
  δ C +0.37 [−0.04, 0.57], A +0.50 [0.26, 0.64], D +0.31 [0.002, 0.47], B +0.37 [0.22, 0.47]. Point values barely move.
- A3 (model-free: same measured window force 3.8–4.6 N, β ≤ 0.07, single-window H fraction, sample 1 vs HF):
  - v 0.05, β ≥ 0.04 (4 N is near F_max here): C 10/20 = 0.50 vs 7/57 = 0.12 (Fisher p 0.001); A 7/20 = 0.35 vs
    3/67 = 0.04 (p 0.001); B 3/10 = 0.30 vs 26/47 = 0.55 (p 0.18). For C and A the **HF session is Helmholtz LESS
    often at the same force** — the opposite of a +0.4 session offset; B is in the +direction but not significant.
  - v 0.20, β < 0.04: C 0.62 vs 0.62, A 0.46 vs 0.55, B 0.25 vs 0.26 (all p ≥ 0.5). Both sessions are only
    intermittently Helmholtz at ~4 N there; in HF the band then continues up to 12 N. So the sample-1 "finite upper
    boundaries" at v 0.20 / low β (which produce most HF-higher contradictions) sit inside an intermittent zone at the
    top of sample 1's force range, not at an F_max that HF contradicts at equal force.
  - v 0.10: ≥ 0.90 Helmholtz in both sessions. D: overlap only 3.95–4.02 N (n ≤ 19) → not usable.
  - HF overlap strokes have 0.03–0.2 N higher median force than sample-1 overlap strokes; small vs the differences.
- Reading of Q2 + additions: the parametric δ > 0 is not supported by any same-force comparison; at equal force the
  sessions agree (v 0.10, 0.20) or HF is lower (v 0.05, C/A). The under-prediction is located where only HF has data
  (> 4.6 N; e.g. 22–29 of 30 v 0.20 columns right-censored at 12 N). This favours a force-range/shape
  (extrapolation) reading over a uniform session offset, but it cannot exclude a session effect that only shows above
  4.6 N — session and force range remain confounded by design.

### Q3 — force-reading artefact cannot supply +0.17…+0.36
- Label-free, r_on = ln(F_window(0.2 s)/F_onset), median by onset-force band (all speeds; n per band 550–1,511 HF):
  HF 3.5–5 N −0.10…−0.11, 5–7 N −0.087…−0.097, 7–9 N −0.074…−0.080, 9–13.5 N −0.064…−0.069 (absolute fall
  0.45–0.72 N). Sample 1: 0.25–0.5 N −0.64…−0.83, 0.5–1 N −0.31…−0.37, 1–2 N −0.16…−0.18, 2–4.5 N −0.090…−0.100.
  The two sessions join continuously in the overlap (2–4.5 N vs 3.5–5 N); the relative fall shrinks with force
  (absolute fall grows: compare A2's 0.17 N over 14 cm at ranks 1–20). Drift within the plateau r_plat (t 0 → 0.2):
  −0.006…−0.018 at HF.
- Boundary strokes (two strokes bracketing finite upper intervals): HF −0.066…−0.075 (n 72/76/86/78) vs sample 1
  −0.042…−0.055 (n 102/106/96/88) → Δr = −0.022 / −0.011 / −0.027 / −0.029 (C/A/D/B). Wrong sign and ≤ 0.03 in size;
  re-expressing boundaries in onset force would leave level errors of +0.39/+0.23/+0.19/+0.39. Decision (planned
  wording): "a force-reading shift of this kind is too small to explain the under-prediction". Not tested: a
  session-to-session load-cell gain/offset error (no reference force is available; it is not identifiable here).

### Q4 — rule sensitivity (summary of `highforce_confirm/posthoc_multiband.json`, not recomputed)
- Multi-band share of banded columns: development 0.37–0.73 (D_T2 0.37, A1 0.38 … D1 0.73), HF 0.47–0.70 → not
  specific to high force; extra bands are short (median 2–4 strokes) and separated by a median 1-stroke gap.
- S1_strict level error by upper rule (C/A/D/B): last band (frozen) +0.36/+0.22/+0.17/+0.36; longest band
  +0.02/−0.14/−0.08/+0.19; first band −0.14/−0.38/−0.24/+0.00. Free e_β: last −0.75/−0.82/−0.48/−0.39, longest
  +0.05/−0.28/−0.15/−0.22, first +0.53/+0.24/+0.13/+0.23. In the 4–12 N range the level error is of the same order as
  the rule-to-rule spread (~0.4–0.6 log), so part of the failure may be rule fragility; the frozen verdict stands.

### Overall (post hoc, association only)
1. Not a force-reading artefact (Q3, |Δr| ≤ 0.03, wrong sign).
2. Not rescued by the pre-existing curvature surface (Q1; worse, +0.53…+0.68).
3. The planned parametric session offset (Q2, δ +0.38…+0.51) is not confirmed by the only same-force comparison
   (A3): no positive session shift at 3.8–4.6 N, and for C/A the opposite at v 0.05. The discrepancy lives above
   sample 1's force ceiling, where the laws extrapolate, and it is entangled with the last-band rule (Q4).
4. Nothing here is a confirmation; HF data are seen. Separating session from force range needs a session that spans
   both ranges (e.g. a repeat sample-1 session up to 12 N), which this dataset does not contain.

### Not done
- No uncertainty propagation for the b2 surface's local exponents (deterministic refit only, as planned).
- Only one δ sensitivity variant (A2); no variant recoding 2–3-stroke top dropouts (would be a forking path).
- D: same-force overlap too narrow (3.95–4.02 N) for A3.


## Verifier corrections (workflow wf_80e89ee6-df4, independent re-run; reproduced = True, forbidden data used = False)

**Corrected headline (use this one):** POST HOC; the HF labels have already been seen. All headline numbers reproduce exactly on re-run. (1) The b2 quadratic surface, fitted before the HF data existed, reproduces exactly (max |Δ| = 0). It predicts HF levels worse than the frozen laws: +0.53 to +0.68 vs +0.17 to +0.36. Its local e_v is 0.13 to 0.17, against 0.57 to 0.91 observed. (2) By the planned decision rule, Q2 reads 'session offset, HF higher': δ = +0.38 to +0.51, CI excludes 0 for 4/4 strings. But δ rests on only 8 to 13 finite sample-1 columns; the rest are right-censored near 4 N. Post-plan checks at the same measured force (3.8 to 4.6 N) show no HF-higher shift. These checks are descriptive and clustered, and the β split was chosen after looking. They cannot reach above 4.6 N, where the miss is. Session and force range stay confounded, and neither explanation is favoured. (3) The onset-to-window force drop differs between sessions by at most about 0.03 log, too small to explain the miss. A calibration difference between sessions cannot be tested. (4) Changing the upper-band rule moves the level errors by about 0.4 to 0.6 log, as much as the failure itself. The frozen verdict (P2 failed 4/4) stands.

- [medium] Headline: 'not a force-reading artifact' — Q3 only tests how far the force drops from onset to the t=0.2 s window. A difference in load-cell gain or offset between sessions was not tested and cannot be identified here; the README says so, but the headline drops that limit. Separately, Δr is not matched on β or force. Sample-1 boundary strokes are pooled over all β. A β≤0.075 variant was computed but not reported: it gives A_T2 r_on −0.095, so Δr for A is +0.029 (right sign) instead of −0.011. That makes 'wrong sign' true for 3/4, not 4/4. |Δr| stays below 0.10, so the planned decision does not change.
  Correction: Say: 'the onset-to-window force drop differs by at most 0.03 between sessions (−0.022/−0.011/−0.027/−0.029 pooled; A becomes +0.029 when sample 1 is restricted to β≤0.075), too small to explain +0.17 to +0.36. A between-session calibration error cannot be tested.' Report the β≤0.075 Δr alongside.
- [medium] A3 same-force comparison (p = 0.001) — The β<0.04 / β≥0.04 split and the Fisher tests (posthoc_hf_a3split.py, 22:44) were decided after the pooled A3 was seen. They are not in the 22:41 additions list. Fisher's test treats strokes as independent, but strokes are clustered within columns and ramps. 24 cells were tested (4 strings × 3 speeds × 2 β halves) with no multiplicity correction. The test compares Helmholtz fraction at about 4 N, not F_max. HF strokes at 3.8 to 4.6 N sit at the bottom of HF ramps, while sample-1 strokes there sit at the top, so ramp position is confounded. 'p = 0.001' in the headline overstates the evidence.
  Correction: Label A3_split as post hoc on post hoc, decided after seeing pooled A3. Drop the p-values from the headline, or give a column-clustered test with a note that 24 cells were tested. Describe it as 'descriptive: no same-force cell shows HF more often Helmholtz'.
- [medium] Headline and 'Overall' framing (extrapolation favoured over session offset) — The planned Q2 decision rule gives 'session-level offset (HF higher) consistent with the data' for 4/4: δ +0.38 to +0.51, all CIs exclude 0, and it reproduces exactly. The headline and README ('this favours a force-range/shape reading') put post-plan additions (A1 to A3) ahead of the planned outcome. A3 cannot say anything above 4.6 N, which is where the miss is. The design still confounds session with force range.
  Correction: Lead with the planned result: 'planned Q2: session offset δ +0.38 to +0.51 (CI excludes 0), but it is identified mainly from right-censored sample-1 columns (8 to 13 finite).' Then: 'post-plan same-force checks at 3.8 to 4.6 N show no positive shift. Session and force range cannot be separated; neither reading is favoured.'
- [low] Q1 exponent inside/outside-CI comparison — The b2 local exponents come from one deterministic fit, so 'outside the observed CI' ignores the surface's own uncertainty. The plan allowed this, but it makes the e_v and e_β mismatch look sharper than it is.
  Correction: Add: 'surface uncertainty not propagated; inside/outside refers to the observed CI only.'
- [low] Plan-before-results evidence — The order inside README.md is correct (PLAN, then additions at 22:41, then RESULTS), and the script (22:38) and run.log (22:40) mtimes are consistent with that. But the claim that the plan was written at 22:37 is self-reported; no earlier snapshot of the README exists.
  Correction: No change to the analysis. Note that the plan timing is self-reported.
