# robust_upper — robustness of the maximum-force (upper boundary) results

Claude, 2026-10-19 (night). **Everything here is POST HOC** on the 11 development diagrams
(A1, A2, B1, C1, D1, A_T2, B_T2s1, B_T2r1, B_T2r2, C_T2, D_T2). A_T2 and D_T2 were scored once earlier and are now seen
data; nothing in this folder is a confirmation. No reserved archive (A4, D2, B_T2 sample 2, coldwarm) and no high-force
label is loaded: the loader reads exactly 11 hard-coded cache paths. Natural log throughout.

## PLAN (written before any computation)

### Common definitions (reference = "baseline", reproduces the existing numbers)
- Columns: `final/compare.intervals` logic at t = 0.2 s, window force, one row per (diagram, speed folder, β column).
  Upper interval (lo, hi) in log N; right-censored at the column ceiling; rows with no H run are dropped (as before).
- E-shape level of a diagram: interval-censored normal offset δ_d with log F = R(v, β) + δ_d, where R is the A1+A2
  free law without Z (c + a log v + b log β, fitted on A1+A2 upper columns).
- Pairs (T1 → T2): A = A_T2 − mean(A1, A2); B = (0.5 B_T2s1 + 0.25 B_T2r1 + 0.25 B_T2r2) − B1 (session-balanced, as
  upper_law / paired.py); C = C_T2 − C1; D = D_T2 − D1. Z exponent g = Δlevel / Δln Z (published Z, POMA Table 1).
- D_T2 strict level error: ln c = offset of the strict law ln Z + log v − log β over the pooled T1 columns
  (A1, A2, B1, C1, D1), then D_T2 offset relative to ln c + ln Z_D + log v − log β.
- A_T2 shift: E-shape level of A_T2 (law R fixed); its Z exponent = shift / ln(1.1884/1.0592).
- Pooled shape: log F = δ_d (11 diagram offsets) + e_v log v + e_β log β, interval censored (no Z; offsets absorb Z).

### Variants (each recomputes every quantity, including the reference law R and ln c, under its own rule)
- **base**: all speeds, K = 2 (check: must reproduce A_T2 shift +0.197, D_T2 −0.0018, pair exponents A 1.71 / B −0.51 /
  C 3.25 / D 0.23, pooled e_v 0.697 / e_β −1.063).
- **U1 (no 0.20 m/s)**: only 0.05 and 0.10 m/s columns, everywhere (reference law, ln c, levels, pooled shape).
- **U2 (K = 3)**: intervals with K = 3 (first/last run of 3 consecutive H windows in force order) at all speeds.
- Secondary within U1/U2 (pre-declared, reported, not used for the decision): levels against the *frozen* baseline law R
  (BASE_A) instead of the refitted one.

### Uncertainty
- Pooled shape: cluster bootstrap over diagrams (11 diagrams drawn with replacement, a duplicated diagram gets its own
  offset), B = 1000, seed 3, for base, U1, U2.
- **Canonical bootstrap (U3)**: β-position clusters paired across speeds (one cluster = one column index of one diagram,
  carrying all its speed rows), resampled with replacement *within each diagram* (stratified), B = 2000,
  seed 20261019. One draw set per variant is shared by all quantities of that variant. Per replicate the reference law
  R and ln c are refitted from the resampled A1+A2 / T1 clusters (full propagation). Applied to: pooled shape (base only,
  as asked), pair Δlevel and g (base, U1, U2), D_T2 strict error (base, U1, U2), A_T2 shift and its g (base).
  For the A_T2 shift two versions are reported: (i) law R held fixed, only A_T2 resampled (like-for-like with the frozen
  design, which used 200 unpaired row resamples), (ii) R refitted per replicate. (i) is the one placed next to the frozen
  interval; (ii) is secondary. For D_T2 likewise (i) ln c fixed, D_T2 resampled; (ii) joint.
- Intervals: 95 % percentile. Failed fits are counted and reported (no silent drop).

### U4 torsion — defensibility rule (decided now)
Formula candidates: Z_tot = Z0 Z_R / (Z0 + Z_R) (Schoonderwaldt 2008 §1.1; Mansour 2016 eq. 2), Z_R = √(K_R I_R) / r².
The absolute Z_tot is computed only if (D2) POMA's "Zto" (manufacturer "torsional stiffness", unit printed g m²/s²) can
be converted to K_R [N m²] without an unstated unit assumption **and** (D3) the polar moment of inertia I_R is available or
given by a cited approximation **and** the resulting Z_R/Z0 for the four T1 strings falls in the literature range 1–4
(Schoonderwaldt: "factor 2 to 4"; van Walstijn 2026 / Matusiak 2024 cello G: ≈ 1.25). If any fails, the absolute Z_tot is
declared not defensible. In that case (decided now, not after seeing results) one bracketed sensitivity is still run,
because it uses Zto only as a unit-free T2/T1 ratio: Z_R(T2)/Z_R(T1) = √(Zto2 μ2 / (Zto1 μ1)) · d1/d2 (assumes I_R ∝ μ r²
within a type), with Z_R(T1) = k Z0(T1) for k ∈ {1.25, 2, 4}; pair exponents g_tot = Δlevel / Δln Z_tot (base levels,
canonical SE). If |Δln Z_tot| < 0.02 the exponent is reported as "not identified" (no number interpreted).

### U5 random effects
Meta-analysis of the pair exponents g_i with SE_i = SD of the canonical-bootstrap g_i (base variant), over {A, B, C} and
{A, B, C, D}. τ² by Paule–Mandel (DerSimonian–Laird also reported). Interval: HKSJ (t, k − 1 df) standard, plus the
truncated variant (q* = max(1, q)) — both pre-declared, both reported. Also Cochran Q and the fixed-effect estimate.
Caveat fixed now: SE_i is within-session sampling only; C_T2's published Z uncertainty (±0.10 kg/s) is not propagated.

### Decision wording (fixed now)
- A result "is robust" to a variant if the point estimate moves by less than the canonical SE of the base estimate
  and the 95 % interval's relation to the reference value (0 for Δlevel; 1 for g; −1/1 for e_β/e_v not used) is
  unchanged. Otherwise "changes" with the direction stated. Specifically:
  pooled e_v: robust if |Δe_v| < SE_base; pooled e_β: same rule.
  pair g: "1 inside/outside" the interval reported per pair and variant.
  D_T2 strict error: "within 0.12" (the frozen tolerance) reported per variant; this is not a re-test.
- RE: "a common Z¹ is compatible" if 1 lies in the HKSJ interval; "Z exponent not identified" if the interval width > 4.
- None of these statements is causal or confirmatory.

## RESULTS (appended after the run; all post hoc)

Files: `robust_upper.py` (all items, log `robust_upper.log`, 258 s single-thread), `results.json`,
`canonical_draws.npz` (all canonical draws), `fig_robust_upper.py` → `fig_robust_upper.png`.
Run: `OPENBLAS_NUM_THREADS=1 python -B robust_upper.py` (venv-mac). Fit failures: 0/2000 canonical, 0/1000 diagram
bootstrap in every variant.

**Checks.** K = 2 intervals equal `final/compare.intervals` exactly (1800 finite bounds, max |diff| 0). Base reproduces
A_T2 shift +0.197, ln c 0.9660, D_T2 strict error −0.0018, pair g A +1.71 / B −0.51 / C +3.25 / D +0.23, pooled
e_v +0.697 / e_β −1.063.

**Denominators.** 11 diagrams; 40 β-position clusters per diagram (the bootstrap unit). Upper-boundary rows:
base 1227 (109–115 per diagram), U1 879 (79–80), U2 1208 (105–114).

### U1 / U2 / base (canonical intervals: β-clusters within diagram, B 2000, seed 20261019; pooled: diagram bootstrap B 1000, seed 3)

| quantity | base | U1 no 0.20 m/s | U2 K = 3 |
|---|---|---|---|
| reference A1+A2 law (c, a_v, b_β) | −0.176, +0.661, −1.111 | +0.064, +0.836, −1.212 | −0.287, +0.581, −1.062 |
| pooled e_v [diag. boot] | +0.697 [0.633, 0.762] | +0.863 [0.802, 0.923] | +0.691 [0.618, 0.760] |
| pooled e_β [diag. boot] | −1.063 [−1.144, −1.001] | −1.191 [−1.254, −1.142] | −1.105 [−1.187, −1.048] |
| g A (Δlevel) | +1.71 [1.00, 2.52] (+0.197) | +1.95 [1.35, 2.55] (+0.225) | +1.83 [1.05, 2.71] (+0.210) |
| g B | −0.51 [−1.54, 0.48] (−0.040) | +0.04 [−0.70, 0.70] (+0.003) | −0.23 [−1.33, 0.77] (−0.018) |
| g C | +3.25 [1.13, 5.26] (+0.189) | +3.81 [1.74, 5.83] (+0.221) | +3.62 [1.42, 5.76] (+0.210) |
| g D | +0.23 [−0.23, 0.71] (+0.033) | +0.33 [−0.12, 0.85] (+0.048) | +0.10 [−0.36, 0.58] (+0.014) |
| D_T2 strict error [c fixed / joint] | −0.0018 [−0.081, 0.093] / [−0.086, 0.094] | −0.0015 [−0.064, 0.062] / [−0.066, 0.064] | −0.0133 [−0.094, 0.089] / [−0.103, 0.089] |
| A_T2 shift, law fixed (g) | +0.197 [0.124, 0.289] (g [1.07, 2.51]) | +0.225 [0.160, 0.287] (g [1.39, 2.49]) | +0.210 [0.128, 0.308] (g [1.12, 2.67]) |

Secondary (frozen BASE_A law instead of the refitted one): U1 g A +1.93, B +0.06, C +3.69, D +0.28 (A_T2 +0.226);
U2 g A +1.86, B −0.22, C +3.67, D +0.14 (A_T2 +0.183). Same pattern as the primary.

Decisions by the pre-declared rule (base canonical SE: e_v 0.027, e_β 0.033; g A 0.39, B 0.51, C 1.09, D 0.24):
- **Pooled shape is NOT robust to dropping 0.20 m/s**: e_v +0.697 → +0.863 (Δ +0.166 ≈ 6 SE), e_β −1.063 → −1.191
  (Δ −0.128 ≈ 4 SE); both move away from the base and the U1 intervals exclude the base point. Without the most
  censored speed the upper boundary is steeper in both v and β (still e_v < 1 and e_β < −1 by interval). K = 3: e_v
  robust (Δ −0.006); e_β changes slightly (Δ −0.042 > 1 SE, steeper).
- Pair exponents: point moves < 1 base SE for A, C, D in both variants and for B under K = 3; **B under U1 moves +0.55
  (> SE 0.51) toward 0**. Relation to 1: A contains 1 only in base (lower bound 0.997; U1 and U2 exclude 1 from above);
  B excludes 1 from below in all three; C excludes 1 from above in all three; D excludes 1 from below in all three. The
  heterogeneity (A, C above 1; B, D below 1) survives both variants.
- D_T2 strict level error stays within 0.12 in all variants (−0.0018 / −0.0015 / −0.0133). Not a re-test (D_T2 is seen).

### U3 canonical bootstrap next to the original intervals (base)

| quantity | original (as published) | canonical (B 2000, β-clusters paired across speeds, stratified) |
|---|---|---|
| pooled e_v | 0.697 [0.644, 0.757] (200 unstratified β-cluster) | [0.647, 0.751], SE 0.027; diagram boot [0.633, 0.762] |
| pooled e_β | −1.063 [−1.138, −0.996] | [−1.129, −1.004], SE 0.033; diagram boot [−1.144, −1.001] |
| g A | 1.71 [0.86, 2.47] (300 unpaired rows) | [0.997, 2.519], SE 0.39 |
| g B | −0.51 [−1.79, 0.75] | [−1.539, 0.478], SE 0.51 |
| g C | 3.25 [1.11, 5.12] | [1.126, 5.262], SE 1.09 |
| g D | 0.23 [−0.25, 0.70] (normal, incl. session-noise MoM) | [−0.234, 0.707], SE 0.24 (no session noise) |
| A_T2 shift (frozen) | +0.197 [0.106, 0.286]; g [0.92, 2.49] (200 unpaired rows, as executed) | law fixed: [0.124, 0.289]; g [1.07, 2.51] · law refitted: [0.115, 0.290]; g [0.998, 2.524] |
| D_T2 strict error (sealed) | −0.0018 [−0.078, 0.091] | c fixed [−0.081, 0.093]; joint [−0.086, 0.094] |

Note (added after the run, not in the plan): whether 1 lies inside the A interval is at Monte-Carlo resolution. Base
pair A: 2.5 % quantile 0.997, binomial 95 % band of that order statistic [0.972, 1.045], P(g < 1) = 0.0255. A_T2 law
fixed: 1.075 [1.031, 1.112], P(g < 1) = 0.013. The frozen interval [0.92, 2.49] stays the executed result; the canonical
law-fixed version puts its lower bound just above 1, the full-propagation version on 1. Neither changes "A > 0".

### U4 torsion

Formula exists locally: Z_tot = Z0 Z_R / (Z0 + Z_R) (Schoonderwaldt 2008 §1.1; Mansour 2016 eq. 2) with
Z_R = √(K_R I_R)/r² (Mansour 2016 §2.2). **Absolute Z_tot from POMA is not defensible**: (D2) POMA's Zto
(manufacturer "torsional stiffness", printed unit g m²/s², no definition, no SD) has neither a stiffness (N m²) nor an
impedance unit; (D3) no polar moment of inertia is published. With I_R = μr²/2, the three readings of Zto give
Z_R/Z0 for the T1 strings of 0.68–0.92, 22–29 and 128–233 — none inside the literature range 1–4 (cello G from
van Walstijn 2026 / Matusiak 2024 Table values: Z_R ≈ 1.47 / 1.43 kg/s). So no absolute effective-impedance result.

Pre-declared bracketed sensitivity (Zto used only as a unit-free T2/T1 ratio; assumes I_R ∝ μr² within a type;
Z_R(T1) = k·Z0). Δln Z_tot vs Δln Z0 and g_tot = Δlevel/Δln Z_tot (canonical 95 %):

| pair | Δln Z0 | Z_R ratio T2/T1 | k = 1.25 | k = 2 | k = 4 |
|---|---|---|---|---|---|
| A | 0.115 | 0.834 | −0.028 → g −7.1 [−10.4, −4.1] | +0.006 → not identified | +0.048 → g +4.1 [2.4, 6.0] |
| B | 0.079 | 1.035 | +0.059 → −0.7 [−2.1, 0.6] | +0.064 → −0.6 [−1.9, 0.6] | +0.070 → −0.6 [−1.7, 0.5] |
| C | 0.058 | 1.037 | +0.048 → +3.9 [1.4, 6.3] | +0.051 → +3.7 [1.3, 6.0] | +0.054 → +3.5 [1.2, 5.7] |
| D | 0.145 | 0.785 | −0.046 → −0.7 [−2.2, 0.7] | −0.001 → not identified | +0.055 → +0.6 [−0.6, 1.9] |

Reading: the manufacturer Zto drops 31 % (A) and 39 % (D)
from T1 to T2, so with torsion the A and D impedance changes shrink to near zero and their **sign depends on the unknown
Z_R/Z0**; their exponents are not identified. B and C change little (Z_R ratio 1.04; Zto +9 % and +3 %): B stays below 1 and C above 1 for
every k. Torsion therefore does not reconcile the pairs with a common exponent of 1; it removes the A and D pairs
from any exponent estimate. Assumption-dependent; not a result about torsion physics.

### U5 random-effects pooled Z exponent (base g_i, SE_i = canonical bootstrap SD)

| set | fixed (±SE) | Q (df), p | χ²(g = 1) (df), p | τ PM / DL | RE (PM) normal | RE (PM) HKSJ | RE (DL) HKSJ |
|---|---|---|---|---|---|---|---|
| A, B, C | +1.08 ± 0.30 | 16.2 (2), 0.0003 | 16.3 (3), 0.001 | 1.72 / 1.54 | +1.35 [−0.75, 3.44] | [−3.25, 5.94] | +1.32 [−3.25, 5.89] |
| A, B, C, D | +0.56 ± 0.19 | 21.2 (3), 0.0001 | 26.6 (4), < 0.0001 | 1.45 / 1.07 | +1.00 [−0.53, 2.54] | [−1.49, 3.50] | +0.93 [−1.47, 3.32] |

With PM, the HKSJ scale q is 1 by construction, so the truncated HKSJ equals the standard one (DL: q 1.20 / 1.57,
truncated = standard). Decision: 1 lies inside every HKSJ interval, and the widths (9.2 and 5.0) exceed 4 →
**"a common Z¹ is compatible, but the pooled Z exponent is not identified"** by these four pairs. The common-slope and
g = 1 tests reject homogeneity. SEs are within-session only; C_T2's published Z uncertainty is not propagated.

### What this does and does not show
- Robust: the A_T2 / A-pair shift is positive (all variants), C above 1 and B, D below 1 (pair heterogeneity), and the
  D_T2 strict level hit within 0.12 (all variants). These are associations on seen data.
- Not robust: the pooled speed and β exponents depend on the 0.20 m/s columns (most censored speed): without them
  e_v 0.86 and e_β −1.19. The headline v^0.70 is partly a censoring/extrapolation product of that speed.
- The canonical bootstrap does not change any qualitative statement; the A exponent's "1 inside" is at Monte-Carlo
  resolution in every bootstrap scheme.
- No causal claim (tension, Z, EI, T2 batch and session remain collinear); no new confirmation.


## Verifier corrections (workflow wf_80e89ee6-df4, independent re-run; reproduced = True, forbidden data used = False)

**Corrected headline (use this one):** Post hoc on the 11 development diagrams (no confirmation). Positive in every variant: the A_T2/A-pair shift, Δlevel 0.20–0.23. Within 0.12 in every variant: the D_T2 strict level error, |err| ≤ 0.013. Across variants C's Z exponent stays above 1 and B's and D's stay below 1, but only against within-session SEs; session noise is not included. A's exponent (≈1.7–1.9) has 1 at the edge of its base interval and inside the frozen one. The pooled shape depends on the speed range: with 0.05 and 0.10 m/s only, e_v rises from 0.70 to 0.86 and e_β falls from −1.06 to −1.19; the cause (censoring or curvature) is not identified. K=3 changes little. An absolute torsion-corrected impedance cannot be computed from POMA Zto; in a ratio-only sensitivity check, the A and D exponents become unidentified. The random-effects pooled Z exponent is not identified (HKSJ widths 5–9) and is compatible with 1.

- [medium] Headline: 'A and C above Z exponent 1' — For A this is overstated. The base canonical interval is [0.997, 2.519]: the 2.5% quantile is 0.997 and P(g<1)=0.0255, which I reproduced from canonical_draws.npz. The original/frozen intervals [0.86, 2.47] and [0.92, 2.49] contain 1. The README itself says 'A contains 1 only in base'. U4 at k=1.25 gives A g = −7.1. Only C is consistently above 1. A is above 1 only in U1 and U2.
  Correction: Say: 'C is above 1 and B and D are below 1 in every variant; A is about 1.7–1.9, with 1 at the edge of its base interval (inside the frozen interval).'
- [medium] Pair heterogeneity ('the pairs disagree', U5 Q=16.2/21.2, homogeneity rejected) — SE_i are within-session β-cluster bootstrap SDs only. Each T1→T2 contrast for A, C and D is one session against one or two sessions, so session-to-session variance is missing from SE_i. The original D interval included a session-noise term. Q and the χ²(g=1) rejections are therefore anti-conservative, and some of the disagreement could be session noise. The README notes the caveat but still presents 'homogeneity rejected' and the heterogeneity as robust findings.
  Correction: Call the heterogeneity 'beyond within-session sampling error only'. Do not call it an established disagreement between string types until session noise is included.
- [low] Interpretation of U1 ('v^0.70 is partly a censoring/extrapolation product of 0.20 m/s') — This is a causal reading the analysis cannot establish. A steeper slope from 0.05 and 0.10 m/s alone is equally consistent with real curvature in ln v (a non-power-law speed dependence), not only with a censoring artefact. The U1 and base fits share 879 of 1227 rows, so 'U1 interval excludes the base point' is not a test.
  Correction: Say: 'The pooled speed and β exponents depend on the speed range: with 0.05 and 0.10 m/s only, e_v=0.86 and e_β=−1.19. The single power law with e_v≈0.70 does not describe all speeds equally well. Censoring at 0.20 m/s or curvature could explain this; neither is distinguished here.'
- [low] Outcome-dependent row exclusion (U2) — Columns with no K-run of H windows are dropped, which is selection on the outcome. With K=3 a further 19 rows drop out. This follows the existing convention, but it is a truncation on the outcome, and the README mentions it only as a 'censoring mix' caveat.
  Correction: State it as truncation on outcome (rows without an H run are excluded) and give the count per variant.
- [low] PLAN-before-results ordering — File order is PLAN then RESULTS and every planned item (U1–U5, the secondary frozen-law analysis, both HKSJ variants, fit-failure counts) is reported. The timestamps cannot verify that the PLAN came first: README mtime 22:47 is later than the script (22:40). One post-hoc addition (the Monte-Carlo band for A) is correctly labelled as added after the run.
  Correction: None needed beyond noting that the PLAN timing is self-declared.
