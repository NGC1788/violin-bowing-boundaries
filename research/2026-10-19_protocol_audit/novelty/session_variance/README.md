# How much of the boundary-level variation is string type, and how much is recording session? (post hoc; plan first)

Claude, 2026-10-20 ~09:50. All sessions used here are already seen; frozen verdicts are unchanged.

## Motivation
The dataset paper judges string differences against a same-day repeat. Our frozen tests showed different-day sessions of
one string type differing far more than same-day repeats. If the session component is as large as the string-type
component, string comparisons in this collection need replicate sessions, and the heterogeneous T1→T2 pair exponents are
expected.

## Data (one level per recording session)
Sessions in the ≤ ~4 N design (3 speeds × 40 β × 50 forces): A1, A2, A3, A4 (A_T1); B1 (B_T1); C1 (C_T1); D1, D2 (D_T1);
A_T2; B_T2s1, B_T2 Part 6 (r1 and r2 are same-day repeats → averaged into one session, their difference reported as the
same-day yardstick); C_T2; D_T2. Sensitivity: + B_T2 sample 2 truncated at ≤ 4 N (40 × 40 design, standard speeds).
High-force sessions are excluded (different force range); reported separately as context only.

## Levels (one function for all sessions)
- Upper: interval-censored single offset of the session's upper columns relative to the frozen strict law
  S1_strict (FROZEN_v2 θ, published Z of the type × tension) — thermal_law.offset_fit.
- Lower: single offset of the session's lower columns relative to the A1+A2 free-exponent lower law without Z
  (compare.physics_models P-free with Z = 1, refitted deterministically on A1+A2), same offset_fit. Zero-width lower
  intervals (both ends at the 1 mN floor) are widened by ±0.01 N for every session (the D_T2 repair rule).
- t = 0.2 s, K = 2, measured window force. SE of each level: β-column cluster bootstrap paired across speeds, B = 200,
  seed 41.

## Model
y_s = μ + a_g + b_s + e_s; g = type × tension group, s = session; a_g ~ N(0, τ²) (string type × tension),
b_s ~ N(0, ω²) (session), e_s ~ N(0, se_s²) known. REML for (μ, τ, ω); likelihood-ratio test of ω = 0 and τ = 0
(boundary χ² mixture, p halved); profile-likelihood 95 % intervals for ω and τ. Same-day yardstick |r1 − r2| reported
next to ω. Reported for upper and lower separately, with and without B_T2 s2 (≤ 4 N).

## Reading (fixed)
- ω ≥ τ (point) with the ω = 0 test rejecting → "session variation is at least as large as string variation".
- ω ≫ |r1 − r2|/√2 → "same-day repeats understate session variability".
Anything else is reported as found. No other groupings, references or rules are tried.

## RESULTS (2026-10-20; run.log, results.json)
Upper (vs strict cZv/β): τ_type 0.058 [0, 0.145] (p τ=0 0.13); ω_session 0.069 [0.04, 0.13] (p ω=0 < 0.001); same-day
|r1 − r2|/√2 = 0.002. With B_T2 s2 (≤ 4 N): τ 0.065, ω 0.066 [0.04, 0.12].
Lower (vs A1+A2 law): τ 0.114 [0, 0.32] (p 0.23); ω 0.158 [0.09, 0.30] (p < 0.001); same-day 0.127.
A_T1 four samples: upper −0.100/−0.104/−0.066/−0.173, lower +0.115/−0.118/−0.196/−0.367.
Reading per plan: ω ≥ τ (point) with ω = 0 rejected for both boundaries → "session (specimen + day) variation is at least
as large as string-type variation"; for the upper boundary ω ≫ same-day yardstick. Specimen and day are confounded.
Simulation check of reml() with this layout and these SEs (sim_check_reml.py, 200 reps each): recovers τ and ω (median
0.057/0.064 for 0.06/0.07); false rejection of ω = 0 when ω = 0: 4 %; power at ω = 0.07: 0.82–0.83; ω CI coverage 0.93–0.98.

## Adversarial verification (workflow wf_7f5846d0-4a8) — the conclusion above is REFUTED as stated
Numbers reproduce exactly (independent re-implementation; reml() identical; MoM and Bayesian estimators agree on points).
But: (i) with 5 within-type df τ and ω cannot be ranked (posterior P(ω ≥ τ) 0.6–0.7; the fixed reading rule fires in
16–36 % of simulations where string type truly dominates; τ CI contains the whole ω CI); (ii) upper ω comes from the single
D1–D2 pair (dropping either: ω 0.02, n.s., τ 0.08–0.09); lower ω comes from A1 (reverse scan) and A4 (dropping A1: τ 0.14 >
ω 0.09); within A_T1 the lower levels are rank-ordered with force floor and date; (iii) without dividing by the published Z,
the upper type component is τ 0.104 [0.025, 0.22] (p 0.02) > ω 0.070; (iv) β-block bootstrap SEs are 1.5–3× larger; (v) a
1-df same-day SD (0.002) is uninformative (95 % CI [0.001, 0.08]).
Supported statement: within some types (D_T1 upper; A_T1 lower, tracking scan protocol, force floor and date) sessions
differ as much as Z-normalised types; whether session variation exceeds type variation is not identified, and the data
do not contradict the dataset paper's "minimal within-type variability" in general. Prior work already measured
session repeatability (Schoonderwaldt et al. 2008 §3.6; Galluzzo 2003; Lampis et al. 2024a,b).
