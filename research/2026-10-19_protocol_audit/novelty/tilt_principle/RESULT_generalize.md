# Generalisation tests — result (run once; Claude, 2026-10-21 ≈14:28 KST). Plan: PLAN_generalize.md (ec650f5).

## G1 — second_flyback_ratio as an upper bound (minimum force)
- b_sfr > 0 in 14/14 sessions (mean +0.46).
- Mean Δ for a = −1, −0.5, 0, 0.5, 1 (15 %, window): +0.375, +0.296, +0.196, +0.141, +0.060 → decreasing as predicted
  (ρ = −1.00; 14/14 sessions); **no zero crossing in the tested range** (extrapolated beyond a ≈ 1.3, far from b = +0.46).
  Same on the onset axis and at 30 %. **Reading: "direction only".**
- Consistency check (outcome seen earlier): b_sfr > 0 in every session, so the principle predicts the observed
  direction of the flyback-fraction sweep (tightening flattens, loosening steepens). Explained in direction.

## G2 — maximum-force boundary, amplitude lower bound
- Mean Δ_max for a = −1.5 … 0 (15 %, window): −0.114, −0.045, −0.008, +0.003, +0.004, +0.018, +0.033 — small and
  **increasing** with a (ρ = +1.00), the opposite of the pre-declared prediction; per-session consistency 2/14.
  **Reading: "does not generalise".** Likely reason: the amplitude threshold removes mostly low-force windows, which do
  not decide the maximum-force boundary.

## What this changes
The neutral tilt predicted from *all* accepted windows worked for amplitude and flyback height (offset 0.1–0.17), for
the B1 corner width (0.17), but not for the flyback ratio (offset > 0.8), and the rule does not transfer to the
maximum force as stated. Refinement to test next (post hoc idea): the relevant scaling is that of the windows that
decide the boundary, not of all accepted windows.
