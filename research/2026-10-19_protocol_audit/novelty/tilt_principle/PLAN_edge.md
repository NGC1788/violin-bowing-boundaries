# Refined tilt principle: the scaling that matters is that of the boundary-deciding windows (pre-registered)

Claude, 2026-10-21 (before any code for this; the commit time is the record).

## Refinement (post hoc idea from RESULT_generalize.md)
The neutral tilt of a threshold equals the β-scaling of the feature among the **edge windows** — for the minimum force,
in every (folder, column) the two Helmholtz windows that start the first K = 2 Helmholtz run (the windows at F_hi of the
lower interval and the next one) — not among all accepted windows.
b_edge = within-folder slope of log X on log β over the edge windows.

## Consistency check (outcomes already known — not confirmatory)
Compare b_edge with the known zero crossings: amplitude −1.09, flyback height −1.06 (window axis, 15 %), B1 corner width
+0.76, flyback ratio (no crossing up to a = 1). Reported as is.

## Confirmatory test on a feature never swept before: periodicity (lower bound)
- Criterion: periodicity ≥ K·β^a, a ∈ {−0.6, −0.4, −0.2, 0, 0.2, 0.4, 0.6}; 15 % of base-Helmholtz windows removed per
  speed folder (30 % reported); 14 sessions, window axis primary, onset reported.
- b_all and b_edge for periodicity computed before the sweep.
- Predictions: Δ increases with a (ρ ≥ 0.9 on the mean; per-session ρ ≥ 0.7 in ≥ 11/14); the zero crossing lies within
  ±0.25 of b_edge. **"Refined rule confirmed"** if both hold and |zero − b_edge| < |zero − b_all|; "direction only" if
  only the first; otherwise "not confirmed". If periodicity barely varies (|Δ| < 0.05 everywhere), "uninformative".
