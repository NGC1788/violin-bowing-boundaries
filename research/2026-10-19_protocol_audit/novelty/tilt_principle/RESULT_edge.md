# Refined tilt principle — result (run once; Claude, 2026-10-21 ≈14:30 KST). Plan: PLAN_edge.md (81d77c4).

## Consistency check (outcomes known)
Edge-window scaling ≈ all-window scaling: amplitude b_edge −0.93 vs b_all −0.92 (known zero −1.09); flyback height
−0.93 vs −0.94 (zero −1.06); flyback ratio +0.35 vs +0.46 (no zero crossing up to a = 1); B1 corner +0.59 vs +0.59
(zero +0.76). **The refinement does not explain the offsets.** Not supported.

## Confirmatory periodicity sweep (feature never swept before)
- Mean Δ for a = −0.6 … 0.6 (15 %, window): −1.37, −1.37, −1.35, +0.23, +0.89, +0.95, +0.99; ρ = +1.00, 14/14 sessions;
  zero crossing −0.029; b_edge −0.007, b_all −0.014. Onset axis and 30 % the same.
- **Pre-declared reading: "direction only"** (the refined rule required b_edge to be closer than b_all; it was not).
- Not the pre-declared reading for this test, stated for completeness: by the original rule (PLAN.md, T1 + T2 with
  b_all) periodicity would pass (|−0.029 − (−0.014)| = 0.015).
- Periodicity varies very little across Helmholtz windows (0.8–1.0), so even a tilt of ±0.2 turns the threshold into
  an almost pure β filter (shifts of −1.35 / +0.89). Post hoc: the distortion per unit of tilt mismatch grows as the
  feature's spread within a β column shrinks.

## Overall state of the tilt principle (all tests so far)
- **Direction rule: confirmed for every feature tested on the minimum force** — amplitude, flyback height, periodicity
  (lower bounds), flyback ratio and B1 corner width (upper bounds; reversed sign) — 14/14 sessions each.
- **Neutral point ≈ the feature's own β-scaling over accepted windows**: within 0.25 for amplitude, flyback height,
  B1 corner width and periodicity; fails for the flyback ratio (no neutral point within a ∈ [−1, 1]).
- **Maximum force: does not transfer** (small, opposite-sign effects).
- Edge-window refinement: not supported.
