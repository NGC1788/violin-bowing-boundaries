# Threshold-tilt principle: when does a Helmholtz criterion distort the minimum-force β law? (pre-registered)

Claude, 2026-10-21 (before any code; the commit time is the record). The principle was found post hoc on B1 corner width
(`novelty/label_sensitivity/b1_flyback/`) and in the simulations; this is its first confirmatory test.

## Principle (stated before testing)
A criterion "quantity X ≥ K·β^a" (lower bound) applied on top of the base Helmholtz label removes windows where X/β^a is
small. If X itself scales as β^b among Helmholtz windows, X/β^a ∝ β^(b−a): with b − a > 0 the removals concentrate at
small β, raising the small-β minimum force and steepening the fitted law; with b − a < 0 they concentrate at large β and
flatten it; at a = b the criterion is β-neutral. For an upper bound "X ≤ K·β^a" the direction reverses.

## Design (fixed)
- Data: the 14 sessions and the base label of `novelty/label_sensitivity/label_sweep.py` (t = 0.2 s window, K = 2
  boundary, per-session interval-censored fit), window-force axis primary, onset axis (11 sessions) reported.
- Quantities: **amplitude_std** (primary) and **flyback_height** (secondary), both lower bounds, all 14 sessions;
  **B1 corner width** d_fb (upper bound) from `b1_flyback.csv`.
- b per session = slope of log X on log β among base-Helmholtz windows with speed-folder fixed effects (demeaned within
  each folder), computed before any variant.
- Tilts: a ∈ {−1.5, −1.25, −1.0, −0.75, −0.5, −0.25, 0} for the lower bounds; a ∈ {0, 0.25, 0.5, 0.75, 1.0} for B1
  corner width.
- Strength fixed: within every speed folder, K is set so that exactly 15 % of base-Helmholtz windows are removed
  (lower bound: lowest 15 % of log X − a log β; upper bound: highest 15 %). Sensitivity: 30 % (reported, not read).
- Δ(a) = exponent(variant) − exponent(base), per session.

## Tests and readings (fixed; primary = amplitude, window axis)
- **T1 monotonicity**: Spearman ρ(a, mean Δ over sessions) ≥ 0.9 for lower bounds (Δ increases with a) — and per
  session ρ ≥ 0.7 in ≥ 11 of 14 sessions.
- **T2 neutral point**: the a where mean Δ crosses zero (linear interpolation) lies within ±0.25 of the mean b.
- **T3 size**: Spearman ρ(|a − mean b|, |mean Δ|) ≥ 0.8.
- Reading: **"principle confirmed"** if T1 and T2 pass; "direction only" if only T1 passes; "not confirmed" if T1 fails.
- Generality (secondary): the same tests for flyback_height; for B1 corner width the predicted direction is reversed
  (Δ decreases with a; ρ ≤ −0.9) with the neutral point within ±0.25 of B1's b.
