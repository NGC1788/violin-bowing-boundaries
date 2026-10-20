# Does the force range of a scan bias the estimated maximum bow force? (post hoc; plan written before computing)

Claude, 2026-10-20. All data used here are already SEEN (B_T2 sample 2 and the four high-force sessions were scored
once under frozen plans; nothing below changes those verdicts).

## Question
Across eight frozen tests the strict cZv/β level error rose with the session's force ceiling (−0.17 … +0.36; post hoc
ρ = +0.74). Is this a property of the sessions, or of estimating F_max from a scan whose top force is limited?
B_T2 sample 2 spans 0.25–7.5 N in ONE session, so truncating its scan at virtual ceilings holds the session fixed and
varies only the force range. The high-force sessions (3.8–12.6 N) allow the same at the top end.

## Plan (fixed before computing)
- Boundaries: frozen convention (t = 0.2 s, K = 2, measured window force, last-band upper rule, compare.intervals).
- Truncation: within each column keep only strokes whose measured window force ≤ c, then re-extract intervals
  (H at the new top → right-censored at the new top force).
- B_T2 sample 2: c ∈ {2, 3, 4, 5, 6, full}; the three standard speeds (0.05/0.10/0.20; primary) and all four speeds.
- High-force sessions: c ∈ {7, 9, full} (their floor is 3.8 N).
- Statistics per (session, c): strict cZv/β level error (offset_fit vs FROZEN_v2 S1_strict), empirical-shape E3_free_Z
  level error, free-fit e_v and e_β (censored, no bootstrap), column counts (finite / right-censored / no band).
  Bootstrap of the B_T2 s2 level-error change between c = 4 and full over β-columns paired across speeds (B 1000,
  seed 31).
- Reading: if the level error moves upward as c rises within one session by an amount comparable to the between-
  session trend (≈ 0.2–0.4 over 4 → 12 N), the ceiling trend is (at least partly) an estimation/shape effect of the
  scan range; if it stays flat, the trend points to sessions.
- No other ceilings, rules or windows are tried; all results reported.
