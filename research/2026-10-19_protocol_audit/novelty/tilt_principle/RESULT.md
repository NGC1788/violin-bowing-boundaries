# Threshold-tilt principle — result (run once; Claude, 2026-10-21 ≈14:12 KST)

`tilt.py` → `results.json`, `run.log`, `fig_tilt.png`. Plan: `PLAN.md` (4d8d11b). Base exponents reproduce the stored
values exactly in all 14 sessions.

## Measured feature scaling among base-Helmholtz windows (within speed folders)
amplitude b = −0.86 to −0.97 (mean −0.92); flyback height −0.86 to −0.99 (mean −0.94); B1 corner width +0.59.

## Pre-registered tests (primary: amplitude, window axis, 15 % removed)
- Mean shift of the minimum-force β exponent for a = −1.5 … 0: −0.65, −0.27, +0.14, +0.35, +0.60, +0.75, +0.85.
- **T1** monotonicity: ρ = +1.00; per-session ρ ≥ 0.7 in **14 / 14** sessions → pass.
- **T2** neutral point: zero crossing at a = −1.09 vs mean b = −0.92 (|diff| 0.17 ≤ 0.25) → pass.
- **T3** size: ρ(|a − b|, |Δ|) = 0.96 → pass.
- **Reading: "principle confirmed".** The same holds on the onset axis (zero −1.05 vs b −0.93), at 30 % removal
  (zero −0.98 / −0.97), and for flyback height (zero −1.06 vs b −0.94; 30 %: −1.00). All eight lower-bound cases
  pass T1–T3.
- **B1 corner width (upper bound): the predicted sign reversal holds** — shifts +0.51, +0.43, +0.36, +0.02, −0.37 for
  a = 0 … 1 (ρ = −1.00), zero crossing 0.76 vs b 0.59 (within 0.25), on both axes and at 30 %.

## What it means
- A threshold on a Helmholtz feature is β-neutral only when it scales like the feature itself; every unit of mismatch
  between the threshold tilt and the feature's own β-scaling moves the fitted minimum-force β exponent by about one
  unit when 15 % of Helmholtz windows are affected (about two units at 30 %). Lower and upper bounds act in opposite
  directions.
- The neutral tilt is predicted from the data before any boundary is fitted (b, estimated among accepted windows);
  the zero crossing sits slightly beyond b (by 0.1–0.17 for the lower bounds, consistently), presumably because the
  windows that decide the boundary (lowest forces) scale a little differently from all accepted windows.
- This explains, with one rule, the fixed amplitude floor (a = 0 vs b ≈ −0.9 → flattens by ≈ +0.85 at 15 %, observed
  +1.4 at its own removal level), the B1 corner-width criterion (a = 1 vs b ≈ 0.5 → steepens), and plausibly the steep
  simulated exponents with the β-scaled amplitude/slip label.
- Diagnostic for any boundary study: estimate each feature's β-scaling among accepted windows and scale its threshold
  accordingly, or report the exponent's sensitivity to the threshold tilt.
