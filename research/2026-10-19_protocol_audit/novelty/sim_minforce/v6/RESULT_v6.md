# Simulation v6 — result (run once, `run_v6.log`, `results_v6.json`; Claude, 2026-10-21 ≈12:10 KST)

- **Gate passed, and cleaner than with Coulomb friction**: H0m (frequency-independent loss, smooth friction, ideal start)
  F_min/F_S = 1.17 at β 0.1 (Δμ_eff 0.77), b_β = −2.05, and the Helmholtz region is one continuous band at every β
  (e.g. β 0.1 `........HHHHHHHHHHHHHHHHHHHHH...`). The patchiness seen in v4 came from the abrupt Coulomb law.
- **Maintenance with the measured damping (H1m)**: r = 4.5 at β 0.1 → pre-declared "substantial (r ≥ 3)"; F_min =
  0.022 N at β 0.1 — **12–18× below the measurement** (0.27–0.40 N), unlike v4 (Coulomb, 0.335 N). b_β ≈ −6.5 (no
  maintenance at β ≤ 0.05; 0.45 / 0.012 / 0.0011 N at β 0.08 / 0.12 / 0.2), even steeper than v4 (−4.1).
- **Establishment from rest (H1r, and the control H0r)**: no K = 2 Helmholtz run at any β — the same as v5 with Coulomb
  friction → pre-declared reading "smooth friction does not restore establishment".
- Label note: at low forces with the measured damping many runs pass the period rule but fail the amplitude check
  (`c` in the label strings): a periodic one-slip motion with a strongly rounded corner (or the decaying start). The
  amplitude check was fixed in v3 to reject exactly this; counting them as Helmholtz would change small-β results.

## What is robust across v4 (Coulomb) and v6 (smooth friction)
1. The model reproduces Schelleng when the loss is frequency independent (gates passed twice).
2. The measured, frequency-dependent string damping **raises** the maintenance minimum (×4.5–27 at β 0.1) and makes it
   **fall much more steeply with β** (β^−4 to β^−6.5 instead of β^−2) — the opposite of the measured flattening
   (β^−1.2 to β^−1.5).
3. Neither friction law establishes Helmholtz motion from rest under the robot's slow ramp.
## What is not robust
- The absolute level: 0.335 N (Coulomb) vs 0.022 N (smooth) at β 0.1 — a 15× dependence on the friction law, so v4's
  agreement with the measured level was a coincidence of that law and must not be claimed.
