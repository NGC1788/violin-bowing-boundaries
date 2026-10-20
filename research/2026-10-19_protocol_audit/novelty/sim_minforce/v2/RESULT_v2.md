# Simulation v2 — result: validation failed again (no reading)

Claude, 2026-10-21 ≈00:15 KST. Run once (`vsim2.py` → `results_v2.json`, 3 s).

- **C0n validation failed.** With all loss lumped in the nut-side loop and the bridge loop lossless, Helmholtz labels
  persisted down to the lowest grid force (0.00045 N) at every β (F_min/F_S = 0.06, b_β ≈ 0). PLAN_v2 requires
  F_min/F_S in [1, 4] and b_β = −2 ± 0.3, so **nothing below is interpreted**.
- Diagnosis (after the fact): (1) Schelleng's minimum-force mechanism needs loss on the *bridge* side of the bow (the
  stick-phase force ramp comes from the imperfect reflection that reaches the bow from the bridge); the verifier's
  distributed and bridge-lumped variants did reproduce Schelleng (Q 1.2–3.0). Lumping at the nut removed the mechanism
  — a design error in PLAN_v2, not a property of the string. (2) The v1 classifier checks only one slip per period and
  the period, not the amplitude: a slowly decaying free Helmholtz motion (τ = 5.65 s for C0) is labelled Helmholtz
  for 0.5 s even when the bow cannot sustain it.
- Uninterpreted numbers, for the record only: C1n F_min 3.70 / 1.50 / 1.11 / 0.61 / 0.18 N at β 0.05 / 0.08 / 0.1 / 0.12 /
  0.2 (none at 0.02, 0.03), b_β −2.17; C1n-flat b_β −1.77; FIR magnitude error ≤ 0.20 (log) for n ≤ 100; D = 2000 gave
  the same F_min at β 0.12 and none at 0.03.
- A valid v3 would need: loss filters in **both** loops in proportion to their length (distributed loss; the bridge
  loop is short, so a low-order filter), an amplitude/energy criterion in the label (e.g. stick-phase bow-point
  velocity and slip-phase amplitude within 10 % of ideal Helmholtz), and the validation rerun before any C1 number is
  read. Not started (usage limit).
