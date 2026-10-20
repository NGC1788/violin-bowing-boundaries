# Simulation outcome after v3/v3b (Claude, 2026-10-21 ≈02:10 KST)
- **What now works:** the corner-exact waveguide maintenance test with all loss on the bridge side reproduces Schelleng
  for frequency-independent loss (C0b: F_min/F_S = 2.08 at β 0.1, b_β = −1.90; pre-registered gate passed).
- **What does not yet work:** putting the *measured* frequency-dependent damping into that short bridge-side loop as a
  linear-phase FIR failed the pre-registered accuracy gate twice (v3 K = 64, v3b K = 300). No C1 number is interpreted.
- **Question still open:** does measured frequency-dependent damping lift the maintenance minimum ~10× and flatten its
  exponent? Next design (v4, not started): smooth the ζ_n target (e.g. a pre-declared power-law fit instead of raw
  bumpy modes) and use a minimum-phase or low-order IIR loss filter, or distribute the loss over both loops; keep the
  same C0b gate and add a fundamental-loss check (filter DC/low-mode gain within 0.1 % of target).
