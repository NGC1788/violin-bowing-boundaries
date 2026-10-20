# Outcome (Claude, 2026-10-21 ≈00:16 KST)
- v1 (`PLAN.md`, builder `RESULT.md`): C0 validation failed under the ramp-from-rest protocol → no reading. The
  verifier (`verify/VERIFY.md`) confirmed the failure with an independent waveguide and modal code, and **withdrew the
  builder's two appended observations**: the "ripple 3–30 × Schelleng" premise failure and the off-protocol C0
  maintenance F_min 0.10–0.34 N with exponent −0.8 to −1.1 are artifacts (Gibbs undershoot of the 300-mode corner;
  creeping slip from the hysteresis rule). Do not cite them. The verifier's corner-exact waveguide maintenance test
  reproduces Schelleng (F_min/F_S 2.1–2.5, b_β −1.95), which validates that solver for C0 only.
- v2 (`v2/PLAN_v2.md`, `v2/RESULT_v2.md`): C0n validation failed (nut-lumped loss removes Schelleng's mechanism;
  amplitude not checked by the label) → no reading.
- Net: the simulation question (does measured frequency-dependent damping lift the maintenance minimum ~10× and flatten
  its exponent?) is **open**. Cost: workflow 2 agents ≈ 0.40 M tokens, 42 min.
