# Simulation v5 — result (run once, `run_v5.log`, `results_v5.json`; Claude, 2026-10-21 ≈12:10 KST)

- **Reading: undefined.** Started from rest with the measured ramp (0.104 m/s²), the v4 damped string (R1) gave no
  K = 2 Helmholtz run at any β or force; only isolated single Helmholtz labels (β 0.08: 2 of 32 forces, β 0.12: 1,
  β 0.2: 1, β ≤ 0.05 and β 0.1: none). The same held for the fast attack (R1-fast, 1.0 m/s²: none), the low-mode damping
  fit (R1-low: one label at β 0.2), v = 0.05 and 0.2, and at D = 8000. The control R0 (frequency-independent loss) gave
  F_min only at β 0.1 (0.82 N) and 0.2 (0.055 N), patchy.
- Meaning: in this model (point bow, Coulomb friction with an abrupt 1.2 → 0.6 drop, no torsion, no bow-hair compliance)
  Helmholtz motion is almost never *established* from rest under the robot's slow ramp, whereas the robot establishes it
  over most of the measured Schelleng plane. The model therefore lacks something that governs establishment; v4's
  maintenance result (level right at β 0.1, β dependence too steep) cannot be extended to the measured protocol with it.
- Known from the literature: van Walstijn et al. 2026 reproduced the measured Helmholtz region on this rig with a
  thermal elasto-plastic friction law plus torsion, bow-hair compliance and finite bow width.
