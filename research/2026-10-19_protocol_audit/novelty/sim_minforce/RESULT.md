# RESULT — simulated minimum bow force ablation (PLAN.md): **C0 validation FAILED → stopped**

Builder: Claude (workflow agent), 2026-10-20 ≈23:25–23:55 KST. PLAN.md not edited. No git actions.
Files: `sim.py` (model, kernel, classifier), `run.py` (stages checks / c0 / conv / all / summary / fig; only checks and c0 were run),
`diag_c0.py` (off-protocol C0 diagnostics → `diag_c0.json`, `fig_sim.png`), `results_sim.json`, `wave_C0_v0.1_b0.1.npz`
(raw bow-point relative velocity, C0, β 0.1, all 32 forces, label window), logs `log_checks.txt`, `log_c0.txt`, `log_diag_c0.txt`.

## Final summary
**Status: C0_failed.** Under the PLAN protocol (start from rest, linear ramp at 0.104 m/s² to 0.1 m/s, label in 0.2–0.4 s after the
plateau starts, K = 2), C0 (Coulomb μs 1.2 / μd 0.6, ζ_n = ζ₁/n, no stiffness) gives **no Helmholtz run at any of the 32 forces
(0.00045–5 N) for any β** (0.02, 0.03, 0.05, 0.08, 0.12, 0.2, plus 0.1). One isolated Helmholtz label (β 0.03, 0.136 N) does not
make a K = 2 run. So F_min, Q_sim(0.1) and b_β do not exist for C0 and the validation fails (pass would have needed Q within ×2
of F_S = 0.0080 N at β 0.1 and b_β = −2 ± 0.3). After 3 debug attempts it still fails. Per PLAN ("if C0 fails, nothing else is
interpreted") **the convergence check, C1, C2, C3 and the 0.05 / 0.2 m/s runs were not run, and no H1/H0 reading is made.**

### Numerics (what was built)
- Modal string, A T1: T 145.31 N, μ 7.721 g/m → Z = 1.0592 kg/s, L = 0.70136 m, B = 4.186e-5; mode mass μL/2 = 2.7076 g, shape sin(nπx/L).
- Per-mode **exact** discrete transition matrix of the damped oscillator with the friction force held over the step (ZOH);
  bow-point velocity at the step end = v_free + Y_eff·f; friction solved implicitly each step (stick if |f_stick| ≤ μs F, else slip on
  the friction law; hysteresis rule: an ongoing slip continues while a same-direction slip solution exists). Numba, 32 forces in parallel.
- fs = 96 kHz; modes up to f ≤ 0.3066 fs = 29.3 kHz → **N = 300** (≥ 2/β_min = 100); C3 would carry 189 stiff modes.
  The cutoff is chosen so the modal ZOH step admittance alone equals 1/(2Z) (Si(N ω₁ Δt) = π/2).
  **Bow-point admittance Y_eff·2Z = 0.998 (C0) / 0.993 (C1)** at β 0.1 (0.997–0.999 / 0.993–0.994 over all β); numerical
  infinite ZOH sum for the flexible string 1.0000; C3 0.663 vs its continuous stiff-string ZOH value 0.678 (dispersion above ~15 kHz is physical).
- Damping (C1–C3): digitized A T1 ζ_n, log-linear in n between digitized modes, beyond mode 17 power law **p = 0.658** (a = −8.528)
  fitted to all 11 modes; S10 recomputed = 0.0011789 = results.json. Free-decay check (C1 and C3, modes 1–60): frequency and ζ
  recovered within 0.14 % (worst C3 mode 60).
- Run time ≈ 0.85 s per β for 32 forces in parallel (≈ 0.03 s per run).

### Debug attempts (all reported, none tuned to data)
1. **Bug fixed.** First version added a memoryless residual admittance dY = 1/(2Z) − Y_modal for truncated modes. It acts as a series
   dashpot (passes DC, dissipates f²·dY during stick): the string stayed permanently stuck at F = 1.6 and 3.2 N. Removed; Y matched by
   the mode cutoff instead (above).
2. **Band limit** (β 0.1): fs 96 / 192 / 384 kHz (N 300 / 601 / 1203). Ramp protocol: still no Helmholtz at any force.
3. **Friction rule × label threshold** (β 0.1): hysteresis vs stick-whenever-possible, slip threshold 1 / 5 / 20 % of v.
   Ramp protocol: no K = 2 run in any of the 6 combinations (one isolated H with stick-if-possible at 20 %).

### Diagnosis (off-protocol, `diag_c0.json`, `fig_sim.png`; not a reading)
- (a) **Establishment under the ramp.** With ζ_n = ζ₁/n every mode decays with τ = 1/(ζ₁ω₁) = 5.65 s, so the irregular multi-slip motion
  created while the bow accelerates from rest never dies within the 1.2–1.4 s run: at every force the bow-point velocity shows many
  short slips per period (fig panel 2, blue).
- (b) **Maintenance is not Schelleng either.** Starting from an ideal Helmholtz state at constant 0.1 m/s (same solver, same label),
  Helmholtz is kept only above F_min = 0.335 N at β 0.1 (Q = 42; 0.452 N at 192 kHz, 0.184 N at 384 kHz: not converged, patchy
  labels). β scan: F_min = 1.11 / 0.61 / 0.45 / 0.45 / 0.136 / 0.184 N for β 0.02…0.2 (Q 5.6–92), **b_β = −0.83** (hysteresis rule);
  stick-if-possible rule 0.82 / 0.34 / 0.18 / 0.14 / 0.10 / 0.055 N (Q 3.8–28), **b_β = −1.08**, F_min(0.1) = 0.101 N (Q 12.6).
- Mechanism seen in the force waveform: the Coulomb jump (μs → μd) at each release launches secondary waves that the nearly lossless
  C0 string does not absorb. Median stick-phase force ripple (RMS after removing a linear trend) = 0.017 / 0.032 / 0.154 N at
  F = 0.335 / 0.611 / 2.03 N (5–8 % of F), against Schelleng's whole assumed stick-phase variation Z²v/(2Rβ²) = 0.0048 N at β 0.1.
  Schelleng's derivation keeps only the bridge-loss term; in this point-bow Coulomb model the ripple term is 3–30× larger, so
  C0 cannot reproduce F_S, and the C0 premise of PLAN.md does not hold for this model class.
- Incidental (exploratory scratch runs while debugging, not saved in this folder, uninterpreted): C1 at β 0.1, v 0.1 under the ramp
  protocol labelled Helmholtz only at 0.61 and 1.11 N on a coarse (every second) grid, with slip fractions 0.77 / 0.67 (not textbook Helmholtz).

### Deviations / implementation choices (explicit)
- Force grid extended by the same log step 8 points below 0.005 N (down to 0.00045 N), because F_S(C0) at β ≥ 0.12 is below the plan
  floor; plan-grid F_min is obtained by restricting to the 24 plan forces (both reported; for C0 both are None).
- Extra β = 0.1 run (for Q_sim(0.1) and the absolute comparison; not in the b_β fit).
- Each run stops at plateau + 0.4 s (end of the label window; causal, so the label cannot change); the last 0.1 s of the 0.5 s plateau is not simulated.
- Classifier: slip ⇔ |Δv| > 1 % of v; stick→slip times t_i in the window; a period counts as "exactly one" when the next interval is
  in [0.5, 1.5) T0; fraction = good / max(#intervals, 18); mean period = (t_last − t_first)/(M − 1). Exact-solver-state variant (ε = 0) also stored.
- Off-protocol diagnostics (ideal start, other band limits, other friction rule, other thresholds) are not used for any reading.
- Not done because of the stop rule: convergence check (C1, β 0.03 / 0.12), C1–C3, speed exponents, results figure for C0–C3
  (`fig_sim.png` is the C0 diagnostic figure instead).

---
## Progress log (auto-appended while running; the first "Stage 1" block used the rejected residual setup)


## Stage 1 — unit checks (auto)
- L = 0.7014 m, Z = 1.0592 kg/s, B = 4.186e-05, zeta1 = 2.879e-04; damping power law p = 0.658 (a = -8.528); S10 recomputed 0.00117891 vs results.json 0.00117891
- fs = 96 kHz, modes f_n <= 24 kHz: N = 245 (C0-C2), 166 (C3). Modal-sum bow admittance x 2Z (beta 0.02/0.1/0.2): C1 0.881/0.871/0.868; after truncation correction 1.000 (C0-C2), C3 0.677/0.679/0.678; numerical infinite ZOH sum x 2Z (flexible) 1.0000/1.0000/1.0000
- free decay (C1): n=1: f 97.80/97.80 Hz, zeta 2.879e-04/2.879e-04; n=2: f 195.60/195.60 Hz, zeta 2.897e-04/2.897e-04; n=3: f 293.40/293.40 Hz, zeta 6.672e-04/6.672e-04; n=5: f 489.00/489.00 Hz, zeta 4.984e-04/4.984e-04; n=10: f 978.00/978.00 Hz, zeta 8.402e-04/8.402e-04; n=17: f 1662.59/1662.60 Hz, zeta 3.496e-03/3.493e-03; n=30: f 2934.00/2934.00 Hz, zeta 1.857e-03/1.857e-03; n=60: f 5867.99/5868.00 Hz, zeta 2.931e-03/2.930e-03
- free decay (C3): n=1: f 97.80/97.80, zeta 2.879e-04/2.879e-04; n=2: f 195.62/195.62, zeta 2.897e-04/2.897e-04; n=3: f 293.46/293.46, zeta 6.672e-04/6.672e-04; n=5: f 489.26/489.26, zeta 4.984e-04/4.984e-04; n=10: f 980.04/980.05, zeta 8.402e-04/8.402e-04; n=17: f 1672.61/1672.63, zeta 3.494e-03/3.493e-03; n=30: f 2988.76/2988.76, zeta 1.857e-03/1.857e-03; n=60: f 6294.65/6294.68, zeta 2.934e-03/2.930e-03
- sanity beta 0.1, v 0.1, F = 0.05..3.2 N: C0 labels [False, False, False, False, True, False, False], C2 labels [False, False, True, True, True, False, False]; wall for 8 parallel runs 0.102 s

## Debug note 1 (numerics, before any C0 grid run)
- First implementation carried modes <= 0.25 fs (N = 245) and added a memoryless residual admittance dY = 1/(2Z) - Y_modal (0.13/(2Z)) to make Y_eff = 1/(2Z). Sanity run (C0, beta 0.1) showed the string PERMANENTLY STUCK at F = 1.6 and 3.2 N: the residual acts as a series dashpot (passes DC: string creeps under a stuck bow once mu_s F > v/dY = 1.63 N) and dissipates f^2 dY during stick, i.e. an artificial bow-point loss. Rejected.
- Fix: no residual term; carry modes up to f_max = 0.3066 fs (N = 301 at 96 kHz), the cutoff at which the modal zero-order-hold step admittance alone equals 1/(2Z) (Si(N w1 dt) = pi/2). The earlier stage-1 block above was computed with the rejected setup; it is re-run below.

## Stage 1 — unit checks (auto)
- L = 0.7014 m, Z = 1.0592 kg/s, B = 4.186e-05, zeta1 = 2.879e-04; damping power law p = 0.658 (a = -8.528); S10 recomputed 0.00117891 vs results.json 0.00117891
- fs = 96 kHz, modes f_n <= 0.3066 fs = 29.4 kHz (no residual term): N = 300 (C0-C2), 189 (C3). Modal-sum bow admittance x 2Z (beta 0.02/0.1/0.2): C1 0.994/0.993/0.993; C3 modal 0.677/0.663/0.663 vs continuous stiff-string ZOH value 0.676/0.678/0.678; numerical infinite ZOH sum x 2Z (flexible) 1.0000/1.0000/1.0000
- free decay (C1): n=1: f 97.80/97.80 Hz, zeta 2.879e-04/2.879e-04; n=2: f 195.60/195.60 Hz, zeta 2.897e-04/2.897e-04; n=3: f 293.40/293.40 Hz, zeta 6.672e-04/6.672e-04; n=5: f 489.00/489.00 Hz, zeta 4.984e-04/4.984e-04; n=10: f 978.00/978.00 Hz, zeta 8.402e-04/8.402e-04; n=17: f 1662.59/1662.60 Hz, zeta 3.496e-03/3.493e-03; n=30: f 2934.00/2934.00 Hz, zeta 1.857e-03/1.857e-03; n=60: f 5867.99/5868.00 Hz, zeta 2.931e-03/2.930e-03
- free decay (C3): n=1: f 97.80/97.80, zeta 2.879e-04/2.879e-04; n=2: f 195.62/195.62, zeta 2.897e-04/2.897e-04; n=3: f 293.46/293.46, zeta 6.672e-04/6.672e-04; n=5: f 489.26/489.26, zeta 4.984e-04/4.984e-04; n=10: f 980.04/980.05, zeta 8.402e-04/8.402e-04; n=17: f 1672.61/1672.63, zeta 3.494e-03/3.493e-03; n=30: f 2988.76/2988.76, zeta 1.857e-03/1.857e-03; n=60: f 6294.65/6294.68, zeta 2.934e-03/2.930e-03
- sanity beta 0.1, v 0.1, F = 0.05..3.2 N: C0 labels [False, True, False, False, False, False, False], C2 labels [False, False, True, True, False, False, False]; wall for 8 parallel runs 0.337 s

## Stage 2 — C0 validation (auto)
- F_min (plan grid) per beta [0.02, 0.03, 0.05, 0.08, 0.12, 0.2]: [None, None, None, None, None, None] (censored at grid floor: [False, False, False, False, False, False]); extended grid: [None, None, None, None, None, None]
- Q_plan per beta: [None, None, None, None, None, None]
- beta = 0.1 direct: F_min None, F_S 0.00798, Q_sim(0.1) = None (interp 0.08/0.12: None)
- b_beta plan grid None, extended grid None, plan uncensored only None
- pass: Q within x2 False; b_beta (plan) False; b_beta (ext) False

