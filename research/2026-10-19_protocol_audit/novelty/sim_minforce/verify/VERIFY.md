# VERIFY — adversarial check of the sim_minforce ablation (PLAN.md / RESULT.md)

Verifier: Claude (separate workflow agent), 2026-10-21 KST. Builder files were not edited; no git actions; no packages installed.
Own code (independent of the builder's): `vsim.py` (two solvers: a digital waveguide and a modal solver whose coefficients come from
scipy `expm`; own Coulomb solver; own PLAN classifier), `run_checks.py` → `checks.json`, `run_indep.py` / `run_final.py` /
`run_last.py` → `indep.json`, `dbg_wg.py` (lossless waveguide sanity check, printed only), and waveforms `wg_C0_*_b0.1.npz` and `mod_C1_b*.npz`.
The builder's `sim.py` / `diag_c0.py` were imported read-only for side-by-side comparisons only (numba cache redirected to `verify/nbcache`).

## Verdict: **C0_failed (status confirmed); the builder's diagnosis is refuted**

The headline result stands. Under the PLAN protocol (start from rest, ramp at 0.104 m/s², label 0.2–0.4 s after the plateau starts,
K = 2), C0 has no F_min at any β. An independent, band-unlimited waveguide reproduces this, with both loss placements, both friction
rules, and both D = 1000 and D = 2000. So nothing after C0 may be interpreted.

The two "uninterpreted observations" in RESULT.md are wrong and should be withdrawn:
1. "In this point-bow Coulomb model the stick-phase ripple from secondary waves launched by the friction jump is 3–30× larger than
   Schelleng's bridge-loss variation. So the premise that C0 reproduces F_S does not hold for this model class." **Refuted.** The same model class
   (point bow, Coulomb 1.2/0.6, ζ_n = ζ₁/n) solved without a band limit (sharp corner) and with the stick-if-possible rule
   *maintains* Helmholtz down to F_min/F_S = 2.07–2.51, with b_β = −1.95 (distributed loss = the PLAN's ζ₁/n). With all the loss at the bridge,
   Q = 1.24–3.0. The large ripple the builder measured comes from its numerics, not from the model class.
2. "Off-protocol C0 maintenance F_min at β 0.1 (0.10–0.34 N) is near the measured level and its exponent (−0.83 to −1.08) is flatter
   than −2." **Artifact.** The values are explained by (i) the Gibbs precursor of the 300-mode truncated corner, which scales ∝ 1/β, and
   (ii) a creeping-slip pathology of the hysteresis rule with Coulomb friction. The sharp-corner value of the same quantity is 0.0166 N at
   β 0.1, which is 16–24× **below** the measured 0.27 / 0.40 N.

## Check 1 — modal normalisation: **pass**
- Constants recomputed independently: Z = 1.059216 kg/s, L = 0.701362 m, mode mass μL/2 = 2.70761 g, ζ₁ = 2.8789e-4, damping
  power law p = 0.658. All match the builder to machine precision. f0 recovered from the eigenvalues of the transition matrices = 97.79999 Hz.
- Transition coefficients: my `expm` of the augmented [q, p, g] system vs the builder's closed-form a_ij, b_i agree to max rel. diff
  1.7e-15 (A) and 2.4e-12 (B), for C0 and C1 at β 0.05 / 0.1 / 0.12.
- Bow-point ZOH admittance Y·2Z (mine = builder's): C0 0.99820 / 0.99812 / 0.99811, C1 0.99307 / 0.99299 / 0.99298 (β 0.05 / 0.1 / 0.12).
  The cutoff trick Si(Nω₁Δt) = π/2 is analytically correct: the infinite flexible sum Σ sin²(nπβ) sin(nθ)/n = π/4 gives 1/(2Z).
- Free decay from the eigenvalues: C0 τ = 5.6526 s for modes 1, 10, 100 and 300, so ζ_n = ζ₁/n exactly (ζ_300 = 9.596e-7). C1 ζ from
  the eigenvalues 2.879e-4 / 8.402e-4 / 4.101e-3 / 8.451e-3 for n = 1 / 10 / 100 / 300, against spec 2.879e-4 / 8.402e-4 / 4.101e-3 / 8.451e-3.
- Waveguide self-check (`dbg_wg.py`). Lossless, started from my analytic Helmholtz travelling waves: exactly one slip per period for 40 periods,
  slip length exactly βD (100 / 50 samples), stick force exactly 0.6000 F, for F = 0.05–2 N at β 0.1 and 0.05.

## Check 2 — stick/slip solver: **concern** (signs correct; the friction *rule* is a hidden degree of freedom)
- Signs: f = −w/Y in stick, v_end = v_free + Y f, slip force −sgn(Δv) μ_d F, and slip speed solved from u + YFμ(u) = s·w. All correct.
  The stick test is |f| ≤ μ_s F. For Coulomb the slip root is unique (u = |w| − YFμ_d). The hyperbolic (C2) larger-root branch was not exercised.
- Energy (builder's solver, ideal start, C0 β 0.1). Power into the string is +0.89 / +1.36 / +4.03 mW at F = 0.335 / 0.61 / 2.03 N (positive).
  Friction heat is 0.0195 W against μ_d F v = 0.0201 W. The stick force peaks at 0.997 / 0.955 / 0.976 μ_s F, so the static margin is
  nearly used up every period.
- **Rule ambiguity.** For YFμ_d < |w| ≤ YFμ_s both stick and slip are solutions. The builder's primary rule (hysteresis: keep slipping
  while a slip root exists) has a pathology with *any* loss and Coulomb friction. After the corner passes, the returning velocity is
  v(1 − ε) < v, so a slip root u = vε > 0 still exists and the string never re-sticks; it creeps at μ_d F. In my waveguide (ideal start,
  distributed loss, β 0.1, F 0.2 N) the exact-stick samples per period go 500 → 100 → 0 in the first three periods, with creep at
  0.29–1.0 mm/s. By period 8 the string slips continuously. PLAN does not specify the rule, and every C0/C1 number below depends on it.

## Check 3 — Helmholtz classifier on raw waveforms: **pass for C0; concern for C1 under the hysteresis rule**
- C0, builder's `wave_C0_v0.1_b0.1.npz` (ramp protocol, β 0.1). F ≤ 0.055 N: continuous slip (slip fraction 1.0). F 0.075–0.136 N:
  slip fraction 0.90–0.99, 21–56 slip runs. F ≥ 0.18 N: 281–2096 slip runs in 0.2 s, median length 2–13 samples, longest 15–52 samples,
  against βT0 = 98 samples for Helmholtz. Every label is correctly "not Helmholtz". My PLAN classifier, also with a flicker-tolerant variant
  that ignores slips shorter than 5 samples, agrees: 0/32 Helmholtz.
- C1 (hysteresis), β 0.1. The "Helmholtz" labels at 0.61–1.50 N have one stick→slip per period (frac 0.95–1.0, period 0.95–1.002 T0) but
  **slip fraction 0.77 / 0.77 / 0.67 / 0.48** against β = 0.1. At β 0.12, F 0.45 N, the label is H with slip fraction 0.989. This satisfies the
  letter of the PLAN rule, but it is creeping slip, not Helmholtz motion. With the stick-if-possible rule the C1 H labels have slip fraction
  0.11–0.17, which is physical.

## Check 4 — independent re-implementation, v = 0.1, PLAN ramp protocol
F_S(C0) = 0.0319 / 0.00798 / 0.00554 N and F_S(C1) = 0.0795 / 0.0199 / 0.0138 N at β 0.05 / 0.1 / 0.12 (S10 = 0.00117891 recomputed).

| run | β 0.05 F_min (Q) | β 0.1 F_min (Q) | β 0.12 F_min (Q) |
|---|---|---|---|
| builder C0 (RESULT) | none | none | none |
| **my waveguide C0, distributed loss, hysteresis, 32 forces** | none | none | none |
| my waveguide C0, bridge loss, hysteresis | none (one isolated H at 5 N) | none | none |
| my waveguide C0, stick-if-possible (both losses; the label at 3.0–3.2 s is identical) | none | 0.825 N (Q 103, isolated band) | none (2 isolated H) |
| my waveguide C0, D = 2000 | none | — | none |
| my modal C0 (expm, 16 forces) | none | — | none |
| **my modal C1, hysteresis, 96 kHz N 300** | 2.03 N (25.5) | 0.611 N (30.7) | 0.825 N (59.8) |
| builder `sim.simulate` C1, same forces | 2.03 N (25.5) | 0.611 N (30.7) | 0.825 N (59.8); max\|Δv\| diff vs mine 4e-13 |
| my modal C1, hysteresis, 192 kHz N 601 | 1.11 N (14.0) | — | 0.825 N (59.8) |
| my modal C1, stick-if-possible, 96 kHz | 0.825 N (10.4) | 0.184 N (9.2) | 0.184 N (13.3) |
| my modal C1, stick-if-possible, 192 kHz N 601 | 0.611 N (7.7) | — | 0.184 N (13.3) |

Off-protocol (ideal Helmholtz start, constant 0.1 m/s, same label window): maintenance F_min (Q) for β 0.02, 0.03, 0.05, 0.08, 0.12, 0.2
and 0.1:
- waveguide C0, distributed loss, stick-if-possible: 0.452 (2.27), 0.184 (2.07), 0.0746 (2.34), 0.0303 (2.43), 0.0123 (2.22),
  0.0050 (2.51), 0.0166 (2.08). **b_β = −1.95.**
- waveguide C0, bridge loss, stick-if-possible: Q 1.24, 1.53, 1.28, 2.43, 3.00, (β 0.2 grid-bottom artefact), 1.54. b_β is −2.38 with
  β 0.2 and −1.52 without it.
- waveguide C0 with hysteresis: fails at almost every force (distributed loss); this is the creep pathology.
- builder's modal C0 (`diag_c0.json`): Q 3.8–28 (stick-if-possible) and 5.6–92 (hysteresis), b_β −1.08 / −0.83. That is 2–40× above the
  sharp-corner values.

Gibbs precursor in the builder's C0 basis (free motion of the ideal Helmholtz state, 2-sample guard at the corners): the stick-phase undershoot is
4.5–4.7 % of the corner jump v/β, i.e. 0.091 / 0.047 / 0.039 m/s at β 0.05 / 0.1 / 0.12. Holding it needs F ≥ 2Z·δv/(μs − μd) =
**0.32 / 0.17 / 0.14 N**, which matches the builder's modal maintenance F_min (0.18–0.45 / 0.10–0.34 / 0.10–0.14 N) and its ~1/β scaling.
Gibbs overshoot does not shrink with bandwidth. That explains the builder's non-convergent band scan (0.335 → 0.452 → 0.184 N at
96 / 192 / 384 kHz).

Mechanism of the ramp failure, seen in the sharp-corner waveguide (stick-if-possible, β 0.1): from rest the string locks into a stationary
**multi-corner state**. Slip length is exactly βD = 100 samples, but there are ~5 slips per period (mean period 0.20 T0) for F 0.005–0.075 N
and around 0.45 N. The labels are unchanged at 3.0–3.2 s after the plateau starts. With ζ_n = ζ₁/n there is no corner rounding to
pick the fundamental. The builder's diagnosis (a), "transients that never decay within the run", is close but incomplete: these
are persistent alternative attractors, not only slow transients.

## Check 5 — convergence and plan deviations: **concern**
- C0: the PLAN-protocol failure is robust to solver class (modal vs waveguide), band limit (96–384 kHz modal; waveguide D 1000 / 2000),
  friction rule, loss placement, classifier (mine vs the builder's; 1 % / 5-sample-flicker variants) and a 3 s later label window. The deviations
  (extended force grid, extra β 0.1, stopping at plateau + 0.4 s, classifier details) cannot change the C0 reading. My waveguide simulated
  the window with its own classifier and got the same result.
- C1 was not run by the builder (correct under the stop rule). My PLAN-style convergence check (β 0.05 and 0.12 instead of 0.03 and 0.12):
  hysteresis −45 % at β 0.05 and 0 % at β 0.12; stick-if-possible −26 % and 0 %. **Both fail the < 10 % criterion at β 0.05.** The C1 Helmholtz
  bands are narrow and patchy (for example `....HHH.H.......` at β 0.1).
- The builder's maintenance band scan (0.335 / 0.452 / 0.184 N) is non-convergence caused by Gibbs, as the builder suspected. It is not physics.

## Check 6 — does the stated reading follow? **Yes for the status; no for the observations**
- PLAN: "If C0 fails, nothing else is interpreted." C0 fails under the PLAN protocol, so there is no H1/H0/partial reading and no exponent
  or absolute comparison. The builder's "reading: None" follows.
- The appended observations do not follow (see the verdict above). The builder's own non-convergence (0.335 / 0.452 / 0.184 N) already showed
  that those numbers could not be quoted, even as hints.
- For the record only (not interpretable): if C0 were ignored, C1 would give Q_sim(0.1) = 30.7 (hysteresis) or 9.2 (stick-if-possible), both
  ≥ 5. That is "H1" by the letter, but it is rule-dependent by 3.3×, unconverged at β 0.05, and physically non-Helmholtz under the hysteresis
  rule. C1 F_min(0.1) = 0.18–0.61 N brackets the measured 0.27 / 0.40 N. Three β values are too few for b_β.

## Corrected reading (one paragraph)
C0 validation fails under the pre-registered protocol, so PLAN forbids any H1/H0 reading. This is confirmed with an independent,
band-unlimited waveguide: starting from rest, the nearly lossless ζ_n = ζ₁/n string with Coulomb friction locks into persistent
multi-corner motion (≈5 slips per period at β 0.1) and never reaches K = 2 Helmholtz runs at any force or β. The failure is a property of
the PLAN's establishment protocol for this idealised string, not proof that the model class misses Schelleng. Maintenance of an existing
Helmholtz motion in the same model, solved without a band limit and with the stick-if-possible Coulomb rule, does reproduce Schelleng's
form (F_min/F_S ≈ 2.1–2.5, b_β = −1.95, F_min(0.1) = 0.017 N). The builder's off-protocol figures (0.10–0.34 N, exponent −0.8 to −1.1, near the
measured level) are numerical artifacts of the truncated modal corner (Gibbs undershoot of 4.7 % of v/β) and of the hysteresis rule's
creeping slip, and must not be cited. Any rerun must (1) fix and pre-register the Coulomb stick/slip rule, (2) use a corner-exact solver
(waveguide) or a band-edge taper for C0, and (3) redefine the C0 validation as a maintenance test, or use a protocol that selects the fundamental
Helmholtz state. Until then the C1 numbers (Q 9–60 depending on rule and β, unconverged at β 0.05) carry no evidential weight.
