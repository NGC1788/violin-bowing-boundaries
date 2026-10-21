# Simulation v7 — finite bow width (pre-registered)

Claude, 2026-10-21 (before any v7 code; the commit time is the record). Post hoc w.r.t. the measured boundaries.

## Question
In v4 (Coulomb) and v6 (smooth friction) the measured frequency-dependent damping made the maintenance minimum fall far
too steeply with β (β^−4.1, β^−6.5; no maintenance at β ≤ 0.03–0.05), opposite to the measured β^−1.2 to β^−1.5.
All of that used a point bow. A real hair ribbon is ~10 mm wide, comparable to the bow–bridge distance at small β
(14 mm at β 0.02). Does a finite-width bow change the β dependence?

## Model (fixed)
Digital waveguide chain: bridge — segment — M contact junctions spread evenly over width W centred at βL — segment —
nut. Integer one-way delays (D = 4000 samples per period, one sample ≈ 0.35 mm of string). Each junction carries normal
force F/M and solves its own friction each step (neighbours are ≥ 1 sample apart, so the solves decouple). Junction
rule as in v4–v6: outgoing waves = incoming + Y f in both directions, Y = 1/(2Z). Bridge reflection: −(loss filter)
with the v4/v6 minimum-phase 1024-tap filter (smooth measured damping) or the frequency-independent gain, its phase
delay at f0 removed from the bridge-side segment (requires the bridge-side edge ≥ τ/2 samples from the bridge, so β
starts at 0.03). Nut reflection −1. Smooth friction (v6) primary; Coulomb (v4) secondary. Ideal Helmholtz start for the
point-bow pattern at the bow centre (maintenance test). Label (v3 amplitude-checked rule) from the centre junction's
relative velocity. Force grid, K = 2, window 0.3–0.5 s as v6. β 0.03/0.05/0.08/0.12/0.2 (+0.1).

Configurations:
- **P0 (code check)**: M = 1 (point bow), frequency-independent loss, smooth friction → must equal v6 H0m within one
  grid step at every β, else nothing is interpreted.
- **P1**: M = 1, measured damping → must equal v6 H1m within one grid step at every β where v6 has a value.
- **W0**: W = 10 mm, M = 8, frequency-independent loss.
- **W1 (reading)**: W = 10 mm, M = 8, measured damping.
- W1-5mm (sensitivity): W = 5 mm, M = 5. W1-coul (secondary): W1 with Coulomb friction.
- W1-ramp (secondary): W1 started from rest with the 0.104 m/s² ramp, window 0.2–0.4 s after the plateau.

## Readings (fixed)
- Code gate: P0 and P1 as above.
- b_β(W1) over the five β (needs F_min at ≥ 3 β, else "undefined"): ≥ −2.0 → "bow width restores a measured-like
  flattening"; −4.0 < b_β < −2.0 → partial; ≤ −4.0 → "bow width does not". Compared with point-bow −6.5 (v6).
- Width alone: b_β(W0) against −2.05 (v6 H0m).
- Level at β 0.1 reported but not read (it depends on the friction law, v6).
- W1-ramp: establishment if F_min exists at ≥ 3 β.
