# String damping → minimum bow force (zero-parameter Schelleng test)

Claude, plan written 2026-10-20 23:08 KST, **before** the damping values were digitized and before any
damping–boundary association was computed. The boundary data themselves have been seen many times, so every
result here is **post hoc** with respect to the boundaries. The plan fixes the damping summary, the constants and
the reading bands in advance so that no variant can be picked after looking.

## Why
- Referee point: the physical property that sets the *minimum* force in Schelleng's theory is the loss
  (bridge resistance R), and our string-property comparisons used Z, EI, T, d but never damping.
- The IWK rig is a monochord with rigid terminations (Lampis et al. 2024, Acta Acustica 8, 44, §2). There is no
  instrument body, so R is set by the string's own losses. Those losses were measured on the same strings by
  plucking: modal damping factors ζ_n (Lampis et al. 2025, POMA 58, 035013, Fig. 1, modes 1–19).
- The friction-coefficient jump on the same rig was also measured: μs − μd1 ≈ 0.6 at low bow force rising to
  ≈ 1.0 at high force (Lampis et al. 2024, §4.1, Fig. 11b).
- So every term of Schelleng's minimum force can be filled from independent measurements on the same rig. No
  parameter is fitted to the boundary data. POMA only reports that damping correlations with playability were
  "not consistent across bow speeds"; no quantitative prediction has been made.

## Model (fixed now)
Schelleng's maintenance minimum (standard form):

    F_S = Z² v / (2 R β² (μs − μd)) = Z · (Z/R) · v / (2 β² Δμ)

Equivalent rigid-end loss from pluck damping. With a resistive end the energy reflection per period is
1 − 4Z/R; a mode with damping ζ_n loses 4π n ζ_n of its energy per period. Helmholtz motion has mode energies
∝ 1/n² (shape independent of β), so the energy-weighted loss per period is 4π Σ ζ_n/n / (π²/6). Setting this
equal to 4Z/R gives

    Z/R_eff = (6/π) · S_N,   S_N = Σ_{n=1..N} ζ_n / n

(check: ζ_n = ζ_1/n, a frequency-independent R, returns Z/R = π ζ_1 exactly.)

Fixed choices:
- **Primary**: N = 10, constant 1/2 (Schelleng), Δμ = 0.6 (low-force end of Lampis 2024, because the lower
  boundary sits at low force), Z from POMA Table 1 for each type.
- Prediction band (reported, not used for the reading): N ∈ {5, 15}; constant 2/3 (my own energy-balance
  derivation with a linear stick-phase friction ramp gives 2/3 instead of 1/2); Δμ ∈ [0.6, 1.0].
- Missing modes n ≤ N: linear interpolation of log ζ against n between the nearest digitized modes; never
  extrapolated (if the last digitized mode < N, the sum stops there and this is reported).
- ζ_n: centre of the "×" marker, digitized from the raster Fig. 1 of POMA (method below).

## Digitization (fixed now)
Fig. 1 of the POMA PDF is a single raster image (2367 × 879 px). Per panel: y calibration from the dotted
horizontal grid lines, which sit at the labelled ticks 0.0004 … 0.018 and are equally spaced in log ζ (the
labels are rounded values of a log-spaced sequence; the end labels 0.0004 and 0.018 are used to fix the scale).
x calibration from the vertical grid lines at mode 5, 10, 15, 20. Marker centre = centroid of the "×" arm pixels
of the panel colour around each integer mode, after removing error-bar cap rows. Validation: (i) overlay plot
rendered and checked by eye; (ii) the text statements of POMA §2 (material differences ≈ 30 % vs tension ≈ 20 %;
modes 4 and 9 vary most, 50–60 %; mode 1 varies more between tensions, 27 % vs 17 %) recomputed from the
digitized values and reported, whatever they show.

## Boundary side (fixed now; same extraction as novelty/lower_law, K = 2, t = 0.2 s)
- Primary axis: onset force (12 sessions with ramp metadata; no D T2). Secondary: window force (all sessions).
- Session level L_s = interval-censored offset of log F_min at fixed common exponents (onset axis: the
  12-session random-effects means b_v = 0.957, b_β = −1.230; window axis: its own random-effects means), read at
  v = 0.1 m/s, β = 0.1. Type level = mean of its sessions.

## Tests and reading bands (fixed now)
**T1 absolute.** Q = F_meas / F_S at v = 0.1 m/s, β = 0.1 for each type (primary constants), plus Q at the two
ends of the measured β range (Q depends on β because measured b_β ≠ −2).
- Q < 0.5: measured minimum is below the maintenance limit → contradiction with the model.
- 0.5 ≤ Q ≤ 2: pluck-loss Schelleng limit matches the measured establishment boundary.
- 2 < Q ≤ 8: measured boundary exceeds the maintenance limit by an establishment-sized factor
  (Galluzzo 2003 reported establishment needing several times the maintenance force).
- Q > 8: the pluck-loss Schelleng limit fails by about an order of magnitude or more.
I will also say whether the prediction band crosses a band boundary.

**T2 across string types.** OLS of type level L on P = log(Z · S_10) (7 types on the onset axis, 8 on the window
axis). Prediction: slope +1.
- "supported" if the 95 % t-interval of the slope excludes 0 and includes 1;
- "contradicted" if it excludes 1;
- "not decided" otherwise.
Also reported: SD of L across types vs SD of (L − P) (does the physics offset shrink the type scatter?) and
Spearman ρ(L, P).

**T3 robustness (report all).** T2 slope with N = 5 and N = 15, with S replaced by ζ_1 alone, and on the
window axis.

No result is called confirmatory. Everything goes into the paper/report as a post hoc, zero-fitted-parameter
comparison.

## Addendum 1 — digitization calibration (2026-10-20 23:20 KST, before any ζ value was extracted)
Inspection of grid-line pixel rows showed two facts that contradict the calibration written above:
1. The horizontal grid lines are **not** equally spaced in log ζ. Their spacings (A T1 panel: 50, 49.5, 50.5, 52,
   44, 55 px) are proportional to the log-differences of the labels themselves (0.018 → … → 0.0004), at
   ≈ 182 px per decade. So the grid lines sit at the labelled values, not at a rounded log-spaced sequence.
2. The panels do **not** share one y scale. The D T2 panel shows six grid lines with spacings 56, 55, 56.5, 58,
   49.5 px (≈ 203 px per decade); only the first column carries tick labels.
Corrected calibration (replaces the y-calibration sentence above): for each panel separately, detect its own
grid-line rows, match them to a contiguous run of the label sequence (0.018, 0.0095, 0.0051, 0.0027, 0.0014,
0.0008, 0.0004) by the alignment with the smallest residual of a straight-line fit of row against log10(value),
and use that fit. The fit residual is reported as the calibration error. Everything else in the plan is unchanged.

## RESULTS (2026-10-20 ~23:55 KST; run once with the fixed constants; all post hoc w.r.t. the boundary data)

Digitization: all 8 panels calibrated with grid-line fit residuals 0.09–0.24 px; every visible marker found (overlay checked by eye; the overlay contains the publisher's figure and is kept local only). POMA text checks, recomputed from the digitized values over all shared modes: material differences 44 % vs tension 34 % (text ~30 % vs ~20 %, same order); mode 1 tension 24 % vs material 22 % (text 27 % vs 17 %, same order, smaller gap); largest material variation at mode 4 (64 %; text: modes 4 and 9, 50–60 %).

| type | ζ₁ (10⁻⁴) | S₁₀ (10⁻⁴) | Z/R_eff | F_S at v = 0.1 m/s, β = 0.1 (N) | onset F_meas (N) | Q onset | window F_meas (N) | Q window |
|---|---|---|---|---|---|---|---|---|
| A T1 | 2.88 | 11.8 | 0.0023 | 0.020 [0.009–0.041] | 0.401 | 20.2 [9.7–46.9] | 0.273 | 13.7 [6.6–32.0] |
| A T2 | 3.92 | 16.0 | 0.0031 | 0.030 [0.014–0.055] | 0.346 | 11.4 [6.3–25.1] | 0.172 | 5.7 [3.1–12.5] |
| B T1 | 3.10 | 14.8 | 0.0028 | 0.026 [0.010–0.054] | 0.371 | 14.3 [6.9–37.6] | 0.215 | 8.3 [4.0–21.8] |
| B T2 | 5.13 | 23.1 | 0.0044 | 0.044 [0.017–0.087] | 0.498 | 11.4 [5.7–28.7] | 0.290 | 6.7 [3.3–16.7] |
| C T1 | 3.36 | 13.5 | 0.0026 | 0.023 [0.010–0.039] | 0.314 | 13.9 [8.1–31.1] | 0.171 | 7.6 [4.4–16.9] |
| C T2 | 3.74 | 20.1 | 0.0038 | 0.036 [0.016–0.058] | 0.309 | 8.7 [5.3–19.1] | 0.163 | 4.6 [2.8–10.1] |
| D T1 | 4.57 | 16.9 | 0.0032 | 0.028 [0.012–0.050] | 0.389 | 13.9 [7.9–33.3] | 0.253 | 9.0 [5.1–21.6] |
| D T2 | 4.79 | 16.1 | 0.0031 | 0.031 [0.014–0.053] | – | – | 0.284 | 9.2 [5.4–20.8] |

**T1 (absolute).** Onset axis: Q = 8.7–20.2 for the 7 types → pre-declared reading **'fails by about an order of magnitude' (Q > 8) for all 7**; the prediction band (N, constant, Δμ) reaches down to Q = 5.3–9.7, i.e. into the 'establishment-sized excess' band for 5 of 7. Window axis: Q = 4.6–13.7 (4 of 8 types > 8). Because the measured β exponent is −1.23 (onset) / −1.54 (window) instead of −2, Q grows with β: onset Q ≈ 2.5–5.8 at β = 0.02 and 15–34 at β = 0.2.

**T2 (across types).** Onset axis (7 types): slope of type level on log(Z·S₁₀) = +0.21 [-0.45, +0.87] → pre-declared verdict **'contradicted' (CI excludes 1)**; subtracting the physics offset raised the type SD from 0.164 to 0.261; Spearman +0.00. Window axis (8 types): +0.08 [-0.90, +1.06] → 'not decided'.

**T3 (robustness, all reported).** Onset slopes: N = 5 +0.11 [-0.58, +0.80]; N = 15 +0.37 [-0.20, +0.94]; ζ₁ only +0.34 [-0.36, +1.03]. Window: N = 5 -0.04, N = 15 +0.27, ζ₁ +0.35 (all CIs about ±1). No variant gives a positive association that excludes 0.

### Post hoc additions (not in the plan; labelled as such)
- **Direct, model-light check** (`posthoc_direct.py`): in every (speed, β) column with an observed non-Helmholtz stroke, that stroke's force exceeded the type's F_S(v, β): onset axis 1216 columns, ratio median 8.0, >4× in 91 %, >8× in 50 %; window axis 1471 columns, median 5.6, >4× in 71 %. So Helmholtz motion failed at forces several times the pluck-loss maintenance minimum; this does not depend on the censored-fit model. Left-censored columns (no non-H stroke seen; 0–51 of 120 per session) carry no information on this and are excluded.
- **Regime check (corrected 2026-10-21 00:15).** The analysis window starts 0.2 s after the plateau onset. A first version of this bullet used the acceleration of the first ramp_meta row only (0.026 m/s²) and concluded that Guettler's attack conditions cannot matter; that was wrong. Per-stroke acceleration spans 0.03–3.4 m/s² (C1: 10th/50th/90th percentiles 0.026/0.088/3.37 at 0.05 m/s; 0.104/0.104/3.37 at 0.1 m/s; 0.42/0.42/3.37 at 0.2 m/s — the attack schedule found in the protocol audit). Guettler-type limits scale as a·T₀·Z/[β(1−β)] (bracketed friction terms O(1)), i.e. ≈ 0.001–0.4 N at β = 0.1 over this range, so for the fastest attacks establishment conditions are of the same order as the measured boundary.
- **Attack-split direct check** (`posthoc_accel.py`; stroke level, onset force, 11 sessions; acceleration classes fixed before computing rates). Share of strokes not Helmholtz at t = 0.2 s:

  | F_onset / F_S | slow attack < 0.15 m/s² | medium 0.15–1 | fast ≥ 1 |
  |---|---|---|---|
  | 4–8 | 57 % of 5,469 | 88 % of 3,537 | 99 % of 270 |
  | 8–16 | 21 % of 4,660 | 57 % of 4,569 | 68 % of 517 |
  | 16–64 | 10 % of 3,475 | 32 % of 7,776 | 15 % of 5,136 |

  Within 0.1 m/s alone: 4–8×: slow 71 %, medium 78 %, fast 100 %; 8–16×: 26 %, 65 %, 98 %. Faster attacks fail more (an establishment contribution exists), but even slow, quasi-static attacks are mostly not Helmholtz at 4–8 × F_S. So the order-of-magnitude excess is not only an attack effect. Caveats: acceleration is confounded with speed (no slow attacks at 0.2 m/s) and with recording order; the 16–64× band can include strokes above the upper boundary at large β, so only the two lower bands are read.
- **Same convention as Schoonderwaldt et al. 2008** (Z/R = π ζ₁, i.e. R = 2Z f₁ τ₁ from the fundamental only): Q at β = 0.1 becomes A T1 50, A T2 28, B T1 42, B T2 31, C T1 34, C T2 28, D T1 31 (onset) and A T1 34, A T2 14, B T1 24, B T2 18, C T1 18, C T2 15, D T1 20, D T2 19 (window). They reported a factor 6–11 on a violin D/E string (monochord and violin, slope ≈ −2). Our ratio is of the same order at small β and larger at large β.
- **Within-type vs between-type spread** of session levels (descriptive only; the formal variance-decomposition claim was refuted earlier, `refute_session_variance/`): onset 0.163 (df 5) vs 0.164; window 0.263 (df 6) vs 0.245. Same-type sessions differ as much as types do, so an across-type damping effect of the predicted size (type SD of log(Z·S₁₀) ≈ 0.2) could not have been resolved; T2 has little power.

## Addendum 2 — prior art found after the result (2026-10-21 00:00 KST)
The 'Why' section above says no quantitative prediction had been made. That is true for this collection (POMA reports only that damping correlations were inconsistent), but **not in general**: Schoonderwaldt, Guettler & Askenfelt (Acta Acust. united Acust. 94, 604–622, 2008, §5, Tables VI–VII; full text read at knutsacoustics.com/files/schoonderwaldt-et-al.-empirical-schelleng.pdf) computed Schelleng's c_lower from pluck-decay R on a violin and a monochord and found the measured lower limit 'almost one order of magnitude larger (a factor 6-11)'. They also found, within one string with added damping, c_lower ∝ R^−2.3. So T1 is a **replication** on a different instrument (cello G strings), rig, lab and damping estimate (multi-mode, digitized), not a new finding. What is new here: (i) the replication across 8 string types and 12–15 sessions with a column-level direct check; (ii) the excess grows with β because the cello exponent is −1.2 to −1.5 rather than −2; (iii) across string types the pluck damping does not order the minimum force (slope +0.2 vs +1 predicted, and vs +2.3 implied by their within-string damping law), with the caveat that within-type session spread equals between-type spread.
- (iv) the excess persists for slow, quasi-static attacks (57 % non-H at 4–8 × F_S), so on this rig it is not only an establishment effect; Schoonderwaldt et al. varied force gradually during sustained strokes, so both rigs point to a maintenance-side discrepancy.
