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
