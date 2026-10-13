# Fractional-delay mechanism test — 2026-10-13

An established allpass interpolation baseline removes the beta-dependent damping of
linear fractional delay. It moderately reduces several regime-label discrepancies in
this fixed synthetic grid, but does not resolve the overall convergence question.

Only the two **feedback** fractional delays change. For delay D=M+δ, the optional
`thiran1` path reads the integer-delayed sample and applies

\[
A(z)=\frac{a+z^{-1}}{1+a z^{-1}},\qquad a=\frac{1-\delta}{1+\delta},
\]
with the recurrence y[n]=a x[n]+x[n−1]−a y[n−1]. Exact integer delay bypasses the filter,
avoiding a spurious pole-zero cancellation at z=−1. The magnitude is unity and DC group
delay is δ. The extra state is one previous input and output for each of two delay lines.
This is an established [Thiran allpass interpolator](https://www.dsprelated.com/freebooks/pasp/Thiran_Allpass_Interpolators.html),
not a new physical mechanism or a new interpolation invention. Allpass phase dispersion
remains. The reference/default linear path is preserved.

Independent verification uses the implemented filter's impulse response: FFT magnitude
is unity to 1e−12 and low-frequency delay agrees to 1e−6 samples for δ=0.2,0.5,0.8.
An integer-delay simulation is byte-identical to the linear path. A separate four-second
free-decay experiment fits the fundamental envelope after a brief bow excitation:

| beta | Linear fitted Q | Thiran1 fitted Q | Requested Q |
|---:|---:|---:|---:|
| 0.03 | 1954.65 | 1999.997 | 2000 |
| 0.10 | 1985.10 | 1999.997 | 2000 |
| 0.20 | 1988.55 | 1999.997 | 2000 |

This isolates the interpolation-loss mechanism in the linear free-vibration subsystem.
It does not prove nonlinear bowing-regime accuracy.

The matched factorial reuses the same 96 synthetic violin-frequency cases, physical
parameters, initial bow profile, 0.6/1.2 s durations and final 0.4 s windows. Rates are
100/200/400/800 kHz. Existing 768 linear outcomes are compared with 768 new allpass
outcomes; no model parameters were retuned.

| Duration | Rate pair, kHz | Linear labels changed /96 | Thiran1 labels changed /96 |
|---|---|---:|---:|
| 0.6 s | 100 → 200 | 19 | 17 |
| 0.6 s | 200 → 400 | 13 | 12 |
| 0.6 s | 400 → 800 | 15 | 12 |
| 1.2 s | 100 → 200 | 20 | 22 |
| 1.2 s | 200 → 400 | 18 | 15 |
| 1.2 s | 400 → 800 | 17 | 11 |

These are multiclass changes. For the actual **binary H/non-H endpoint**, the matched
counts and cross-resolution IoUs are:

| Duration | Rate pair, kHz | Linear H changes /96 (IoU) | Thiran1 H changes /96 (IoU) |
|---|---|---:|---:|
| 0.6 s | 100 → 200 | 9 (0.690) | 10 (0.697) |
| 0.6 s | 200 → 400 | 7 (0.767) | 6 (0.812) |
| 0.6 s | 400 → 800 | 8 (0.733) | 6 (0.800) |
| 1.2 s | 100 → 200 | 10 (0.730) | 13 (0.683) |
| 1.2 s | 200 → 400 | 3 (0.917) | 7 (0.825) |
| 1.2 s | 400 → 800 | 7 (0.806) | 5 (0.865) |

The binary result makes the mixed outcome especially clear: Thiran worsens the 1.2 s,
200→400 kHz pair. A post-hoc upper-pitch gate (detected f0≤1.25 reference f0) changes
the finest-pair binary counts from linear 8/7 to 8/6, and Thiran 6/5 to 6/4 for
0.6/1.2 s, respectively. Its finest-pair Thiran IoUs are 0.786/0.879. This gate is an
observer diagnostic, not a validated label correction or latent mechanical truth.

The outcome is mixed: the longer-duration first rate pair gets worse. At 1.2 s,
400→800 kHz, median stick-fraction change improves from 0.000248 to 0.000169; median
spectral TV changes only from 0.00938 to 0.00900. Eleven provisional labels still change.
No claim of universal improvement, superiority on measured data, or continuum certification
is supported by these numbers.

The **observation** path deliberately retains linear bridge interpolation and the same
box-averaged 50 kHz output. Box averaging is not a high-rejection anti-aliasing filter;
it may affect high-frequency observations. A subsequent isolated ablation should use
a documented low-pass/downsampling operator shared by every solver before assigning
remaining errors to physical mechanisms or the classifier. It must not alter feedback
and observation filtering simultaneously and attribute the entire gain to one of them.

Reproduce after `convergence_factorial.py`:

```bash
.venv-mac/bin/python research/2026-10-13/delay_method_factorial.py
```

The script also reproduces the free-decay table. Aggregate results are in
`research/2026-10-13/delay_method_summary.json`; full per-case comparisons are in
`reports/physics_audit/2026-10-13/delay_method_factorial.json`.
