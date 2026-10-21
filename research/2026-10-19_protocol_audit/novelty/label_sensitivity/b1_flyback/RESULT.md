# B1 corner-width labels — result (run once; Claude, 2026-10-21 ≈13:52 KST)

`stream_b1.py` (974 s, 6000 strokes streamed from the 14.9 GB archive, nothing extracted to disk, archive untouched) →
`b1_flyback.csv`; `analyze_b1.py` → `results_b1.json`, `fig_b1_flyback.png`. Plan: `PLAN.md` (ae9a370).

## Gate
Recomputed periodicity, flybacks_per_period and second_flyback_ratio equal the stored B1 features for **6000 / 6000**
strokes (0 stream errors).

## Corner width (flyback duration at the bridge), base-Helmholtz strokes at t = 0.2 s (n 2535)
- Median 0.78 ms (IQR 0.54–1.06); ratio to the bow–bridge round trip βT0: median 0.99 (IQR 0.73–1.26).
- log d_fb = 0.84 + 0.48 log β − 0.04 log v − 0.12 log F (R² 0.63): the corner width grows only as β^0.48, so it is
  about 2βT0 at β 0.02 and about βT0 or less at β 0.2. (Proxy caveat: the tallest flyback leg can also reflect
  Schelleng-ripple structure, whose period is βT0.)

## Variants (B1 minimum-force β exponent; base −1.663 window / −1.323 onset)
| variant | window β (shift, 95 % paired bootstrap) | onset β (shift) | base-H removed by β tercile | flag |
|---|---|---|---|---|
| abs1: d_fb ≤ βT0 | −2.527 (−0.864 [−1.150, −0.626]) | −2.217 (−0.894 [−1.139, −0.689]) | 97 / 62 / 19 % | **β-confounded** |
| abs2: d_fb ≤ 2βT0 | −1.719 (−0.056 [−0.091, −0.018]) | −1.368 (−0.046 [−0.075, −0.015]) | 8 / 1 / 0 % | — |
| abs4: d_fb ≤ 4βT0 | −1.663 (0.000) | −1.323 (0.000) | 0 / 0 / 0 % | — |
| rel1.5: ≤ 1.5 × column median | −1.444 (+0.219 [+0.104, +0.355]) | −1.211 (+0.112 [+0.047, +0.188]) | 3 / 5 / 13 % | — |
| rel2: ≤ 2 × column median | −1.615 (+0.048 [+0.000, +0.146]) | −1.287 (+0.035 [+0.002, +0.100]) | 0 / 0 / 2 % | — |

Readings: unflagged variants shift the exponent by −0.06 to +0.22 (window; "small" to "moderate") and −0.05 to +0.11
(onset); none moves it to Schelleng (steepest −1.72 / −1.37). The only variant that does (abs1, to −2.5) is
β-confounded: because the corner width scales as β^0.48 rather than β, a threshold proportional to βT0 removes almost
all Helmholtz strokes at small β (97 %) and few at large β (19 %), which steepens the fitted law.

## Link to the simulations (interpretation, post hoc)
The simulations' amplitude/slip label (v3–v8) also used thresholds scaled with β (slip fraction in [0.5β, 2β], slip
speed ≥ 0.5 v/β) and produced the steepest exponents (−4.1 to −6.5). In the measured data a threshold of the same kind
(corner width ≤ βT0) steepens the B1 exponent from −1.66 to −2.53 by the same route — removing rounded-corner Helmholtz
motion preferentially at small β. A β-scaled acceptance threshold applied to a quantity that does not scale with β
manufactures a steeper β law; this is a plausible common cause of the steep simulated exponents and a caution for any
Helmholtz criterion tied to β. One session only.
