# Label-threshold sensitivity — result (run once; Claude, 2026-10-21 ≈12:29 KST)

## Gate
Base variant rebuilt from the feature files reproduces every stored per-session β exponent **exactly** (max |diff| 0.000;
13 window and 11 onset sessions — D_T2 had no stored window value and entered the means ungated). The sweep is the paper pipeline with one threshold changed.

## Pre-registered sweep (`label_sweep.py`, `results.json`)
Mean of per-session minimum-force β exponents (window axis, 14 sessions | onset axis, 11), paired shift from base:
- base −1.532 | −1.236.
- Periodicity threshold 0.7–0.9: shifts ≤ +0.011 | +0.021 (labels changed ≤ 1.5 %) → no effect.
- Flyback fraction 0.3 / 0.4 / 0.6 / 0.7: +0.169 / +0.066 / −0.095 / −0.171 | +0.112 / +0.049 / −0.044 / −0.058
  (labels changed 2.6–9 %). Stricter single-flyback rule → flatter in every session; looser → steeper on average (window
  axis; 3/14 sessions go flatter, per-session fly0.7 shifts −0.55 to +0.19; on the onset axis the fly0.7 shift CI
  [−0.14, +0.03] includes 0).
- Fixed amplitude floor 1× / 2×: +1.396 / +2.237 | +1.168 / +1.867 (labels changed 12–27 %).
- **S = 2.41 (window), 1.93 (onset) → pre-declared reading "label-dependent (S ≥ 0.3)"**, entirely because of the fixed
  amplitude floor. The fixed-floor variants were included in the plan without noting that the classifier source
  (commit 8e3f6c0, 2026-09-17) already identifies a fixed floor as β-biased; their exclusion below is post hoc.

## Post hoc (`posthoc_amplitude.py`, `posthoc_amplitude.json`; decided after the sweep)
- Why the floor dominates: it removes 0–6 % of base-Helmholtz windows in the smallest-β tercile but 42–58 % in the
  largest-β tercile in every session (Helmholtz amplitude scales with v/β), and preferentially at low force (correlation of removal with log F over all
  Helmholtz windows of a session −0.34 to −0.61, inflated by the β–force confound; −0.12 to −0.52 after removing
  (speed, β)-column means, by the two verifiers). The classifier's own source comment warns that a fixed floor "would erase weak but
  regular oscillation at large beta and low speed". The floor variant therefore changes the β axis itself.
- Relative amplitude criterion (a loose analogue of the simulation's amplitude check, which compares the slip-phase
  velocity with 0.5 v/β and also checks the slip fraction; the measured criterion is normalised within each column and
  cannot detect a column-wide amplitude deficit): amplitude ≥ r × the median of base-Helmholtz windows in the same column: r 0.5 → +0.014 | +0.007; r 0.7 → +0.164 | +0.102.
- Among the variants that do not remove windows by β (periodicity, flyback, relative amplitude): range of means
  0.34 (window, −1.70 to −1.36) and 0.17 (onset, −1.30 to −1.12) — "label-dependent" on the window axis and "moderate" on
  the onset axis by the pre-declared bands. No such variant brings the mean near −2 (steepest −1.70).

## Reading in words
The measured law is sensitive to how strictly "one flyback per period" is applied (±0.17 around base in the mean
exponent, a span of 0.34 on the window axis and 0.17 on the onset axis — comparable to or larger than the
force-reference-time effect, 0.21 [0.11, 0.32]; by the pre-declared bands "label-dependent" (window) and "moderate"
(onset); per-session shifts up to −0.55), insensitive to the periodicity threshold, and very
sensitive to a fixed amplitude floor, which removes Helmholtz windows by β and is excluded by the classifier's design.
Under every β-neutral label the measured minimum force falls more gently than Schelleng's β^−2. Threshold changes inside the
measured classifier moved the exponent much less than changing the label definition moved the simulated one (−2.7 vs
−4.9 in v8 with smooth friction; −2.7 to −6.5 across friction laws); amplitude checks scaled to the column median (or,
in an independent check, to v/β) moved the measured mean by ≤ +0.16. No slip-fraction or stationarity label was tested
on the measured data, so the comparison holds for thresholds only.

## Independent verification (workflow wf_7ab9b5e3-38d, 2 agents; `verify/`)
- **Recomputation (own code, label_sweep.py not imported): confirmed.** A1, C1, A_T2, B_T2r1 × base / fly0.3 / fly0.7 /
  amp1x / rel0.7 reproduce results.json to the printed digit (A1 within 4e-8); the flyback rule equals the classifier
  with 0 mismatches over 725k windows at f = 0.5 (other f are a post-processing proxy, not a re-extraction); the
  amp1x β-confound reproduces and works by turning 17–19 of 40 large-β columns into right-censored columns with no
  Helmholtz windows left; onset mapping aligned exactly. Corrected: the "within a column" correlation (now fixed above).
- **Claims audit: weakened (wording).** All numbers match. "Robust" misused the pre-registered vocabulary (fixed above;
  the commit message of cdd11fe says "robust to beta-neutral thresholds" and overstates — it cannot be edited without a
  force-push). Extensions by the verifier: flyback fraction 0.8 / 0.9 gives −1.30 / −0.59 (rules that admit
  near-equal double flybacks — not genuine Helmholtz labels, but they show the ±0.17 depends on the chosen range); an
  amplitude floor scaled to v/β (closer to the simulation) shifts the mean by +0.000 (r 0.5) and +0.13 (r 0.7).
- **Caveat added by the recomputation:** at large β, 19–29 of 40 columns in C1, A_T2 and B_T2r1 are left-censored at
  the lowest force tested (fly0.7: 25–31), so the large-β end of each fit — and much of the flyback sensitivity — rests on
  censored intervals; this limits how strongly the data alone support "gentler than β^−2" (see also the earlier
  censoring variants, −0.9 to −1.6, none near −2).
