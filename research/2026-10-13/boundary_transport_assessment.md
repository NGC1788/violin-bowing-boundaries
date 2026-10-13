# Separate lower/upper boundary transport: completed negative diagnostic

## Question and prior

Would a known asymmetric physical prior improve transfer from one bow speed over uniformly translating a playable band with force proportional to speed?

[Schoonderwaldt, Guettler & Askenfelt (2008), An Empirical Investigation of Bow-Force Limits in the Schelleng Diagram](https://dael.euracoustics.org/landing_pages/aaua/64660.html), DOI 10.3813/AAA.918070, studies steel violin D/E strings with machine bowing. Its abstract reports an approximately speed-independent minimum force and agreement of the upper limit with a modified Schelleng equation. Therefore the proposed `(lower exponent, upper exponent)=(0,1)` is a literature-inspired baseline. It is neither a newly discovered law nor automatically valid for this project's different cello-string constructions and provisional labels.

## Frozen experiment

Implementation: `scripts/boundary_transport.py`; mathematical checks: `tests/test_boundary_transport.py`. Results: `reports/boundary_transport/2026-10-13/`.

Let `b` be standardized log beta and `q=log(F/F_ref)`, with both transformations fitted only on source predictors. Fit

`g(b,q) = a0 + a1*b + a2*b² + (d0+d1*b)*q - c*q²`, with `c >= 1e-6`.

Logistic loss plus L2 regularization is convex in its six coefficients, with a linear concavity constraint. Five contiguous source-beta folds select regularization from `.001, .01, .1, 1` using pooled source out-of-fold nominal IoU. The final model is fitted once per source A/B/C/D at nominal speed 0.1 m/s. Its two real roots, when present, are transported by

`r_lower(v)=r_lower(v_ref)+alpha_lower*log(v/v_ref)` and
`r_upper(v)=r_upper(v_ref)+alpha_upper*log(v/v_ref)`.

The four pairs `(1,1), (0,0), (0,1), (1,0)` were written to a manifest before dataset loading and target evaluation. No exponent pair was selected after seeing target performance. All four use exactly the same fitted source coefficients and selected regularization. Measured target velocities enter prediction. Source velocities are set to the nominal reference for fitting, so small source-speed deviations cannot determine an apparent transport law.

No-real-root cases retain a negative quadratic; transport that reverses root order cannot manufacture a positive interval. Roots outside source acquisition support are reported as extrapolations. Positive components and positive acquisition edges are recorded without filling holes or treating an edge as a measured crossing. This is still a single-band model: retaining a diagnostic does not remove that structural restriction.

This is explicitly **post-hoc development on previously exposed sample1 data**. It does not touch prospective Type A sample2. The label reference remains the benchmark's fixed A/0.1 observation rule, including for B/C/D source sweeps. Thus these sweeps are conditional on the same existing label convention, not wholly independent label calibration exercises.

## Results

Each number below is mean nominal IoU over the eleven non-source specimen/speed panels.

| Source | Uniform `(1,1)` | Static `(0,0)` | Asymmetric `(0,1)` | Reverse `(1,0)` |
|---|---:|---:|---:|---:|
| A / 0.1 | 0.7514 | 0.6046 | 0.6253 | 0.7228 |
| B / 0.1 | 0.7097 | 0.6065 | 0.6203 | 0.6934 |
| C / 0.1 | 0.7091 | 0.6066 | 0.6239 | 0.6866 |
| D / 0.1 | 0.7479 | 0.6051 | 0.6268 | 0.7203 |

The asymmetric hypothesis loses to uniform translation for every source. Its reverse also performs better for every source. The same ordering holds on the following threshold-family diagnostic:

| Source | Static mean panelwise minimum IoU | Uniform mean panelwise minimum IoU | Asymmetric mean panelwise minimum IoU |
|---|---:|---:|---:|
| A / 0.1 | 0.5496 | 0.6907 | 0.5632 |
| B / 0.1 | 0.5627 | 0.6578 | 0.5727 |
| C / 0.1 | 0.5598 | 0.6576 | 0.5718 |
| D / 0.1 | 0.5505 | 0.6881 | 0.5654 |

These are **means of each panel's minimum over 15 rules**. That allows different worst rules on different panels and is more conservative than using one rule for all evaluation panels. The latter is separately preserved under `global_same_rule` in the output JSON. Neither score is physical truth coverage.

The matched static-to-uniform contrast supports a practical benefit of the known `F/v` coordinate in this restricted classifier. It does not establish a new scaling law or the best possible classifier. Comparing against a raw three-dimensional model whose almost constant source-speed feature is standardized can additionally exaggerate an out-of-domain baseline failure; a matched force/beta model with explicit fixed speed transform is the cleaner contrast.

## Interpretation limits

The fitted upper root is outside the source acquisition range in 29/40 A, 32/40 B, 26/40 C, and 26/40 D beta columns. The source labels have multiple positive components in 36/40, 38/40, 37/40, and 30/40 columns respectively. These may reflect physical regimes, observer noise, or both; they are not evidence for a single physical band.

Consequently the experiment falsifies the proposed **predictive advantage of this asymmetric transfer baseline on the current benchmark**. It cannot falsify the 2008 physical observation: instruments, strings, force ranges, observer rules and boundary identifiability differ. Extrapolated upper roots particularly weaken physical interpretations of the exponent comparison. Do not use the negative result to claim that either true physical border universally scales with speed.

Six independent checks passed: analytic-loss gradient, source reconstruction, exact exponent shifts, uniform coordinate equivalence, no-band handling, preservation of censoring/components, and simple synthetic-band recovery (some checks share one test). No target-based threshold or regularization tuning was performed.
