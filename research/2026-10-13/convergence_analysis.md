# Rate and duration factorial audit — 2026-10-13

The corrected passive simulator still does not justify a pointwise continuum-regime
claim. Finer rates improve the median contact-state and spectrum differences, while a
substantial minority of shape labels remains unstable. Doubling the settling duration
does not remove that instability.

Protocol: reuse the **prespecified** 96 violin-frequency cases (force ranks 6 and 16 of
20, each of 4 pitches × 4 beta values × 3 speeds). Keep all physical/friction parameters
fixed. Cross simulation rates 50, 100, 200, 400, 800 kHz with durations 0.6 and 1.2 s.
Observe the final 0.4 s at the same 50 kHz output rate; no noise is added. Six E5,
beta 0.03 cases are inadmissible at 50 kHz and explicitly rejected in each duration.
There are 948 valid stroke simulations. This is a deterministic stress grid, not an
independent sample of real violins or a statistical estimate of their performance.

| Rate pair, kHz | Labels changed, 0.6 s | Labels changed, 1.2 s | Median absolute stick-fraction change, 0.6 s / 1.2 s |
|---|---:|---:|---:|
| 50 → 100 | 21/90 | 22/90 | 0.002088 / 0.001888 |
| 100 → 200 | 19/96 | 20/96 | 0.000900 / 0.000750 |
| 200 → 400 | 13/96 | 18/96 | 0.000547 / 0.000469 |
| 400 → 800 | 15/96 | 17/96 | 0.000345 / 0.000248 |

The table above counts **all regime-label changes**; it is not a count of errors in the
binary Helmholtz boundary. The correct endpoint-specific diagnostics are:

| Rate pair, kHz | Binary H changes, 0.6 s / 1.2 s | H-region IoU, 0.6 s / 1.2 s |
|---|---:|---:|
| 50 → 100 | 13/90 / 12/90 | 0.594 / 0.684 |
| 100 → 200 | 9/96 / 10/96 | 0.690 / 0.730 |
| 200 → 400 | 7/96 / 3/96 | 0.767 / 0.917 |
| 400 → 800 | 8/96 / 7/96 | 0.733 / 0.806 |

These IoUs compare numerical resolutions, not predictions against experimental truth.
The result is still nonmonotonic, even after restricting attention to the intended binary
endpoint. All per-case memberships and intersections/unions are saved.

An additional **post-hoc diagnostic** excludes a provisional H label if its detected pitch
exceeds 1.25 times the reference. It does not modify historical labels or claim a validated
new observer. In the main 960-case synthetic sweep, 11 of 306 provisional H cases violate
this upper-pitch gate, reaching about three times the fundamental. At 400→800 kHz, the
gate changes binary disagreement from 8→8 at 0.6 s and 7→6 at 1.2 s; gated IoUs are
0.714 and 0.812, respectively. Rejecting obvious superharmonic candidates therefore does
not by itself fix resolution dependence or the sine/triangle negative-control failure.

To distinguish a label-threshold artifact from a change in the simulated observable,
the audit also compares phase-invariant power spectra in **fixed nominal-frequency
bands**. The signal is mean-subtracted, Hann-windowed and normalized by total power;
the bins are not aligned to the estimated pitch. Total variation is one half the L1
distance between the normalized band-power vectors.

At 400 → 800 kHz:

- Median spectral TV is 0.02670 at 0.6 s and 0.00938 at 1.2 s.
- 26/96 and 24/96 cases still have spectral TV above 0.1, respectively.
- Only 1/96 and 2/96 cases change their internal stick fraction by more than 0.02.
- Three changed-label cases in each duration have spectral TV below 0.02 **and** stick
  fraction change below 0.002. This subset is consistent with a fragile observation
  decision despite relatively stable coarse dynamics.

Changing duration from 0.6 to 1.2 s at the same 800 kHz rate changes 25/96 labels;
43/96 spectra have TV above 0.1 and 12/96 stick fractions change by more than 0.02.
Therefore this is not resolved by assuming that the original 0.2 s settling period was
the sole problem. A duration of 1.2 s is not asserted to be asymptotic steady state.

These results support three separate statements, with different limits:

1. The local contact balance and passivity corrections are necessary implementation
   checks, and they now pass. They do not prove accurate global nonlinear trajectories.
2. The median contact invariant becomes more stable over the tested rate sequence.
   Such an aggregate cannot distinguish all multi-slip or chaotic attractors.
3. The shape-label boundary is not numerically validated by two agreeing sample rates.
   Spectral changes in some cases show that label thresholds alone cannot explain all
   instability; threshold changes in other cases show that dynamics alone cannot explain
   every label flip. This finite experiment proves neither mathematical nonconvergence
   nor a particular continuous-time attractor.

Next useful methodological test: retain a **rate/duration/observation sensitivity set**
and evaluate the inner/outer boundary width and coverage, including an abstention cost.
Call it sensitivity- or assumption-conditioned until an independently justified numerical
error bound exists. Test independently fixed solver families (for example an energy-based
finite-difference or modal scheme) before attributing disagreement to physical mechanisms.
Do not erase difficult points, recalibrate each rate on test observations, or increase the
rate until a favorable label appears.

Reproduce after the main synthetic benchmark:

```bash
.venv-mac/bin/python research/2026-10-13/convergence_factorial.py
```

Aggregate: `research/2026-10-13/convergence_summary.json`.
Full rows, per-case comparisons and figure: `reports/physics_audit/2026-10-13/`.
The protocol and continuous metrics are implemented in `convergence_factorial.py`.
