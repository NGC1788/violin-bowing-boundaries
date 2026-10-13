# Endpoint reconstruction: a useful baseline with a calibration blind spot

Date: 2026-10-13. This is a synthetic audit of the corrected frozen simulator,
not a new method claim and not a redefinition of the empirical labels. The
production simulator and existing data splits were not changed.

The promising finding is conditional: with the exact reflection filter and
geometry, endpoint forces recover a precisely known smoothed contact velocity.
The dangerous finding is equally clear: agreeing bridge and nut reconstructions
can both be wrong about sticking when they share an impedance calibration error.
Dual-end agreement is a diagnostic, not an accuracy certificate.

## Prior work and scope

Woodhouse, Schumacher and Garoff (2000), *Reconstruction of bowing point friction
force in a bowed string*, JASA 108, 357–368, already reconstruct both contact
force and velocity. The [primary article abstract](https://pubmed.ncbi.nlm.nih.gov/10923898/)
describes two approximate reconstruction methods and simulated and observed
examples; DOI [10.1121/1.429529](https://doi.org/10.1121/1.429529).
The wave identity below therefore must not be sold as a novel inverse method.

Lampis et al. (2025), *Friction force reconstruction in bowed cello strings*,
DOI [10.1121/10.0037894](https://doi.org/10.1121/10.0037894), is a JASA
157(4 Supplement) A203 meeting abstract. The [funded project's own record](https://www.fwf.ac.at/en/research-radar/10.55776/P34852)
also identifies the invited presentation. It establishes directly relevant
ongoing work, but an abstract is insufficient to audit every algorithmic detail.
No full-text implementation comparison is claimed here.

## Lossless identity and sign convention

Let a_b(t) be the outgoing velocity wave from the bow towards the bridge,
Z the transverse characteristic impedance, and tau_b the one-way travel time.
At an ideal rigid endpoint, the returning velocity wave has opposite sign:

    v_c(t) = a_b(t) - a_b(t - 2 tau_b)
    F_b(t) = 2 Z a_b(t - tau_b)

Consequently,

    v_c(t) = [F_b(t + tau_b) - F_b(t - tau_b)] / (2 Z).

The same identity holds at the nut if its force polarity is defined in the same
wave-coordinate convention. Actual transducer signs and gains require checking.
This is offline reconstruction: the first term needs a future endpoint sample.
It assumes a uniform, non-dispersive string, a rigid reflection and calibrated
force, geometry and impedance. It is not an identity for arbitrary microphone
audio or an unknown violin body transfer function.

An integer-delay random-wave test gives maximum absolute error below 1e-14.

## Exact identity for the frozen lossy digital model

The simulator places the entire attenuation/rounding filter on the return path.
At either endpoint i, let O_i be its one-way *observation* linear delay, and let
R_i = H_i D_i be the complete return operator. Its coefficients are known:

    H_i(z) = g_i (1 - p_i) / (1 - p_i z^-1)
    d_i = 2 h_i - p_i / (1 - p_i)
    h_b = beta fs / (2 f0), h_n = (1 - beta) fs / (2 f0).

D_i is the linear fractional delay with delay d_i. In this model,

    F_i / (2 Z) = O_i a_i
    v_c = (I - R_i) a_i.

All operators are linear time-invariant and commute away from record edges,
giving the **stable forward-filter** identity

    (I - R_i) F_i / (2 Z) = O_i v_c.

No inverse reflection filter, small spectral denominator or unstable
deconvolution is required. This statement follows the existing simulator's
lumped-loss observation convention; it is not a proof that the same convention
matches a real termination force sensor. A real lossy termination's force
reaction depends on its actual reflection coefficient and measurement location.

For alignment, the prototype applies the adjoint one-way delay O_i^T, which is
an offline linear fractional advance. This produces

    vhat_i = K_i v_c,   K_i = O_i^T O_i.

If h_i has fractional part delta_i, then K_i is exactly the symmetric three-tap
filter

    K_i v[n] = a_i v[n-1] + (1-2 a_i) v[n] + a_i v[n+1],
    a_i = delta_i (1-delta_i).

Thus the pointwise discrepancy near a slip transition is identifiable numerical
smoothing, even when the internal filter parameters are perfect. The estimator
does not recover an arbitrary instantaneous velocity exactly.

The crossfiltered redundant residual

    r = K_n vhat_b - K_b vhat_n

is exactly zero in the clean known model. Plain vhat_b - vhat_n can be nonzero
solely because the two fractional observation delays smooth differently.

## Fixed experiment and results

The protocol JSON is saved before simulation. There are 96 cases at each of
100 and 200 kHz, 192 simulations in total: f0 in {98.08, 195.9977, 440,
659.2551} Hz, beta in {.03, .05, .1, .2}, bow speed in {.05, .1, .2} m/s,
and beta F/(Z v) in {.45, 5.6}. Q=200, loss corner=6000 Hz and Z=1 are explicit
generic assumptions, not measured violin parameters. Duration is .6 s, speed
ramps for .1 s, and analysis uses .2–.58 s to exclude future-data edges.

A private prototype copy exposes bridge force, nut force, true contact velocity
and the solver's exact stick indicator at every time sample. Bridge outputs
match the frozen production solver **exactly after float32 output conversion**;
the contact indicators also pass a separate parity test. Full records remain
in memory for each batch and are not saved. Only metrics are retained.

Across all cases, reconstructed velocity differs from the analytical smoothed
truth by at most 1.38e-10 times bow speed. Crossfiltered residuals are at most
8.53e-11 times speed. These tiny numbers verify the implementation of the model
identity, not the accuracy of its physical assumptions.

The following are medians over 96 cases at each rate. The near-stick decision
uses |v_bow-vhat| <= .01 v_bow. It is compared to the exact internal Boolean
stick state, so the error includes both smoothing and the deliberate nonzero
tolerance; this is not a experimentally validated stick detector.

| Estimator | Velocity RMSE / speed, 100 kHz | 200 kHz | Indicator error, 100 kHz | 200 kHz |
|---|---:|---:|---:|---:|
| Known reflection, bridge | .05237 | .02937 | 1.013% | .473% |
| Known reflection, nut | .06416 | .03473 | 1.064% | .503% |
| Known reflection, average | .06443 | .04367 | 1.039% | .522% |
| Naive lossless identity, bridge | .23732 | .26954 | 69.099% | 68.520% |
| Naive lossless identity, nut | .25480 | .30158 | 67.776% | 69.245% |
| Known filter, shared Z 10% too high | .20146 | .19502 | 79.316% | 79.401% |

The comparison isolates the importance of the reflection/phase model in this
simulator. The lossless identity is a deliberately mismatched baseline, not the
best implementation of the published reconstruction literature. The median of
per-case averages need not lie between the medians of each single-end method.
No benefit from simple two-end averaging is established.

Sharp transition errors can be much larger than RMSE: at 200 kHz the median
maximum pointwise error for the bridge estimator is .3773 times bow speed.
Reporting only overall correlation or RMSE would hide this problem.

## What can and cannot be bounded

For bounded endpoint force noise |e_i[n]| <= epsilon_i, passive g_i <= 1 and
the exact parameters, positivity of the one-pole kernel and linear-delay
weights gives

    ||delta vhat_i||_infinity <= (1 + g_i) epsilon_i / (2 Z).

The audit injects a fixed bounded oscillatory noise sequence with
epsilon_i=.002 Z v_bow, without tuning its amplitude to outcomes. The analytic
bound is below .002 times bow speed; observed maxima are .0017032 at 100 kHz
and .0014809 at 200 kHz. All cases satisfy the bound.

Smoothing has a separate exact local bound:

    |K_i v[n] - v[n]| <= a_i (|v[n-1]-v[n]| + |v[n+1]-v[n]|).

This is verified against latent simulator increments. It is an **oracle bound**:
those increments are not observed in the experimental dataset. A justified
physical increment/curvature bound or a carefully stated bandlimit would be
needed before using it as an empirical uncertainty guarantee. White Gaussian
noise also has no finite deterministic amplitude bound without a probability
level and a finite time horizon. The bounded-noise calculation cannot simply
be transferred to it.

If a valid total velocity error bound e_i is available, each endpoint defines a
slip interval [v_bow-vhat_i-e_i, v_bow-vhat_i+e_i]. Their intersection can support
selective *near-zero-slip* decisions: accept only if the entire intersection
is within a declared tolerance, reject only if it excludes that region, and
otherwise abstain. Empty intersections detect disagreement. None of this
establishes exact mathematical sticking at nonzero error tolerance.

Two counterexamples prevent a blanket guarantee:

1. **Shared impedance error.** If both reconstructions use 1.1 Z, both inferred
   velocities divide by 1.1. Their crossfiltered residual remains zero, yet a
   true sticking plateau v_c=v_bow becomes an apparent slip .0909 v_bow. In
   the experiment the median indicator error exceeds 79%, while the dual-end
   residual stays below 8e-11 times speed. Absolute force calibration, bow speed
   calibration and common time/model errors require independent constraints.
2. **Common observation nullspace.** A half-sample linear observation delay
   annihilates a Nyquist alternating sequence. Both one-way delays can have
   fractional part .5, e.g. 17.5 and 82.5 samples, so adding a second endpoint
   does not always recover that hidden component. Four unit tests explicitly
   include this case. Bandlimited reconstruction may be possible, but must state
   its bandwidth and conditioning instead of claiming unrestricted recovery.

## Decision for the research

Endpoint reconstruction is worth retaining as an established mechanical
observation baseline. It offers substantially more interpretable information
than one-flyback waveform labels **under validated transfer parameters**.
This audit does not yet provide those parameters or new empirical labels.

A defensible new direction would have to address uncertainty that survives
redundancy: jointly constrained calibration, interval/identified-set contact
states, explicit abstention, and falsification on deliberately misspecified
models and independent endpoint measurements. A shared-impedance error must be
one of its negative controls. Simply averaging endpoints, adding a residual
threshold, or presenting the wave identity would not be a sufficient advance.

Before any empirical use, inspect force-channel polarity, gain, timing, sensor
frequency response, tension/density-derived Z uncertainty, reflection geometry,
and dispersion. These are dataset metadata and system-identification questions;
they do not require buying sensors or recording the user's violins. If the
necessary information is missing, report partial identification instead of
promoting model-dependent contact states to ground truth.

## Reproduce

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2 .venv-mac/bin/python research/2026-10-13/endpoint_observer_audit.py
.venv-mac/bin/python -m pytest tests/test_endpoint_observer_audit.py -q
```

Observed runtime: 4.09 s including compilation/cache activity. Four tests pass.
The small public aggregate is `endpoint_observer_summary.json`; detailed cases
and the saved protocol are under ignored `reports/physics_audit/2026-10-13/`.
