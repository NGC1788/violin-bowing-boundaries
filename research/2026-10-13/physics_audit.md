# Physics and observation audit — 2026-10-13

The historical calibration is not a valid passive-physics baseline. Two separate defects
must be distinguished: the implemented reflection recurrence differed from its design
equation, and some requested decay/corner combinations require an active reflection
even under the corrected recurrence. Both were reproduced mathematically and checked in
code before the new synthetic experiments. Old results remain historical artifacts.

## Governing model and the actual defect

At the bow, free relative velocity is \(W_0=v_b-v_h\), string admittance is
\(Y=1/(2Z)\), and sliding satisfies

\[
w+Y F_N\left(\mu_d+(\mu_s-\mu_d)\frac{v_0}{v_0+|w|}\right)=W_0.
\]

The code selects a stick/slip branch with history and a closed-form quadratic root.
For the assumed friction curve, local mechanical dissipation is \(F_{friction}w\ge0\).
This contact property does not imply passivity of the complete string: the reflection
filters must also be passive.

The filter design calculates

\[
p=e^{-2\pi f_c/f_s},\quad
L(e^{i\omega})=\frac{1-p}{1-p e^{-i\omega}},\quad
g=\frac{e^{-\pi/(2Q_1)}}{|L(e^{i\omega_0})|}.
\]

It therefore intends \(H(z)=g(1-p)/(1-pz^{-1})\). However, the old recurrence was
\(y_n=g[(1-p)x_n+p y_{n-1}]\), giving
\(H_{old}(z)=g(1-p)/(1-gp z^{-1})\). Its pole, DC gain, attenuation and group
delay all differ. The corrected recurrence is
\(y_n=g(1-p)x_n+p y_{n-1}\).

Numerical reproduction from `physics_check.py` (effective Q here is a response-based
diagnostic before interpolation, not a fitted full-system pole):

| Parameters | Rate | Old reflection magnitude at f0 | Old response-equivalent Q |
|---|---:|---:|---:|
| Default Q=2000, corner=6000 Hz | 50 kHz | 0.99863151 | 1147.05 |
| Default Q=2000, corner=6000 Hz | 100 kHz | 0.99779205 | 710.64 |
| Historical best Q=3726.62, corner=1896.41 Hz | 50 kHz | 1.00295335 | −532.66 |
| Historical best Q=3726.62, corner=1896.41 Hz | 100 kHz | 1.00682167 | −231.05 |

A magnitude greater than one means this linear termination injects energy at that
frequency. It does not assert that every nonlinear bowed trajectory diverges, but it
invalidates its interpretation as a lossy passive termination. Optimization could have
used this numerical energy source; its degree of contribution to every historical IoU
has not yet been measured by a matched rerun.

## Passivity is a separate constraint

For this first-order design, maximum response is at DC and equals g. Passive reflection
requires \(g\le1\), equivalent to
\(Q_1\le\pi/[-2\log|L(e^{i\omega_0})|]\). A low corner and arbitrarily high Q cannot
be specified independently. The old best also violates this condition after fixing the
recurrence. Clipping g would silently change the meaning of Q, so the code rejects the
pair with a clear error. `reflection_is_passive` enables search-time filtering.

Linear fractional delay introduces additional passive, beta- and rate-dependent loss:
\[
D_\delta(e^{i\omega})=e^{-i\omega\lfloor\delta\rfloor}
[(1-\alpha)+\alpha e^{-i\omega}],\quad\alpha=\delta-\lfloor\delta\rfloor.
\]
The full round-trip magnitude is \(|H^2D_bD_n|\). The new
`linear_decay_diagnostics` exposes its nominal-frequency attenuation and effective Q.
At default parameters, beta 0.1, 50 kHz, this Q is approximately 1985.10. An independent
four-second free-decay experiment, fitting the fundamental amplitude after contact is
removed, recovers it within 1%; the regression test rejects the old recurrence.

The buffer now accommodates both delay lines, and both must exceed two samples after
reflection group-delay compensation. Previously only the bridge line was checked and
the buffer length came only from the nut line. Inputs and friction domains are checked
before state allocation.

## The observation operator is not yet identifiable

The current classifier returns `helmholtz` for all three controlled signals:

| Signal, 98.08 Hz | Periodicity | Counted flybacks | Return duration |
|---|---:|---:|---:|
| Sine (negative control) | 0.97449 | 1 | 5.085 ms |
| Symmetric triangle (negative control) | 0.97452 | 1 | 5.084 ms |
| Sawtooth (positive shape control) | 0.97223 | 1 | 0.120 ms |

One monotonic return per averaged cycle is not specific to a Helmholtz corner. These
tests falsify the sufficiency of the current rule; they do not prove that every measured
label is wrong. A sharper-return/asymmetry feature may reject these controls, but its
threshold must be validated against bandwidth/noise changes and actual waveforms.
Cycle folding can suppress intermittent or alternating slip structure. The flyback law
is useful independent physical support, not a per-trial reference label.

The compiled simulator additionally reports latent stick fraction, slip indicator and
raw slip-onset count. These are internal states, not experimental labels. Small contact
transitions can produce many onsets even when the bridge waveform has one large return;
equating all zero-crossing events with perceptual multiple slipping is also unjustified.

The historical fixed 40–160 Hz pitch search cannot be reused for violin pitches. The
new synthetic benchmark explicitly scales the search to 0.4–3.5 times each reference
frequency while preserving the old label logic for this audit.

## Executed Mac experiments

`scripts/bowed_string_cpu.py` is a numba implementation of the corrected reference with
no fastmath, at most two worker threads and a 1 GiB estimated-array limit. It shares
filter/geometry/input checks with the reference. The waveform parity test uses per-stroke
parameters, positive and negative bow motion, no contact, fractional delays and output
averaging. Thread counts give identical waveforms. Eighteen targeted physics/CPU tests
passed during this audit.

`violin_synthetic_benchmark.py` ran 960 main cases at 100 kHz: four standard violin
open-string pitches, four beta values, three speeds and twenty dimensionless force
values \(\lambda=\beta F_N/(Zv_b)\) from 0.1 to 12. Equal-tempered frequencies derive
from A4=440 Hz; the [Yamaha instrument guide](https://www.yamaha.com/en/musical_instrument_guide/violin/mechanism/mechanism002.html)
confirms the G–D–A–E string assignment. Z=1 kg/s, Q=200, corner=6000 Hz and length
0.328 m are **declared generic-string assumptions**, not measurements of violin strings.

Each stroke lasts 0.6 s with a 0.1 s velocity ramp and a final 0.4 s observation window.
A prespecified 96-case subset (force ranks 6 and 16 at every pitch/beta/speed) was rerun
at 50 and 200 kHz; six E5, beta 0.03 cases cannot meet the 50 kHz delay condition and
were explicitly rejected. All outputs use a 50 kHz observation rate.

| Check | Result |
|---|---:|
| Simulated strokes, including rate repeats | 1146 |
| Label disagreements, 50 vs 100 kHz | 21 / 90 valid pairs |
| Label disagreements, 100 vs 200 kHz | 19 / 96 valid pairs |
| Agreement across all three rates | 65 / 90 |
| Maximum local contact load-line residual | 3.55e−15 m/s |
| Minimum friction dissipation | 0 W |

Main-grid shape labels include 306 `helmholtz` cases; these are **provisional classifier
outputs**, not established Helmholtz solutions. Correct passivity is necessary but does
not establish pointwise regime convergence. The smaller cello-frequency audit likewise
changes 8/36 and 6/36 labels over successive rate doublings. These finite collections
do not estimate a universal convergence rate or bound the continuum-model error.

Reproduce:

```bash
.venv-mac/bin/python -m pytest tests/test_bowed_string.py tests/test_bowed_string_cpu.py -q
.venv-mac/bin/python research/2026-10-13/physics_check.py
.venv-mac/bin/python research/2026-10-13/violin_synthetic_benchmark.py
```

Detailed cases and figures are in `reports/physics_audit/2026-10-13/`; the small aggregate
is `research/2026-10-13/violin_synthetic_summary.json`. Timing is a local execution
measurement and not a reproducible performance guarantee.

## Claims to retire and next methodological step

- “The upper boundary fails because of the point bow” is a hypothesis, not an identified
  cause. Correct numerical energy balance and rate sensitivity must be assessed before
  attributing residuals to bow width, torsion or thermal memory. Matched ablations with
  calibration budget held fixed are needed to distinguish those explanations.
- A failed modal implementation is not a proof that modal stiff-string models cannot
  converge. The [2026 thermal elasto-plastic paper](https://doi.org/10.1051/aacus/2026042)
  explicitly uses a modal formulation; its primary-source indexed sections 2.3 and 2.6
  also discuss existence and uniqueness. Full publisher-page retrieval was blocked in
  this audit, so no additional unverified implementation details are asserted here.
- Extending a prior simulator with finite bow width or thermal friction alone cannot be
  presented as a new mechanism: such model classes already exist in the above paper
  and the [2019 stiff-string elasto-plastic work](https://www.dafx.de/paper-archive/2019/DAFx2019_paper_18.pdf).
- Search optima near a parameter-box edge are a reason to investigate sensitivity, not
  proof that the true optimum is outside the box. The present example shows an even
  stronger alternative: optimization can favor numerically nonpassive combinations.

The defensible new research target is to separate three uncertainty sources in a
boundary prediction: physical parameter uncertainty, finite-rate numerical uncertainty,
and observation-label uncertainty. An inner region can require agreement across a
prespecified admissible model/observation set; an outer region can include any plausible
member's prediction. Evaluate coverage **and** set width/abstention cost. Any guarantee
is conditional on that declared set unless an independent error bound is proved.

Before claiming superiority: compare passive physics, data-only classifiers, discrepancy
correction, and the proposed set-valued method with equal information/budgets; freeze an
unseen experimental archive; use stress tests and independent label evidence. Synthetic
violin-frequency checks support numerical transport experiments, not validation on real
violins. An accurate, candid negative result is preferable to a high IoU obtained by
the same flawed classifier and an energy-injecting physical model.
