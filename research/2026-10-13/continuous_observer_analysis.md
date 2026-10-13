# Independent analysis of the continuous observation-rule oracle

Reviewed 2026-10-13: `scripts/continuous_observer.py`, its nine existing tests,
`continuous_observer_audit.py`, and `continuous_operator_prior_audit.md`.
The oracle code was not changed in this review. The timing audit was upgraded to
seven repetitions with separately reported preparation costs; its summary was
regenerated without fitting or changing any predictor.

**Verdict:** the objective and finite-state reduction are sound under the
assumptions below, and the implementation handles the important strictness and
tie cases. This is a useful exact-state sensitivity evaluator assembled from
established fractional-programming and computational-geometry techniques.
Current real-data differences from the 15-rule grid are small and do not change
any within-source model ranking. Neither a new general optimization method nor a
substantive empirical improvement in model selection has been established.

## 1. Exact objective and assumptions

For fixed data rows i=1,...,N, let r_i be the flyback ratio, q_i the periodicity,
e_i a fixed eligibility mask, p_i a fixed prediction in [0,1], and w_i >= 0 fixed
weights with positive total. A **single common** parameter pair (f,c) lies in the
closed declared rectangle [f_min,f_max] × [c_min,c_max]. Its label vector is

    y_i(f,c) = e_i 1[r_i < f] 1[q_i >= c].

Ineligible rows remain zero for every operator. The current rectangle is
[0.3,0.7] × [0.75,0.85]. All other observation decisions, feature extraction,
frequency eligibility and missing-feature handling are fixed. These ranges
describe a sensitivity model; they are not a validated probability distribution
or a demonstrated coverage region for the correct mechanical classifier.

Define

    a = sum_i w_i p_i,
    A(y) = sum_i w_i p_i y_i,
    B(y) = a + sum_i w_i (1-p_i)y_i,
    J(y) = A(y)/B(y).

The oracle minimizes or maximizes J over admissible y. With binary p this is
weighted ordinary Jaccard/IoU. With soft p this is a deterministic soft-Jaccard
ratio. It is not, in general, the expectation of IoU obtained by independently
sampling Bernoulli predictions with these probabilities.

If a>0, every denominator B(y)>=a is strictly positive. If a=0, all positive
weight predictions are zero: nonempty positive-weight targets score zero, while
the empty union scores one. The endpoint-based special case in the implementation
is correct because the smallest admissible target is at (f_min,c_max) and the
largest at (f_max,c_min). Zero-weight rows do not change that conclusion.

No independent or identically distributed sampling assumption is required for
these deterministic finite-dataset extrema. Such assumptions would be needed
for a separate statistical generalization or confidence claim.

## 2. Why the continuous rectangle has a finite exact representation

The label vector changes only when f passes an observed r_i or c passes an
observed q_i. Endpoints must be retained. Equal feature values enter as a group;
otherwise intermediate label vectors would describe impossible global operators.

For the strict r_i<f predicate, the state immediately above each eligible
internal r_i is needed. `nextafter(r_i,+inf)` supplies a binary64 representative,
including when r_i equals the lower endpoint. A point at r_i=f_max can never
enter. For q_i>=c, the observed q_i itself represents the inclusive transition.
The implementation starts at c_max, includes every row with q_i>=c_max, then
sweeps complete equal-q groups down to c_min.

With K fraction states and Q periodicity states there are at most KQ candidate
pairs, where K,Q=O(N). Different pairs can induce the same label vector. The
procedure is exact for the finite represented feature values and declared
predicate; it is not a numerical-convergence result about the underlying string
trajectory or the feature extractor.

The tests include independent midpoint/breakpoint enumeration, degenerate
rectangles, equal features, strict/inclusive endpoints, zero predictions and
weights, invalid values, and fresh-process Numba cache reuse. Those checks are
appropriate evidence for implementation correctness, not a formal verification
of all floating-point executions.

## 3. Fractional reduction, sweep and proof of the residual bound

For a candidate ratio lambda,

    A(y) - lambda B(y)
      = sum_i w_i [p_i-lambda(1-p_i)] y_i - lambda a.

The last term is independent of the operator. Minimization therefore reduces
to a signed weighted quadrant problem on (r_i,q_i); maximization follows by
negating the signed weights. The sweep maintains every fraction-state score,
using a suffix range-add/min segment tree as each periodicity group enters.
Negative weights are essential and are supported.

For the minimum, let J*=min_y J(y) and
F(lambda)=min_y[A(y)-lambda B(y)]. Then F(lambda)=0 if and only if lambda=J*.
Starting at lambda=1 bounds all feasible J from above. If y_lambda is a global
minimizer and F(lambda)<0, updating lambda to J(y_lambda) decreases lambda but
keeps it at least J*. For the maximum the same argument uses max, starts at zero,
and increases the ratio toward its optimum. A finite feasible state family and
strict improvement before termination give finite termination in exact
arithmetic; no universal small iteration count is implied.

For the minimum along these updates, write epsilon=|F(lambda)|. At an optimal
label y*,

    F(lambda) <= B(y*)[J*-lambda] <= -a(lambda-J*).

Consequently lambda-J* <= epsilon/a. The returned updated ratio lies between
J* and lambda, so its gap is also at most epsilon/a. The maximum case is
symmetric. This justifies the code's `abs(residual)/base` bound and stopping
threshold `tolerance*base`, **provided the signed weighted subproblem is solved
globally and arithmetic is exact**.

The implementation recomputes A, B and the residual on the selected label vector;
this is a good consistency check. That recomputation alone cannot certify that
the segment tree selected a global subproblem optimum. The bound is therefore
correctly described as an exact-arithmetic bound, conditional on the oracle
subproblem, rather than a rigorously enclosed floating-point error bound. No
interval-arithmetic enclosure of tree summation error is implemented. Normalizing
weights by their maximum and then their sum avoids overflow, but does not turn
binary64 arithmetic into exact real arithmetic.

Preparation costs O(N log N), each fractional subproblem O(N log N), and storage
O(N); total solve cost is O(k N log N) for k fractional iterations. The dense
breakpoint histogram/cumulative-sum comparator costs O(N+KQ) once its geometry is
prepared and requires O(KQ) storage. Do not drop k from the oracle complexity or
claim measured peak memory from these asymptotic bounds. Observed iterations on
the 56 real prediction arrays were 2–3 for minima and 2–4 for maxima.

## 4. Global coupling, rectangular relaxation, and pooled versus macro IoU

The global operator cannot flip ambiguous labels independently. Let Y_grid be
the 15 sampled label maps, Y_cont the full continuous-threshold family, and
Y_box every binary vector between the pointwise endpoint envelopes L and U.
Then

    Y_grid subset Y_cont subset Y_box.

It follows directly that

    min_box J <= min_cont J <= min_grid J
       <= max_grid J <= max_cont J <= max_box J.

This relation is a set-containment result, not a statistical coverage statement.
The existing exact box formulas are for hard binary predictions. The real audit
uses hard predictions and verifies the full nesting chain. The oracle supports
soft predictions separately, but hard-box formulas should not silently be
reused as soft-box extrema.

Pooling all rows permits **one shared** (f,c) across every test panel. It optimizes
sum_g A_g / sum_g B_g. It does not optimize the macro mean
(1/G)sum_g A_g/B_g, a sum of ratios with operator-dependent denominators.
Giving each panel its own optimum is another problem again:

    mean_g min_theta J_g(theta) <= min_theta mean_g J_g(theta).

There is no general equality between these quantities or the pooled extremum.
The current audit correctly evaluates pooled IoU. Its pooled winner may differ
from the model with the best macro score in another report without contradiction.

Model-wise worst operators can differ. Comparing two models' individually worst
scores is a minimax ranking, not proof one model dominates the other under every
common operator. Uniform pairwise dominance would require analyzing
J_model1(theta)-J_model2(theta) under the same theta, which this single-ratio
oracle does not solve directly.

## 5. Nearest primary prior art

- [Dinkelbach (1967), On Nonlinear Fractional Programming](https://doi.org/10.1287/mnsc.13.7.492):
  the ratio-to-parametric-objective transformation and iterative ratio update are
  established fractional programming. The publisher abstract was checked.
- [Narasimhan, Kar and Jain (ICML 2015), Optimizing Non-decomposable Performance Measures](https://proceedings.mlr.press/v37/narasimhana15.pdf):
  explicitly treats Jaccard as a pseudo-linear metric and connects adaptive
  linearization to Dinkelbach–Jagannathan updates. Using a label operator as the
  feasible variable changes the application, not the foundational ratio method.
- [Pérez-Lantero (2010), Geometric Optimization for Classification Problems](https://personal.us.es/dbanez/papers/pablo-thesis.pdf),
  §2.6.2, printed p47 / PDF p57: the signed maximum-weight quadrant problem has a
  sweep with dynamic prefix optimization in O(N log N) time and O(N) space. The
  exact primary thesis was downloaded and this passage checked. The restricted
  threshold rectangle and sign-reversed minimum are specializations of known
  geometry, not a new generic quadrant algorithm.
- [Maia Polo et al. (NeurIPS 2024), Weak Supervision Performance Evaluation via Partial Identification](https://proceedings.neurips.cc/paper_files/paper/2024/file/f4c6bec746b0aeca8c2cd15096f1ad1f-Paper-Conference.pdf):
  bounds performance using distributions compatible with prescribed marginals
  and studies estimation/misspecification. Our deterministic shared-operator
  family imposes different restrictions and supplies no estimate of unknown
  physical label probabilities. Broad claims to invent evaluation without gold
  labels or label-uncertainty performance bounds are unsupported.

The existing prior-art note is appropriately cautious. A targeted search that
does not find this exact combination does not establish firstness. A defensible
description is “an efficient exact-state sensitivity audit for a globally shared
two-threshold observation rule, using established fractional programming and
weighted quadrant optimization.” Whether that application is a publishable
methodological contribution still depends on consequences beyond those shown.

## 6. What the current data actually show

The audit uses 56 already-frozen prediction arrays across four retrospective
source splits, with no fitting or model changes. The 15-rule minimum exceeds the
continuous-family minimum in 31/56 cases, but the largest difference is only
0.0003570247233 IoU and the mean is 0.00005983336825. These equal approximately
0.0357 and 0.0060 percentage points respectively.

Within each source, checking every pair of audited models found **zero strict
ranking reversals and zero coarse-grid ties broken**, using 1e-12 to recognize
ties. The pooled-IoU winner remains unchanged in all four sources. These are
descriptive results on reused development data, not independent replications.
The small constructed three-point test demonstrates that a coarse grid can
miss an interior minimum by 1/6; it must not be presented as the magnitude found
in the real bowed-string corpus.

The real candidate-pair counts are 292,436–319,300. No changed real-data model
selection or physical conclusion has yet been demonstrated by replacing the
15-rule grid with this continuous oracle. Its current contribution is completeness
of the sensitivity calculation and an implementation whose scaling is testable.

## 7. Repeated timing audit

`continuous_observer_audit.py` now uses seven warm repetitions on the same seeded
synthetic input at each N, alternates sweep/dense call order, and verifies both
extrema against the independent dense calculation every repetition. Common
threshold-geometry preparation is reported separately, as is dense-specific bin
preparation. Allocation of each solver's working arrays is inside solve timing.
Numba warmup is outside timing. Full repetitions, quartiles, versions and code
hashes are saved in `continuous_observer_summary.json`.

| N | Sweep median, ms | Dense median, ms | Solve-only median ratio | Including preparation ratio |
|---:|---:|---:|---:|---:|
| 250 | 0.124 | 0.939 | 7.57× | 5.64× |
| 500 | 0.178 | 3.801 | 21.34× | 15.05× |
| 1,000 | 0.612 | 15.923 | 26.02× | 19.94× |
| 2,000 | 0.977 | 65.815 | 67.35× | 47.08× |

These replace the earlier single-measurement 77× figure. They are local timings
on this Mac and this synthetic input distribution, not a hardware-independent
speedup, independent input replications, a measured memory comparison or a
cold-start deployment claim. The synthetic N=2,000 case has 4,008,004 candidate
pairs, substantially more than the actual audit arrays; its speed ratio should
not be transferred to the real-data audit. Small absolute times are especially
sensitive to timing noise and concurrent processes.

## 8. Remaining implementation/scope cautions

1. `from_features` excludes nonfinite flyback counts; historical `label_family`
   only checks count>=1 and would allow positive infinity. The released data
   parity test passes, so no present result changes. Equality for arbitrary
   malformed input is not established. Both paths would need an explicit common
   validity policy before claiming universal equivalence.
2. Membership of physical ground-truth labels in Y_cont is unverified. Frequency
   gate misspecification, hysteresis, sensor filtering, waveform feature error,
   acquisition bias and numerical artifacts can all lie outside this family.
3. No claim about more than two threshold features follows from the current
   segment-tree solver without another complexity analysis and implementation.
4. The oracle evaluates fixed predictions. A better optimizer for the adversary
   does not itself improve prediction accuracy or rescue the previously negative
   robust-training comparison.

The strongest present conclusion is limited but reproducible: a complete
shared-rule sensitivity calculation is feasible on the laptop; the existing
15-rule grid was already a close approximation for these frozen predictions.
