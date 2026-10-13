# Affine speed transport: a bounded candidate test

Date: 2026-10-13. Status: **post-hoc development, not confirmatory**. No A2
prospective data were accessed. The candidate was evaluated on every declared
source/speed pair; no pair was removed after seeing its outcome.

**Result:** the candidate helps on the published D-string speed-transfer task,
but is essentially tied with existing baselines on the primary cello comparison.
It does not establish a broadly better method or an original flow formulation.

## Method hypothesis

A single force rescaling F/v^alpha translates the whole playable region in log
force. It cannot change its width. Existing violin-string experiments report
different speed dependence of upper and lower force limits, so a deformation
of both center and width is a reasonable hypothesis to test. That physical
motivation is already in Schoonderwaldt et al. (2008), DOI
[10.3813/AAA.918070](https://doi.org/10.3813/AAA.918070); it is not a discovery here.

Set b=log(beta/.1), u=log(F/1 N), and t=log(v/.1 m/s). The proposed candidate
uses the autonomous affine flow

    du/dt = a0 + a1 b + k u.

Its reference-speed invariant is

    z = exp(-k t) u - (a0+a1 b) (1-exp(-k t))/k.

At k=0 the continuous extension is z=u-(a0+a1 b)t. The forward transformation
is u(t)=exp(k t)u(0)+(a0+a1 b)expm1(k t)/k, and compositions add their log-speed
increments exactly. These are standard linear differential-equation results;
neural ODEs, group actions and this affine flow are not claimed to be new.

A six-coefficient quadratic logit in normalized b,z is jointly optimized with
the three flow parameters. Its coefficient of z squared is constrained to be
at most -1e-6. Consequently, at each bridge position and speed, the predicted
positive set in log force is an interval or empty. The transformation preserves
this band topology. The log-force width scales by exp(k t); the center also
moves with a0+a1 b. The total model has nine fitted parameters.

These are predictive deformation parameters, not uniquely identified physical
mechanisms. At the reference speed t=0, the likelihood derivative with respect
to all three flow parameters is exactly zero; a dedicated test checks this.
At one other fixed speed, an affine coordinate change can be absorbed by the
base quadratic. Two speeds supply predictive deformation information, but do
not prove a unique mechanistic interpretation: flow parameters and the fitted
base band can remain correlated, acquisition may censor either root, and the
regularizer can strongly affect the chosen representation. No theorem of
physical parameter identification is asserted.

Normalization is computed once from the current training rows using the prior
coordinates (b,u-t), then held fixed during warp optimization. Each cross-
validation fold computes its own normalization. No target features or labels
enter normalization, fitting or selection.

Training minimizes mean nominal binary cross-entropy plus

    .5 lambda ||quadratic coefficients excluding intercept||^2
      + .5 lambda_w [(a0-1)^2 + a1^2 + k^2].

The declared grids are lambda and lambda_w in {.001,.01,.1}; bounds are
a0 in [-1,3], a1 and k in [-2,2]. The prior is ordinary force/speed scaling
(a0=1,a1=k=0). Three starts k=-.5,0,.5 are run for every fit. The successful
start with lowest **training objective** is selected, never by target or
validation performance. Bounded L-BFGS-B uses analytic derivatives, maxiter=500,
ftol=1e-11 and gtol=2e-7. Every optimizer start is logged.

## Evaluation protocol

For cello sample1, all four source specimens and all three two-speed pairs
(.05,.1), (.05,.2), (.1,.2) are used: twelve source tasks. Targets comprise the
source's third speed plus all three speeds on each other specimen. Five
contiguous beta blocks are held out across both source speeds together.
Hyperparameters maximize pooled source out-of-fold minimum IoU across the same
15 coherent observer rules used by the preceding benchmark. These rules share
an A-derived nuisance calibration; this remains a limitation for B/C/D sources.

For the published 2008 D-string monochord, all six pairs among
.05,.1,.15,.2 m/s are used, with the other two D speeds as the headline target.
Only systematically analyzed published cells are included. The empirical
published class is a singleton label family; its score is ordinary nominal IoU.
This is violin-string evidence on a rigid monochord, not an intact-violin test.

The published E-string figure has deliberately unanalysed interior Helmholtz
cells. Its remaining strict cells are concentrated near boundaries. Therefore
E is **excluded from the headline and combined D score**. Conditional E scores
are saved as diagnostics only; they must not be reported as ordinary full-grid
generalization.

Comparators are the preceding source-selected uniform exponent model and raw
3D quadratic logistic regression. Cello uses their previously frozen models,
checking exact source-ID hashes. Published comparators use the same alpha grid
0:.25:2, C grid {.1,1,10}, and five source beta folds. All eighteen flow fits and
published comparator selections are frozen with full coefficients, IDs and
hashes before target scoring. Baseline features/regularization differ from the
candidate, so an advantage would not by itself isolate the contribution of the
flow; matched concavity and regularization ablations would still be necessary.

## All aggregate outcomes

Values below average panel IoUs and then average the declared source tasks.
Caution: the tasks have overlapping target panels and are not independent
experimental replicates.

| Target | Affine flow | Uniform learned alpha | Raw 3D quadratic |
|---|---:|---:|---:|
| Cello combined, minimum-family IoU | .694794 | .694609 | .694037 |
| Cello source third speed, minimum-family IoU | .713565 | .720903 | .689714 |
| Cello other specimens, minimum-family IoU | .692709 | .691687 | .694518 |
| Cello combined, nominal IoU | .747885 | .749343 | .747928 |
| Published D other two speeds, nominal IoU | .861265 | .842085 | .813398 |

The cello aggregate improvement is only .000185 over learned alpha and .000757
over raw 3D. It loses to learned alpha on the source's held-out speed and to raw
3D across other specimens. On global pooled same-rule cello family IoU, flow
scores .721195 versus .717082 for alpha and .723313 for raw 3D. There is no
consistent superiority under the alternative reasonable aggregations.

The published D improvement is .019180 over learned alpha (5 wins, 1 loss),
and .047867 over raw 3D (3 wins, 3 losses). The larger mean raw-3D gain is strongly
influenced by failures when extrapolating from low source speeds. Two observed
speeds also weakly identify unconstrained velocity curvature in raw 3D, so this
is not evidence against all well-regularized flexible baselines.

| Published D source speeds | Flow | Learned alpha | Raw 3D |
|---|---:|---:|---:|
| .05, .10 | .891149 | .823587 | .714826 |
| .05, .15 | .910249 | .866701 | .793963 |
| .05, .20 | .883351 | .879682 | .886652 |
| .10, .15 | .873634 | .863132 | .849077 |
| .10, .20 | .854459 | .847981 | .863691 |
| .15, .20 | .754746 | .771424 | .772178 |

The last pair is an explicit failure: extrapolating downward from the two highest
speeds worsens both comparisons. Cello also contains substantial local losses,
including B/.1,.2 (.653029 versus alpha .691600 and raw .717223), and A/.1,.2
(.648144 versus .682134 and .699434). These cannot be hidden behind the means.
The public JSON includes all eighteen tasks; the figure plots every one.

E's conditional observed-cell IoUs are approximately .4301, .4310 and .4264
for flow, alpha and raw 3D respectively. They show no flow advantage and are
not comparable to full-grid D IoU because of the selection mechanism.

## Additional established asymmetric comparator

After the first comparison, a stronger literature-informed baseline was added
without changing any frozen flow result. At each of the two source speeds,
fit the existing six-parameter concave quadratic band independently. Transport
its lower and upper roots with fixed exponents, then average the two models'
probabilities. Static (0,0), uniform (1,1), and known asymmetric (0,1) use the
same folds and candidate regularizations {.001,.01,.1,1}; each prior selects
its own regularization by pooled source-only OOF IoU. Both training speeds
exclude the held-out beta block together. All six model sets are frozen before
their target scoring. There is no selection of which prior to display.

| Two-speed D baseline | Mean held-out D-panel IoU |
|---|---:|
| Affine flow, unchanged | .861265 |
| Known asymmetric ensemble | .844512 |
| Source-learned uniform alpha, unchanged | .842085 |
| Static ensemble | .816941 |
| Raw 3D, unchanged | .813398 |
| Uniform ensemble | .761346 |

Flow exceeds the known asymmetric ensemble by .016752 on average, with five
wins and one loss. On the highest-speed training pair, the known prior scores
.803740 versus flow .754746. The published-D gain therefore survives this
specific stronger comparator, but is modest, post-hoc, and not consistent across
all pairs or across the cello domain. Two independent bands have twelve base
coefficients versus the flow's nine total parameters; this is a strong known
baseline, not a matched parameter-count ablation. Averaging probabilities is
also only one reasonable way to combine the two sources.

The known-prior run took .26 seconds; two ensemble-specific tests pass. Full
logs and frozen models are in `reports/twospeed_known_prior/2026-10-13/`, and
the public `twospeed_known_prior_summary.json` preserves every pair and all
three prior choices.

## Matched width ablation

A final declared test isolates whether k, the width-deformation parameter,
provides the benefit. The ablation fixes only k=0 and retains the identical
objective, concave quadratic basis, source normalization, a0/a1 bounds, both
regularization grids, source rows, beta folds and source selection metric.
It therefore learns a beta-dependent translation with eight free parameters.
All eighteen ablation models were frozen before target scoring; the original
full-flow models were reused unchanged. No further candidates were searched.

| Target and metric | Full flow | Matched translation k=0 | Difference |
|---|---:|---:|---:|
| Cello combined family-min macro IoU | .694794 | .693502 | +.001292 |
| Cello source third-speed family-min IoU | .713565 | .719325 | -.005760 |
| Cello other specimens family-min macro IoU | .692709 | .690633 | +.002076 |
| Published D held-out-speed nominal macro IoU | .861265 | .840464 | +.020801 |

Full flow wins 8/12 cello tasks and 5/6 published D tasks against this matched
ablation, but has a large cello failure on B/.1,.2: .653029 versus .704581.
The results support a width benefit for the particular published D transfer
experiment. They do not support an across-domain gain or physical identification
of k. Root censoring and the selected regularizer can still influence the result.

The ablation has one unique initialization at k=0; repeating all three starts
after fixing k would duplicate the same computation. Full flow has three
distinct starts. To quantify this optimizer difference, all 810 full-flow CV
fits were audited: only one gains more than 1e-6 in training objective from a
nonzero start. It is an unselected cello-D/.05,.1 candidate (lambda=.1,
lambda_w=.001). Every selected final model's k=0-start objective is within
2.54e-11 of its best start. Thus the observed comparison is not evidently
driven by a selected local-optimization failure, although no global optimum
theorem is claimed.

This experiment took 6.76 seconds, with 810 CV fits and 18 final fits. Two new
tests check that only k is fixed and the exact shared objective is used, frozen
predictions survive JSON, and source selection is unchanged after poisoning
target labels. All eighteen paired results are preserved in
`affine_width_ablation_summary.json`, with full logs under
`reports/affine_width_ablation/2026-10-13/`.

## Numerical checks and reproducibility

Twelve affine tests pass: central finite differences at k=0, tiny k and both signs;
exact composition and inversion; concavity and interval topology; JSON frozen
prediction parity; training-objective start selection; target-label poisoning
leaves the complete source selection unchanged; and the published-panel shared
evaluator produces finite serializable summaries.
The additional zero-likelihood-gradient test demonstrates the single-reference-
speed identifiability failure directly. Two separate known-prior tests verify
the frozen ensemble calculation and source selection under poisoned targets.
The two matched-width tests bring the three new method modules to sixteen
targeted passing tests in total.

The final run took 21.37 seconds on the Mac using one numerical-library thread.
It performed 2,430 flow cross-validation optimizer starts with zero unsuccessful
starts, plus 54 final-fit starts. Published comparators required 900 CV fits and
12 final fits. Cello comparators were reused with provenance hashes.

The first attempt is retained in `reports/affine_transport/2026-10-13/`. All
model fits succeeded; scoring serialization then failed because published panel
IDs used an incompatible delimiter in the shared evaluator. This was fixed
without changing any model, candidate grid or source selection, and a new run
was written to `2026-10-13-v2/`. Before inspecting performance, the E headline
exclusion was also made explicit because its sampling mechanism is known.
The final run's parameter values match the first fit pass.

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv-mac/bin/python scripts/affine_transport.py \
  --output reports/affine_transport/new-run
OPENBLAS_NUM_THREADS=1 .venv-mac/bin/python -m pytest tests/test_affine_transport.py -q
.venv-mac/bin/python research/2026-10-13/affine_transport_assessment.py
```

The last command renders the fixed archived v2 results, not a fresh experiment.
The source protocol, full CV logs, all fitted parameters and the pre-target
manifest are in the run directory. Public aggregate, all-task scores and a
comparison figure are stored beside this document.

## Research decision

The affine-flow candidate is a compact, physically motivated hypothesis with
some useful published D-string results. The primary cello evidence does not
justify a general improvement claim. It should remain a transparent exploratory
candidate rather than be promoted as the central paper contribution.

A future superiority claim would need matched shape/regularization ablations,
further defensible ways to jointly fit the already-tested asymmetric prior,
independent specimens or recordings, and a prospective evaluation chosen before
seeing its labels. Switching the headline to the favorable D subset after
observing the cello tie would be selective reporting. No such claim is made.
