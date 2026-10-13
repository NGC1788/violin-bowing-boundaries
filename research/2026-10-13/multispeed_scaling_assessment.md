# Two-speed source-only exponent selection: completed retrospective diagnostic

Learning a common speed exponent from two source speeds improves over imposing exponent
one on this benchmark. It performs approximately the same as a generic three-dimensional
quadratic logistic classifier overall, and slightly worse on other specimens. This does
not establish a new competitive method or a new physical scaling law.

## Protocol and information boundary

`scripts/multispeed_scaling.py` evaluates all four source specimens A–D and all three
training-speed pairs: (0.05,0.10), (0.05,0.20), (0.10,0.20) m/s. Each model trains on exactly
two speed panels from one specimen. Evaluation has ten panels: the source specimen's
third speed plus all three speeds of each other specimen. These two evaluation scopes
are also reported separately. No prospective A2 data are opened.

The reduced coordinates are

\[
z_1=\log(\beta/0.1),\qquad z_2=\log(F/1\mathrm{N})-\alpha\log(v/0.1\mathrm{m\,s}^{-1}).
\]

A degree-two polynomial, source-fitted standardization and logistic regression are used
for every model. The four declared baselines are static α=0, fixed α=1, learned α from
{0,0.25,…,2}, and a raw three-dimensional quadratic in log beta/force/speed. All use
C∈{0.1,1,10}, the same source rows, the same logistic solver and probability threshold
0.5. Measured speed is used consistently for training and prediction.

The **same contiguous beta block is held out across both training speeds together**.
Five such folds produce one out-of-fold probability per source row. Selection maximizes
the minimum pooled OOF IoU across the same 15 coherent observation rules. Ties follow
declared grid order. Each fixed baseline selects its own C; the learned model selects
α and C using source folds only. Fixed-α fits are reused from the same grid, preventing
different solver settings or normalization from creating a contrast.

All twelve source selections and final fitted coefficients, normalization parameters,
polynomial powers and source IDs are saved before **any target metric is computed**.
`all_models_frozen.json` fingerprints these files. Evaluation reconstructs predictions
from the frozen JSON coefficients, not from a mutable in-memory estimator. There were
1800 CV fits and 46 distinct final fits (1846 fit records). One CPU thread was used.

This is post-hoc development on already-exposed sample1 data. It follows the earlier
single-speed scaling diagnostic; it is not preregistered independent confirmation.
The A/0.1-derived observer nuisance settings remain fixed across every experiment, even
when B/C/D is the training specimen. Thus model selection uses only its source labels,
but these experiments do not independently recalibrate or validate the label definition.

## Results

The aggregate below first computes each panel's minimum IoU over the 15 rules, averages
panels within an experiment, then averages the twelve predeclared source/pair experiments.
It permits different worst rules on different panels. Exact pooled metrics requiring a
**single common rule** across every evaluation row are separately retained in the JSON.

| Model | Combined ten panels | Source specimen's third speed | Other specimens, nine panels |
|---|---:|---:|---:|
| Static α=0 | 0.5450 | 0.5004 | 0.5499 |
| Fixed α=1 | 0.6703 | 0.6703 | 0.6703 |
| Source-learned α | 0.6946 | 0.7209 | 0.6917 |
| Raw 3D quadratic | 0.6940 | 0.6897 | 0.6945 |

Combined nominal-label macro IoUs are respectively 0.5927, 0.7239, 0.7493 and 0.7479.
Mean exact pooled single-rule minimum IoUs are 0.5669, 0.6965, 0.7171 and 0.7233.
The raw model wins this latter aggregate. Changing aggregation cannot be used to hide it.

Every source experiment, using the panel-macro minimum-family score:

| Source | Training speeds | Selected α | Static | Fixed α=1 | Learned α | Raw 3D |
|---|---|---:|---:|---:|---:|---:|
| A | .05,.10 | 1.25 | .5528 | .6891 | .6881 | .6950 |
| A | .05,.20 | 1.00 | .5484 | .6887 | .6887 | .5951 |
| A | .10,.20 | 1.25 | .5291 | .6619 | .6821 | .6994 |
| B | .05,.10 | 2.00 | .5361 | .6226 | .7001 | .6932 |
| B | .05,.20 | 2.00 | .5608 | .6685 | .6986 | .7091 |
| B | .10,.20 | 1.25 | .5185 | .6665 | .6916 | .7172 |
| C | .05,.10 | 1.75 | .5154 | .6277 | .6969 | .6830 |
| C | .05,.20 | 1.75 | .5867 | .6758 | .7031 | .7233 |
| C | .10,.20 | 1.50 | .5531 | .6923 | .7129 | .7166 |
| D | .05,.10 | 1.50 | .5425 | .6799 | .7028 | .6995 |
| D | .05,.20 | 1.75 | .5875 | .6952 | .6952 | .6946 |
| D | .10,.20 | 1.00 | .5088 | .6752 | .6752 | .7023 |

Learned versus fixed improves the combined score by 0.02432 on average (8 wins, 2 ties,
2 losses). Versus raw 3D the average gain is only 0.000572 (5 wins, 7 losses). The large
A/.05,.20 raw-model failure contributes substantially to this average. For other
specimens alone the learned model loses by 0.00283, with 4 wins and 8 losses.

## Interpretation and limitations

Two source speeds provide information that a single fixed speed cannot provide about
an exponent in this constrained representation. The test recovers a planted exponent
1.5 from a synthetic two-speed band. Real fitted exponents, however, depend on training
pair and specimen, and two B fits reach the declared ceiling 2. This is not evidence
that the true physical exponent equals 2 or lies outside the grid. No search-range
expansion was made after seeing evaluation performance.

The raw quadratic has more polynomial features (nine versus five) and can represent
speed interactions outside the common-exponent restriction. With only two nominal
speeds, its independent quadratic speed curvature is poorly supported; regularization
and small within-panel speed variation influence extrapolation. This limitation is
part of the baseline interpretation, not grounds for removing its favorable results.

All results remain relative to provisional weak labels. The twelve experiments reuse
four physical specimens and overlapping evaluation panels; they are not twelve
independent instrument replications. No row-level significance claim or publication-level
novelty claim follows. Known dimensional analysis, grid search and polynomial logistic
regression are established methods.

Four meaningful tests passed: shared-beta fold isolation and single-speed refusal;
planted-exponent recovery; frozen-JSON prediction equivalence; and poisoning every
target label leaves source CV, selections and fitted coefficients unchanged while a
guard verifies that all frozen manifests predate target evaluation.

Reproduce:

```bash
.venv-mac/bin/python -m pytest tests/test_multispeed_scaling.py -q
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
  .venv-mac/bin/python scripts/multispeed_scaling.py --output reports/multispeed_scaling/new-run
```

Full output: `reports/multispeed_scaling/2026-10-13/`.
Public aggregate/per-experiment summary: `research/2026-10-13/multispeed_scaling_summary.json`.
Existing frozen benchmark files were not changed.
