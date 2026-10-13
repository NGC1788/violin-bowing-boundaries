# External published violin D-string comparison

The fixed asymmetric boundary prior performs best on the prespecified primary
mean across target speeds. Uniform force/speed coordinates perform worse than
matched static coordinates. This reverses the pattern in the previously examined
cello data and rules out presenting uniform scaling as a generally established
violin-string advantage. It does not establish a new law or a new methodology.

## Execution and interpretation

The source is Figure 4 of [Schoonderwaldt, Guettler and Askenfelt (2008)](https://doi.org/10.3813/AAA.918070),
recovered as described in `README.md`. We use 1,041 strictly classified conditions
from a real violin D string on a monochord. The 15 unplotted cells at 0.20 m/s
remain unknown. There are no new waveforms, no own measurements, and no assembled
violin validation in this classified-cell benchmark.

The comparison plan, source/fold IDs and code hashes were written before model
evaluation. Nevertheless, the figures were examined during extraction, and the
asymmetric prior comes from the same publication. This is a retrospective
external computational comparison, **not a blind test, preregistration, or an
independent discovery of the original authors' physical interpretation**. It is
external to the cello corpus; that does not make every aspect of the hypothesis
independent of this source.

The primary source speed is 0.10 m/s. Five contiguous source beta-column folds
select C from 0.1, 1 and 10, or boundary regularization from 0.001, 0.01, 0.1 and
1. Static and uniform variants receive identical source coordinates and select
identical hyperparameters. The four boundary priors share exactly one source
fit. Every variant uses threshold 0.5. No hyperparameter or prior is chosen using
target scores. All fixed comparisons are reported below.

## All-source sensitivity results

Entries are the equal-weight mean IoU over the other three target speeds. The
four columns share the same physical string and overlapping training/evaluation
conditions; they are not four independent experimental replications.

| Model | Source 0.10 (primary) | Source 0.05 | Source 0.15 | Source 0.20 |
|---|---:|---:|---:|---:|
| Quadratic, static | 0.831140 | 0.773998 | 0.812161 | 0.781462 |
| Quadratic, uniform F/v | 0.774699 | 0.619317 | 0.799369 | 0.756472 |
| Quadratic, raw 3D | 0.831140 | 0.773998 | 0.812161 | 0.781462 |
| RBF SVM, static | 0.824358 | 0.743275 | 0.821725 | 0.779593 |
| RBF SVM, uniform F/v | 0.739681 | 0.708256 | 0.799546 | 0.746344 |
| Concave boundary, uniform (1,1) | 0.763821 | 0.608699 | 0.786685 | 0.758432 |
| Concave boundary, static (0,0) | 0.825181 | 0.757228 | 0.810804 | 0.782427 |
| Concave boundary, asymmetric (0,1) | 0.860716 | 0.825363 | 0.848280 | 0.834848 |
| Concave boundary, reverse (1,0) | 0.724152 | 0.509365 | 0.758897 | 0.724118 |

Here the raw 3D quadratic equals its static counterpart: source speed is exactly
constant, so speed and speed-interaction columns carry no learnable variation.
That outcome also illustrates why a single source speed cannot identify a speed
law from labels alone. A fixed nonzero speed exponent contributes prior
assumptions, rather than information learned from that source.

For the primary shared boundary fit, target IoUs are:

| Fixed boundary exponents (lower, upper) | Target 0.05 | Target 0.15 | Target 0.20 |
|---|---:|---:|---:|
| Uniform (1,1) | 0.696970 | 0.829787 | 0.764706 |
| Static (0,0) | 0.839506 | 0.857143 | 0.778894 |
| Asymmetric (0,1) | 0.777778 | 0.900524 | 0.903846 |
| Reverse (1,0) | 0.759563 | 0.782123 | 0.630769 |

The asymmetric prior loses to static transport on 0.10→0.05. Its average
advantage is therefore not an improvement on every transfer direction. The
source-selected boundary regularization is 0.01, with source OOF IoU 0.859756;
the selected quadratic C is 1 and RBF C is 10. Source CV differences do not enter
the target macro scores.

## Limits that affect the research claim

- A prior is not identifiable from a single speed. Every exponent pair acts
  identically at its reference speed; a source-only rule cannot discover the
  correct pair from these identical predictions.
- Cross-corpus differences do not identify their cause. Cello versus violin
  strings, apparatus, input range, author classification versus the newer
  automatic observer, and collection conditions all change together. The
  reversal cannot be attributed exclusively to instrument, observer or friction.
- An accuracy comparison against a prior motivated by this same 2008 paper is
  a reproduction/transport baseline. A new method must outperform these strong
  fixed controls under a separate, defensible evaluation design.
- E-string plotted H interiors were not systematically analyzed; the strict
  remaining 130 cells are spatially selected. They are released with provenance
  but not used to claim full-domain predictive performance.
- Nominal beta uses rounded reported distances. Boundary errors below this
  coordinate precision are not meaningful. No interpolated continuous boundary
  is treated as measured truth.
- The D source at 0.10 has one column with multiple observed positive runs.
  Positive low-force acquisition edges occur at 5, 3, 5 and 4 columns for speeds
  0.05, 0.10, 0.15 and 0.20 respectively. The 0.20 panel has unknown low-force
  cells as well as high-force cells. `published_grid_audit.json` retains exact
  adjacent transition brackets and never fills unknown gaps.
- We do not report independent-instrument confidence intervals or infer
  generalization to assembled violins, all strings, or unseen performers.

## Reproduce and inspect

From the repository root:

```sh
.venv-mac/bin/python -W ignore::FutureWarning \
  research/2026-10-13/schoonderwaldt2008/benchmark_published.py \
  --output reports/published_violin/reproduction
.venv-mac/bin/python -m pytest tests/test_published_violin.py -q
```

Use a new empty output directory. The run takes approximately one second on this
Mac after imports. `benchmark/` retains the original manifest, source/fold IDs,
all scores and per-beta errors. Complete per-cell predictions are in ignored
`reports/published_violin/2026-10-13/predictions.csv`, with SHA256 retained in
`benchmark/prediction_file_hash.json`; the script regenerates them. The five
contract tests passed, checking strict exclusions, the 24 manually inspected
markers, contiguous disjoint folds, source-coordinate equality and preservation
of an unknown gap. These checks validate the extraction/evaluation contract;
they do not prove the original author's classifications physically correct.
