# Source boundary identification audit

Date: 2026-10-14. Retrospective development evidence only. No A2 waveform, outcome, model or frozen evaluation specification was read or changed. The audit does not fit predictions or overwrite labels.

**Decision:** these observations do not establish an exact common physical width law. The published violin D-string grids are almost single bands, but their exact width brackets contradict a common affine width rate in five of six source-speed pairs. Cello operational labels are less band-like and heavily censored. An affine predictor can improve IoU without identifying the physical mechanism that generated the boundary.

## Computation and assumptions

For each specimen, nominal speed, nominal bow-position column and declared observer, sort positive measured bow forces. Keep every observed connected positive component. Under a single-band interpretation, adjacent negative/positive observations bracket each endpoint in log force. A positive edge observation leaves the exterior endpoint unbounded. Missing or excluded observations are not supplied with invented labels. Closed brackets are conservative outer relaxations of endpoint inclusion conventions.

Independently enumerate every contiguous positive prediction, including the empty prediction, and retain every minimum-Hamming alternative. This gives the **minimum number of observed labels that disagree with any single interval**. It is a model-mismatch diagnostic, not a correction, an estimated error rate, or permission to train on repaired labels. Nonunique alternatives remain distinct in the detailed output; their summary is an outer envelope that can include combinations attained by no single alternative. If the empty prediction ties, no unconditional boundary envelope is asserted.

For two speeds with log separation Δt, endpoint brackets [L₁,U₁], [L₂,U₂] imply the slope interval [(L₂−U₁)/Δt, (U₂−L₁)/Δt]. A band of log-force width W transported by du/dt=a(β)+ku has W₂/W₁=exp(kΔt). This gives per-column intervals for k. Intersecting those intervals across columns yields a necessary outer constraint on common k; a nonempty intersection does not prove joint boundary feasibility. Empty intersections do refute that exact constraint system under its maintained observation assumptions. These are elementary interval calculations, not a novel estimation theorem.

The cello calculation treats measured positive forces within nominal β/speed columns as an ordered slice. Small measured speed differences and variation within nominal β columns are not modeled. Published coordinates are rounded nominal values, author classes are not error-free physical truth, and the high-speed D panel has 15 unknown cells. Therefore interval incompatibility is **not** evidence that physical affine behavior is impossible after allowing control, annotation or model discrepancy. It rules out an exact interpretation of these retained brackets.

## Single-band mismatch and censoring

Nominal cello rule: fraction .5, periodicity .8; the fixed A-derived frequency eligibility is unchanged. All fifteen declared observer rules are also retained in the machine-readable summary. Published D uses its original singleton author class, not the cello observer.

| Specimen | Speed (m/s) | Multiple-component columns | Both endpoints bracketed exactly | Minimum disagreements | Fraction |
|---|---:|---:|---:|---:|---:|
| A | 0.05 | 33/40 | 1/40 | 57/1949 | 2.925% |
| A | 0.1 | 36/40 | 0/40 | 90/1750 | 5.143% |
| A | 0.2 | 19/40 | 4/40 | 48/1626 | 2.952% |
| B | 0.05 | 32/40 | 0/40 | 81/1986 | 4.079% |
| B | 0.1 | 38/40 | 0/40 | 129/1990 | 6.482% |
| B | 0.2 | 27/40 | 0/40 | 107/2000 | 5.350% |
| C | 0.05 | 32/40 | 1/40 | 116/2000 | 5.800% |
| C | 0.1 | 37/40 | 1/40 | 112/2000 | 5.600% |
| C | 0.2 | 31/40 | 2/40 | 117/2000 | 5.850% |
| D | 0.05 | 24/40 | 0/40 | 58/2000 | 2.900% |
| D | 0.1 | 30/40 | 0/40 | 113/2000 | 5.650% |
| D | 0.2 | 27/40 | 1/40 | 103/2000 | 5.150% |
| Published violin D | 0.05 | 0/11 | 6/11 | 0/264 | 0.000% |
| Published violin D | 0.1 | 1/11 | 7/11 | 1/264 | 0.379% |
| Published violin D | 0.15 | 0/11 | 6/11 | 0/264 | 0.000% |
| Published violin D | 0.2 | 0/11 | 2/11 | 0/249 | 0.000% |

Across nominal cello panels, 366/480 columns have multiple observed positive components, but the minimum mismatch totals only **1,131/23,301 = 4.854%** of cells. A high component count can therefore be driven by relatively few holes. It does not show that the underlying physical regime has dozens of disconnected bands. Only 10/480 columns have an exact single band with both endpoints finitely bracketed, and no source-speed pair has such a fully bracketed column at both speeds.

Published violin D has **1/1,041 = 0.096%** minimum mismatch, with one disconnected column among 44. Its .2 m/s panel has just 2/11 exactly fully bracketed columns: five touch the low-force observation edge and four touch the high-force edge. The sparse bracket information is a concrete reason that a smooth fitted boundary is not a precisely measured boundary.

Across all fifteen cello rules the pooled minimum mismatch ranges from **4.111% to 8.137%**; all 180 source/specimen/speed-pair cases still have zero columns whose exact single band is fully bracketed at both speeds. The same observations are reused across rules; 7,244 audited column/rule cases are not independent experimental repetitions.

## Width law and extrapolation

The following table uses every published D source-speed pair. The common-k constraints are identical for exact bands and minimum-disagreement envelopes in these six cases. A displayed interval with lower endpoint greater than upper endpoint is empty. No column is removed based on whether its width interval helps common-k feasibility. Columns that do not supply a band are omitted from this necessary constraint and cannot be counted as evidence supporting compatibility.

| Source speeds (m/s) | Exactly fully bracketed at both speeds | Common-k outer interval | Largest absolute extrapolation-weight sum | Historical affine IoU |
|---|---:|---:|---:|---:|
| 0.05, 0.1 | 5/11 | [0.099451, 0.092484] | 3.000 | 0.891149 |
| 0.05, 0.15 | 6/11 | [0.411136, 0.262348] | 1.524 | 0.910249 |
| 0.05, 0.2 | 1/11 | [0.354236, 0.161013] | 1.000 | 0.883351 |
| 0.1, 0.15 | 5/11 | [0.787145, 0.422486] | 4.419 | 0.873634 |
| 0.1, 0.2 | 1/11 | [0.541784, 0.414497] | 3.000 | 0.854459 |
| 0.15, 0.2 | 1/11 | [0.467472, 0.539710] | 8.638 | 0.754746 |

The high-speed .15/.2 pair has the weakest historical affine prediction (IoU .754746), only one exactly fully bracketed column at both speeds, and maximum extrapolation weight sum **8.638** toward .05 m/s. Yet its common-k interval is nonempty while the other five are empty. The observed compatibility indicator does not order these six retrospective prediction scores; this comparison does not identify the cause.

For nominal cello labels, common k is already incompatible in 9/12 source-speed pairs even after allowing every minimum-disagreement envelope. A single β-independent lower-endpoint exponent and a single β-independent upper-endpoint exponent each fail in all twelve nominal cases. Across fifteen rules, lower-exponent compatibility fails in all 180 cases; this is conditional on the approximating-band construction and nominal-slice assumption, not a calibrated mechanical rejection.

For source log speeds t₁,t₂ and target t, the straight-line endpoint extrapolator has weights 1−r,r, where r=(t−t₁)/(t₂−t₁). Source bracket widths amplify by |1−r| and |r|. If a global curvature bound |b″|≤M were established, the usual additional interpolation/extrapolation remainder is at most M·|(t−t₁)(t−t₂)|/2. **Two source speeds cannot identify M.** The audit does not assign one from the target scores or claim finite out-of-support coverage.

Historical score associations are descriptive: fully bracketed fraction versus affine error has Spearman correlation +.0748 on twelve cello tasks and −.6172 on six published-D tasks; speed leverage versus error is +.3585 and +.6957 respectively. Tasks share specimens and target grids. No p-value, validated error predictor or source-selection rule is claimed from these small, dependent comparisons.

## Specimen transfer

Minimum-disagreement endpoint envelopes often differ between cello specimens at the same nominal speed/position. For example, A versus B at .05 m/s has disjoint lower envelopes in 32/40 columns and disjoint upper envelopes in 18/40. Every one of the eighteen specimen-pair/speed cases has some disjoint lower and upper envelopes. A within-specimen speed law therefore does not by itself constrain another specimen’s base boundaries. This calculation adds no measured bound on specimen variability, and cannot support guaranteed cross-specimen transfer.

## What this changes in the research

An uncertainty method that presents narrow extrapolated bands must justify both its observation discrepancy and its speed-law assumptions. Neither can be inferred merely because two fitted source slices agree. A future method would need an explicit, testable error model and source information that constrains transport; simply wrapping the affine model in intervals would produce either false certainty or very wide answers. These diagnostics support an identification-first research direction, but are not themselves a new machine-learning method or a successful prospective evaluation.

## Independent review and reproduction

The existing source audit had one correctness issue outside the current real-data cases: conflicting repeated forces can make an empty prediction and a finite nonempty interval equally optimal. Reporting the latter’s brackets unconditionally overstated identification. The corrected implementation keeps the nonempty conditional envelope but supplies no unconditional band when empty is also optimal. The retained real-data audit has zero conflicting exact-force ties, so this fix changes no reported empirical values.

Targeted verification: **30 tests passed** across source geometry and shared-observer audit, including all 128 binary sequences on seven distinct force levels, conflicting ties, empty alternatives, censoring, invalid interval/speed inputs, direct observer enumeration and shared-versus-separate panel semantics. Independently rebuilding labels and intersection/union counts at both stored witness settings in all 5,596 shared-observer comparison rows reproduced **22,384 reported IoUs**, with maximum absolute error **2.22 × 10⁻¹⁶**. This verifies the reported witnesses and aggregation; exhaustive random-fixture tests separately check extremality. No shared-observer implementation changes were needed.

- [Complete compressed summary, all observer/panel summaries and provenance](source_identifiability_summary.json.gz).
- Publication metadata uses a repository-relative published-source path. The bundle
  records this transformation and the original uncompressed payload SHA-256;
  scientific values and original ignored run files are unchanged.
- Script: `scripts/source_identifiability_audit.py`.
- Tests: `tests/test_source_identifiability_audit.py`.
- Archived full run: `reports/source_identifiability/2026-10-14-audited/` (ignored, regenerable).

```sh
bash scripts/run_mac.sh python scripts/source_identifiability_audit.py \
  --flow-run reports/affine_transport/2026-10-13-v2 \
  --output reports/source_identifiability/new-run
bash scripts/run_mac.sh python -m pytest \
  tests/test_source_identifiability_audit.py tests/test_observer_dominance_audit.py -q
```

Use the actual affine development run path when reproducing; the output directory must be new or empty. Derived public inputs suffice for this audit. No sample2 archive or new recording is required.
