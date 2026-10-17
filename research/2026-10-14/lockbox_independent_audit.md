# Independent audit of the completed A2 evaluations

Audit completed on **15 October 2026, 02:42 KST** (14 October, 17:42 UTC).
The numerical and provenance checks pass. **Neither prespecified primary
method comparison supports improvement over its source-selected baseline.**
This audit does not turn the operational labels into physical ground truth.

The audit used a separate CSV parser, scalar observation-rule implementation,
Python-set IoU calculations and independent prediction transforms. It did not
call the production label-family, metric, panel-evaluation, affine-flow or
multispeed prediction functions. Original standard classifiers still use their
frozen scikit-learn estimator prediction routines; the audit does not independently
reimplement tree traversal, SVM probability calibration or logistic regression.
There was no fitting, target adaptation or parameter selection in this audit.

## Findings

- Recomputed the MD5 of the entire **14,775,691,753-byte** local archive:
  `ec520fecf6c5011ef8491b335edc1cee`, matching the pinned Zenodo archive and the
  pre-extraction receipt. This is a new whole-file check, not merely a copied
  expected checksum.
- Checked canonical fingerprints of all four frozen specifications and current
  SHA256 values of **15 implementation files**. The fourteen original checkpoint
  file hashes match their freeze; all twelve transport models remain the
  coefficients embedded in the addendum.
- Independently recovered **6,000 raw rows and 5,892 eligible rows**. The fixed
  exclusions are 108 nonpositive-force rows and zero off-plateau rows. Eligible
  counts at .05/.10/.20 m/s are **1,986 / 2,000 / 1,906**. There are no duplicate
  trial IDs or grid cells, and no ID overlap with the development corpus.
- Reconstructed all fifteen labels for every eligible row, including the strict
  flyback-ratio inequality and inclusive periodicity gate. These, the nominal
  label, panel names, beta levels and ordered IDs exactly match both saved
  prediction bundles. Original target provisional labels are not used to form
  the audit targets; their observed disagreement count is separately checked.
- Independently reconstructed **153,192 probability values from 26 models**.
  The maximum difference from first-saved predictions is **5.55 × 10⁻¹⁶**.
  Classification at the frozen threshold is unchanged.
- Recomputed all original and additional panel, macro, specimen and global
  metrics, all fifteen rule-specific IoUs, box bounds, Brier scores, copied
  endpoints and contrasts. **4,694 numeric occurrences** match exactly. This
  count includes intentionally duplicated summary values and manifest copies;
  it is not a count of independent measurements.
- Checked source-only row IDs, dataset identity, source predictors/labels
  fingerprints for the three two-speed pairs, winning CV indexes and selected
  regularization/baseline settings. The original freeze refers to
  `benchmark_A_portable`, whose manifest and selection bytes match the freeze.
  The initial `benchmark_A` is a different historical manifest; it is not used
  as a substitute for that provenance.
- Verified the five released result/provenance JSON files are byte-identical to
  their original outputs. The decompressed released CSV is byte-identical to
  the original input. In the feature summary, repository prefixes were removed
  from **`source_trials` and four `plots` paths**; every other value is identical.
  The earlier description saying only `source_trials` changed was incomplete
  and has been corrected in the release documentation and manifest. No numerical
  evidence changed.

## Primary results and all additional source pairs

The endpoint remains the unweighted mean of each of the three panels' minimum
IoU over the fifteen frozen observation rules. It is not pooled IoU and does not
require the minimizing observer to be the same across panels. The separately
reported global score does use one shared rule over all rows.

For the original one-speed protocol, source-selected covariant logistic scores
**0.6595695000** and coherent-family robust scores **0.6354515329**. The candidate
minus baseline difference is **−0.0241179672**. All fourteen original model
endpoints are retained in the audit JSON; all original metrics remain in the
[released results](lockbox_A2/original_results.json).

| Two-speed source pair | Full affine flow | Matched k=0 | Learned exponent | Raw 3D | Flow − k=0 | Flow − source-selected simple |
|---|---:|---:|---:|---:|---:|---:|
| **.05, .10 — primary** | **.653585559** | .667870240 | .662272727 | .676429926 | **−.014284681** | **−.008687168** |
| .05, .20 — secondary | .657760226 | .655558093 | .647781678 | .555658859 | +.002202133 | +.009978548 |
| .10, .20 — secondary | .640013869 | .602838778 | .640146703 | .670739694 | +.037175092 | −.000132834 |

The source-selected simple model is learned exponent for all three pairs.
Positive secondary comparisons do not replace the negative primary comparisons.
Raw 3D has the highest primary-pair target score, but was not the source-selected
simple baseline. Different one-speed and two-speed training budgets are kept
separate.

## Recorded chronology and scope

| Local recorded event | UTC |
|---|---|
| Original model freeze | 13 October 07:34:07 |
| Evaluation freeze | 13 October 07:42:27 |
| Preprocessing freeze | 13 October 08:39:08 |
| Transport addendum freeze | 13 October 09:02:25 |
| Whole-archive verification receipt | 14 October 06:19:49 |
| Feature output timestamp | 14 October 06:40:41 |
| Original evaluation completed | 14 October 06:40:45 |
| Additional evaluation completed | 14 October 06:40:49 |

The checked records place every freeze before the verified archive and saved
feature/evaluation outputs. The preprocessing binding also matches the archive,
fixed configuration and original output hashes. This audit does not independently
repeat raw waveform extraction. Its chronology check does not prove that every
human avoided external exposure, does not establish the first human reading of
these results, and is not formal public preregistration.

A2 is **one additional cello-string specimen/session with three dependent speed
panels**. The labels remain generated by the declared observer family, not
independent physical Helmholtz truth. Neither population-level significance nor
intact-violin generalization follows from these counts. A2 has now been examined
and must be treated as development evidence for subsequently designed methods.

## Reproduction and tests

The audit uses local original output bundles and the local source-CV archive in
addition to the public release. Run from the repository root:

```sh
bash scripts/run_mac.sh python research/2026-10-14/lockbox_independent_audit.py --verify-archive
bash scripts/run_mac.sh python -m pytest tests/test_lockbox_independent_audit.py -q
```

The four targeted tests pass. They check the independent box-bound oracle against
exhaustive binary completions of small sets, empty-set and threshold ties,
unweighted macro versus pooled scores, and rejection of altered metric/schema
values. The machine-readable result is
[lockbox_independent_audit.json](lockbox_independent_audit.json).

For a fresh clone without the large raw archive or local original bundles, use
[the public numerical replay](lockbox_A2/README.md#replay-without-downloading-the-large-archive)
instead. That replay and this independent audit have different evidence scopes.
