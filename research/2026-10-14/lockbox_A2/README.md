# Completed A2 evaluation: portable derived evidence

The two frozen evaluations completed on **14 October 2026 at 06:40 UTC
(15:40 KST)**. Both prespecified primary method comparisons are negative.
These are the first saved outputs, retained without target adaptation.

The original protocol trained on specimen A at 0.10 m/s. Its source-selected
covariant logistic baseline scores **0.659570**, versus **0.635452** for the
coherent-family robust candidate (difference **−0.024118**). All fourteen
declared models are in `original_results.json`; the baseline also has the
largest original-protocol target score, but its selection was made on source
cross-validation before A2, not on this ranking.

The separate transport addendum trained on two source-A speeds. Its primary
pair was fixed as (0.05,0.10) m/s before decoding A2:

| Source speed pair | Full flow | Matched k=0 | Learned exponent | Raw 3D |
|---|---:|---:|---:|---:|
| **0.05, 0.10 — primary** | **0.653586** | 0.667870 | 0.662273 | 0.676430 |
| 0.05, 0.20 — secondary | 0.657760 | 0.655558 | 0.647782 | 0.555659 |
| 0.10, 0.20 — secondary | 0.640014 | 0.602839 | 0.640147 | 0.670740 |

The primary full-flow differences are **−0.014285** against the matched
translation ablation and **−0.008687** against the source-selected simple
baseline, learned exponent. Positive secondary comparisons do not replace this
failed primary result. Different one-/two-speed training budgets must not be
mixed into one ranking. No conditional-position extension was added later.

The endpoint is the unweighted mean of three speed panels' minimum IoU over
the fifteen fixed coherent observer rules. Nominal, pooled, Brier and all
individual-panel scores remain in the complete JSON files. IoU refers to the
eligible acquisition grid and the specified operational labels, not independent
physical Helmholtz truth or continuous parameter-space area.

## Provenance and limitations

- Public source: **Alessio Lampis**, *A collection of bowed responses of cello
  strings (part 2)*, [version record17782542](https://doi.org/10.5281/zenodo.17782542),
  Type A sample2, `2024-04-19_TypeA_sample2.7z`.
- Whole archive: 14,775,691,753 bytes, MD5
  `ec520fecf6c5011ef8491b335edc1cee`. It was verified before decoding; the
  original receipt and preprocessing/output identities are retained.
- 6,000 raw trials in three 2,000-trial panels. Fixed eligibility excludes
  108 nonpositive-force rows and no off-plateau rows, leaving 5,892:
  1,986 / 2,000 / 1,906 at .05 / .1 / .2 m/s.
- Model, observer, evaluator and preprocessing freezes precede decoding. The
  original fourteen-model plan and later twelve-model addendum stay separate.
  This local chronology is not formal public preregistration or proof of every
  person's external exposure history.
- This is one additional cello-string specimen, not a new violin experiment,
  many independent instruments, or population-level statistical confirmation.
- A2 has now been examined. Methods designed after these results use it as
  development evidence; it cannot become untouched again.

## Files and license

The derived CSV and feature summary are provided under **CC BY 4.0**, with
the Lampis attribution above, the [license](https://creativecommons.org/licenses/by/4.0/),
and notice of these changes: the project extracted fixed waveform features,
estimated force/speed grid coordinates and applied its own provisional labels.
They are not author-supplied reference classifications. No endorsement is implied.

`regimes.csv.gz` is an exact lossless copy of the 6,000-row derived CSV; its
decompressed SHA256 matches the original evaluation manifest. The local
repository prefix in the feature summary's `source_trials` and four `plots` paths was removed.
The original summary hash is retained, and its released hash is different.
No scientific values were altered. The five original result/provenance JSON
files are byte-identical. `release_manifest.json` records every released file's
SHA256 and the transformations. No raw recordings or model pickle files are
included.

## Replay without downloading the large archive

From the repository root:

```sh
bash scripts/run_mac.sh python scripts/restore_frozen_models.py
bash scripts/run_mac.sh python research/2026-10-14/lockbox_release.py replay
```

The first command restores the fourteen exact original model checkpoints from
public source data and frozen parameters; it never uses A2 for fitting. The
second checks released features and frozen code, predicts with all twenty-six
models, and compares every saved metric at tolerance1e−12. Transport model
coefficients are embedded in the unchanged addendum specification.

This replays **numerical prediction and evaluation from derived features**.
It does not claim to repeat raw waveform extraction or freshly verify an
archive absent from the reader's computer. Full raw reproduction remains in
the [Mac workflow](../../../docs/mac_research.md).
