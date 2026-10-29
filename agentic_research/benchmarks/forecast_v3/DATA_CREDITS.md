# Data credits and redistribution scope

This directory contains one processed benchmark derived from **Alessio Lampis**, *A collection of bowed responses of cello strings*, parts 1, 2, 3, 4, and 6. Original creator affiliation: University of Music and Performing Arts Vienna. ORCID: [0009-0003-0813-9031](https://orcid.org/0009-0003-0813-9031).

The five current official Zenodo API records were checked during packaging and each returned the license identifier `cc-by-4.0`. The processed data are provided under [Creative Commons Attribution 4.0 International](https://creativecommons.org/licenses/by/4.0/). Retain the creator attribution, source links, license link, and the description of modifications when reusing or redistributing these data. No endorsement by the original creator is implied.

| Original collection part | DOI | Selected diagrams |
|---|---|---|
| Part 1 | [10.5281/zenodo.17749111](https://doi.org/10.5281/zenodo.17749111) | A1, B1, C1 |
| Part 2 | [10.5281/zenodo.17782542](https://doi.org/10.5281/zenodo.17782542) | D1, A2, A3 |
| Part 3 | [10.5281/zenodo.17782552](https://doi.org/10.5281/zenodo.17782552) | A4, D2 |
| Part 4 | [10.5281/zenodo.17782565](https://doi.org/10.5281/zenodo.17782565) | C_T2, B_T2s1, A_T2 |
| Part 6 | [10.5281/zenodo.17795326](https://doi.org/10.5281/zenodo.17795326) | B_T2r1, B_T2r2 |

`provenance.json` records exact titles, creators, DOIs, license identifiers, API URLs and response hashes, selected archive download URLs/checksums, and the hashes of the twelve intermediate feature CSVs. The historical parent fixture cites Part 5 as well; no Part 5 archive is used in this benchmark.

## Modifications

The research team—Kim Taegeon, Jeong Jiyun, and Jeong Jihoon—prepared this derived benchmark by extracting mechanical-signal window features, selecting 78,000 original strokes, applying fixed feature transforms and prefix-only missing indicators, constructing the nine declared observer outputs and ten-pattern IDs, and attaching whole-family training/development partitions and reproducibility metadata. The artifact is processed data, not the original raw recordings.

Inputs summarize windows ending by 0.20 s; the future target covers 0.25–0.45 s relative to a retrospectively recovered speed landmark. There are 54,000 training strokes from families A/B and 24,000 development strokes from C/D. The nine targets are deterministic observer classifications, **not independently verified physical ground truth**. The underlying apparatus bowed cello G2 strings; the artifact is not violin microphone audio. The landmark uses the full speed trace, and development material was historically exposed. Thus this is an offline development benchmark, not a validated online system or an untouched final test.

## License boundary

The CC BY 4.0 attribution above concerns the original and processed **data**. This file does not choose, replace, or infer a license for software elsewhere in this repository. Users must inspect applicable software licenses separately.

No personal recording data, local home-directory paths, email addresses, or detected credential strings were found in the benchmark's NPZ string fields or receipt. No content was redacted: `dataset.npz` is byte-identical to the verified prepared artifact with SHA256 `598a6964ce7ef91c5bb6b3cde1aa37daf4b615219438f896a4dc188f0d427975`.
