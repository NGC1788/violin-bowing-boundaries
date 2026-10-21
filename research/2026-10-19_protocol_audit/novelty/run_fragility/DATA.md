# Compact derived input: source and transformations

`input_columns.json.gz` contains 1,560 force-ordered columns (78,000 individual stroke windows) from 13 already analyzed diagrams of public robotic **cello G2** data. It contains no waveform samples and is not a new violin measurement dataset.

## Attribution and reuse

Original creator: **Alessio Lampis**, University of Music and Performing Arts Vienna (IWK), with project leader Vasileios Chatziioannou and researcher Alexander Mayer credited in the records. Collection: *A collection of bowed responses of cello strings* (2025). The used source records are:

| Record | Used diagrams |
|---|---|
| [Part 1, version record 17749111](https://zenodo.org/records/17749111) | A1, B1, C1 |
| [Part 2, 17782542](https://zenodo.org/records/17782542) | D1, A2, A3 |
| [Part 3, 17782552](https://zenodo.org/records/17782552) | A4, D2 |
| [Part 4, 17782565](https://zenodo.org/records/17782565) | A_T2, B_T2s1, C_T2 |
| [Part 6, 17795326](https://zenodo.org/records/17795326) | B_T2r1, B_T2r2 |

The Zenodo JSON metadata for these five records was checked on 2026-10-21 and reported `cc-by-4.0`. This derived data fixture is distributed with attribution under [Creative Commons Attribution 4.0 International](https://creativecommons.org/licenses/by/4.0/). Changes made here are listed below. No endorsement by the original creators is implied. This notice concerns the data fixture; it does not change licenses of other repository code or documents.

Main source publication: A. Lampis, V. Chatziioannou and G. Scavone (2025), “Experimental analysis of cello string types: influence on playability and tonal characteristics using Schelleng diagrams,” *Proceedings of Meetings on Acoustics* 58, 035013. [DOI](https://doi.org/10.1121/2.0002111).

The fixture's `source_records` lists the wider loader provenance, including Part 5. No Part 5 column is retained: D_T2 was explicitly excluded before this calculation, and B_T2 sample 2 is not in this standard-grid cohort. The table above describes the actual included columns. Part 1's collection/concept record 17749110 and the downloaded version record 17749111 are distinct identifiers.

## Changes and scope

1. Existing extraction computed features and measured mean bow force in a 0.2-s window starting 0.2 s after the bow-speed plateau begins.
2. Existing base classifier thresholds, originally set on A2, yielded the binary H/non-H labels. These are algorithmic waveform proxies, not human truth labels supplied by the dataset creator.
3. The existing beta grouping assigned 40 columns per speed folder and 50 strokes per column. The export retains each original trial ID, folder, speed, median beta, individual mean force and binary label, then sorts each column by mean force.
4. The export includes nonpositive measured forces as recorded (510 windows), without clipping or changing them. The analysis excludes original nonfinite/nonpositive/zero-width brackets, no-run and left-censored columns, and only admits positive finite edited brackets. The full force vector can still contain nonpositive windows outside the chosen bracket.
5. Original files and labels remain unchanged. `results.json` stores hypothetical edits separately. The export is deterministically gzip-compressed with mtime=0.

Each row-level identity is `(session, folder, trial)`, not just trial number. `column` is the existing beta-column assignment; `force_rank` in a saved edit is zero-based within the exported increasing-force vector. Original acquisition order is not the same as this force order.

## Provenance and reproduction limits

The fixture contains SHA256 hashes of original feature files and export/loader wrappers. `verification/input_audit.json` adds the available classifier/grouping/threshold source hashes from the original research environment. `verification/verified_source_sha256.json` pins the independently checked implementation. The final artifact manifest covers the packaged files.

The included fixture is sufficient to reproduce the exact costs, witnesses, diagnostics, synthetic study and independent algorithm checks without raw archives. `export_inputs.py` is retained for provenance but intentionally requires the older full research cache and classifier code. Do not run it to reproduce this compact artifact. Revalidating raw-signal extraction or the physical accuracy of the original classifier is a separate task; successful compact reproduction does not establish those.
