# Data contract

This document separates published dataset facts from requirements proposed for this project. The starter has inspected official metadata and the first archive's member header; it has **not** verified the full signal contents of the large archives or established any research result.

## Keep the two sources separate

| Source | Observations | Valid initial use |
| --- | --- | --- |
| Public bowed-cello-string dataset | Robotic monochord measurements of mechanical forces and bow velocity | Test mechanical-state analysis and boundary estimation, after verifying schema and reference labels |
| Project violin recordings (revised scope) | Final supplementary microphone recordings and available setup notes; no new force sensor | Describe acoustic observations under documented conditions; not quantitative force-boundary validation |

The public dataset is not a collection of microphone recordings of complete violins. Mechanical force and microphone pressure have different observation models. Success on public mechanical signals does not validate a microphone classifier or establish the physical slip regime of a real violin. Do not pool these sources without an explicit domain model and separate evaluation.

## Public-data schema: verify before parsing

The [official Part 1 metadata](https://zenodo.org/records/17749110) describes a nominal 98 Hz cello G2 string, 50 kHz sampling, and the following `whole.csv` columns, in order:

| Column | Quantity | Published unit |
| --- | --- | --- |
| 1 | Bow force | N |
| 2 | Bow velocity | m/s |
| 3 | Bridge force | N |
| 4 | Nut force | N |

None of these four columns is microphone audio, a time column, or a regime label. Do not convert the force columns into WAV files and describe them as recorded violin sound.

The metadata also describes `betas.csv` for relative bow–bridge distance β and a timestamps file for the start/end of a classification window. **Window boundaries are not state labels.** Their units, indexing convention, CSV delimiter/header, alignment, and actual file structure must be established from archive contents and accompanying documentation before implementation. Folder numbers 1, 2, 3 are described as target bow speeds 0.1, 0.05, 0.2 m/s respectively; preserve measured velocity separately from targets.

General collection metadata describes forces 0.1–4 N and β 0.02–0.2. [Part 7](https://zenodo.org/records/17822016) and [Part 8](https://zenodo.org/records/17822037) extend forces to 4–12 N with reduced β range. These are collection-level descriptions, not proof that every archive or trial contains every combination. Check actual coverage.

### Observed first-archive header, 2026-09-13

A bounded HTTP Range inspection of the official `2024-03-25_TypeA_sample1.7z` header transferred 253,329 bytes, then its member listing was checked with 7-Zip. This was **header inspection only**, not a full download, payload checksum test, or CSV read. It found 18,000 files and 4 directories, totaling 74,179,785,108 uncompressed bytes (69.085 GiB), in one solid block. Under the archive root, `2024-03-27_r_v1`, `2024-03-27_r_v2`, and `2024-03-27_r_v3` each contain 2,000 numbered `whole_N.csv` files with matching `beta_N.csv` and `timestamp_N.csv` files.

These actual names differ from the generalized names in the metadata. The parser must support the observed names. Neither the dates nor `v1/v2/v3` establish measured speeds or independent repeat groups on their own. No separate regime-label file was apparent from names alone; this does not prove that no label information exists elsewhere. Use [the first-data preparation procedure](today_server.md) to verify the complete download and inspect actual CSV prefixes on the server.

If regime labels are absent or their meaning cannot be established, mark them unavailable. Any new manual labels or rule-based labels must record their creator/procedure, version, evidence, uncertainty, and independence from the method being tested. A model's own output is not an independent reference label.

## Provenance and immutable originals

For each download retain:

- Requested record ID, resolved version record ID, metadata retrieval time in UTC, source URL, and complete metadata snapshot.
- Exact archive filename, published byte count and MD5, verification result, and resolved-version storage path. MD5 checks data integrity; it is not a security guarantee.
- Dataset license and citation metadata as stated by the source; publishing this repository does not relicense source data.
- Archive member path, source trial/session identifiers where established, schema/unit interpretation, and extraction/conversion code version for each derived file.

Record IDs can resolve to another version ID. For example, the initial query for Part 1 ID `17749110` resolved to `17749111`; retain both rather than assuming the requested ID pins immutable bytes. Start with `python scripts/zenodo_catalog.py catalog`, which retrieves metadata only. Downloading a file and extracting an archive require separate storage planning. Never edit raw originals; write derived data separately.

Initial selection is [Part 1](https://zenodo.org/records/17749110), [Part 2](https://zenodo.org/records/17782542), and [Part 6](https://zenodo.org/records/17795326), whose metadata describe string variation and setup repeats. A filename or folder index alone does not establish statistical independence; map samples and repeated setup measurements first.

## Project-recording requirements

The 2026-09-13 scope uses a final, limited recording session with existing equipment; new force-sensor construction and large-scale own-data collection are no longer required. See [the final violin check](final_violin_check.md). These recordings are supplementary and are not a force-boundary validation dataset.

Each trial needs stable, non-identifying `trial_id`, `session_id`, `instrument_id`, `player_id`, and `bow_id`; string identity, target conditions, available measured conditions, sample rates, time origin/synchronization, microphone setup, and exclusion reason where relevant. Explicitly mark unmeasured force, velocity and β as unavailable rather than treating instructions as measurements. If calibrated sensor data are ever included, retain their orientation and independent calibration records. Keep failed or missing measurements identifiable rather than silently deleting them.

Microphone amplitude is not calibrated pressure unless a pressure calibration was performed. Audio periodicity alone is not proof of Helmholtz motion. Define an acoustic state by a documented, reproducible rule and retain reference-label disagreement. Keep the mapping from participant codes to identities outside this public repository. Do not commit raw participant video/audio, private paths, credentials, or unpublished personal documents.

## Split before deriving training examples

1. Assign raw trials to groups first. Every overlapping window, crop, resampling, augmentation, and paired sensor stream derived from a trial stays in that trial's split.
2. Use whole sessions/dates for testing robustness to recalibration and recording changes. Use held-out instruments, players, string samples, or setup repeats only for the corresponding stated generalization claim. Preserve crossed player–instrument dependence in the evaluation design.
3. Fit normalization, imputation, calibration learned from research examples, label thresholds, feature selection, and hyperparameters only with their designated training/development data. Predetermined physical sensor calibration needs separate provenance and its own independent verification.
4. Freeze the split manifest and evaluation protocol before final testing. Boundary estimation from the same trial's output and boundary prediction before hearing that output are different tasks; record which one is evaluated and which inputs it receives.
5. Report uncertainty using the independent units supported by the design. Thousands of windows do not create thousands of independent trials, and repeated recordings do not increase the number of instruments.
6. Preserve unobserved or censored boundaries, missing conditions, ambiguous states, and disconnected stable regions. Do not use the last sampled force as a measured boundary or force a single interval when the evidence does not support one.

Before model training, verify trial IDs and grouping, channel units and signs, finite values, sample counts and timing, reference-label provenance, condition coverage, and split overlap. The starter does not yet perform this dataset-specific audit; implement it against verified archive contents and project pilot recordings.
