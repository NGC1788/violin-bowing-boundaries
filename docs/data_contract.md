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

### Observed CSV contents, 2026-09-17

Source: a user-supplied server transcript. `prepare-first` at commit `4102225` reported PASS: it used a validated saved metadata snapshot, found the existing archive with the published size and matching MD5, and reused an earlier completed extraction after matching member names and sizes. The observations below come from **trial 1 of each folder** plus **all 6,000 `beta_N.csv` and `timestamp_N.csv` files**. They are not yet a whole-signal audit; `bash scripts/run.sh trial-audit` checks every trial.

- `whole_N.csv`: four numeric columns, **no header row**, **CRLF line endings** (`beta_1.csv` holds a 17-character value in 19 bytes). Trial 1 has 258,001 / 158,001 / 108,001 rows in `r_v1` / `r_v2` / `r_v3`, i.e. exactly 5.16 / 3.16 / 2.16 s if the published 50 kHz rate applies. The files do not state a sampling rate.
- `timestamp_N.csv`: one CRLF row `start,end` of integer sample indices. Across all 2,000 trials per folder the width `end - start` is only 19,997 or 19,998 (≈ 0.4 s at 50 kHz), while the start varies (r_v1 109,803–183,202; r_v2 61,203–107,402; r_v3 42,002–70,902). This is consistent with the published description of a classification window. **It is not a state label, and no state-label file has been found.** Whether `end` is inclusive is not established; the audit treats `[start, end]` as inclusive 0-based indices.
- `beta_N.csv`: one value per trial. Every folder spans 0.0200–0.2001 with 41–45 distinct values at three decimals, consistent with the Part 1 β range. Individual values deviate slightly from round nominal levels (e.g. 0.200108, 0.200091, 0.200041).
- Column 2 (published: bow velocity) in trial 1 starts at exactly `0` and peaks at **0.0500 / 0.1000 / 0.2000** in `r_v1` / `r_v2` / `r_v3`. Its mean inside the classification window is 0.0500 / 0.1000 / **0.1879**.
- **Conflict with the metadata mapping.** The metadata states folders 1, 2, 3 correspond to 0.1, 0.05, 0.2 m/s. Measured column 2 instead orders `r_v1` < `r_v2` < `r_v3` as 0.05 < 0.1 < 0.2, and trial durations are ordered consistently with the measured values. Use measured column 2, never the folder name or the published mapping, as the velocity condition. This rests on trial 1 per folder until the full audit confirms it for all trials.
- In `r_v3` trial 1 the window-mean velocity is below the peak, so **a classification window is not guaranteed to lie on a constant-velocity plateau.** Check this per trial before treating window statistics as steady-state conditions.
- Column 1 (published: bow force) in trial 1 ranges about 3.9–5.1 and is exceptionally smooth; adjacent samples can differ by ~1e-12, which suggests a filtered or derived signal rather than raw sensor samples. The first rows of columns 3 and 4 looked stepped (differences near multiples of ≈0.0032 and ≈0.0011456). **This does not establish quantization:** the audit below found thousands of distinct column-4 values per window, so a continuous component is present. An earlier version of this section overstated quantization.

#### First 20 trials per folder (`trial-audit --limit 20`)

- 60 of 60 trials readable, no orphan companion files, every window within its trial.
- Trial length varies within a folder by 18,000 rows: r_v1 240,001–258,001; r_v2 140,001–158,001; r_v3 90,001–108,001.
- The column-2 peak is exactly 0.05 / 0.1 / 0.2 in all 20 trials of r_v1 / r_v2 / r_v3, consistent with the measured mapping above.
- β is 0.2001 (0.2000–0.2001 in r_v3) in all of these trials while the column-1 window mean takes 20 distinct levels (about 0.60–4.15 in the published unit N). Trials therefore appear to sweep force within a β level. The complete (β, force) grid is not yet established.
- Median distinct column-4 values in a ≈20,000-sample window: 2,424 / 5,227 / 8,252.
- **Windows off the velocity plateau.** All 20 windows lie inside the column-2 plateau (1% tolerance) in r_v1 and r_v2, but only 9 of 20 in r_v3, where the smallest steady fraction is 0.4984 and the window-mean column 2 falls to 0.1789. At 0.2 m/s, window statistics must not be treated as constant-velocity conditions without per-trial selection.

#### All 6,000 trials (`trial-audit`)

- PASS: 6,000 of 6,000 trials readable, no orphan companion files, every window within its trial.
- The column-2 peak is exactly 0.05 in all 2,000 `r_v1` trials, 0.1 in all `r_v2` trials and 0.2 in all `r_v3` trials. **The measured folder-to-speed mapping holds for every trial; the published mapping does not.**
- Trial length: `r_v1` 236,001–352,001 rows; `r_v2` 138,001–202,001; `r_v3` 90,001–128,001.
- β rounded to three decimals gives 45 / 41 / 45 bins with a median of 50 trials per bin and minima of 2 / 3 / 1. That is the pattern 40 levels × 50 trials would produce if a few levels straddle rounding boundaries, but rounding cannot establish it. `grid-summary` separates levels at the natural break between jitter and level spacing instead.
- Column-1 window means range from −0.065 to 4.21 with medians of 0.32–0.38. **Some are negative.** A pressing bow force cannot be negative, so at the lowest levels the recorded value carries an offset, drift or loss of contact comparable to the intended force. Do not use absolute column-1 values there without checking.
- Windows not fully on the column-2 plateau (1% tolerance): 22 / 135 / 364 of 2,000 (1.1% / 6.8% / 18.2%); minimum steady fraction 0.82 / 0.19 / 0.49; minimum window-mean speed 0.0497 / 0.086 / 0.178. If these trials cluster in one part of the (β, force) plane, excluding them biases boundary estimates there. `grid-summary` maps them.
- The smallest gap between distinct column-4 window values has a median of 3.31e-09 in every folder, consistent with the text precision of stored values rather than a sensor resolution.
- The median column-3 window standard deviation rises with speed: 0.236 / 0.394 / 0.484. This is directionally consistent with column 3 being a string force whose oscillation grows with bow speed. It pools all states and does not validate units or scale.

If regime labels are absent or their meaning cannot be established, mark them unavailable. Any new manual labels or rule-based labels must record their creator/procedure, version, evidence, uncertainty, and independence from the method being tested. A model's own output is not an independent reference label.

### Provisional rule-based labels, 2026-09-17

No state-label file exists in the first archive. The dataset authors classify with the Woodhouse/Galluzzo detrended-staircase algorithm (van Walstijn et al., Acta Acustica 2026, Sect. 3.2.2, citing Lampis et al., ISMRA 2025). Their per-trial labels are not published. `scripts/regime_map.py` assigns **provisional** labels from the shape of column 3 inside each classification window:

- Periodicity: the autocorrelation peak within 40–160 Hz.
- Flybacks per period: counted on the mean cycle folded at that period.
- Oscillation floor: 3 × the median window std of column-1 ≤ 0 windows. This single global constant is the only use of column 1.

Thresholds are recorded in every report and documented in [the regime-map guide](regime_map.md). Created by this repository's code; version is the commit that produced the report. Uncertainty is kept as `ambiguous`, and boundary fits treat runs near the grid edge as censored. A label never uses its own trial's column 1, column 2 or β, which later models take as inputs. The flyback law check uses column 2 and β only after labelling. These are not published reference labels.

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

Before model training, verify trial IDs and grouping, channel units and signs, finite values, sample counts and timing, reference-label provenance, condition coverage, and split overlap. `trial-audit` now performs the structural part of this audit (files, companion pairing, finiteness, column count, window bounds, per-trial channel ranges and velocity-plateau coverage). Reference labels, units, sign conventions, trial grouping and split design are not yet established.
