# Validation record

Initial checks: 2026-09-07. Updated data preparation: 2026-09-13. This distinguishes local preparation checks from a subsequent user-provided Ubuntu server transcript. Neither is an experimental research result.

## Local preparation checks

| Check | Observed result | Scope |
|---|---|---|
| `uv lock --check` | PASS; 153 packages resolved | Linux x86_64 dependency metadata and lock consistency; not a Linux installation |
| `bash -n scripts/bootstrap_ubuntu.sh` and `bash -n scripts/run.sh` | PASS | Shell syntax; bootstrap not executed on the remote server |
| `python3 -m unittest discover -s tests -v` | PASS, 20 tests | 9 mocked downloader tests + 11 mocked environment/report tests, including distinct installation/runtime disk thresholds |
| `python scripts/zenodo_catalog.py catalog` | PASS, three official API records | Actual HTTPS metadata retrieval for Parts 1, 2, 6; no large archive downloaded |
| `python scripts/smoke_test.py` | PASS | Synthetic audio, image, table, classification fit, and plot IO on macOS / Python 3.11.15 |
| `python scripts/doctor.py` | CPU operations PASS; overall WARNING | macOS, no CUDA: actual PyTorch forward/backward, GPyTorch covariance, TorchVision NMS and TorchAudio resampling |

The local CPU run used torch 2.14.0, torchvision 0.29.0, torchaudio 2.11.0 and gpytorch 1.15.2. It used macOS wheels in a separate test environment, **not** the Linux CUDA wheels from the server lock.

Synthetic PCM-24 WAV round-trip maximum absolute error was approximately `1.1882e-7` for a 0.2-amplitude sine. The resampled sine peak was 440 Hz and mel output shape was 40 × 101. These are software checks with known synthetic input; they do not validate pitch estimation, violin stability labels, sensor accuracy, or a scientific model.

The TorchAudio transform emitted an upstream TorchScript deprecation warning. The tested resampling operation passed. This does not justify enabling compilation or changing the server environment without further checks.

## Ubuntu GPU run reported by the user

The user subsequently supplied a terminal transcript showing successful installation, `SETUP PASS`, and a separate repeat doctor run with `CUDA execution verified: True`. This is evidence from the supplied transcript; the maintainer did not remotely execute these commands or independently inspect the saved server JSON files. Private account names, machine identifiers, paths and raw logs are not reproduced here.

- Target: Linux x86_64, NVIDIA RTX A5000, driver 595.84.
- Environment: Python 3.11.15, torch 2.14.0+cu126, torchvision 0.29.0+cu126, torchaudio 2.11.0+cu126, gpytorch 1.15.2; uv 0.12.6 installed the lock successfully.
- Actual CUDA forward/backward and synchronization: PASS. TorchVision NMS on the CUDA device: PASS.
- GPyTorch CPU covariance and TorchAudio CPU resampling: PASS. These checks do not establish GPU GP training performance.
- Synthetic WAV, resampling, mel features, image/table IO, classifier fit and headless plotting: PASS.
- Metadata catalog for Parts 1, 2 and 6: PASS. No raw archive was downloaded in the supplied transcript.

The initial Ubuntu setup gate is therefore passed for this reported run. Preserve the original reports locally; the next step is [inspection of one archive](first_archive.md).

## Still unverified

- Whole-signal audit results, units, sign conventions, reference labels and trial grouping. The first archive's download, MD5 check and extraction were reported PASS on 2026-09-17 (below); a whole-trial structural audit tool exists but its server results are pending.
- Video tracking, synchronization or real violin acoustic-state validation. New force-sensor construction/calibration is outside the revised 2026-09-13 scope; see the research roadmap.
- Training, benchmark comparisons, uncertainty calibration, or claims of improvement.

For a new server installation, run `bash scripts/bootstrap_ubuntu.sh` and preserve its environment reports. `SETUP PASS` means only that its installation and smoke checks passed; it is not a research result.

## Updated first-data preparation checks — 2026-09-13

The user reported an interrupted first-file download at 3,816,485,424 of 14,083,546,513 bytes. Completion of that download has not subsequently been reported. The new downloader retains `.part` files, retries only transient transport failures, verifies exact HTTP resume ranges, and checks the full published MD5 before publishing the final file.

- Local Python 3.11.15 regression suite: PASS, 59 tests, covering bounded retry/resume, permanent-error rejection, environment checks, archive inventories, extraction preservation/recovery, bounded CSV inspection, and saved failure/report behavior.
- A real tiny solid 7z fixture was created, inventoried, extracted, checked against expected bytes, scouted and reused using 7-Zip 26.03 on macOS. Disk-space conditions were controlled in the test. This checks the code path and format handling, not Linux throughput or the integrity of the large dataset.
- A bounded HTTP Range read of the official first archive transferred 253,329 header bytes. The resulting member listing was checked by 7-Zip and parsed by `archive_audit.py`: 18,000 files, 4 directories, 74,179,785,108 uncompressed bytes; one solid block; inventory path guards passed. Each of `whole_N.csv`, `beta_N.csv`, and `timestamp_N.csv` has 6,000 members. See the [data contract](data_contract.md) for the observed directory layout.
- The header-only inspection did not retrieve, decompress, or hash the 13.116 GiB payload and did not inspect any actual CSV values. The server preparation command must still perform full MD5 verification and extraction before its CSV reports are evidence of real file contents.

`TODAY DATA PREPARATION: PASS` will mean the stages actually requested by that run completed. With `--extract`, those stages include successful 7z extraction, exact extracted file names/sizes, and readable nonempty prefixes of selected CSV files. It does not mean that all signal values, units, labels or research claims have been validated. Reusing an extraction checks names and sizes against its completion record; it does not freshly hash each extracted file. Preserve the raw archive for reproducible reconstruction.

## Installation without sudo — 2026-09-14

The school account does not have administrator privileges. `setup-tools` now installs official 7-Zip 26.03 for Linux x86_64 under the project's ignored `.local-tools/` directory. `run.sh` adds that directory to the child process PATH; no system package installation or shell-profile edit occurs. The guide uses Ubuntu's `nohup` instead of requiring tmux.

The official Linux release asset is 1,575,072 bytes with SHA256 `dc99eff5008f1ab79bd7084c68513701547a808a89502bf4133683535ab3c695`, matching the GitHub release-asset metadata. The extracted regular `7zz` member is 2,882,120 bytes with SHA256 `3d52c92deb7e9f1bd059692eefc33f86a144cdc548acc4b6f4c809a4dd7bc369`. Both are pinned in the installer. The actual official package was downloaded and checked locally; its Linux executable was not run on macOS. Server setup must still pass its real executable/version check.

The local regression suite passes 67 tests, including eight new installer tests for platform rejection before download, size/hash checks, duplicate/symlink rejection, installation/reuse, altered existing files, and failed execution preservation. Linux execution in those installer tests is mocked; the existing real macOS 7-Zip tiny-archive integration test also passes. Shell syntax for the revised `run.sh` passes.

## User-reported Linux success and metadata outage — 2026-09-14

The user subsequently provided a server transcript at commit `836fe1f`: official user-space 7-Zip setup passed its real executable check, all 67 tests passed (including the real tiny-archive extraction test), and the filesystem reported approximately 162 GiB available. These observations establish the no-sudo setup on that reported server run.

The same transcript shows the full-data preparation failed before payload download: the metadata API returned HTTP 504 after eight retries. No full dataset preparation success was reported. A subsequent bounded check from the maintainer's separate connection received HTTP 200 for record metadata and HTTP 206 for the first 32 archive bytes; this does not establish availability from the school's network or completion of the file transfer.

`prepare-first` now reuses a previously saved official snapshot only after checking its fixed requested/resolved version, record ID, source URL, schema, filename, size and MD5. It retains the original retrieval time, records the metadata used, and invokes the existing downloader with that same validated object. It no longer makes a second metadata request after download. The generic catalog/download CLI still queries live metadata. Raw-file MD5, resume-range, size and extraction checks remain required.

Local regression suite after this change: PASS, 75 tests. New checks cover zero metadata requests with a valid cache, malformed/oversized/mismatched snapshots, live fallback and pin validation, removal of the second API request, and an end-to-end orchestration check that accepts a checksum-matching existing tiny file without any network but rejects changed contents of the same byte count. Full-data preparation still awaits a successful server run.

## First real data and trial audit — 2026-09-17

**Server transcript (user-supplied, commit `4102225`).** `prepare-first --extract` reported `Using validated saved metadata`, `Already present and MD5 verified` for the 14,083,546,513-byte archive, `Inventory PASS: 18000 files; 69.085 GiB uncompressed; Solid=+`, reuse of an existing completed extraction, and `TODAY DATA PREPARATION: PASS`. The filesystem then reported 93 GB available (90% used). Follow-up read-only shell inspection produced the CSV observations recorded in the [data contract](data_contract.md), including a conflict between measured column-2 speeds and the published folder-to-speed mapping. The maintainer did not independently access the server.

**New tool.** `scripts/trial_audit.py` (`bash scripts/run.sh trial-audit`) reads every `whole_N.csv` once and writes `reports/trial_audit/<run>/trials.csv` (one row per trial) and `summary.json`. It fails on missing or orphan companion files, unreadable or non-four-column signals, non-finite values and out-of-bounds windows. A window that is not fully inside the column-2 velocity plateau is reported as a warning, not a failure. It writes only small reports and never copies signal data.

**Local checks (macOS, Python 3.11.15, numpy 2.4.6, pyarrow 25.0.1 test environment).**

| Check | Result | Scope |
|---|---|---|
| `python -m unittest discover -s tests` | PASS, 91 tests (1 skipped: no 7-Zip) | 75 existing + 16 new trial-audit tests on synthetic trials with exactly known plateau, window, force, quantization step and file structure |
| Deliberate code mutations: plateau flag forced true, orphan files ignored, off-by-one window bound, lexicographic trial order, non-finite check removed | Each made the new tests FAIL; original restored and re-verified | Shows the tests detect these errors; not proof of absence of others |
| pyarrow vs numpy parser on a 258,001-row CRLF signal | Bit-identical arrays | Parser agreement on synthetic decimal text |
| Parse + per-trial analysis, 14.9 MB synthetic trial | 74 ms (pyarrow), 269 ms (numpy) single-threaded | Mac M2 timing only; not server throughput |
| `bash -n scripts/run.sh` | PASS | Shell syntax after adding `trial-audit` |

These checks establish that the audit code computes the intended quantities on known inputs. They do not establish anything about the real dataset until the server run is reported.

### Server results and a correction — 2026-09-17, later

User transcript at `5f12fec`: fast-forward pull; **91 tests OK** in the server's locked environment; `trial-audit --limit 20` **PASS** for 60/60 trials with the pyarrow parser and 8 workers. Its observations are recorded in the [data contract](data_contract.md).

The same output exposed two defects, both fixed:

- **Reporting.** The column-4 smallest-gap median was stored with 7-decimal rounding and displayed as `0.0`. Summary quantities of this size now keep full precision and the display uses significant figures. The per-trial `trials.csv` was already unrounded.
- **Documentation.** This repository had stated that column 4 is quantized in ≈0.0011456 steps, based on a few rows. Thousands of distinct values per window contradict that; the statement is corrected.

A per-β-level trial count was added to check grid coverage. Local suite: **PASS, 93 tests** (1 skipped without 7-Zip). Two new deliberate mutations (forcing the old rounding; counting each β level once) were each detected by the tests.

### Full trial audit and grid reconstruction — 2026-09-17, later

User transcript at `c69ae53`: `trial-audit` **PASS for 6,000/6,000 trials**. Observations are recorded in the [data contract](data_contract.md).

**New tool.** `scripts/grid_summary.py` (`bash scripts/run.sh grid-summary`) reads only `trials.csv` from the latest PASS audit and opens no signal file. It separates β levels at the natural break between within-level jitter and between-level spacing, ranks measured column-1 window means within each level, checks whether trial numbers sweep β and force in order, and maps off-plateau and non-positive-force windows onto the reconstructed grid. It writes `grid.csv`, `summary.json` and `grid.png`. The reconstruction is inferred from measured values, not taken from a published design.

| Check | Result | Scope |
|---|---|---|
| `python -m unittest discover -s tests` | PASS, 102 tests (1 skipped: no 7-Zip) | 9 new tests: a planted 4 × 5 grid including a β level that straddles a three-decimal rounding boundary, known sweep order and problem cells; command-line report and table; plot output; a pipeline test feeding real `trial_audit.py` output into `grid_summary.py` |
| Deliberate mutations: break detection disabled; force ranks reversed; contiguity check off by one; three-decimal rounding in place of the natural break; numpy scalars written unconverted | Each detected by the tests | Shows detection of these errors only |
| Plot of a synthetic 3 × 40 × 50 grid with a planted problem region | Inspected visually; region and markers rendered as planted; overlapping labels fixed | Synthetic data only |

A defect was caught during development: with numpy 2, `repr(np.float64(x))` is `np.float64(x)`, which would have written unparseable text into `grid.csv`. Values are now converted to Python scalars and the tests parse every numeric field. `trial_audit.py` already wrote Python scalars; the pipeline test confirms its table is readable.

### Provisional regime labels and boundary maps — 2026-09-17, later

User transcript at `7843620`: `grid-summary` reconstructed **40 β levels × 50 trials per level in each of the three speed folders** (break ratios 49, 48, 79), β swept in descending order by trial number with contiguous trial numbers in every level, and force descending within every β level. Off-plateau windows cluster at high force and intermediate β; non-positive column-1 windows cluster at the lowest force ranks. Details are in the [data contract](data_contract.md).

**New tool.** `scripts/regime_map.py` (`bash scripts/run.sh regime-map`) assigns provisional labels from column 3 and maps them over the reconstructed grid, fits lower and upper Helmholtz boundaries on log–log axes with standard errors, and saves `regimes.csv`, `summary.json`, `regime_map.png` and `examples.png`. See the [guide](regime_map.md).

| Check | Result | Scope |
|---|---|---|
| `python -m unittest discover -s tests` | PASS, 114 tests (1 skipped: no 7-Zip) | 12 regime-map tests: sawtooth period and periodicity; one slip per period under smoothing, noise and ringing; two slips per period; five planted regimes; Schelleng grids with known slopes and boundary positions; interior misclassifications; misclassified edge cells; censoring; slope standard error; an audit → grid → regime pipeline with planted labels |
| Deliberate mutations: slip grouping disabled; subharmonic rule disabled; amplitude floor disabled; censoring disabled; boundary placed one rank off; Helmholtz slip band widened; interior bridging disabled | Each detected | The slip-grouping mutation initially passed because the ringing fixture never split into separate runs; the fixture was strengthened and a guard asserts that it splits without grouping |
| Synthetic 3 × 40 × 50 map with planted slopes and 3% interior misclassification | Recovered −1.99±0.02, −2.00±0.01, −2.01±0.01 (lower) and −1.00±0.01, −1.01±0.03, −1.02±0.20 (upper); the last uses only 5 uncensored levels | Synthetic data only |

Defects found and fixed during development, each first reproduced by a failing test:

- **Split runs.** Using the longest strictly contiguous Helmholtz run let one misclassified interior cell move a boundary; slopes on the synthetic map came out near −1.1 instead of −2. Gaps of up to two ranks are now bridged.
- **Edge holes.** A misclassified cell at the highest force rank made a boundary lying above the sampled range look uncensored, giving an upper slope of −0.36. Runs ending within the bridging distance of a grid edge are now censored.
- **Plot claims.** Fitted lines were drawn across the full β range, beyond the fitted data, and the legend covered the low-force, large-β corner. Lines are now drawn only across the fitted β range and legends sit below the axes.
- A wrong module reference in the summary code was caught by the pipeline test before any server run.

These checks show that the tool recovers known answers on synthetic signals and grids. They do not show that the provisional thresholds classify the real bridge-force signals correctly.

### First real regime map failed; flyback counting replaces slip runs — 2026-09-17, later

User transcript at `0afda0a`, run `20260917T104233Z_0e8ffc33`: f0 detection was correct (reference 98.05–98.11 Hz), but only 39 / 125 / 120 trials per folder were labelled Helmholtz and 1,473 / 1,344 / 1,276 ambiguous. Boundary slopes were not interpretable. Among periodic trials, slips per period were 0–0.1 in 3,203, 0.75–1.25 in only 284, and 2–4 in 1,035.

**Evidence from the real waveforms (`examples.png`).** Unmistakable clean sawtooths, with one sharp return per 10.2 ms, scored 0.03, 0.61, 1.65, 1.84 and 2.83 slips per period. Low-force windows at the ±0.1 level were visibly noise without a sawtooth, but were labelled aperiodic or ambiguous because the 0.02 amplitude cut lies below that level.

**Cause, reproduced synthetically.** A per-sample `|diff|` threshold at 25 % of the per-period maximum is crossed by sample noise along the ramp whenever the return is spread over many samples. Runs closer than a tenth of a period were chained, so one noisy window collapsed into a single event. On a sawtooth with a 1.5 return, 21-sample smoothing and noise 0.02, the old counter gives 0.03 for 5 of 5 seeds. That is the same value as the real clean sawtooths.

**Published method.** The dataset authors assign regimes with an algorithm from Lampis, Chatziioannou and Scavone (ISMRA 2025), based on Galluzzo (PhD thesis, 2004). In their description, the bridge force is detrended into a step function and the spacing of histogram peaks is compared with the theoretical Helmholtz flyback (van Walstijn et al., Acta Acustica 2026, Sect. 3.2.2; Lampis et al., Acta Acustica 2024, Sect. 2.3). Their categories are Helmholtz, double slip, raucous, multiple slip, S-motion, multiple flyback and unclassified. No per-trial labels are in the archive.

**Change.** Each window is folded at its dominant period into a mean cycle (fractional period cuts). The cycle is split into legs at turning points with hysteresis of 3 standard errors of the mean cycle. Flybacks are legs in the steepest tall direction; a flyback counts if it is at least half the largest. Helmholtz means exactly one flyback. The second-largest / largest ratio is saved, and boundary slopes are reported for flyback fractions 0.3–0.7. The oscillation floor is 3 × the median window std of column-1 ≤ 0 (no-contact) windows, with 0.02 as fallback and lower bound. Labels still use column 3 shape only. As an independent check that is not used for labelling, the Helmholtz flyback law ΔF = 2 Z v_b / β is fitted on Helmholtz-labelled trials. The ideal exponents are +1 for speed and −1 for β. The result is compared with Z = 1.175 kg/s derived from Table 1 of van Walstijn et al. (2026), which may describe a different string.

| Check | Result | Scope |
|---|---|---|
| `python -m unittest discover -s tests` | PASS, 122 tests (1 skipped: no 7-Zip) | New: spread return under sample noise (the real failure); hysteresis under heavy noise, with a guard that the fixture splits without it; ripple and moderate ringing; rising returns; double and triple slipping with phases; the second ratio deciding the count at any fraction; fractional-period folding; noise floor from no-contact windows; flyback-law exponents and impedance; fraction sensitivity matching labels; pipeline with planted Helmholtz, aperiodic and no-contact trials |
| Deliberate mutations: hysteresis removed; flyback sign flipped; legs counted in both directions; integer-period folding; floor factor ignored; floor not applied in classification; sensitivity restricted to Helmholtz labels; β sign flipped in the law fit; second ratio taken from the smallest leg | 9 of 9 detected | — |
| Example panels rendered with the mean cycle and counted flybacks overlaid | Markers at each return for one, two and three flybacks, including a rising-return case | Synthetic signals only |

Heavy ringing (0.35 of a 0.8 return) now counts as two flybacks. The earlier fixture was designed to break run grouping and is replaced by moderate ringing (0.15). Whether such ringing occurs in the real signals must be checked in the new `examples.png`.

These checks show that the new feature recovers known answers on synthetic signals. Whether it classifies the real bridge force correctly must be judged from the next server run's `examples.png`, `features.png` and flyback-law check.

