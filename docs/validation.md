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
