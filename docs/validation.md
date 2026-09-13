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

- Full published-archive download/extraction on the server, schema/label verification, or whole-signal dataset-specific parsing. A small fixture extraction and real archive header inspection are described below.
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
