# Validation record

Date: 2026-09-07. This records checks actually run during repository preparation, not experimental results.

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

Not yet performed:

- Actual Ubuntu installation and A5000 CUDA execution.
- Archive extraction, schema/label verification, or dataset-specific parsing.
- Sensor calibration, video tracking, synchronization or real violin acoustic-state validation.
- Training, benchmark comparisons, uncertainty calibration, or claims of improvement.

On the server, run `bash scripts/bootstrap_ubuntu.sh` and preserve its environment reports. `SETUP PASS` means only that its installation and smoke checks passed; it is not a research result.
