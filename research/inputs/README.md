# Compact derived benchmark inputs

These eight gzip files make the retrospective benchmark executable from a fresh repository clone without the lost server's raw archives or ignored `reports/` directory. Total compressed size is approximately **4.13 MB**. Each is a lossless copy of one saved 6,000-row CSV; gzip timestamps are fixed to zero. `manifest.json` records both compressed and decompressed SHA256 hashes, original relative paths and row counts. The loader verifies the pinned **decompressed** SHA256 before using either the original report or this release copy.

## Attribution and changes

The underlying robotic bowed-cello measurements are by **Alessio Lampis**, University of Music and Performing Arts Vienna, ORCID [0009-0003-0813-9031](https://orcid.org/0009-0003-0813-9031):

- A/B/C sample1: *A collection of bowed responses of cello strings (part 1)*, version record [10.5281/zenodo.17749111](https://doi.org/10.5281/zenodo.17749111).
- D sample1: *A collection of bowed responses of cello strings (part 2)*, record [10.5281/zenodo.17782542](https://doi.org/10.5281/zenodo.17782542).

Both provider records identify **Creative Commons Attribution 4.0 International**, [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Part1 attribution/license was checked against the retained official API metadata; Part2 was checked against the live official API on2026-10-13. This compact derived-table release is provided under CC BY4.0 with the attribution above. No endorsement by the data creator is implied.

The `*_measured_*` tables are **project-derived summaries and provisional rule labels**, not author-provided reference classifications. The project extracted analysis windows, computed signal features, inferred beta-level/force-rank grids, marked velocity-plateau membership, and assigned labels using its own folded-waveform flyback classifier. These are the September2026 outputs retained after the server was lost; they are preserved exactly, including NaN diagnostics and ambiguous labels. See `research/2026-10-13/evidence_audit.md` and `scripts/regime_map.py`.

The `*_simulated_*` tables are project-generated historical digital-waveguide outputs on the same trial IDs. They used target-derived nuisance parameters and contain results from the numerical implementation subsequently found defective. They are included solely to reproduce the **historical diagnostic** comparison; they are not validated physical ground truth or a fair zero-shot simulator baseline. The October benchmark separates these diagnostics from its main model rankings.

No raw waveforms, microphone recordings, personal participant data, private machine paths, credentials or model checkpoints are included here. The trials concern robotic cello strings on a rigid monochord, not an acoustic violin. A/B/C/D refer to string designs/specimens, not four different instruments.

## Reproduce

From the repository root with the Mac Python environment described in the main documentation:

```bash
PYTHONPATH=scripts .venv-mac/bin/python scripts/benchmark_data.py --include-ambiguous
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 PYTHONPATH=scripts \
  .venv-mac/bin/python scripts/benchmark_sets.py --source A --out reports/research_20261013/reproduction_A
```

Missing `reports/collection/...` and `reports/simulation/...` input CSVs are resolved to these gzip copies. A present report with a changed checksum is rejected; the loader does not silently fall back to hide a mismatch. The raw24,000rows become23,301positive-force plateau rows, retaining211ambiguous weak labels. All data have already been inspected during development; a fresh clone does not make them an untouched test set.
