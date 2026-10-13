# Evidence audit — 2026-10-13

This audit inspected saved CSV/JSON artifacts and the actual Python code on the Mac. No raw waveform was downloaded or reconstructed. Existing artifacts are development evidence, not a newly untouched test set. Original reports were not modified.

## What is actually available

Exactly four measured `sample1` diagrams have backed-up per-trial labels: A, B, C and D; each contains three speed panels × 2,000 trials = 6,000 rows. Each speed panel is a 40 beta-level × 50 force-rank grid. The four selected CSVs contain 24,000 unique global trial IDs and 24,000 distinct specimen/condition/beta-level/force-rank cells.

| Specimen label | Raw rows | Positive force and plateau rows | Ambiguous among eligible | Helmholtz among eligible |
|---|---:|---:|---:|---:|
| A | 6,000 | 5,325 | 50 | 1,827 |
| B | 6,000 | 5,976 | 25 | 3,080 |
| C | 6,000 | 6,000 | 74 | 3,229 |
| D | 6,000 | 6,000 | 62 | 3,171 |
| Total | 24,000 | 23,301 | 211 | 11,307 |

There are 521 off-plateau rows, all in A, and 178 nonpositive-force rows (A154, B24). These two exclusions do not overlap in these saved tables. Excluding the 211 ambiguous eligible rows leaves 23,090 rows; preserving them explicitly as unknown targets leaves 23,301.

The measured velocity verifies `v1=0.05`, `v2=0.1`, `v3=0.2 m/s` for all four diagrams (A names have the date prefix). Some A windows are outside the steady-speed plateau; their window means must not be substituted for genuine constant-speed trials without qualification.

Raw CSVs and cached windows/profiles are absent. `regimes.csv` nevertheless preserves beta, force mean, velocity mean, inferred grid position, plateau flag, amplitude, periodicity, pitch, flyback count, second-flyback ratio and several other output features. Thus **ordinary tabular baselines and uncertainty-aware boundary methods can run now without a raw download**. Waveform-level audits, new hysteresis settings, new period estimators, time-resolved features, and reproducing the driven simulator need waveform/profile recovery.

## Priority 1: claims that must change

### Current physics results are target-adapted

`calibrate.load_cells` reads impedance from the diagram-wide flyback-law summary, noise/floor from its diagram-wide no-contact summary, and reference pitch from each condition. These summaries were computed using measured outputs across all three speeds before the train/hold-out selection.

`simulate_grid.run` repeats that process for each target specimen. B/C/D predictions therefore use their own measured pitch and impedance; B also uses target-measured observation noise. The impedance estimate is especially outcome-dependent: it is the median of `flyback_height * beta / (2 * velocity)` **among trials labelled Helmholtz**. This is a legitimate target adaptation protocol if stated explicitly, but it does not support “no target data used” or “fully untouched transfer.”

Saved nuisance settings:

| Diagram | Target-derived Z (kg/s) | Observation-noise standard deviation |
|---|---:|---:|
| A | 0.9463 | 0.068045 |
| B | 0.9450 | 0.147751 |
| C | 0.9563 | 0.0 (fallback when no controls) |
| D | 0.9584 | 0.0 (fallback when no controls) |

Zero here means **missing control-based estimate replaced by zero**, not experimentally noiseless data. No-contact windows are inferred from negative mean force; they are not independently identified blank recordings. The historical outputs should be shown in a separate “target-adapted physics diagnostic” row, not used as an unqualified fair comparator against a model with only source training data.

### Calibration and full-grid evaluation did not use identical fitted classifiers

Calibration labels simulations with measured floor and measured reference pitch. The full-grid simulation instead calls `regime_map.run`, which recomputes the floor and reference pitch from the synthetic signals. For B, the synthetic floor becomes **3.8900809**, versus measured **0.4432525**, and synthetic v3 has no estimated reference pitch. For calibrated A, synthetic floor is **0.2962004**, versus measured **0.2041361**.

Re-labelling the backed-up simulated features using nuisance settings fit only on A/0.1 fixes this diagnostic discrepancy without raw data. On the default 23,090 eligible unambiguous rows, it changes **651 multiclass labels but zero binary Helmholtz predictions**. Hence this is a real multiclass comparability defect, **not an explanation for an improved Helmholtz IoU**. The AND amplitude/periodicity gate is why noise-floor changes mainly redistribute non-Helmholtz categories.

### Labels are provisional outputs of an evolving rule

The original authors' per-trial reference classes are not present in the local artifacts. These labels are not independent ground truth. The code was revised repeatedly after examining real maps and example signals; thresholds are therefore historical development choices. A flyback-law consistency check uses related quantities from the same measured waveforms, so it supports plausibility but is not an independent confusion-matrix estimate.

For new methods, say “predicting an operational Helmholtz indicator” or “weak-label boundary inference.” The label-rule family can be varied using the saved second-flyback ratio and periodicity; re-running waveform fold/hysteresis settings cannot be done from these summaries. Robustness across a rule family does not by itself give calibrated coverage of the true physical boundary.

### Current transfer results are retrospective

All four diagrams, their maps, and their physics scores have already been examined. Do not retroactively designate any of them an untouched test set. Fixed splits/hashes now are useful for reproducibility and preventing further accidental tuning, but cannot remove earlier exposure. Stronger future validation requires a previously unused public specimen/archive, with code, hyperparameters and decision rules frozen first.

## Priority 2: evaluation and accounting defects

- The calibration JSONL has **1,500 evaluations**, not the earlier partial count 1,424. Maximum training IoU is **0.7382**, index550, on500 sampled A/0.1 cells. The saved held-out score is0.6611 on1,000 A/other-speed cells. There are **11 visible hold-out reports**, one for each new training-best set. Training selection itself maximizes the training metric; repeated hold-out reporting introduces opportunity for subsequent human adaptation and cannot justify “the gain is not fitting noise.”
- Search rate was **50kHz**; reported full grids use **100kHz**. The stored500-cell and6,000-cell metrics differ in rate, sample membership, and full-grid classifier nuisance fitting. Their difference cannot be attributed only to generalization. The saved two-point50/200kHz check does not establish convergence across a boundary surface.
- A has **two identical calibrated full-grid runs** (`20260919T060704Z_47169eda`, `20260919T063758Z_d9d6ed1a`). They are duplicates, not independent replicates. Select one explicit run. `latest_run` is unsuitable for a benchmark because the latest A run is the literature baseline, not calibrated physics.
- The literature full-grid baseline exists only for A: IoU0.4391; calibrated A0.5969. B/C/D calibrated IoUs0.6337/0.7347/0.7361 have no paired literature baseline in the backup. Do not claim all-specimen improvement over physics baselines.
- `simulate_grid.compare` and calibration scoring include nonpositive force and off-plateau windows. The existing boundary slope code excludes off-plateau cells, so those metrics use different admissible populations. Report explicit denominators for each endpoint.
- The handoff statement “remaining11panels18,000strokes” is arithmetically inconsistent. Of12panels, reserving1full panel leaves22,000raw rows;18,000 is only B/C/D. The actual calibration used500of the2,000A/0.1cells. With the current static eligibility filter, A/0.1 has1,750rows (1,742if excluding ambiguous).
- `overnight.sh` selects the maximum IoU over **all** calibration JSONLs, irrespective of diagram, sampling or rate. There happens to be one JSONL now, so its current choice is recoverable, but future runs need a single explicitly bound manifest.
- Calibration JSONL rows do not store train IDs, selected levels, data hashes, nuisance settings, seed, code hash or rate. The filename and retained script support reconstruction, not a complete self-contained execution record. The10beta levels implied by `overnight.sh` are0,4,8,13,17,21,26,30,34,39, with all50force ranks. Treat this as a reconstructed protocol, not freshly verified historical execution arguments.
- Boundary slopes use geometric midpoints and hole bridging (`max_hole=2`). Multiple disconnected regimes are collapsed to the widest run. The stored point estimates are not true exact boundaries, and the regression standard errors omit spatial dependence, force measurement uncertainty, label-rule uncertainty and censoring selection. Preserve interval bounds/censoring in a new method.
- Near-boundary best search parameters can motivate extending a search box but do **not prove** that the true optimum lies outside it. Existing documentation overstates that inference.

## Independent units and scientifically defensible splits

Each trial is one robot stroke at a chosen grid cell. The 50force values within a beta column are adjacent conditions in a sweep; they are not independent remeasurements. The three speed panels share a specimen. These backups contain four `sample1` string specimens with one retained acquisition diagram each, not24,000independent instruments and not12independent strings.

For a source A/0.1 training protocol, split complete beta columns within that panel into training/tuning/calibration blocks. Keep all cells in a selected column together. Report results separately for each of the11other specimen/speed panels; identify same-specimen speed transfer and cross-specimen transfer separately. A block bootstrap of beta columns measures conditional grid-sampling variability, not population-level generalization over new strings. Four specimens give limited information for the latter; do not create narrow cross-string confidence intervals by bootstrapping individual cells.

If source-only prediction is claimed, learned normalization, label nuisance parameters, hyperparameters and simulation calibration must use source rows only. The measured target F/v/beta can enter X because they are the conditions whose response is being predicted; target bridge-force amplitudes, pitch estimates, periodicity, derived impedance and labels cannot enter prediction. `beta_level`, force rank, trial order and specimen identity are grouping variables, not default predictive features.

## Implemented frozen data interface

`scripts/benchmark_data.py` now exposes:

```python
data = load_dataset(include_ambiguous=True)
X = data['X']  # raw columns beta, force_N, velocity_m_s; shape (23301, 3)
y = data['y']  # provisional Helmholtz=1, other=0, ambiguous=-1
train = data['train_mask']  # A/0.1, 1750 rows
```

Additional arrays: `ids`, `string`, `speed`, `group`, `beta_level`, `force_rank`, `labels`, `y_fixed`, `fixed_labels`, historical/reclassified simulated labels and binary predictions. `output_features` contains the five requested measured output summaries **for label-family construction only**; it is separate from X. `reference_f0_hz=98.08739519244497` comes solely from A/0.1. Its control-derived floor is0.23330562561750412 from115source controls. Re-estimating this nuisance parameter on all source rows is acceptable only if the source-panel protocol permits them; if a strict within-source calibration split is needed, estimate it on that split instead.

The loader pins four measured and four simulated run IDs and SHA256 digests. It rejects changed source bytes, duplicate trial/grid IDs, unmatched measured/simulated memberships, invalid booleans, non-finite predictors and inconsistent speed mapping. It never opens `latest.json` or chooses the best score dynamically. The generated manifest includes explicit source paths/hashes, excluded counts, source-only nuisance estimates and split-membership hashes. Legacy simulated results are labelled target-adapted diagnostics.

Run:

```bash
PYTHONPATH=scripts .venv-mac/bin/python scripts/benchmark_data.py --include-ambiguous --out research/2026-10-13/benchmark_manifest_all_eligible.json
PYTHONPATH=scripts .venv-mac/bin/python -m unittest discover -s tests -p test_benchmark_data.py -v
```

Eight tests passed, including an actual-artifact test of24,000rows,23,301eligible rows,211ambiguous rows,1,750source rows and all eight checksums. A counterfactual target-output perturbation test confirms it changes labels but neither X nor source-only classifier settings. No waveform data were needed.

The saved artifacts trace back to the Zenodo collection cited in the repository, but the local CSVs are derived labels rather than original provider files. Before public redistribution of derived per-trial tables, attach source record/version and license provenance from the provider metadata; a model benchmark manifest alone is not a license audit.

## Completed benchmark and portability follow-up

The fixed A/B/C/D source sweeps subsequently completed in39.36seconds total:1,645fit records,105saved model outputs including declared alternative selections and diagnostics, and zero optimizer failures. Independent recomputation from saved prediction arrays matched all1,155per-panel coherent-family metric vectors to absolute tolerance1e-14. `scripts/benchmark_sets.py` has five meaningful tests, including a counterfactual target perturbation that leaves source CV inputs unchanged and a test distinguishing one globally shared rule from independent panelwise adversaries.

The coherent-family candidate **did not beat the strongest declared simple baseline for any of the four source sweeps**. Primary A-source macro family-minimum IoU is0.6696 for the candidate versus0.6884 for covariant quadratic logistic regression. The corresponding B/C/D candidate values are0.6653/0.6331/0.6469. Force-speed covariance improves each matched raw-coordinate comparison, but is a declared prior not learned from a single-speed source. These are discrete eligible-grid IoUs; they are not areas in a uniformly weighted continuous physical parameter space. The full negative result and nominal-CV alternatives are preserved.

Fresh-clone reproducibility was then added using eight losslessly compressed input tables under `research/inputs/` (4,129,839bytes total). The decompressed bytes retain the original pinned checksums. Upstream Part1/Part2 metadata identifies Alessio Lampis and CC BY4.0; attribution and the distinction between measured derivatives and historical synthetic outputs are documented in the release README. A ninth loader test verifies that a report-free clone produces identical data/manifests and that tampered decompressed content is rejected.

The unchanged A-source benchmark was repeated as `benchmark_A_portable` solely to capture the final portable loader hash. Fourteen selected models were fitted and saved before sample2 decoding; checkpoint predictions match the original A run within1e-10, and immediate save/load round trips pass. The final public freeze is `lockbox_frozen_manifest_portable.json`, fingerprint `6191519763c552ef57f22062933714b0ad71403c159e479171d210c659544913`. The strongest simple baseline was chosen using **source CV** (covariant quadratic logistic,0.71004), not target performance. The earlier freeze is preserved as historical provenance.
