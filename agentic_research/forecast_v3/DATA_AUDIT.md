# Forecast v3: data and observer audit

Status: implemented and verified on the original 78,000 strokes. This document reports data construction and synthetic verification, **not model performance**. No learned development scores were computed by the data audit.

## Prediction target and legitimate scope

Each example is one previously recorded mechanical bowing stroke. Inputs use windows `[0,.20]`, `[0,.10]`, `[.05,.15]`, and `[.10,.20]` seconds. The target is the separate window `[.25,.45]`: a 50 ms gap follows the latest permitted input. Nine deterministic observer outcomes use periodicity thresholds `.7/.8/.9` crossed with second-flyback-ratio thresholds `.3/.5/.7`. They map to ten admissible joint patterns. Persistence uses only the latest short prefix `[.10,.20]` under exactly the same observer.

The task is **retrospective future-window prediction relative to a full-trace landmark**. It is not a validated online system. In `timeres/replicate.py:windows_of`, speed is smoothed with a 50-sample convolution; the first/last samples exceeding 98% of the *entire trace's* maximum define `on/off`. CSV `t` equals `(start-on)/rate`, i.e. the window's **start**, not its end. `plateau_s=(off-on)/rate` is also future-derived. We exclude `plateau_s` and all future feature values from predictors, but that cannot undo the retrospectively chosen landmark. A causal detector would require new raw-signal extraction and a different protocol.

Inputs are bridge-force periodicity/shape features and measured controls, not microphone audio. Outcomes are the historical observer's classifications, not independently verified Helmholtz ground truth. Success would establish forecasting of this declared observer across exposed families; it would not establish violin generalization, causal online control, or a new physical law.

## Source identity and split

The authoritative selection is `novelty/joint_rule/input_joint.json.gz`, SHA256:

`85ab696054670f80bc9690873291c39bb119b56670b4513da59bae994a2b0d70`

The loader expands its 1,560 beta columns × 50 strokes to exactly 78,000 unique `(session, folder, trial)` records. Records are ordered by `(session, folder, numeric trial)`, independently of targets and model seeds. Twelve CSV archives contain thirteen diagrams; the setup-variability archive contains two repetitions.

| Partition | Diagram IDs | Strokes | Beta columns |
|---|---|---:|---:|
| Training | A1, A2, A3, A4, A_T2, B1, B_T2s1, B_T2r1, B_T2r2 | 54,000 | 1,080 |
| Development | C1, C_T2, D1, D2 | 24,000 | 480 |

Every listed diagram was historically exposed in earlier research. The new task therefore has a fixed family split, **no untouched final test set**. Splitting windows or strokes randomly would mix nearly identical acquisition conditions and is not allowed.

`groups` conservatively contains the string families A/B/C/D. A/B and C/D are disjoint. `sessions` retains thirteen diagram IDs; `measurement_sessions` retains 39 acquisition folders; `columns` retains 1,560 beta-column IDs. We do not call these folders independent physical specimens. In particular, the two B setup repetitions can belong to one specimen and remain on the same training side. Two development families are insufficient for broad population-level confidence claims. Report diagram and family results descriptively; many strokes do not create many independent families.

IDs, session/folder names, trial number, column rank, and full-grid force ordering never enter X. Trial number reuse in different folders is supported. An identical full stroke identity or duplicate selected window fails closed.

## Observer parity, not a newly invented label

The implementation in `data.py:observer` follows these existing sources:

1. `novelty/label_sensitivity/label_sweep.py:helm`, base variant `(0.8,0.5,0)`.
2. `novelty/joint_rule/export_inputs.py`, shared physical guard and fixed override.
3. The frozen A2 preprocessing `summary.json`, with amplitude floor `0.4063327834010124`, aperiodic threshold `.5`, subharmonic ratio `.75`, reference f0 `98.08 Hz`.

For finite periodicity P and ratio R, the shared guard requires `P>=.5`, finite flybacks-per-period at least one, and no *finite* f0 below `.75*98.08`. The missing-f0 behavior is preserved exactly; missing f0 is not silently changed to a rejection. A rule is `fixed OR (guard AND P>=tau_P AND R<tau_R)`.

The baseline also retains the original amplitude-OR-periodicity gate. Since the base observer requires `P>=.8`, amplitude alone cannot rescue a failure. `fixed = baseline AND NOT finite(P,R)` is retained for exporter compatibility; with this particular baseline it is identically false. We did not create an alternative amplitude veto or fill missing targets with guessed labels.

Before any future target is accepted, every source stroke's old `[.20,.40]` window is independently reconstructed and checked against the historical fixture. **Actual check result: 78,000 strokes, zero baseline mismatches, zero guard mismatches, zero fixed mismatches, and zero periodicity/ratio/force mismatches.** All twelve source CSV SHA256 hashes exactly match the fixture's stored provenance. Missing required windows, extractor errors in selected rows, changed beta, altered source files, and mismatched observers stop construction.

## Predictors and leakage controls

X has 60 columns: 15 per input window. Each window contributes:

- `asinh(force/.1)`, `log(beta)`, `log(measured prefix speed)`.
- Periodicity, f0, amplitude standard deviation, flybacks-per-period, second-flyback ratio, flyback height, each with its own missing indicator. f0 uses `asinh(f0/98.08)`; amplitude and flyback height use `asinh(value/.1)`; other values retain their original scale.

Missing signal values are set to zero only in their own prefix row and marked by a binary indicator. Controls must be finite, with positive beta and speed. No future-dependent imputation, feature selection, normalization, target balancing, or fit occurs in the loader. The model caller must fit preprocessing on the training split alone. The four overlapping prefix windows are alternative summaries of the same past, not independent examples.

Future force/speed, future periodicity/ratio, plateau length, identities, and target labels cannot reach feature assembly. X is constructed before future rows are accessed. `prefix_Y9` and `prefix_pattern_ids` are permissible prefix-only persistence features; `Y9` and `pattern_ids` are the future targets and must never be appended to X.

## Reproducibility and verified tests

`load_or_build_dataset(features_dir, joint_input_path, cache_dir)` validates sources, audits the observer, and creates a derived NPZ. `save_prepared_dataset(dataset,path)` and `load_prepared_dataset(path)` write/read the same format. NPZ contains numeric/string arrays and a scalar JSON metadata string; loading uses `allow_pickle=False`. A mandatory same-stem JSON receipt binds the file SHA256 and metadata. Metadata binds all source hashes, loader and structural-head hashes, ordered ID hash, split manifest hash, feature-name hash, observer constants, temporal contract, and limitations. Caches are under ignored `forecast_v3/data/`; raw CSVs are never copied into Git.

Verified from the repository root:

```bash
.venv-mac/bin/python -m pytest -q \
  agentic_research/forecast_v3/tests/test_data.py \
  agentic_research/forecast_v3/tests/test_rule_patterns.py
```

Result: **26 passed**. The seven data tests establish future mutations leave X bitwise identical, prefix mutations leave future targets identical, prefix-only missing flags, identity exclusion, disjoint family splits, safe handling of repeated trial numbers, duplicate/missing-window rejection, cache corruption rejection, failure on observer disagreement, and exact scalar/vector observer agreement at threshold edges and missing values. The other nineteen tests verify the ten-pattern structure and numerically stable differentiable losses. Test success is implementation evidence, not predictive improvement.

The input fixture cites Zenodo records 17749111, 17782542, 17782552, 17782565, 17782538, and 17795326 and describes the derived data as CC BY 4.0. No derived data were uploaded by this work; redistribution needs the provenance/license check handled by the release owner.

## Precise provenance for the selected twelve archives

The stored Zenodo API manifests identify **Alessio Lampis**, University of Music and Performing Arts Vienna, ORCID `0009-0003-0813-9031`, as creator. The collection title is *A collection of bowed responses of cello strings*, with the part number shown below. Each of these five actually used records has `metadata.license.id = cc-by-4.0` in the local manifest. This is a record of inspected metadata, not a substitute for the release owner's current license verification. The parent fixture also cites Part 5, but none of its archives is selected here.

| Part / DOI | CSV stems selected (append `.csv.gz`) | Diagram IDs |
|---|---|---|
| 1 / `10.5281/zenodo.17749111` | `2024-03-25_TypeA_sample1`, `2024-04-10_TypeB_sample1`, `2024-04-12_TypeC_sample1` | A1, B1, C1 |
| 2 / `10.5281/zenodo.17782542` | `2024-04-15_TypeD_sample1`, `2024-04-19_TypeA_sample2`, `2024-04-22_TypeA_sample3` | D1, A2, A3 |
| 3 / `10.5281/zenodo.17782552` | `2024-04-24_TypeA_sample4`, `2024-04-26_TypeD_sample2` | A4, D2 |
| 4 / `10.5281/zenodo.17782565` | `2024-06-11_TypeC_T2_sample1`, `2024-06-13_TypeB_T2_sample1`, `2024-06-19_TypeA_T2_sample1` | C_T2, B_T2s1, A_T2 |
| 6 / `10.5281/zenodo.17795326` | `2024-10-23_variability_setup` | B_T2r1, B_T2r2 |

Local API evidence (under `violin-rne/data/manifests`):

| Manifest filename | SHA256 |
|---|---|
| `zenodo_17749110_20261020T004500.859472Z.json` (concept ID resolves to 17749111) | `12de4b648021dec7f20f58659434a1ffec4a16e3c0f27f94e90edb718cc36dc9` |
| `zenodo_17782542_20261019T005115.420365Z.json` | `3554421346c63c3ebdcfe52108948b5497c2957489957702a716704749820135` |
| `zenodo_17782552_20261019T140247.799286Z.json` | `9a7f3d225fa9d29abd6eef3f9274e39e4ba5b1d07278854a5f3aceb554347947` |
| `zenodo_17782565_20261018T135133.435184Z.json` | `d659682e528cda5fa0be868812ec7fc29f7a9e2bf429703fc702e8ecba4b89ea` |
| `zenodo_17795326_20261018T031735.564420Z.json` | `0d080765e53ee56dca6f72d04b541e1953dbdd130ccad8114bd68cc40fed265f` |

Derived changes are explicit: raw signals were reduced to window features by the existing extractor; a fixed set of original strokes and prefix/target windows was selected; declared transforms and missing indicators produced X; the existing observer produced nine rule labels and ten-pattern IDs; whole-family training/development labels were added. This processed artifact is not the creator's original raw dataset and implies no creator endorsement.

Prepared artifact: `data/dataset.npz`, 9,031,011 bytes, SHA256 `598a6964ce7ef91c5bb6b3cde1aa37daf4b615219438f896a4dc188f0d427975`. Its receipt is `data/dataset.json`; current and stored loader hashes agree at `94feedce80d4b5e41232a16dc084447740c672ce9bed252d88006115ccda9663`.
