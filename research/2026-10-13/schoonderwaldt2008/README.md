# Published violin-string regime classifications (Schoonderwaldt et al., 2008)

This is a reproducible extraction of **published author classifications**, not
recovered waveforms, new measurements, or physical ground truth. It is independent
of the 2025 cello corpus used elsewhere in this repository. No new recordings or
sensors are needed. This dataset supplies an external violin-string benchmark;
the extraction itself is not a new learning methodology.

## Primary source and provenance

E. Schoonderwaldt, K. Guettler and A. Askenfelt, *An Empirical Investigation of
Bow-Force Limits in the Schelleng Diagram*, Acta Acustica united with Acustica 94,
604–622 (2008), [DOI 10.3813/AAA.918070](https://doi.org/10.3813/AAA.918070).

- [Publisher record](https://dael.euracoustics.org/landing_pages/aaua/64660.html).
- [Primary coauthor-hosted PDF](https://www.knutsacoustics.com/files/schoonderwaldt-et-al.-empirical-schelleng.pdf).
- PDF SHA256: `9e7f018421ae5512a7fb3c4a54e2c53c9146393167721a3fe7c169eb6358387a`.
- Figure 4: PDF page 7 / journal page 610; D4 steel violin string on a rigid
  monochord, speeds 0.05, 0.10, 0.15, 0.20 m/s (panels a–d).
- Figure 7: PDF page 10 / journal page 613; E5 steel violin string on a rigid
  monochord, speeds 0.10, 0.20 m/s (panels a–b).
- Tables VI and VII: PDF pages 15 and 16 / journal pages 618 and 619; includes
  assembled-violin measurements separately from monochord measurements.

The original PDF and rendered excerpts remain in ignored `work/` and must not be
redistributed with this repository. We have not verified an open-content license
for the article. Repository code licensing does not license the original article.
The CSVs contain attributed factual values and classifications derived from it;
they do not reproduce the article or its figures.

## What was recovered

`published_cells.csv` preserves all 1,584 acquisition-grid slots (6 panels × 11
bow positions × 24 forces), including missing ones. The extractor reads vector
PDF primitives, identifies grid axes and markers, and records object indexes,
PDF-point coordinates and alignment residuals. It does not classify raster
pixels or assign a marker to an empty cell.

| Panel | H | Constant slip | Multiple slip | Raucous | ALF | S-motion | Gray H | Unknown |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| D / 0.05 | 139 | 0 | 44 | 74 | 7 | 0 | 0 | 0 |
| D / 0.10 | 151 | 12 | 36 | 40 | 25 | 0 | 0 | 0 |
| D / 0.15 | 179 | 9 | 31 | 32 | 13 | 0 | 0 | 0 |
| D / 0.20 | 195 | 0 | 35 | 14 | 2 | 3 | 0 | 15 |
| E / 0.10 | 43 | 0 | 8 | 16 | 4 | 0 | 116 | 77 |
| E / 0.20 | 39 | 0 | 6 | 7 | 7 | 0 | 145 | 60 |

H means the authors' Helmholtz-motion classification; ALF means anomalous low
frequency. Marker legend: open black square H; filled dot constant slip; plus
multiple slip; diagonal cross raucous; open circle ALF; star S-motion. Gray open
squares in Figure 7 are author-indicated H regions that were **not systematically
analyzed**. Preserve their separate category. Large circles in Figure 4 annotate
a separate experiment and are not acquisition-grid ALF observations.

For a strict H-versus-other benchmark, filter
`strict_systematically_analyzed == True`; this leaves 1,171 cells, 746 H and 425
other. D contributes 1,041 cells; E contributes 130. Unknown and gray cells are
neither negative examples nor regular measured positives. Missing low- and
high-force cells in D/0.20 require care when assessing either boundary. E's sparsely
analyzed exterior/boundary cells are selection-biased; scores there would not
estimate performance over the full E acquisition domain.

Each source condition had repeated strokes, but the published figures do not
expose individual replicate labels or a reconstructable aggregation rule. Rows
are published conditions, **not independent stroke replicates**. Dependence
within one physical string prevents treating the row count as a sample count of
independent instruments.

## Coordinates and uncertainty

String length is reported as 325 mm. Bow-bridge positions are reported, rounded
to millimeters, as 11, 13, 15, 18, 21, 24, 29, 34, 39, 46 and 55 mm. The column
`beta_from_reported_rounded_distance` is their ratio to 325; it is approximate,
not a claim to have recovered exact measured beta. The text reports positioning
accuracy around ±0.5 mm. Exact continuous beta is not inferred from rounded tick
labels. Grid column indexes preserve the original ordinal layout.

The 24 nominal bow-force settings are 49, 58, 70, 84, 100, 120, 143, 171, 205,
245, 293, 350, 419, 501, 599, 716, 857, 1025, 1226, 1466, 1753, 2097, 2508 and
3000 mN. These and the panel speeds are nominal settings, not recovered
calibration measurements. The experiment uses real violin strings on a
monochord, so it does **not** validate an assembled violin or all violin strings.

`published_table_vi.csv` and `published_table_vii.csv` transcribe the separate
reported damping and boundary coefficients. Table VII's fitted relations are
F_upper = c_upper/beta and F_lower = c_lower/beta², with mN coefficients at
0.10 m/s. Those aggregate fits cannot be expanded into independent cell labels.
Table VI errors are based on the standard deviation of decay time; Table VII
reports standard errors. Blank stopped-string decay errors mean unreported,
not zero. The assembled-violin rows provide limited factual context, not the
classified monochord benchmark's missing assembled-instrument validation.

## Reproduction and audit

The original extraction used `pdfplumber==0.11.9` (recorded in the audit) and
NumPy. Install those in a separate extraction environment and run from the
repository. The core Mac benchmark environment does not require PDF parsing.

```sh
python research/2026-10-13/schoonderwaldt2008/extract_vector_labels.py \
  --pdf work/schoonderwaldt2008_pdf/schoonderwaldt2008.pdf
```

The script requires the exact PDF hash. It verifies six x and six y ticks per
panel, assigns only known small marker geometries, rejects off-grid primitives,
and preserves any ambiguous classification. See `extraction_audit.json` for
counts and residuals. The maximum marker alignment residual is 0.083 PDF points,
well below the grid spacing. There are no unresolved mixed markers or off-grid
marker primitives in this PDF.

`manual_checks.csv` records 24 stratified cells visually inspected against
rendered source crops, covering every marker class, both strings, all six panels,
gray squares and empty slots. A second assistant independently checked the same
24 targets, legends and both numerical tables without finding a mismatch; see
`SECOND_VISUAL_AUDIT.md`. This is not a human annotation study or exhaustive
manual verification. Original PDF object indexes plus the source hash make
every extracted marker traceable. Do not use small numerical residuals as evidence
that the original experimental classifications were correct.

## Benchmark protocol

`benchmark_published.py` compares model definitions already implemented for the
cello data, without retuning them on this dataset's target speeds. Primary source
speed is D/0.10 m/s; the three other D speeds are targets. All four possible D
source speeds are reported as a declared sensitivity analysis. Hyperparameters
use only five contiguous source beta-column folds. The full model list, grids,
source and fold IDs and code hashes are written before target prediction. The
dataset was examined during extraction, so this is a retrospective external
comparison with a recorded execution plan, **not preregistration or a blind test**.

Only one author classification per strict cell exists here. There is no family
of 15 observation rules and no raw-signal observer sensitivity analysis. Metrics
are ordinary IoU, error counts and Brier score at threshold 0.5. Report all fixed
priors, including static and reverse controls. A favorable result for known
Schelleng scaling is not a new law or method. No inferential confidence intervals
over independent instruments are supported by this single-string experiment.
