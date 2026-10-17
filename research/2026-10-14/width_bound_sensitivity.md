# Restricting the nominal width diagnostic to the fitted exponent bounds

The result was calculated on 2026-10-15 and reproduced on 2026-10-17 KST. All saved values, witnesses, input hashes and code hashes reproduce exactly after excluding the new execution timestamp. The original result JSON and the preceding unbounded experiment remain unchanged.

**Result:** imposing `k ∈ [−2,2]`, the exponent bounds used by the affine predictor, increases the relaxed source disagreement minimum in only **6/186** cases. Each increase is 1 or 3 rows; all six occur for cello specimen B under fraction threshold 0.7. No nominal-observer cello result or published violin D-string result changes. This closes one mismatch between the earlier diagnostic and fitted model, but does not turn the diagnostic into a prediction-error bound.

## What is being restricted

The [original width diagnostic](width_law_assessment.md) minimizes Hamming disagreement between observed classifications and positive force bands at two source speeds. Each nominal bow-position column retains its own free band centers and initial width. The only cross-speed/cross-position constraint is

\[
W_2=\exp(k\log(v_2/v_1))W_1,
\]

with one exponent shared across nominal position columns. Let `L_ind` be the independent-band optimum, `L_free` the original optimum over all real `k`, and `L_bound` the optimum over `−2 ≤ k ≤ 2`. This sensitivity reports both

\[
\Delta_{\rm bound}=L_{\rm bound}-L_{\rm free},\qquad
G_{\rm bound}=L_{\rm bound}-L_{\rm ind}.
\]

These are different quantities. The first isolates the extra restriction from the exponent bounds; the second includes the original shared-width restriction. The calculation checks `L_free ≤ L_bound ≤ L_k=0` for every task. The interval `[-2,2]` is the final parameter bound in [`affine_transport.py`](../../scripts/affine_transport.py), with `theta[8]` representing `k`. Other fitted-model constraints remain relaxed.

The original file stores one optimizer per task. **23 saved optimizers lie outside the bounds**, but this does not mean the optimum is unattainable inside them. A different bounded optimizer attains the same cost in 17 of those cases. New constructive witnesses were generated for all 23, including the six with a larger minimum. The other 163 already have a verified bounded optimum.

## Complete 186-case accounting

Cello tasks are all four specimens × three speed pairs × fifteen fixed observer rules: **180 cases**. Published violin D-string data add all six speed pairs under the author classifications. The nominal cello rule is fraction threshold 0.5 and periodicity threshold 0.8; it is a subset of the 180, not an additional experiment.

| Scope | Cases | Saved `k` outside bounds | Cases with larger minimum | Largest `Δ_bound` | Largest `G_bound` |
|---|---:|---:|---:|---:|---:|
| All cello rules | 180 | 23 | 6 | 3 rows | 12 rows |
| Nominal cello rule, included above | 12 | 1 | 0 | 0 rows | 7 rows |
| Published violin D-string | 6 | 0 | 0 | 0 rows | 3 rows |
| Combined | 186 | 23 | 6 | 3 rows | 12 rows |

The 12 cello pairs below each summarize all fifteen rules. An unchanged cost can still require a different optimizer.

| Cello specimen | Speeds (m/s) | Rows in pair | Nominal `G_bound` | `G_bound` range over 15 rules | Increased-cost rules | Saved `k` outside bounds |
|---|---|---:|---:|---:|---:|---:|
| A | 0.05, 0.10 | 3,699 | 7 | 3–10 | 0/15 | 0/15 |
| A | 0.05, 0.20 | 3,575 | 4 | 0–12 | 0/15 | 0/15 |
| A | 0.10, 0.20 | 3,376 | 4 | 0–11 | 0/15 | 0/15 |
| B | 0.05, 0.10 | 3,976 | 0 | 0 | 0/15 | 0/15 |
| B | 0.05, 0.20 | 3,986 | 1 | 1–2 | 3/15 | 12/15 |
| B | 0.10, 0.20 | 3,990 | 4 | 2–7 | 3/15 | 3/15 |
| C | 0.05, 0.10 | 4,000 | 0 | 0–5 | 0/15 | 0/15 |
| C | 0.05, 0.20 | 4,000 | 3 | 1–6 | 0/15 | 3/15 |
| C | 0.10, 0.20 | 4,000 | 5 | 2–5 | 0/15 | 2/15 |
| D | 0.05, 0.10 | 4,000 | 0 | 0 | 0/15 | 0/15 |
| D | 0.05, 0.20 | 4,000 | 1 | 0–3 | 0/15 | 3/15 |
| D | 0.10, 0.20 | 4,000 | 1 | 0–4 | 0/15 | 0/15 |

All six increases are listed individually. No case or observer was removed after seeing the results.

| Specimen | Speeds (m/s) | Fraction / periodicity thresholds | `L_ind` | `L_free` | `L_bound` | `Δ_bound` |
|---|---|---|---:|---:|---:|---:|
| B | 0.05, 0.20 | 0.7 / 0.75 | 290 | 291 | 292 | 1 |
| B | 0.05, 0.20 | 0.7 / 0.80 | 287 | 288 | 289 | 1 |
| B | 0.05, 0.20 | 0.7 / 0.85 | 282 | 283 | 284 | 1 |
| B | 0.10, 0.20 | 0.7 / 0.75 | 385 | 388 | 391 | 3 |
| B | 0.10, 0.20 | 0.7 / 0.80 | 383 | 386 | 389 | 3 |
| B | 0.10, 0.20 | 0.7 / 0.85 | 378 | 381 | 384 | 3 |

The maximum increase is `3/3990 = 0.0752%` of a pair's rows. The summed increase is 12 **with observation reuse** across rules and speed pairs; it is not a count of twelve distinct bad labels or twelve new experiments. The maximum total `G_bound` remains `12/3575 = 0.336%` across cello rules and `7/3699 = 0.189%` for the nominal rule. The number of positive cello gaps remains 128/180.

| Published violin D-string speeds (m/s) | Rows | `L_ind` | `L_bound = L_free` | `G_bound` |
|---|---:|---:|---:|---:|
| 0.05, 0.10 | 528 | 1 | 2 | 1 |
| 0.05, 0.15 | 528 | 0 | 3 | 3 |
| 0.05, 0.20 | 513 | 0 | 2 | 2 |
| 0.10, 0.15 | 528 | 1 | 4 | 3 |
| 0.10, 0.20 | 513 | 1 | 2 | 1 |
| 0.15, 0.20 | 513 | 0 | 0 | 0 |

The largest published-data gap remains `3/528 = 0.568%`. Every per-task count, selected exponent and witness location is available in [`width_bound_sensitivity.json`](width_bound_sensitivity.json).

## Finite calculation and numerical verification

The original artifact stores the summed minimum-cost envelope on open cells of the `k` axis. This analysis intersects every cell with the bounded domain and chooses a strictly interior point from a minimum-cost cell. It does not clamp the original optimizer and assume that its old cost still applies.

Although the parameter domain is closed, its endpoints cannot have a strictly smaller cost than all neighboring interior points. A feasible band-pair candidate has an open feasibility interval in `k`; if it is feasible at either finite domain endpoint, it is also feasible just inside the domain. With finitely many columns the intersection retains such a neighborhood. Conversely, an envelope breakpoint may have a higher cost because alternatives disappear there. Therefore constructive representatives deliberately avoid breakpoints, including `−2` and `2`.

The replay first invokes the original independent checker on all **186** tasks. For each of the 23 replacement witnesses it reconstructs finite band endpoints, and the independent checker directly reclassifies the data, verifies the width ratios, source membership and independent-band counts, and compares the attained cost with the clipped minimum. The replacement witnesses cover 920 paired columns and 91,802 row evaluations **with reuse**. The complete original checker covers 7,266 paired columns and 702,153 row evaluations with reuse.

The 12 targeted tests include outside-only cheaper intervals, endpoint cost spikes, invalid domains, 120 random abstract-envelope comparisons including domain endpoints, and 60 random full width problems compared against uncompressed enumeration of all force-band predictions. The latter retain force ties and directly reconstruct/reclassify the selected bounded witnesses. All 12 pass. Global optimality still relies on the original finite enumeration argument and these exhaustive finite comparisons; witness attainment alone proves only an upper bound. Computation uses checked floating-point log coordinates, not interval arithmetic.

The replay's `source_sha256` values match the current sensitivity, original report and width-certificate scripts. Its `input_bundle_sha256` matches the preserved original bundle. This establishes exact reproduction of the recorded calculation, not independent verification of the physical labels.

From the repository root, using a new ignored output path:

```bash
bash scripts/run_mac.sh python -m pytest tests/test_width_bound_sensitivity.py -q
bash scripts/run_mac.sh python research/2026-10-14/width_bound_sensitivity.py \
  --output reports/width_bound_sensitivity_my_replay.json
```

Compare the replay with the public JSON after excluding only `created_utc`. Existing public files need not be overwritten to reproduce the result.

## Interpretation and limits

This sensitivity shows that the exponent search bounds do not materially enlarge this particular relaxed source disagreement count. It does **not** show that parameter bounds cannot affect fitted-model selection, extrapolation or held-out accuracy. Centers remain independent across columns and speeds; quadratic shape, coupled center dynamics, regularization and estimation are absent. No model was fitted, no labels were changed, and A2 was not used.

The cello calculation still assumes exactly nominal bow position and speed within each column, while the implemented predictor uses slightly varying measured controls. Consequently it is not a certified loss bound on that predictor. Published violin coordinates match the nominal diagnostic grid, but their extracted classifications are still not contact-motion ground truth. Source Hamming disagreement also cannot be converted into a held-out IoU bound.

The result is supporting methodological verification of a known finite-envelope diagnostic. It does not establish a new physical law, a generally improved predictor, or a major new optimization method.
