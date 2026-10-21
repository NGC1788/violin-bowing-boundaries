# Independent recomputation of the label-sensitivity result (verifier, 2026-10-21)

요약: 4개 세션(A1, C1, A_T2, B_T2r1)에서 label_sweep.py 없이 다시 계산했고, 주장된 지수와 라벨 변경률이 모두 재현된다(|차이| <= 4e-8).
flyback 동치는 72.5만 행에서 불일치 0건이다. amp1x의 β 교락도 재현된다. 다만 RESULT.md에서 "within a column"이라고 적은 상관(−0.34 ~ −0.61)은
실제로는 열을 모두 합쳐서(pooled) 계산한 값이고, 열 안에서 계산하면 −0.12 ~ −0.21로 줄어든다.

Code: `recompute.py` (imports only `final/compare.py` for `censored_fit` / cross-check `intervals`, and
`claude_extra/verify_astra_boundaries.py` for `columns`, `APPLIED`, `rm.classify`). Output: `recompute.json`.
Own relabel (`myH`), own K = 2 interval extraction, own onset mapping.

## 1. Exponents (window axis | onset axis), mine vs claimed (results.json / posthoc_amplitude.json)
| session | base | fly0.3 | fly0.7 | amp1x | rel0.7 |
|---|---|---|---|---|---|
| A1 | −0.9706 / −0.9532 | −0.8831 / −0.8686 | −1.1542 / −1.1761 | +0.0204 / +0.0051 | −0.9143 / −0.8999 |
| C1 | −1.6066 / −1.2615 | −1.3976 / −1.1333 | −1.6126 / −1.1655 | −0.1501 / −0.0584 | −1.4910 / −1.2023 |
| A_T2 | −1.9505 / −1.4124 | −1.9266 / −1.3824 | −2.3971 / −1.4993 | −0.3191 / −0.1617 | −1.8309 / −1.3361 |
| B_T2r1 | −2.1752 / – | −1.8939 / – | −1.9897 / – | −0.2705 / – | −1.9567 / – |

All equal to the claimed values: C1, A_T2 and B_T2r1 match to every printed digit; A1 differs by at most 4e-8 (I take the
speed median over t = 0–0.3 as `final/data.py` does). The changed-label shares match exactly. Base = the stored
lower_law/onset_axis exponents. My own interval code and `compare.intervals` give identical base exponents.
The gate claim holds.

## 2. Flyback equivalence (all 0.2 s-long rows, every t: 181k rows per session, 725k in total)
`int(fpp) == 1` vs `fpp >= 1 and sfr < 0.5`: **0 mismatches** in every session. fpp is always an integer, there is no
fpp == 1 with sfr NaN or sfr >= 0.5, and no fpp >= 2 with sfr < 0.5. The full classify-vs-reimplementation label also
has 0 mismatches. Caveat (cannot be checked from the features): a variant f != 0.5 is a post-processing proxy. Nothing
here shows it equals re-running the feature extraction with `flyback_frac = f` (the count and sfr come from one
hysteresis pass).

## 3. amp1x β-confound and column types
Share of base-Helmholtz windows removed by amp1x, by log-β tercile (pooled, as claimed; per-folder terciles agree
within 1 pp):
A1 1.0/14.1/46.0 %, C1 3.0/18.3/47.2 %, A_T2 1.5/21.7/51.9 %, B_T2r1 5.3/25.8/50.6 %. These match posthoc_amplitude.json.
Over all 14 sessions: 0.2–6.5 % in the smallest-β tercile and 41.7–58.4 % in the largest. **Confirmed.** For comparison,
rel0.7 removes 7–13 % / 9–14 % / 5–11 % and fly0.3 removes 5–12 % / 7–12 % / 8–15 %, so neither is β-confounded.

Amplitude regression among base-H windows, log amp on (log v, log β, log F): b_v 0.85–0.94, b_β −0.70 to −0.80,
b_F 0.13–0.33. This is consistent with amplitude ∝ v/β. 23–35 % of base-H windows lie below the 0.406 floor.

Column types (left-censored / interval / none = no K = 2 run, right-censored at max force), all 120 columns:
| session | base | amp1x | base, large-β tercile (40 cols) | amp1x, large-β tercile |
|---|---|---|---|---|
| A1 | 0/112/8 | 0/89/31 | 0/40/0 | 0/21/19 |
| C1 | 27/86/7 | 7/85/28 | 19/21/0 | 4/19/17 |
| A_T2 | 32/82/6 | 8/86/26 | 21/19/0 | 3/20/17 |
| B_T2r1 | 46/63/11 | 13/72/35 | 29/11/0 | 6/15/19 |

So, yes: amp1x turns 17–19 of 40 large-β columns into "no boundary" columns, almost all with zero Helmholtz windows left
(≤ 3 per session still have an isolated H window). It also removes most left-censored columns. The exponent flip is
caused by large-β columns being right-censored at the top of the force grid. The small-β tercile is unchanged.

**Correction.** RESULT.md says the floor removes windows "preferentially at low force within a column (corr −0.34
to −0.61)". posthoc_amplitude.py computes this correlation pooled across all columns. Pooled, I reproduce −0.55, −0.46,
−0.43 and −0.40. Within a column (log F demeaned per (folder, column)) it is only −0.21, −0.14, −0.14 and −0.12. Most
of the pooled value is the β–force confound itself: corr(log β, log F) among base-H windows is −0.49 to −0.53.

## 4. Onset-axis mapping
For A1, C1 and A_T2, every (folder, trial) in the t = 0.2 s windows is found in the ramp meta (6000/6000). β agrees
exactly (max |Δβ| = 0). Shifting the trial by one gives |Δβ| of about 9e-6, so the keys are aligned and not off by one.
Onset/window force ratio, median: 1.12 (A1), 1.26 (C1), 1.35 (A_T2). Correlation of the logs: 0.98 (C1) and 0.97 (A_T2).
For A1 it is NaN because of a non-positive window force, which the pipeline clips at 1e-3. My onset exponents match the
claimed values for all 5 variants.

## Further sceptical notes (do not overturn the claims)
- "about ±0.17" is a mean. Per-session fly0.7 shifts range from −0.55 to +0.19 (window), and 3/14 sessions go
  *flatter*, B_T2r1 among them (+0.19). On the onset axis the fly0.7 shift CI is [−0.142, +0.025], which includes 0.
  "Looser → steeper" is not established on the onset axis.
- The rel0.7 mean shift is +0.164 (window). Per-session shifts are +0.02 to +0.35, all 14 positive.
- In base, 19–29 of 40 large-β columns are left-censored in C1, A_T2 and B_T2r1: Helmholtz already appears at the
  lowest force tested. fly0.7 raises this to 25–31. Much of the flyback sensitivity, and the large-β end of every fit,
  is set by the bottom of the force grid. This bounds what the data can say about "gentler than β^−2" at large β.
- amp1x uses the A2 lockbox floor (0.406) for every session. A per-session floor could not be recomputed from the
  features.
