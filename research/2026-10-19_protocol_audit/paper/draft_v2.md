# Protocol, session and force-range effects on bow-force limits in a public robotic cello-bowing dataset: an audit with frozen once-only tests

**Kim Taegeon, Jeong Jiyun, and Jeong Jihoon**
Gangwon Science High School, Republic of Korea

Draft v2.5, 2026-10-20 (v2.4 + §3.8 minimum-force law across sessions); v2.2 09:10 (v2 + frozen D2, A4-upper, B_T2 sample 2 and FROZEN_v3 pooled results; revision of draft v1 after two internal referee reports and a root review; JASA Express Letters format, not submitted). Changes are listed in `response_to_reviewers_v2.md`. Internal notes at the end must be removed before submission.

---

## Abstract

Public robotic bowing data are increasingly used to test bow-force limit laws. We audited 11 Schelleng diagrams (66 000 strokes; cello G strings, four constructions, two tensions) and four high-force sessions (18 000 strokes) of one public collection, and scored locally frozen predictions once. At low force the measured bow force falls with bow travel, the attack acceleration follows the recording order (one diagram was scanned in reverse), and each bow speed is a separate session. Across nine frozen once-only tests of the maximum-force law cZv/β the level errors ranged from −0.17 to +0.36 (natural-log units); four passed, and the error rose with the session's force ceiling (all four 4–12 N sessions failed); within-type tension pairs rejected a common impedance exponent of one. Minimum-force levels tracked a same-session load-decrease descriptor post hoc, but the association depended on the force axis and on the reverse-scanned diagram, and frozen tests were untestable or mixed. Within some string types, sessions differed as much as impedance-normalised types; in a frozen test one string's minimum-force boundary moved upward during play-in (by roughly 0.1–0.3, size not established). On the force at bow onset, sessions agreed on a minimum force ∝ v^0.95 β^−1.2 (post hoc). No model beat a model-free lookup.

---

## 1. Introduction

Helmholtz motion of a bowed string occurs only between a minimum and a maximum bow force. Schelleng's analysis gives a maximum force proportional to Z v/β and a minimum force proportional to Z² v/(R β²) [1], with Z the characteristic impedance, v the bow speed, β the relative bow–bridge distance and R the termination resistance. Measurements and simulations disagree about the minimum force: it was nearly independent of bow speed on violin strings [2], had a non-zero baseline at very low speed [3], and scaled as Z v/β in a recent simulation preprint [4]. Theory describes the force needed to *sustain* Helmholtz motion [5], whereas a stroke starting from rest must first *establish* it; the attack (initial acceleration and force) controls establishment [6,7].

A public collection from the IWK (University of Music and Performing Arts Vienna) provides complete Schelleng diagrams for cello G strings of four constructions at two tensions, plus high-force sessions, recorded by a bowing robot [8,9]. It has been used to validate a thermal friction model [10]. Boundaries read from finite windows of individual strokes mix string physics with protocol: how force is applied, how the stroke starts, how long it has settled, and which strokes share a session.

This letter reports what the collection can and cannot show. We contribute (i) a quantitative protocol audit, (ii) three sets of locally frozen, once-only tests of maximum-force laws (one primary success in the training force range, one failure in a new force range and new sessions), (iii) a post hoc minimum-force association that frozen tests did not confirm, and (iv) a negative prediction result. Every result is labelled frozen or post hoc. "log" denotes the natural logarithm throughout.

## 2. Data and methods

**Data.** We used parts 1–8 of the IWK collection [9] (CC BY 4.0). The dataset paper calls each 2000-stroke single-speed scan a "Schelleng diagram" [8]; here a *diagram* is the 6000-stroke set of one string sample over three bow speeds (0.05, 0.10, 0.20 m/s) × 40 log-spaced bow positions (β = 0.02–0.20) × 50 log-spaced nominal forces (0.1–4 N). The sampling rate is 50 kHz (Zenodo record descriptions [9]). The **development set** is 11 diagrams: five at the lower tension T1 (A1, A2, B1, C1, D1) and six at T2 (A_T2, B_T2s1, C_T2, D_T2, and the same-day repeats B_T2r1/r2 from the variability set). D_T2 was held out for a frozen test and is now spent. A third T1 sample of type A (A3) was processed but is not analysed here (it was excluded from the frozen tests because processing began before the predictions were frozen); A4 (a fourth A_T1 sample) was a frozen target (Sec. 3.3). Z and bending stiffness EI are the published values [8]. The dataset paper documents that each string type was mounted new and that additional A_T1 samples and repeated measurements exist [8]; we count diagrams, not independent strings.

**High-force sessions (parts 7–8).** Each of four strings (C_T2, A_T2, D_T2, B_T2; "sample 2", recorded 2024-06-22/24/27/29) was scanned at the same three speeds × 30 log-spaced β (0.02–0.07) × 50 log-spaced forces (3.8–12.6 N measured). These are different sessions from the T2 development diagrams, and the force range starts where the development range ends, so they are not an extension of the same force sweep. The design was verified from control signals alone before any label was parsed (`highforce_confirm/controls_audit.json`).

**Labels and windows.** Each 0.2-s window is labelled by a fixed bridge-force waveform classifier (one flyback per period = "Helmholtz", H), with amplitude thresholds set once on A2. The label is a proxy for string motion. Primary windows start t = 0.2 s after the bow-speed plateau begins; the force of a window is its mean measured bow force.

**Boundaries.** In each speed × β column the strokes are sorted by window force. With K = 2 force-adjacent H strokes as a run, the lower boundary is the interval below the first run and the upper boundary the interval above the last run (the dataset paper uses K = 3 [8]). Boundaries outside the measured range are censored, not imputed. A column without any run ("no band") is recorded as right-censored for the lower boundary and unobservable for the upper boundary. This is an *assumption*: with binary labels and the same error rate outside the band, an H region entirely below and entirely above the measured range give identical likelihoods, so the binary data cannot identify the censoring direction (`astra_20261019/observation_model/README.md`). Laws log F = c + a log v + b log β + Δ_d are fitted by interval-censored Gaussian likelihood; Δ_d is a diagram's offset from a reference shape.

**Load-decrease descriptor.** dec5_pl is label-free but measured on the same session as the boundary: for the 20 lowest nominal forces at each speed, the median change of the measured force between the first 0.1-s plateau window and the first window at least 5 cm of travel later (negative = decrease), averaged over speeds. At 0.05 m/s that later window starts about 1 s into the plateau, i.e. after the boundary window. It is a same-session covariate, not an a-priori predictor of an unseen string.

**Models (Sec. 3.4).** Trained on A1+A2: a model-free lookup (median boundary of the 5 nearest β columns at the same speed); Schelleng and Beigi forms with one constant; a free-exponent law with Z as a fixed offset; logistic regression and gradient boosting. Error = median |log(F_pred/F_mid)| over columns with finite intervals (F_mid = geometric interval midpoint), averaged over diagrams. This comparison followed a plan written beforehand (`final/ANALYSIS_PLAN_FINAL.md`) whose primary analysis used flyback-derived Z and few-shot adaptation at k = 10; the few-shot results were invalidated by a slope-fitting bug and the flyback Z was biased against the published values, so the published-Z comparison reported here is a post hoc change.

**Frozen tests and evidence level.** Plans and predictions were saved locally before the targets' labels were parsed and were scored once; none was publicly pre-registered.

| Test | Frozen | Evidence of timing | Primary endpoint |
|---|---|---|---|
| A_T2 | plan text 2026-10-18 21:38 KST (no hash; extraction had started 21:35, no label or boundary computed) | local file time | free-exponent law with published Z beats lookup on both boundaries |
| D_T2 | plan + coefficients 2026-10-19 01:19, with hashes; predictions from controls before any outcome column was parsed | local hashes | upper: strict cZv/β level within 0.12; lower: descriptor-level regression vs property regressions, margin 0.23 |
| A4 | protocol-aware lower estimator, frozen 10:42 with hashes; scored 12:52 | local hashes + GitHub commit d6b4081 (12:02) | interval distance and log score vs lookup_clip (H1, H1b); finite-column error (H2) |
| D2 (D_T1 sample 2) | FROZEN_v2 upper laws (10:31) and PACE* lower estimator (10:42); processed 22:24 via the bounded stream path, auto-scored 22:25 | identical hashes in GitHub commit d6b4081 (12:02) | upper: level error within each model's PI90; lower: as A4 |
| B_T2 sample 2 (4 speeds, 40 × 40, ≤7.5 N) | FROZEN_v2 upper laws (10:31); dedicated 40 × 40 parser frozen 2026-10-20 08:51 before labels (generic scorers crashed on the design); PACE* declared not applicable (50-step grid assumption) | hashes; same-date friction session seen before freezing (partial contamination) | upper level within PI90 |
| FROZEN_v3 pooled (A4, D2, B_T2 s2) | v3 addendum (10:39); dedicated scorer frozen 2026-10-19 22:3x, loader addendum for B_T2 s2 2026-10-20 08:51 | GitHub d6b4081 (v3) + local hashes | pooled speed curvature and lv·lb interaction, sign + cluster-bootstrap CI |
| High force (4 strings) | thermal and Schelleng upper laws, FROZEN_v2 10:31 + v3 addendum 10:39; scorer protocol frozen 21:39 before labels | identical file hashes in GitHub commit d6b4081 (12:02), before the archives were processed (≥14:00) | per-string level error within each model's 90 % prediction interval; β-exponent zone; speed curvature |

For B_T2 and the high-force B string, one friction-reconstruction session of the same string type (2024-07-03) had been seen and the thermal friction law was fitted on it. All other analyses, including the descriptor choice (screened among about ten descriptors × three response definitions), are post hoc. Intervals are 95 % percentile bootstraps over β columns unless stated; bootstrap settings differ between analyses (100–2000 resamples) and are conditional on the recorded sessions.

## 3. Results

### 3.1 Protocol audit (post hoc)

**Force control.** The dataset paper states that bow force was set "solely from the elevation of the bow" and remained "almost constant during the strokes" [8]; a later model study notes "a small normal load reduction" [10]. In A2, for nominal-force ranks 1–20, the measured force fell by 0.044, 0.088, 0.125, 0.153 and 0.174 N after 2, 5, 8, 11 and 14 cm of travel, nearly identically at the three speeds when plotted against travel (between-speed SD 0.0016 N; against time 0.040 N). In A1 the decrease was 2.5–4 times smaller and the travel collapse weaker (0.0077 vs 0.0135 N). While the bow moved the reading fell at 0.020–0.180 N/s; before it moved, by at most 0.0021 N/s.

**Attack schedule follows recording order.** Within every β column the attack acceleration rises with the *trial index* (within-column Spearman +0.98 to +1.00 in A1 and A2). A2 was scanned with β and force ascending, so there the acceleration also rises with force. A1 was scanned in reverse (β and force descending), so its acceleration *falls* with force (ρ −0.98 to −1.00). Consequently the 20 lowest-force strokes per column received a median attack of 0.563 m/s² in A1 at every speed, against 0.038, 0.104 and 0.417 m/s² in A2 (15×, 5.4× and 1.35× harsher in A1). Among the eight development diagrams with ramp metadata, A1 is the only reverse scan; the other seven share A2's schedule exactly (`paper/repro/attack_scan_all.log`). In A2, 45 %, 51 % and 67 % of strokes at the three speeds use a constant 4.8-cm ramp; for low-force strokes the 0.20 m/s attack is 16× harsher than at 0.05 m/s for β < 0.043, 11× at 0.043–0.093 and 1.4× above 0.093.

**Sessions and window placement.** Each speed is a separate complete β–force scan, so speed is confounded with recording block. The 0.4-s windows distributed with the dataset begin 1.55, 0.35 and 0.08 s after plateau onset at the three speeds (A2). A pooled censored fit to those windows (A1, B1, C1, D1, A2) gave a minimum-force speed exponent of +1.77 [1.64, 1.90]; aligning windows at equal time since plateau onset gave +0.57–0.67 (A1) and +0.94–1.04 (A2), and per-diagram matched-time exponents of +1.09 to +1.42 for the eight other diagrams. Because the force decrease collapses on travel and equal times mean unequal travel, this *window-placement* confound mixes settling time with travel-dependent force loss; it is not identified which.

**Low-force reliability.** Readings after contact was lost were not zero (folder medians −14.9 to +61.0 mN). In one A1 folder the post-contact reading varied from ≈ +4 mN (large β, recorded first) to ≈ +158 mN (small β, recorded last; correlation with trial index +0.95, with β −0.88). Recording order and β are confounded; across folders with opposite scan directions the offset falls with increasing β, which points to a pose-dependent offset rather than drift. In D_T2, strokes inside an H run read −23 to −27 mN. Forces below about 0.2 N are therefore uncertain by tens of millinewtons.

### 3.2 Maximum bow force

**Shape and censoring (post hoc).** Pooled over the 11 diagrams, the upper boundary scaled as v^0.70 [0.63, 0.76] β^−1.07 [−1.14, −1.01] (diagram-cluster bootstrap; with β-position clusters the β interval is [−1.14, −1.00]). The 4-N ceiling right-censors 25–42 %, 50–68 % and 56–83 % of available columns at 0.05, 0.10 and 0.20 m/s; the dataset authors did not compute the upper limit at 0.20 m/s [8]. Level fits therefore rely on the parametric shape in censored columns.

**A_T2 (frozen secondary endpoints).** Relative to the A1+A2 law the upper level rose by +0.197 [+0.106, +0.286] (A1−A2 yardstick 0.005), a Z exponent of +1.71 [+0.92, +2.49] (Schelleng: 1). Column-median errors: free-exponent law with published Z 0.121, lookup 0.167, gradient boosting 0.176 (pre-correction code), same law without Z 0.189; the frozen law's level error was 0.082. The primary hypothesis was rejected because it failed on the lower boundary (Sec. 3.3).

**D_T2 (frozen primary endpoint).** The strict law F = cZv/β with ln c = 0.966 from the five T1 diagrams predicted the D_T2 upper level within 0.002 (conditional interval [−0.078, 0.091]; tolerance 0.12) — passed. This is a censoring-weighted average over speeds of a shape that misfits D_T2's speed dependence: median signed column residuals were +0.07 (0.05 m/s, n 23), −0.01 (0.10, n 15) and −0.55 (0.20, n 10), with fitted σ 0.396. Column medians were 0.124 (strict), 0.113 (empirical shape + Z, frozen secondary) and 0.226 (no Z, uncalibrated); the strict law's column errors on the other T2 diagrams were 0.113–0.223. The test had little leverage on Z: the bootstrap level interval maps to Z exponents 0.37–1.74, a constant T2 offset without Z (post hoc) predicts D_T2 within 0.010, and D1 lay 0.094 above the strict T1 line while D_T2 lay on it. The result is a cross-type calibration success of the T1 constant, not a within-type Z¹ shift.

**D2 (frozen).** For a second D_T1 sample (≤4 N), all ten frozen upper laws were inside their 90 % intervals (strict cZv/β −0.126 against ±0.148; thermal −0.127; empirical shape −0.114; 109 columns, 60 finite, 49 right-censored); the β exponent was −1.13 [−1.21, −1.07] ("steep" zone, decidable) and the speed exponent 0.87 [0.79, 0.96].

**A4 and B_T2 sample 2 (frozen).** For a fourth A_T1 sample the strict law over-predicted the level (−0.173, outside ±0.148; only the zero-parameter thermal variant, with its wide ±0.41 interval, was inside); β exponent −1.12 [−1.30, −0.96]. For B_T2 sample 2 — four speeds including 0.15 m/s, 40 × 40 strokes, forces up to ≈7.5 N; scored with a dedicated parser frozen before its labels were read; partially contaminated because a friction-reconstruction session of the same date had been seen — the strict law passed narrowly (+0.129; +0.145 on the three standard speeds) while all thermal variants failed (+0.16 to +0.22); β exponent −0.98 [−1.20, −0.81], speed exponent 0.75 [0.60, 0.88].

**FROZEN_v3 pooled hypotheses (A4, D2, B_T2 sample 2).** The pooled speed curvature e(0.10→0.20) − e(0.05→0.10) was −0.11 (cluster-bootstrap interval [−0.30, +0.015]) and the lv·lb interaction +0.29 ([−0.09, +0.39]); neither the empirical (concave, positive interaction) nor the thermal hypotheses were decided. Per-diagram curvatures (−0.30, +0.01, −0.01) all lay inside the frozen single-diagram 90 % interval (−0.69, +0.06).

**Summary of all frozen tests of cZv/β (Table 3, Fig. 5).** Level errors over nine held-out sessions were −0.002 (D_T2), −0.173 (A4), −0.126 (D2), −0.066 (A3, frozen under FROZEN_v1 while it was being processed), +0.129 (B_T2 s2) and +0.165 to +0.364 (four high-force sessions); 4/9 passed and the SD was 0.20, larger than the in-sample residual SD (0.08–0.12) used to set the intervals. Post hoc, the signed error increased with the session's force ceiling (Spearman +0.73, n = 9, p = 0.03); within-session truncation of B_T2 s2 at 4 N changed its error by only +0.03 [−0.03, +0.09], and two high-force sessions stayed at +0.34/+0.42 even below 7 N, so the trend reflects sessions more than scan range.

**Table 3.** Every frozen once-only test of the strict upper law (level error = observed − predicted, ln). Ceiling = 95th percentile of the measured t = 0.2 s window force.

| Session | Ceiling (N) | Level error | Tolerance | Result |
|---|---:|---:|---:|---|
| D_T2 | 3.0 | −0.002 | 0.12 (plan) | pass |
| A4 | 3.5 | −0.173 | 0.148 | fail |
| D2 | 3.3 | −0.126 | 0.148 | pass |
| A3 (FROZEN_v1) | 3.4 | −0.066 | v1 level rule | pass |
| B_T2 s2 (partly contaminated) | 5.8 | +0.129 | 0.148 | pass |
| HF C_T2 s2 | 11.4 | +0.364 | 0.148 | fail |
| HF A_T2 s2 | 11.3 | +0.224 | 0.148 | fail |
| HF D_T2 s2 | 11.5 | +0.165 | 0.148 | fail |
| HF B_T2 s2 | 11.1 | +0.357 | 0.148 | fail |

**Tension pairs (post hoc).** Z exponents of the T1→T2 pairs were A +1.71 [0.86, 2.47], B −0.51 [−1.79, 0.75], C +3.25 [1.11, 5.12] and D +0.23 [−0.25, 0.70]. A common exponent of 1 is rejected for A, B, C (χ² = 14.2, 3 df, p = 0.003). A random-effects estimate over A, B, C is +1.37 [−0.49, 3.23] (normal) or [−2.7, 5.5] (t, 2 df); the between-pair τ ≈ 0.12 matches the T1 strict-law residual SD (0.118). Three further limits apply. Z is exactly proportional to tension in this dataset (T/Z = 137.2 m/s for all eight strings), so Z, tension and the T2 batch are collinear. Across strings EI is associated with the upper level at least as strongly as Z (r 0.80 vs 0.71). The published C_T2 Z (1.1154 ± 0.1023 kg/s) has an uncertainty larger than the C pair's Δln Z (0.058), so the C exponent is not identified.

**High-force sessions (frozen, scored once).** All ten frozen upper-boundary models under-predicted the 4–12 N boundary of all four new sessions (Table 2, Fig. 4). For strict cZv/β the level errors (observed − predicted) were +0.36 (C), +0.22 (A), +0.17 (D) and +0.36 (B), all outside the frozen 90 % interval (±0.15). Plain Schelleng beat the thermal-friction variants on level error and log-likelihood for all four strings. The free β exponents were −0.75 [−1.27, −0.44] (C), −0.82 [−1.30, −0.47] (A), −0.48 [−0.79, −0.24] (D) and −0.39 [−0.68, −0.16] (B). The frozen rule labels D and B "thermal zone" (e_β ≥ −0.95), but their slopes are far shallower than every frozen model, including the thermal ones (−0.86 to −0.94), so we read this only as "shallower than β^−1". Speed exponents were 0.57–0.91. The frozen speed-curvature test was not decided (pooled −0.19, cluster-bootstrap interval [−0.46, +0.17]; per string A +0.13, C +0.21, B −0.53, D −0.44). Pre-declared sensitivity (no-band columns, 9–13 per string, treated as F_max < lowest force) changed none of the decisions except that the curvature interval excluded 0 under one of the two pre-declared bootstraps; by the frozen rule this does not upgrade the result. Post hoc, the upper estimate in this range is fragile to how single-stroke label dropouts are handled: replacing the last-band rule by the longest-band rule moves the strict-law errors to +0.02, −0.14, −0.08 and +0.19. Columns with ≥ 2 separated H bands are not specific to high force (37–73 % of banded columns in the development diagrams, 47–70 % here; median gap one stroke). Session, force range and extraction rule are confounded, so the cause of the failure is not identified; the frozen verdict stands.

**Table 2.** High-force frozen level errors (observed − predicted, log) for selected models; 90 % interval ±0.15 for all three. N = upper-boundary columns (of 90; the rest had no band).

| String (Z, kg/s) | N | strict cZv/β | thermal (T1k_break) | empirical shape + Z |
|---|---:|---:|---:|---:|
| C_T2 s2 (1.1154) | 77 | +0.364 | +0.447 | +0.337 |
| A_T2 s2 (1.1884) | 77 | +0.224 | +0.316 | +0.204 |
| D_T2 s2 (1.2021) | 80 | +0.165 | +0.262 | +0.165 |
| B_T2 s2 (1.1899) | 81 | +0.357 | +0.432 | +0.311 |

### 3.3 Minimum bow force

**Levels and repeatability (post hoc).** Lower-boundary levels varied across the ten development diagrams with SD 0.218. A1 and A2 differed by 0.233. The B_T2 same-day repeats differed by 0.18, but their intervals overlap ([−0.485, −0.183] vs [−0.269, −0.024]). In a variance decomposition the within-repeat SD was 0.121 (3 df) against 0.211 for condition means (F = 4.40, permutation p = 0.096). On the onset-force axis the between-diagram SD falls to 0.125.

**Per-diagram shape (post hoc).** Free per-diagram fits gave lower-boundary exponents v^1.09–1.42 and β^−1.61 to −2.18 for the eight non-A_T1 diagrams, against v^0.67/0.94 and β^−0.97/−1.09 for A1/A2. The A1+A2 reference shape used for all levels is therefore atypical; outside it the shape resembles Schelleng's v/β² in direction, with speed still confounded with session.

**Association with the session descriptor (post hoc).** The ten levels correlated with dec5_pl (range −0.120 to −0.032 N): r = 0.88 (Fisher interval about 0.56–0.97; nominal p = 0.0008, ×10 screen 0.008). The association is fragile:

- *Leverage.* Without A1, r = 0.77; without A1 and A2 (which define the reference shape), 0.69 (p 0.06), while EI gives +0.86 (p 0.01) and torsional stiffness +0.85 in the same eight diagrams. On levels taken at a fixed reference point (v = β = 0.1) of per-diagram free fits, r = 0.90 for the ten diagrams but 0.60 (p 0.12) for the eight.
- *Force axis.* Measured-force K/t variants give 0.815–0.880; the nominal axis 0.675–0.707. Reading boundaries on a decaying channel couples the descriptor and the boundary arithmetically.
- *Detection limit.* Treating readings below c as left-censored gives r = 0.80 (c = 0.2 N) and 0.70 (c = 0.3 N).
- *Attack.* A1, the extreme point, is the reverse-scanned diagram whose low-force strokes had 15× harsher attacks at 0.05 m/s; its higher level at 0.05 m/s (+0.24 vs A2 −0.25; +0.04/+0.05 vs −0.05/−0.04 at 0.10/0.20 m/s) has the speed profile expected from harsher attacks needing more force to establish Helmholtz motion [6]. Attack scheduling can explain at most the A1 offset, since the other diagrams share one schedule.

Refitting the reference shape inside each fold, the descriptor reduced the level error from 0.213 to 0.101 (leave-one-diagram-out) and from 0.221 to 0.131 (leave-one-type-out), but increased the finite-column error from 0.294 to 0.316 and from 0.319 to 0.354 (843 of 1200 columns finite); the censoring-inclusive interval distance changed from 0.315 to 0.303. Descriptor selection was outside the cross-validation loop, so these are optimistic.

**A_T2 (frozen secondary).** The lower level was 0.539 [0.382, 0.722] below the A1+A2 law (42 % lower; −0.65 relative to A1, −0.42 relative to A2; A1−A2 yardstick 0.233, i.e. ±2.0 in Z-exponent units). The implied Z exponent was −4.69 [−6.27, −3.32]. Treating readings below c as left-censored with the reference refitted the same way gives −0.42 (c = 0.2 N) and −0.29 (0.3 N); on the onset-force axis −0.19. The draft-v1 value −0.10 truncated on the outcome and is withdrawn. A_T2 also had the most negative dec5_pl (−0.120 N); this is an observation, not evidence that the session caused the shift.

**D_T2 (frozen).** The primary level fit failed because one column had a zero-width interval at the 1-mN floor, so the primary endpoint is **untestable**; the lower bootstrap failed in 1272 of 2000 resamples for the same reason. Secondary finite-column errors (111 columns) favoured the descriptor numerically (0.286, against 0.328 for the mean, 0.325 log EI, 0.376 log Z, 0.386 log Z + log EI), but the predeclared level-scale margin does not define a decision for this secondary metric. The planned K = 3, t = 0.1/0.3 s and nominal-axis secondaries were not run. D_T2's descriptor (−0.038 N) sat next to A1, the most influential training point; post hoc repairs of the zero-width column gave levels −0.09 to −0.13, errors 0.17–0.20 for the descriptor model and 0.17–0.21 for the mean.

**A4 (frozen).** A protocol-aware lower-boundary estimator (PACE*) was scored once against frozen baselines (120 columns, 85 finite, 8 no-band). It beat the clipped lookup on interval distance (0.165 vs 0.207; difference −0.042) and mean log score (−2.003 vs −2.225), so H1 and H1b were met; it was worse on finite-column error (0.241 vs 0.172; H2 failed), and the session/protocol terms added nothing (H3). A mean-only model was better than PACE* on all three metrics (0.204, 0.157, −1.961). The lookup failed to predict 3 columns, so its interval and log-score denominators are 117 against 120. We do not read this as a method advantage.

**D2 (frozen).** On a second D_T1 sample (120 columns; 102 finite, 11 no-band, 7 left-censored) PACE* again predicted the level (error 0.012, within 0.23) but lost to the clipped lookup on interval distance (0.293 vs 0.270; H1 failed) and finite-column error (0.340 vs 0.222; H2 failed); only its log score was better (H1b); session and property terms added nothing (H3). The lookup failed on 3 columns (denominator 117 for its interval and log-score metrics).

### 3.4 Predicting the boundaries of an unseen string

**Table 1.** Median |log(F_pred/F_mid)| over finite columns, averaged over diagrams; training on A1+A2. Development values (B1, C1, D1, B_T2r1, B_T2r2; post hoc, corrected censoring and support handling) give scorable columns in parentheses; the counts differ by model because predictions can fail or fall outside the support. A_T2 values were scored once under the frozen plan with the evaluation code of that time (82 finite lower and 53 finite upper columns of 120); the A_T2 gradient-boosting entries used pre-correction code that assigned a fixed error to non-finite predictions. Beigi has no upper law of its own, so its upper entries equal Schelleng's. Point errors are conditional on the finite intersection; the corrected comparison's point-hit statistic allows ± one grid step (log 40/49) and differs from strict interval coverage.

| Model | Lower, development | Upper, development | Lower, A_T2 (frozen) | Upper, A_T2 (frozen) |
|---|---:|---:|---:|---:|
| Lookup (5 nearest β, same speed) | 0.269 (376) | 0.188 (259) | 0.250 | 0.167 |
| Schelleng + published Z | 0.647 (376) | 0.195 (259) | 0.939 | 0.215 |
| Beigi + published Z | 0.306 (376) | 0.195 (259) | 0.455 | 0.215 |
| Free exponents + published Z | 0.268 (376) | 0.155 (259) | 0.520 | 0.121 |
| Free exponents, no Z | — | — | 0.347 | 0.189 |
| Logistic regression | 0.307 (363) | 0.258 (181) | — | — |
| Gradient boosting | 0.288 (364) | 0.169 (241) | 0.271 | 0.176 |

On the development diagrams the lookup and the free-exponent law tied on the lower boundary (0.269 vs 0.268; a legacy-edge lookup gives 0.266). In a leave-one-diagram-out check over all 11 diagrams (post hoc, including D_T2 after its frozen scoring), no variant beat the plain lookup on finite-column error (0.258; 0.251 clipped; best of 37 alternatives 0.275; paired differences included zero, e.g. lookup + log Z + log EI: −0.016 [−0.064, +0.024]); leave-one-type-out gave the same ranking (lookup 0.283). Some descriptor variants had smaller interval distance (0.221 vs 0.279) and a better log score, so the ranking depends on the metric.

### 3.5 Boundary extraction (method candidate, ended)

A Bayesian single-band change-point extractor (integrated error rates; a known method [11]) was tested once, under a plan frozen beforehand, for test–retest agreement between the same-day repeats B_T2r1/r2 (development data). Against the K = 2 rule the median |Δ log F| changed by −0.009 [−0.076, +0.060] (lower, N 48) and +0.001 [−0.024, +0.046] (upper, N 41): no consistent improvement, so this direction was stopped. On synthetic columns the same extractor helps for independent label dropouts but worsens the upper boundary when the true H region has two bands, and no-band columns remain direction-unidentifiable.

### 3.6 Post hoc robustness and diagnosis (all on seen data; independent re-runs reproduced every number)

**Why the high-force transfer failed (`posthoc_hf/`).** (i) The onset-to-window force drop differs between the high-force and sample-1 sessions by at most about 0.03 (pooled −0.022/−0.011/−0.027/−0.029 for C/A/D/B; A becomes +0.029 when sample 1 is restricted to β ≤ 0.075), far too small to explain +0.17 to +0.36; a between-session load-cell calibration difference cannot be tested. (ii) A quadratic speed–β surface fitted to the development diagrams before the high-force data existed (reproduced exactly) predicts the high-force levels worse (+0.53 to +0.68) and a speed exponent of 0.13–0.17 against 0.57–0.91 observed. (iii) By the planned decision rule, a joint fit of each string's sample-1 columns (β ≤ 0.075) and high-force columns reads "session offset, high-force higher" (δ = +0.38 to +0.51, intervals exclude 0), but δ rests on only 8–13 finite sample-1 columns, the rest being right-censored near 4 N; descriptive same-force comparisons at 3.8–4.6 N (chosen after seeing the planned result) show no high-force-higher shift. The miss lies above 4.6 N, where only the high-force sessions have data, so session and force range remain confounded and neither explanation is favoured. (iv) The upper-band rule moves the level errors by 0.4–0.6, as much as the failure itself.

**Minimum-force association (`robust_lower/`).** The dec5_pl association (r = 0.88, n = 10) survives a max-|r| correction over a 27-pair screen of descriptors × response variants (adjusted p = 0.010; 0.028 over 135 pairs), but the screen omits the 7-diagram descriptors and dec5_pl's own construction choices, so these p-values are optimistic. Treating no-band columns as missing leaves it unchanged (r = 0.90). It weakens under detection-limit censoring (0.80/0.70/0.53 at 0.2/0.3/0.4 N), on the onset-force axis (0.82, n = 7), after zero-offset correction (0.56 vs 0.95 on the same 7 diagrams) and without A1+A2 (0.69, Fisher interval [−0.03, 0.94]). A pure reading-drop null (boundary fixed in onset force, reading drops by the session's decrease) does not account for the per-diagram levels in the primary stroke set (R² with slope 1 = −0.62, n = 7) and partly accounts for them in an onset-band set (+0.30); its implied slope (+4.8 to +6.6 log/N, wide intervals) is compatible with the fitted +7.67, so a mechanical explanation is not ruled out. No protocol effect on the boundary is established.

**Maximum-force results (`robust_upper/`).** The A_T2 upward shift stays positive (0.20–0.23) and the D_T2 strict level error stays within 0.013 in every variant (0.05 + 0.10 m/s only; K = 3; canonical β-cluster bootstrap, B = 2000). With 0.05 and 0.10 m/s only, the pooled exponents become v^0.86 β^−1.19 (all speeds v^0.70 β^−1.06): a single power law does not describe all speeds equally, by censoring at 0.20 m/s or by real curvature in ln v. Pair exponents: C above 1 and B, D below 1 in every variant, but against within-session errors only (session noise not included); A ≈ 1.7–1.9 with 1 at the edge of its canonical interval and inside the frozen one. A random-effects pooled exponent is not identified (HKSJ interval widths 5–9) and is compatible with 1. An absolute torsion-corrected impedance cannot be computed from the published torsional data; in a ratio-only sensitivity check the A and D exponents become unidentified.

### 3.7 Session variance and string settling

**Variance decomposition (post hoc; partly refuted by an independent check).** For the 13 sessions recorded with the ≤4 N design (A1–A4, B1, C1, D1, D2, A_T2, B_T2s1, B_T2 Part 6, C_T2, D_T2; A3 scored once under a freeze written while it was being processed: strict level −0.066, inside its frozen interval), a REML model with known level SEs gave, for the upper level relative to strict cZv/β, a string type × tension component τ = 0.058 [0, 0.145] and a within-type session component ω = 0.069 [0.04, 0.13]; for the lower level τ = 0.114 [0, 0.32] and ω = 0.158 [0.09, 0.30]. An independent re-implementation reproduced every number but showed that the design cannot rank the two components: only five within-type degrees of freedom exist (posterior P(ω ≥ τ) = 0.6–0.7); the upper ω comes from the single D1–D2 pair and the lower ω from A1 (reverse scan) and A4; without dividing by the published Z, the type component is larger (τ = 0.10 [0.03, 0.22]); and β-block bootstrap SEs are 1.5–3 times larger. The supported statement is narrower: within some types (D_T1 upper; A_T1 lower, where the levels also track scan protocol, force floor and date), sessions differ as much as impedance-normalised types. Session repeatability has been examined before [2, 7, 17, 18]; the heterogeneous T1→T2 pair exponents are what such session differences would produce.

**Settling (frozen, one string).** The collection contains the A1 string recorded at 0.10 m/s immediately after mounting ("cold", 03-25) and after a play-in of several thousand strokes ("warm", 03-26), before the A1 diagram (03-27). A prediction was frozen while this archive was being processed (the cold folder's features already existed but were unopened; thresholds = ω/2). By the frozen rule the lower level rose ("matters": +0.256, interval [+0.003, +0.542]; [+0.071, +0.466] after correcting a bootstrap that paired only 29 of 40 β columns), the upper level did not change detectably (−0.041 [−0.099, +0.014]), and from warm to the A1 diagram the lower change was +0.109 (not decided as frozen; [+0.002, +0.229] with corrected pairing). An adversarial check supported the direction (19 vs 6 β columns with non-overlapping intervals; Helmholtz fraction at 0.30–0.60 N fell from 0.50 to 0.29; classifier thresholds cannot flip these labels) but not the size: 17 of 40 cold lower boundaries are left-censored, so the cold level is a censored extrapolation; the complete-case shift is +0.10 [−0.04, +0.26], the onset-force-axis shift +0.12 [+0.02, +0.22], column-wise sign tests are not significant, and a −0.05 N zero offset on the warm forces would remove it. The lowest settings also delivered about three times more force in cold (median onset 0.48 vs 0.17 N): the elevation-to-force mapping changed as the string settled. We therefore report a probable upward shift of the minimum-force boundary during play-in of roughly 0.1–0.3, confounded with the force mapping, and no detectable change of the maximum force. A remount effect on transient playability [18] and a lower minimum force in a diagram repeated five months later [2] were reported before; a controlled new-versus-played-in comparison of steady-state limits was not found.

### 3.8 The minimum-force law across sessions (post hoc)

Per-session interval-censored fits log F_min = a + b_v log v + b_β log β (t = 0.2 s, K = 2) did not reproduce the development pattern in the four sessions never used for this statistic (window-force β: A3 −1.22, A4 −1.42, D2 −1.11, B_T2 s2 −1.31; development non-A_T1 −1.6 to −2.2), and the same-day repeat pair differed by 0.36, so a single diagram cannot separate β^−1 from β^−2. Replacing the window force by the label-free force at bow onset (11–12 sessions with ramp metadata) gave F_min ∝ v^0.95 [0.86, 1.05] β^−1.22 [−1.36, −1.09] (random effects, HKSJ), between-session SD 0.13/0.18; on the window-force axis the same sessions gave v^1.04–1.09 β^−1.45 to −1.54 with twice the scatter, and the onset-minus-window difference was +0.21 [+0.11, +0.32] in β. An independent check confirmed the stroke mapping, gave the same onset-axis result with β-block bootstrap (−1.26 [−1.39, −1.12] for blocks of 10), showed that a nominal-command axis is as scattered as the window axis, and found censoring variants moving single sessions between −0.9 and −1.6 but none near −2. Thus, on the force applied at bow onset, these cello sessions follow approximately F_min ∝ v/β^1.2 — close to the simulated v/β form [4] and inconsistent with Schelleng's v/β² [1] — while the travel-affected window force steepens the apparent exponent. The onset axis was chosen after seeing the data, speed is confounded with recording block, and a within-scan β/time-order test was not completed.

## 4. Discussion

**What the collection supports.** (1) Measurement protocol enters the boundaries through at least four routes: travel-dependent force loss, an attack schedule tied to recording order (and so to force only for ascending scans), speed-as-session, and window placement. (2) A Z¹-scaled maximum-force law with one calibrated constant predicted the level of held-out sessions only within about ±0.2 (4 of 9 frozen tests inside their frozen intervals), with errors growing with the force ceiling; within-type tension pairs were heterogeneous; Z, tension, EI and the T2 batch cannot be separated here. (3) The between-diagram minimum-force variation is associated post hoc with a same-session load-decrease descriptor (surviving a screen-wide max-|r| correction), but the association weakens on the onset-force axis, under detection-limit censoring and without the reverse-scanned A1; a pure reading-drop mechanism is not ruled out; and frozen tests were untestable (D_T2) or favoured the lookup on boundary-level metrics (A4, D2). (4) For the minimum force of an unseen string, a lookup of an already-measured string was not beaten.

**Recommendations for users of this collection.**

- Report boundaries on both onset-force and window-force axes until the cause of the travel-dependent decrease is resolved.
- Record the scan direction and the attack of each stroke; treat each speed folder as a session.
- Align analysis windows by time since plateau onset *and* report bow travel.
- Report censored, no-band and multi-band columns separately; no-band is not a censoring direction.
- Include a model-free lookup baseline and state metric denominators.

**Sessions and play-in.** Within some types, sessions differed as much as impedance-normalised types, and play-in probably moved one string's minimum-force boundary while also changing the elevation-to-force mapping. String comparisons in this collection — and plausibly with other elevation-controlled bowing machines — need replicate specimens or sessions with recorded play-in state and a force calibration per session, not only same-day repeats.

**What is and is not new.** Z scaling of the maximum force follows from Schelleng's analysis [1] and has been examined across strings [2,12,13]; establishment versus maintenance [6,7], thermal friction [10,14], and batch-effect adjustment [15,16] are known. The dataset group lists session effects qualitatively for attack measurements [17]. Within the literature we searched we did not find a quantitative audit of these protocol features in this collection, nor once-only frozen tests of boundary laws on its unseen diagrams and high-force sessions; the failed out-of-range transfer is, to our knowledge, new information about how far low-force Schelleng fits extend.

**Friction.** In a public friction reconstruction of one string of uncertain identity (closest to B_T2) [14], the mean dynamic friction coefficient scaled as (F v)^−0.18; the friction contrast Δμ, which enters Schelleng's F_max, did not collapse on F v. The thermal-friction upper laws lost to plain Schelleng in all four high-force sessions. We therefore make no thermal-friction claim.

## 5. Limitations

- **Scope.** One robot, one monochord, cello G strings. Nothing here is a violin measurement; violin statements in the literature [2,3] are cited, not tested.
- **Small samples and confounding.** 11 diagrams and 4 high-force sessions over 8 type–tension conditions; speed, session, date, tension and scan direction are confounded.
- **Labels and forces.** Waveform-proxy labels; zero-offset uncertainty of tens of mN below 0.2 N; descriptor and boundaries share one force channel.
- **Physics predictors.** Losses were not included: on a monochord with low-mobility terminations, F_min should be controlled by internal string damping rather than a bridge resistance [10], and the dataset paper reports damping differences of 20–30 % [8]. "String properties" here means Z, EI, T, d and torsional stiffness only.
- **Conventions.** K = 2 and the single-band, no-band-as-lower-right-censored convention are operational assumptions; the upper estimate at high force is sensitive to the band rule.
- **Tests.** Local freezes (one with an external commit timestamp), one diagram or session per test; the ML baselines are trained on several windows per stroke with random-row internal validation (pseudo-replication).
- **Coverage.** All frozen predictions have now been scored once, except PACE* on B_T2 sample 2 (not applicable to its design) and the coldwarm archive (not processed).

## Author declarations

Conflict of interest: [to be completed]. Use of AI tools: analysis code, audits and parts of the text were drafted with AI assistants (Claude, Anthropic; a second coding agent); all numbers were checked against saved outputs [authors to confirm wording against the journal policy].

## Data availability

Raw data: the public IWK collection [9]. Code, frozen plans with hashes and outputs: authors' repository (snapshot on branch research/protocol-audit-20261019; later changes local, to be pushed with the authors' approval).

## References

[1] J. C. Schelleng, "The bowed string and the player," J. Acoust. Soc. Am. 53, 26–41 (1973).
[2] E. Schoonderwaldt, K. Guettler, and A. Askenfelt, "An empirical investigation of bow-force limits in the Schelleng diagram," Acta Acust. united Acust. 94, 604–622 (2008).
[3] R. Mores, "Complementary empirical data on minimum bow force," J. Acoust. Soc. Am. 142, 728–736 (2017).
[4] H. Beigi and G. Conneely, "Learned bow control on a measured bowed-string model: a revised minimum-bow-force law, a recurrent controller, and the domain of a supervision ceiling," arXiv:2609.14990v3 (2026) (preprint, simulation).
[5] H. Mansour, J. Woodhouse, and G. Scavone, "On minimum bow force for bowed strings," Acta Acust. united Acust. 103(2), 317–330 (2017).
[6] K. Guettler, "On the creation of the Helmholtz motion in bowed strings," Acta Acust. united Acust. 88, 970–985 (2002).
[7] P. M. Galluzzo, "On the playability of stringed instruments," Ph.D. thesis, University of Cambridge (2003) [section to be cited after full-text check].
[8] A. Lampis, V. Chatziioannou, and G. Scavone, "Experimental analysis of cello string types: influence on playability and tonal characteristics using Schelleng diagrams," Proc. Mtgs. Acoust. 58, 035013 (2025).
[9] A. Lampis (data collector), "A collection of bowed responses of cello strings," University of Music and Performing Arts Vienna, Zenodo records 17749110 (part 1), 17782542 (part 2), 17782552 (part 3), 17782565 (part 4), 17782538 (part 5), 17795326 (part 6), 17822016 (part 7), 17822037 (part 8) (2025).
[10] M. van Walstijn, V. Chatziioannou, A. Lampis, and E. Matusiak, "A thermal elasto-plastic friction model for bowed-string simulation," Acta Acust. 10, 47 (2026).
[11] P. Fearnhead, "Exact and efficient Bayesian inference for multiple changepoint problems," Stat. Comput. 16, 203–213 (2006).
[12] R. T. Schumacher, "Measurements of some parameters of bowing," J. Acoust. Soc. Am. 96, 1985–1998 (1994).
[13] R. Mores, "Maximum bow force revisited," J. Acoust. Soc. Am. 140, 1162–1171 (2016).
[14] Friction-reconstruction data of the IWK group, Zenodo record 15879128 (data citation).
[15] W. E. Johnson, C. Li, and A. Rabinovic, "Adjusting batch effects in microarray expression data using empirical Bayes methods," Biostatistics 8, 118–127 (2007).
[16] J. T. Leek et al., "Tackling the widespread and critical impact of batch effects in high-throughput data," Nat. Rev. Genet. 11, 733–739 (2010).
[17] A. Lampis, A. Mayer, and V. Chatziioannou, "Assessing playability limits of bowed-string transients using experimental measurements," Acta Acust. 8, 44 (2024).
[18] A. Lampis, A. Mayer, and V. Chatziioannou, "An experimental approach for comparing the influence of cello string type on bowed attack response," JASA Express Lett. 4, 113201 (2024).

## Figure captions

**Fig. 1.** Protocol audit. (a, b) Change of the measured bow force for nominal-force ranks 1–20 of A2 against bow travel (a) and elapsed time (b) at three speeds (the collapse on travel is weaker in A1). (c) Attack acceleration against onset-force rank in one β column (β ≈ 0.08, 0.05 m/s) for A2 (ascending scan) and A1 (reverse scan).

**Fig. 2.** Maximum bow force (post hoc). Upper-boundary level relative to the A1+A2 law against published ln Z (left) and ln EI (right); arrows show the A, B and C T1→T2 pairs; dashed line: slope 1 in ln Z with one constant from the five T1 diagrams, drawn on empirical-shape levels.

**Fig. 3.** Minimum bow force (post hoc). (a) Level against dec5_pl, A1 (reverse scan) marked. (b, c) Level and finite-column error with and without the descriptor, reference shape refitted per fold. (d) Cross-validated R² on the measured and nominal force axes.

**Fig. 5.** All nine frozen once-only tests of the strict upper law: level error against the session's force ceiling (`paper/fig5_frozen_upper_summary.png`; trend post hoc).

**Fig. 4.** High-force frozen test. (a) Level error of each frozen model for the four strings, with the 90 % prediction band. (b) Free β exponents with 95 % intervals against the frozen zones and model predictions.

---

## Internal notes (remove before submission)

**Figures.** English figures generated from saved results by `paper/scripts/fig_paper_en.py`: Fig. 1 = `paper/fig1_protocol_en.png` (A2 force change vs travel and time; attack vs onset-force rank for A1 reverse scan vs A2), Fig. 2 = `paper/fig2_upper_en.png`, Fig. 3 = `paper/fig3_lower_en.png`; Fig. 4 = `paper/fig_highforce.png` (`paper/scripts/fig_highforce.py`). Fig. 2 does not yet show the D pair (D_T2 is not in upper_law_t0.2.json).

**Ledger.** Frozen and scored once: A_T2 (all A_T2 columns of Table 1, Z exponents of the level shifts); D_T2 (strict upper level 0.002 and its interval, frozen no-Z comparator 0.182, column medians, lower primary untestable); A4 (H1/H1b/H2/H3 and denominators); high force (Table 2, β exponents, curvature, the pre-declared S1 sensitivity). Everything else is post hoc, including the random-effects estimate, the D pair, the per-speed residuals, the attack/scan analysis, detection-limit and onset-axis variants, the band-rule sensitivity at high force and the band-model test–retest (whose plan was frozen but which used development data).

**Still to do before submission.** Re-render Figs. 1–3 in English; complete [4], [5], [7], [9]; check the JASA-EL abstract limit; decide whether to analyse A3 post hoc; score D2 and B_T2 sample 2 when processed and add them without changing the frozen rules.
