# Measurement protocol and string impedance in the bow-force limits of a public robotic cello-bowing dataset

**Kim Taegeon, Jeong Jiyun, and Jeong Jihoon**
Gangwon Science High School, Republic of Korea

Draft v1, 2026-10-19 (JASA Express Letters format, not yet submitted). Internal notes at the end must be removed before submission.

---

## Abstract

Public robotic bowing data are increasingly used to test bow-force limit laws, but protocol effects can be mistaken for string physics. We re-analysed 11 Schelleng diagrams (66 000 strokes; cello G strings, four constructions, two tensions) from a public collection. Bow force is set by bow elevation, and at low force the measured force falls with bow travel (0.17 N over 14 cm in one diagram). The attack acceleration is a design function of force and bow position, and each bow speed is a separate recording block. Settling-time misalignment had produced a spurious minimum-force speed exponent. Under locally frozen plans, the maximum-force level transferred to two held-out higher-tension strings, consistent with Schelleng's impedance scaling but not separable from a tension-group offset. Post hoc, minimum-force levels correlated with a session load-decrease descriptor (r = 0.88). Neither physical laws nor machine learning beat a model-free lookup for unseen-string minimum force.

---

## 1. Introduction

Helmholtz motion of a bowed string occurs only between a minimum and a maximum bow force. Schelleng's classical analysis gives a maximum force proportional to Z v/β and a minimum force proportional to Z² v/(R β²) [1]. Here Z is the characteristic impedance of the string, v the bow speed, β the relative bow–bridge distance, and R the resistance of the termination. Measurements and simulations disagree about the minimum force. Schoonderwaldt et al. found a minimum force nearly independent of bow speed on violin strings [2]. Mores reported a non-zero baseline even at very low bow speed [3]. A recent simulation study proposed a minimum force proportional to Z v/β [4]. Part of the disagreement may come from what is being measured. Theory describes the force needed to *sustain* Helmholtz motion [5]. A stroke that starts from rest must first *establish* it, and simulations and bowing-machine experiments show that establishment and maintenance differ [6,7]. The attack (initial acceleration and force) also controls whether Helmholtz motion is established [8].

A public collection from the IWK (University of Music and Performing Arts Vienna) now provides complete Schelleng diagrams for cello G strings of four constructions at two tensions, recorded by a bowing robot [9]. It has already been used to validate a thermal friction model [10]. In such data the boundaries are read from finite windows of individual strokes. A fitted "law" therefore mixes string physics with the details of the protocol: how the force is applied, how the stroke starts, how long the stroke has settled, and which strokes share a recording session.

This letter reports what the collection can and cannot show about bow-force limits. We contribute four results:

1. a quantitative audit of the protocol, including a settling-time confound that produced a spurious speed exponent;
2. two locally frozen out-of-sample tests showing that the maximum-force level transfers across tensions in a way consistent with Schelleng's Z scaling, together with the reasons this is not a confirmation of Z scaling;
3. a post hoc association between minimum-force levels and a session-level load-decrease descriptor, which a frozen test could not confirm;
4. a negative result: no model beat a model-free lookup for the minimum force of an unseen string.

We state for every result whether it was pre-specified or post hoc.

## 2. Data and methods

**Data.** We used parts 1, 2, 4, 5 and 6 of the IWK collection [11] (CC BY 4.0). Each Schelleng diagram contains 6000 strokes: three bow speeds (0.05, 0.10, 0.20 m/s) × 40 bow positions (β = 0.02–0.20) × 50 nominal forces (0.1–4 N, logarithmically spaced) [9]. Bow force, bow speed, bridge force and nut force are sampled at 50 kHz.

We analysed 11 diagrams. Five are at the lower tension T1: A1, A2 (two type-A samples), B1, C1 and D1. Five are at the higher tension T2: A_T2, B_T2s1, C_T2, and B_T2r1/r2 (two same-day repeats from the variability set). D_T2 was held out. Impedance Z and bending stiffness EI are the published values (Table 1 of Ref. 9). The number of physically distinct strings is not documented, so we count diagrams, not independent strings.

**Labels and windows.** Each 0.2-s window is labelled by a fixed bridge-force waveform classifier. The classifier folds the signal into periods and counts flybacks per period. "Helmholtz" (H) means one flyback per period. Its amplitude thresholds were set once on A2 and applied unchanged to all diagrams. The label is a proxy, not a direct measurement of string motion.

The primary windows start t = 0.2 s after the bow-speed plateau begins. The force assigned to a window is the mean measured bow force in that window, not the nominal force.

**Boundaries and levels.** For each speed and β column, the 50 different strokes are sorted by window force. The lower boundary is the interval between the first run of K = 2 force-adjacent H strokes and the force just below it. The upper boundary is the interval between the last such run and the next force. Boundaries outside the measured range are recorded as censored, not imputed. A column with no H run counts as right-censored for the lower boundary and as unobservable for the upper boundary. This follows a single-band convention.

Boundary laws log F = c + a log v + b log β + Δ_d are fitted by interval-censored Gaussian maximum likelihood. The diagram level Δ_d is the offset of diagram d from a reference shape.

**Load-decrease descriptor.** The descriptor dec5_pl is computed without labels. It uses the lowest 20 nominal forces at each speed (800 strokes per speed) and takes the median difference between two mean forces: the first 0.1-s plateau window, and the first 0.1-s window whose start lies at least 5 cm of travel into the plateau. The difference is taken as the later minus the earlier force, so negative values mean a decrease. The three speed medians are then averaged.

**Models.** All models were trained on A1+A2:

- a model-free lookup: the median boundary of the 5 nearest β columns at the same speed;
- Schelleng (Z² v/β² lower, Z v/β upper) and Beigi (Z v/β) forms with one constant each;
- a "free-exponent" physical law (c, a, b free), with Z entering as a fixed offset (2 log Z lower, log Z upper);
- logistic regression and histogram gradient boosting on force, speed, β and window time.

The error is the median |log(F_pred/F_mid)| over columns with finite intervals, averaged over diagrams. F_mid is the geometric midpoint of the observed interval.

**Frozen tests and post hoc analyses.** Two out-of-sample tests followed analysis plans that were saved and time-stamped before the target data were processed. These plans were frozen locally, not publicly pre-registered. Each test was scored once.

- A_T2: plan frozen 2026-10-18 21:38 KST. Primary hypothesis: the free-exponent law with published Z beats the lookup on both boundaries. Secondary: Z exponent = level shift / log(Z_T2/Z_T1).
- D_T2: plan and coefficients frozen 2026-10-19 01:19 KST with file hashes. The predictions were saved from control signals before any label was computed.
  - Upper boundary: strict law F = cZv/β with c from the five T1 diagrams; tolerance 0.12 log.
  - Lower boundary: dec5_pl regression vs property regressions; margin 0.23 log.

Everything else reported here is post hoc. This includes the descriptor choice, which was screened among about ten session descriptors. Unless stated otherwise, intervals are 95% percentile bootstraps over β columns (paired across speeds where stated). They are conditional on the recorded sessions and omit training and between-session uncertainty.

## 3. Results

### 3.1 Protocol audit

**Force control.** The dataset paper states that the bow force "was not directly controlled... but solely from the elevation of the bow" [9]. A later model study notes "a small normal load reduction" during the steady stage [10]. We quantify this reduction.

- **Size.** In A2, for nominal-force ranks 1–20 at 0.05 m/s, the measured force fell by 0.044, 0.088, 0.125, 0.153 and 0.174 N after 2, 5, 8, 11 and 14 cm of travel (Fig. 1a). The decrease was nearly identical at the three speeds when plotted against travel (between-speed SD 0.0016 N). Against elapsed time the speeds separated (SD 0.040 N).
- **Dependence on bow motion.** In A2, while the bow moved, the reading fell at 0.020–0.180 N/s. Before the bow started it changed by at most 0.0021 N/s.
- **Session dependence.** In A1 the decrease was 2.5–4 times smaller (0.044–0.069 N at 14 cm). In 92.4–100% of A2's lowest-onset-force strokes per β column, the final window read less than half the onset force; in A1 the share was 24.3–33.5%.
- **Low-force reliability.** Whether this is a real force change or a pose-dependent sensor error is unresolved. Readings after contact was lost were not zero: folder medians ranged from −14.9 to +61.0 mN. In one A1 folder the reading drifted from +158 to +4 mN over the recording order (correlation with trial index +0.95). Forces below about 0.2 N are therefore uncertain by tens of millinewtons.

**Attack.** The bow-speed ramps are linear. In A2, 45%, 51% and 67% of strokes at 0.05, 0.10 and 0.20 m/s use a constant 4.80-cm ramp distance. This gives accelerations of 0.026, 0.104 and 0.417 m/s². Near the lower boundary, the attack at 0.05 m/s is therefore 16 times gentler than at 0.20 m/s. The remaining strokes follow a ∝ F_onset^(1.00–1.09) β^(2.04–2.65) (R² 0.95–0.97), up to 17–20 m/s². In the 22–23 β columns per speed where acceleration is not fixed, it rises monotonically with force rank (median within-column rank correlation +0.98 to +1.00). The attack is therefore a design covariate tied to force, β and speed (Fig. 1b). This is consistent with the stated aim of choosing accelerations that favour Helmholtz motion [10].

**Sessions.** Each speed was recorded as a separate complete β–force scan, so speed is confounded with recording block. One folder shows a folder-specific state. Over the 20 lowest force ranks, the median onset-to-nominal force ratio was 0.48 for A1 at 0.05 m/s. It was 0.73–0.75 for the other A1 folders and 0.80–1.11 for A2.

**Settling-time confound.** The 0.4-s analysis windows distributed with the dataset begin 1.55, 0.35 and 0.08 s after plateau onset at 0.05, 0.10 and 0.20 m/s (A2). A censored fit to these windows (A, B, C, D, A2 pooled) gave a minimum-force speed exponent of +1.77 [1.64, 1.90]. Aligning windows at equal time since plateau onset (t = 0–0.3 s) gave +0.57 to +0.67 for A1 and +0.94 to +1.04 for A2. The +1.77 therefore reflects the window placement, not a speed law. The matched-time exponents still differ between two samples of the same type. Because speed coincides with recording block, the speed exponent of a single diagram cannot be separated from session variation.

### 3.2 Maximum bow force

**Shape.** Pooled over the 11 diagrams (post hoc; cluster bootstrap over diagrams), the upper boundary scaled as v^0.70 [0.63, 0.76] β^−1.07 [−1.14, −1.01]. The β exponent is close to Schelleng's −1; the speed exponent is below 1.

**Repeatability within a string condition.** Diagram-level differences between same-type, same-tension diagrams were small: A1−A2 +0.005 log, B_T2r1−r2 −0.009 log, B_T2s1 vs the repeat pair +0.028 log. Column-level differences were larger (median |Δ| 0.083 log between the same-day repeats). Two repeats cannot fix the size of between-session variation.

**A_T2 (frozen).** For the higher-tension A string (Z ratio 1.122), the upper level rose by +0.197 [+0.106, +0.286] log relative to the A1+A2 law. This gives a Z exponent of +1.71 [+0.92, +2.49]; Schelleng predicts 1 (Fig. 2b). Upper-boundary errors were:

- free-exponent law with published Z: 0.121;
- lookup: 0.167;
- gradient boosting: 0.176;
- same law without Z: 0.189.

The primary hypothesis was nevertheless rejected because it failed on the lower boundary (Sec. 3.4).

**D_T2 (frozen).** The strict law F = cZv/β, with ln c = 0.966 calibrated on the five T1 diagrams, predicted the fitted D_T2 upper level within 0.002 log (conditional interval [−0.078, 0.091]). This is inside the frozen 0.12 tolerance.

The frozen no-Z comparator erred by 0.182 log, but it drops log Z without recalibrating c. A recalibrated no-Z model (post hoc) erred by 0.122 log, at the tolerance. The test also has little leverage:

- D_T2's ln Z differs from the T1 mean by only 0.124, so any Z exponent between about 0.02 and 1.95 would have passed.
- A constant T2 offset without Z, learned post hoc from A_T2, B_T2 and C_T2, predicts D_T2 within 0.010 log.

At the column level, the strict law's median error was 0.124 log, against 0.226 for the uncalibrated no-Z variant, over 48 finite columns. In addition, 63 of 111 scored columns were right-censored at the force ceiling.

**Pooled tension pairs (post hoc).** The Z exponents of the three T1→T2 pairs (inverse-variance fit) were heterogeneous: A +1.71 [0.86, 2.47], B −0.51 [−1.79, 0.75], C +3.25 [1.11, 5.12]. The pooled value was +1.28 [0.61, 1.85], with χ² = 13.4 on 2 degrees of freedom for a common slope (Fig. 2a). Within T1, the strict-law residuals had SD 0.118 log, which Z does not explain.

### 3.3 Minimum bow force

**Variation between diagrams.** Lower-boundary levels varied across the ten development diagrams with SD 0.218 log. They varied even within a string condition:

- A1 and A2 differed by 0.233 log (intervals [0.045, 0.187] and [−0.227, 0.001]).
- The B_T2 same-day repeats differed by 0.18 log (−0.318 vs −0.139).

**Association with the session descriptor (post hoc).** The levels correlated with dec5_pl (range −0.120 to −0.032 N): r = 0.88, reproduced as 0.8798 by an independent implementation (Fig. 3a). The published string properties correlated more weakly (|r| ≤ 0.52), and all had negative leave-one-diagram-out R² (−0.34 to −0.60). The association depends on the force axis:

- measured force, six K/t variants: r = 0.815–0.880;
- nominal force: r = 0.675–0.707.

**Predictive value (post hoc).** We refitted the reference shape and the level regression inside each fold. The descriptor reduced the level error from 0.213 to 0.101 log (leave-one-diagram-out) and from 0.221 to 0.131 log (leave-one-type-out). It increased the median error over finite columns from 0.294 to 0.316 and from 0.319 to 0.354 (843 of 1200 columns finite). The interval distance, which includes censored columns, changed only from 0.315 to 0.303. The descriptor therefore predicts the overall level shift of a diagram, not the individual boundaries.

**What the descriptor misses.** It does not explain the spread among the three B_T2 diagrams: dec5_pl was −0.091, −0.089 and −0.092 N while the levels were −0.26, −0.318 and −0.139. The descriptor and the boundary also share the same force channel, and reverse causation is not excluded.

**A_T2 (frozen secondary).** The lower level of A_T2 was 0.539 [0.382, 0.722] log below the A1+A2 law, i.e. 42% lower. The implied Z exponent was −4.69 [−6.27, −3.32], opposite in sign to Schelleng (+2) and Beigi (+1). A_T2 also had the most negative dec5_pl of the ten diagrams (−0.120 N). This is an observation, not evidence that the session caused the shift. Restricting the fit post hoc to columns with boundaries above 0.2 N reduced the shift to −0.10 log.

**D_T2 (frozen).** The descriptor of D_T2 (−0.038 N) lay inside the training range. The frozen level predictions were +0.074 (descriptor model), −0.298 (mean) and −0.372 (log Z + log EI).

- **Primary endpoint: untestable.** The level fit failed because one column had a zero-width interval, with both ends at the 1-mN force floor.
- **Secondary column medians (111 finite columns).** Errors were 0.286 (descriptor), 0.328 (mean), 0.325 (log EI), 0.376 (log Z) and 0.386 (log Z + log EI). All differences were below the frozen 0.23 margin, so the models were indistinguishable.
- **Post hoc repairs.** Repairing the zero-width interval gave levels of −0.09 to −0.13. Absolute errors were then 0.17–0.20 for the descriptor model and 0.17–0.21 for the mean, again indistinguishable.

### 3.4 Predicting the boundaries of an unseen string

Table 1 summarizes the comparison with training on A1+A2.

**Development diagrams (post hoc).** On five diagrams that had already been inspected, evaluated with corrected censoring and support handling, the lookup and the free-exponent law with published Z tied on the lower boundary (0.269 vs 0.268 log). Every other model was worse. On the upper boundary, the free-exponent law scored best (0.155) over its scorable columns. On the 181 columns where all models gave point predictions, gradient boosting was marginally better (0.176 vs 0.180).

**A_T2 (frozen).** On the lower boundary the lookup was best (0.250), followed by gradient boosting (0.271). The free-exponent law with published Z was worse (0.520), and strict Schelleng with Z was worst (0.939).

**Wider post hoc check.** We also ran a leave-one-diagram-out check over all 11 diagrams, including D_T2 after its frozen scoring. On finite-column error, the plain lookup (0.258, or 0.251 with clipping) was not beaten by any of 37 other variants. These included lookup-plus-covariate, mean-level, string-property and session-descriptor models; the best scored 0.275. Some descriptor variants had smaller interval distance (0.221 vs 0.279) and a better log score, so the ranking depends on the metric.

**Table 1.** Median |log(F_pred/F_mid)| over finite columns, averaged over diagrams; training on A1+A2. Development values (B1, C1, D1, B_T2r1, B_T2r2) come from the corrected post hoc evaluation, with the number of scorable columns in parentheses. A_T2 values were scored once under the frozen plan with the evaluation code of that time (A_T2 had 82 finite lower and 53 finite upper columns of 120). Dashes mark models not in the A_T2 plan.

| Model | Lower, development | Upper, development | Lower, A_T2 (frozen) | Upper, A_T2 (frozen) |
|---|---:|---:|---:|---:|
| Lookup (5 nearest β, same speed) | 0.269 (376) | 0.188 (259) | **0.250** | 0.167 |
| Schelleng + published Z | 0.647 (376) | 0.195 (259) | 0.939 | 0.215 |
| Beigi + published Z | 0.306 (376) | 0.195 (259) | 0.455 | 0.215 |
| Free exponents + published Z | **0.268** (376) | **0.155** (259) | 0.520 | **0.121** |
| Free exponents, no Z | — | — | 0.347 | 0.189 |
| Logistic regression | 0.307 (363) | 0.258 (181) | — | — |
| Gradient boosting | 0.288 (364) | 0.169 (241) | 0.271 | 0.176 |

## 4. Discussion

**Recommendations for users of this collection.**

- Use the per-window measured force, not the nominal force.
- Align analysis windows by time since plateau onset, and report bow travel.
- Treat each speed folder as a separate session.
- Report censored and empty columns.
- Include a model-free lookup baseline.

The window-alignment step alone changes the minimum-force speed exponent from +1.77 (published windows, pooled) to +0.57–1.04 (A1, A2), and the remaining difference between two samples of one type is not separable from session.

**Two boundaries, two behaviours.** The two boundaries behave differently in these data.

- *Maximum force.* The upper-boundary level was repeatable among diagrams of the same string condition (|Δ| ≤ 0.028 log). Under frozen plans it transferred to two held-out higher-tension strings, with errors of 0.121 log (A_T2 columns) and 0.002 log (D_T2 level).
- *Minimum force.* The lower-boundary level differed by up to 0.23 log between diagrams of the same condition, and its between-diagram variation tracked a session descriptor more closely than the string properties.

This pattern may also explain why a lookup was hard to beat for the minimum force. Within-condition differences in the lower level (0.18–0.23 log) were as large as the between-diagram SD (0.218 log). String-property models therefore had little stable signal to learn.

**What is and is not new.** Z scaling of the maximum force follows from Schelleng's analysis [1] and has been examined across strings before [2,12,13]. What we add is a time-stamped out-of-sample test within one dataset where pitch is fixed and tension varies. It is consistent with Z scaling but cannot separate Z from the tension group, whose sessions are also months apart in the folder dates.

The sub-linear speed exponent of the upper boundary (0.70) agrees in direction with earlier velocity-dependent modifications of Schelleng's maximum-force law [2,12,13]. In a public friction reconstruction for one string [14], the mean dynamic friction coefficient during Helmholtz motion scaled as (F v)^−0.18 (R² 0.95, 31 windows), with no residual dependence on slip speed v/β. This is consistent with power-controlled (thermal) rosin friction [6,15]. Inserted into Schelleng's maximum-force condition, it implies speed exponents of 0.48–0.94, depending on the force subset; the measured exponents of ten diagrams span 0.49–0.86. However, the one-parameter thermal law also implies a relation between the speed and β exponents, and that relation was violated (residual −0.216 [−0.282, −0.175]). The thermal account is therefore at most partial (post hoc, one string).

The session-descriptor adjustment of minimum-force levels is an application of standard batch adjustment [16,17]. It is not a new estimator. The same group lists session-level systematic effects qualitatively for attack measurements [18]. The dataset paper judges string differences against a same-day repeat [9]. Schoonderwaldt et al. considered whether imperfect force control biased their minimum forces [2]. Within the literature we searched, we did not find a previous quantitative audit of these protocol features in this dataset, or of their association with the minimum-force levels.

## 5. Limitations

- **Scope.** All measurements come from one robot and one monochord with cello G strings. Nothing here is a violin result.
- **Small sample.** There are 11 diagrams, 8 type–tension conditions, and an unknown number of physically distinct strings. Speed, session, date and tension are confounded.
- **Labels and forces.** The labels are a waveform proxy. Forces below about 0.2 N carry zero-offset uncertainty. The descriptor and the boundaries share one force channel. The cause of the travel-dependent decrease (real force change or pose-dependent reading error) is unresolved.
- **Conventions and censoring.** The K = 2 rule and the single-band censoring convention are operational. Upper boundaries are often right-censored at the force ceiling (63 of 111 columns in D_T2).
- **Test design.** The frozen tests were local rather than publicly registered, and each covers one diagram. The descriptor was selected post hoc from about ten candidates. Bootstrap intervals are conditional on the recorded sessions.
- **Interpretation.** No causal decomposition of string physics and protocol is claimed.

Frozen predictions for further unseen diagrams (including high-force parts that lift the 4-N ceiling) have been stored and will be scored once when processed.

## Author declarations

Conflict of interest: [to be completed by the authors]. Use of AI tools: analysis code, audits and parts of the text were drafted with AI assistants (Claude, Anthropic; a second coding agent), and all numbers were checked against saved outputs; [authors to confirm wording against the journal policy].

## Data availability

The raw data are the public IWK collection [11]. Analysis code, frozen plans with hashes, and outputs are in the authors' repository [URL to be added; current snapshot on branch research/protocol-audit-20261019].

## References

[1] J. C. Schelleng, "The bowed string and the player," J. Acoust. Soc. Am. 53, 26–41 (1973).
[2] E. Schoonderwaldt, K. Guettler, and A. Askenfelt, "An empirical investigation of bow-force limits in the Schelleng diagram," Acta Acust. united Acust. 94, 604–622 (2008).
[3] R. Mores, "Complementary empirical data on minimum bow force," J. Acoust. Soc. Am. 142, 728–736 (2017), doi:10.1121/1.4996141.
[4] N. Beigi and B. Conneely, arXiv:2609.14990v3 (2026) [title to be added].
[5] H. Mansour, J. Woodhouse, and G. Scavone, "On minimum bow force for bowed strings," Acta Acust. united Acust. 103, 317 (2017), doi:10.3813/AAA.919060.
[6] J. Woodhouse, "Bowed string simulation using a thermal friction model," Acta Acust. united Acust. 89, 355–368 (2003).
[7] P. M. Galluzzo, "On the playability of stringed instruments," Ph.D. thesis, University of Cambridge (2003).
[8] K. Guettler, "On the creation of the Helmholtz motion in bowed strings," Acta Acust. united Acust. 88, 970–985 (2002).
[9] A. Lampis, V. Chatziioannou, and G. Scavone, "Experimental analysis of cello string types: influence on playability and tonal characteristics using Schelleng diagrams," Proc. Mtgs. Acoust. 58, 035013 (2025), doi:10.1121/2.0002111.
[10] M. van Walstijn, V. Chatziioannou, A. Lampis, and E. Matusiak, "A thermal elasto-plastic friction model for bowed-string simulation," Acta Acust. 10, 47 (2026), doi:10.1051/aacus/2026042.
[11] IWK bowed cello string collection, Zenodo records 17749110 (part 1), 17782542 (part 2), 17782565 (part 4), 17782538 (part 5), 17795326 (part 6) (2025).
[12] R. T. Schumacher, "Measurements of some parameters of bowing," J. Acoust. Soc. Am. 96, 1985–1998 (1994).
[13] R. Mores, "Maximum bow force revisited," J. Acoust. Soc. Am. 140, 1162–1171 (2016), doi:10.1121/1.4960626.
[14] Friction-reconstruction data of the IWK group, Zenodo record 15879128; A. Lampis, E. Matusiak, A. Mayer, and V. Chatziioannou, "Friction force reconstruction in bowed cello strings," J. Acoust. Soc. Am. 157 (4_Suppl), A203 (2025) (abstract).
[15] J. H. Smith and J. Woodhouse, "The tribology of rosin," J. Mech. Phys. Solids 48, 1633–1681 (2000).
[16] W. E. Johnson, C. Li, and A. Rabinovic, "Adjusting batch effects in microarray expression data using empirical Bayes methods," Biostatistics 8, 118–127 (2007).
[17] J. T. Leek et al., "Tackling the widespread and critical impact of batch effects in high-throughput data," Nat. Rev. Genet. 11, 733–739 (2010).
[18] A. Lampis, A. Mayer, and V. Chatziioannou, "Assessing playability limits of bowed-string transients using experimental measurements," Acta Acust. 8, 44 (2024), doi:10.1051/aacus/2024034.

## Figure captions

**Fig. 1.** Protocol audit (A2, with A1 for comparison).
(a) Change of the measured bow force for nominal-force ranks 1–20, plotted against bow travel and against elapsed time, at three bow speeds. The curves collapse on travel, not on time.
(b) Top: median measured force over the plateau by force-rank band. Bottom: attack acceleration (log₁₀ m/s²) over the β × force-rank grid at each speed.

**Fig. 2.** Maximum bow force.
(a) Upper-boundary level relative to the A1+A2 law, against published ln Z (left) and ln EI (right). Arrows show T1→T2 pairs. The dashed line has slope 1 in ln Z (Z¹ scaling), with one constant calibrated on the five T1 diagrams. It is drawn on levels relative to the A1+A2 empirical shape, so it is not the strict cZv/β law. Bars: column-bootstrap 95% intervals.
(b) Frozen A_T2 test: boundary errors by model, and Z exponents of the upper and lower level shifts against Schelleng's predictions (1 and 2).

**Fig. 3.** Minimum bow force (post hoc).
(a) Lower-boundary level against the load-decrease descriptor dec5_pl for the ten development diagrams.
(b–c) Level error and finite-column error, with and without the descriptor; the reference shape is refitted in every fold.
(d) Cross-validated R² of the level regression on the measured and nominal force axes; panels (a) and (d) use the fixed A1+A2 reference shape.

---

## Internal notes (remove before submission)

### A. Figures: existing files and required changes

All existing figures have Korean axis labels and titles. They must be re-rendered in English from the listed scripts before submission. No figure was regenerated for this draft.

| Fig. | Existing file (absolute path) | Generating script | Status / required change |
|---|---|---|---|
| 1a | /Users/jeongjihoon/violin-rne/claude_extra/fig_drift_test.png | /Users/jeongjihoon/violin-rne/claude_extra/fig_drift_test.py | Korean labels → English. Content matches Sec. 3.1. |
| 1b | /Users/jeongjihoon/violin-rne/claude_extra/fig_force_decay.png | /Users/jeongjihoon/violin-rne/claude_extra/fig_force_decay.py | Korean labels → English. Consider merging with 1a into one two-row figure. |
| 2a | /Users/jeongjihoon/violin-rne/novelty/upper_law/fig_upper_law.png | /Users/jeongjihoon/violin-rne/novelty/upper_law/fig_upper_law.py | Korean → English. Relabel the dashed line as "Z¹ line on empirical-shape levels, one constant from 5 T1 diagrams", not "Schelleng" (astra_20261019/report_audit.md §1). D_T2 is not plotted; add it as the frozen point. |
| 2b | /Users/jeongjihoon/violin-rne/final/fig_confirm_A_T2.png | /Users/jeongjihoon/violin-rne/final/fig_confirm.py | Korean → English. Mark the panel as "frozen plan, scored once". |
| 3 | /Users/jeongjihoon/violin-rne/astra_20261019/fig_protocol_audit.png | /Users/jeongjihoon/violin-rne/astra_20261019/build_audit_outputs.py | Korean → English. Already distinguishes refit vs fixed shape and measured vs nominal axis. |
| (opt.) | /Users/jeongjihoon/violin-rne/novelty/thermal_upper/fig_thermal_upper.png | /Users/jeongjihoon/violin-rne/novelty/thermal_upper/fig_thermal_upper.py | Supplementary only; supports the one-sentence thermal remark. |
| do not use | /Users/jeongjihoon/violin-rne/final/fig_final_prediction.png | /Users/jeongjihoon/violin-rne/final/fig_final.py | Pre-correction comparison (no lookup baseline, old censoring and grid handling). Superseded by Table 1. |

### B. Pre-specified vs post hoc

**Locally frozen and scored once:**

- A_T2: the primary H1 and all values in Table 1's A_T2 columns; the Z exponents of the level shifts.
- D_T2: the strict upper-level error 0.002, its conditional interval, the frozen no-Z comparator 0.182, and the column medians (upper 0.124 / 0.226 / 0.113; lower 0.286 / 0.328 / 0.325 / 0.376 / 0.386).

**Post hoc:**

- everything in Sec. 3.1;
- the pooled upper exponents;
- the pair exponents and pooled +1.28;
- the fair no-Z 0.122 and the leverage analysis;
- the T2 batch-offset alternative;
- r = 0.88 and all descriptor results;
- the nested refits;
- the D_T2 repairs;
- the development-comparison numbers;
- the 11-diagram leave-one-out check;
- the thermal-friction remarks.

**Caveat on the development comparison.** The original development comparison was frozen in final/ANALYSIS_PLAN_FINAL.md, but the lookup, the published-Z variants and the corrected censoring are post hoc. Do not import its earlier bootstrap significance values (final/verify/robustness/run_boot.log) into Table 1. They were computed under the pre-correction evaluation code.

### C. Wording constraints (from novelty/prior_art/prior_art.json)

**Do not write:**

- "confirmed Z scaling";
- "Z gain 0.18" (use the fair 0.122);
- "maximum = physics, minimum = protocol";
- "minimum force is determined by protocol";
- "A_T2 −42% caused by the session";
- "first audit" without "within the literature we searched";
- "new estimator";
- "explained by thermal friction".

**Plural "held-out strings".** Use it only for the level-transfer statement. For the strict cZv/β law, only D_T2 was frozen. For A_T2 the frozen model was the free-exponent law with published Z.

### D. Length

The main text (Sec. 1–5 including Table 1) is about 3,500 words, which is longer than about 4 journal pages with three figures. Cut candidates for v2:
- merge the Sec. 3.1 bullets into two paragraphs;
- move the D_T2 post hoc repairs and the 11-diagram leave-one-out check to supplementary material;
- shorten the thermal-friction paragraph to one sentence.

### E. Reference verification status

**Checked against local full text or reference lists:**

- [1], [2], [6], [7], [8], [9], [10], [12], [15], [18].
- [6] is also verified by its own title page.
- [12], [15], [7] and [8] are checked via the reference lists of Woodhouse 2003 and Woodhouse & Galluzzo 2025 in /Users/jeongjihoon/violin-rne/novelty/prior_art/texts/.

**Abstract level only:**

- [3], [5], [13] (prior_art.json); check page ranges.
- [5] has no end page.
- [4]: the title is missing.
- [14]: the title is from prior_art.json; the abstract was not read.

**From general knowledge, not in local files; verify before submission:**

- [16], [17] (titles, pages).

**Not yet checked (unread risks listed in prior_art.json):**

- the Lampis PhD thesis;
- the full Lampis et al. 2025 friction-reconstruction abstract.

**Note.** The final Korean report cites Zenodo part 1 as 17749111. The download pipeline (timeres/replicate.py) uses 17749110. [11] follows the pipeline; confirm on Zenodo.
