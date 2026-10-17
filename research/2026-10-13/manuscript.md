# Assessing speed transfer and observation uncertainty in bowed-string playable-set prediction

**Working research draft — updated 17 October 2026. Not submission-ready.** This draft distinguishes retrospective development experiments from the completed, separately frozen A2 evaluations. No author list, acceptance claim or independently verified physical reference labels are supplied. The companion [Korean report](README.md) is the main project entry point.

## Abstract

Predicting the bowing conditions that support stable string motion requires assumptions about both physical similarity and the observation used to identify the motion. We examine these assumptions using 23,301 eligible development conditions from a robotic cello-string dataset, 1,041 classified violin D-string conditions recovered from a published monochord experiment, and 5,892 eligible conditions from an additional cello specimen evaluated under frozen protocols. The datasets differ in apparatus, acquisition range and classification procedure; their scores are interpreted separately. In retrospective cello transfer, force–speed coordinates improve simple classifiers, while a coherent observation-family robust objective does not surpass the strongest simple baseline. On the published violin D string, uniform force–speed scaling instead underperforms static coordinates. An affine log-force transport candidate improves on a literature-informed asymmetric ensemble on average, but provides little aggregate improvement on cello. Both prespecified A2 primary method evaluations are negative: the robust candidate scores 0.6355 versus its source-selected baseline's 0.6596; the primary two-speed flow scores 0.6536 versus matched fixed-width and learned-exponent scores of 0.6679 and 0.6623. Exact observer evaluation changes individual worst-case IoUs only slightly, yet shared-observer comparisons reveal model-ranking reversals and consequential aggregation differences. Boundary audits also distinguish predictive fit from identification of a width law. These results support an auditable transfer study, not a generally superior new method or validated performance on assembled violins.

## 1. Research question and prior work

The Schelleng diagram describes bowing regimes as a function of relative bow position and normal bow force, conditional on speed and string properties. Published violin-string experiments already establish that the lower and upper force limits need not obey the same speed dependence [1]. The recent public cello-string collection provides controlled bowing responses across string constructions and speeds, and its associated analysis estimates playable-region borders and discusses acquisition truncation [2,3]. Plotting these maps or fitting alternative power laws therefore cannot alone constitute the present contribution.

Modern bowed-string models already include finite bow width, torsion, hair compliance and elasto-plastic or thermal friction [4,5]. Neural solvers for a bowed oscillator and recent computational violin control also exist [6,7]. This study does not introduce those mechanisms or claim that its inexpensive point-contact simulator is the state of the art. Its question is narrower: **which transfer conclusions survive changes in physical similarity assumptions, numerical implementation and observation definition?**

We examine entire observed positive sets rather than treating a fitted boundary outside the acquisition range as measured truth. We also distinguish a globally shared observer setting from an independent unknown label at every cell. Performance evaluation without ground truth has established partial-identification precedents, including [12], whose assumptions about latent-label marginals differ from our restricted deterministic observer family. These are experimental design choices with statistical and optimization precedents, not by themselves claims of algorithmic novelty.

## 2. Data and estimands

### 2.1 Public cello-string measurements

The preserved tables contain 24,000 trials from four constructions/specimens of cello G strings and three nominal speeds. A positive-force and velocity-plateau eligibility rule retains 23,301 rows. Predictors receive relative bow position, measured normal force and measured speed; no waveform feature enters their input. Supervision is derived from the project's previously computed bridge-force features. It is not the dataset author's independent reference classification.

All four sample1 grids were previously examined. Their comparisons are retrospective development, even when a particular model uses source-only fitting and selection. Classification nuisance settings are fixed from source A; experiments that fit a predictor on B, C or D therefore do not constitute wholly independent refitting of the complete observation pipeline. Full provenance, checksums and exclusions are in `evidence_audit.md` and the compact input manifest.

The additional Type A sample2 archive contains 6,000 trials in three speed panels. The unchanged eligibility rule excludes 108 nonpositive-force rows and no off-plateau rows, retaining 5,892 conditions: 1,986, 2,000 and 1,906 at 0.05, 0.10 and 0.20 m/s. Its complete archive checksum was verified before decoding. The original model, observer, evaluator and preprocessing freezes, and the separate later transport addendum, all precede local decoding. This is documented local chronology, not formal public preregistration or certification of every possible external exposure. A2 is one additional cello-string specimen, not an independent collection of violin instruments.

### 2.2 Published violin-string classifications

We recover vector markers from Figures 4 and 7 of [1], preserving figure, panel, marker primitives, physical coordinates and extraction residuals. Of 1,584 grid slots, 1,171 have strict systematically analyzed classifications. The D-string comparison uses 1,041 cells over four speeds. Fifteen unplotted D cells remain unknown. The E figure's gray interiors were not systematically analyzed, so its remaining 130 strict cells are unsuitable for ordinary whole-grid IoU and are excluded from headline evaluation.

Relative position is calculated from rounded reported bow–bridge distances divided by the reported 325 mm length; force and speed are nominal settings. Small PDF alignment residuals do not imply equally precise physical coordinates. Transition locations outside a sampled bracket or beyond the acquisition range are not observed.

The apparatus is a rigid monochord carrying violin strings, not an assembled violin. Published repeated trials have already been reduced to the authors' classifications; individual repetitions are unavailable. Two assistant visual checks of 24 sampled markers and checks of all separately recovered Table VI/VII entries found no mismatches. They are extraction checks, not an expert relabeling study. The asymmetric speed prior and these observations come from the same paper, so this comparison cannot independently discover that prior.

### 2.3 Target definitions

For the cello observations, an admissible family changes the relative flyback-amplitude fraction over [0.3,0.7] and periodicity cutoff over [0.75,0.85], while holding the earlier folding and reference-frequency procedure fixed. A single parameter pair applies to all cells in a map. The default rule is retained, and all family members are scored; no observer is selected to favor a predictor.

Primary cello evaluation averages the minimum IoU across 15 observer settings separately within each target panel. A pooled IoU minimized under one globally shared setting is also reported, as are bounds under a larger independent-cell envelope. These estimands are different. None supplies coverage of an unknown physical ground-truth set. Published violin-string scores instead use the authors' classifications and have a singleton label family.

## 3. Predictive models and exact observer evaluation

### 3.1 Baselines and robust objectives

Single-speed comparisons include simple polynomial logistic classifiers, RBF SVMs and standard tree baselines, with static and force–speed coordinates where applicable. A common RBF basis compares nominal cross-entropy, nominal soft IoU, consensus-label learning, a rectangular-envelope objective and a coherent-family robust objective. Source-only blocked cross-validation holds out contiguous relative-position columns. Fixed regularization grids and all fitted candidates are retained.

For predictions P and nested observed sets L and U, the exact independent-cell hard-IoU extrema are |P∩L|/|P∪U| and |P∩U|/|P∪L|, subject to the declared empty-union convention. Such a rectangular envelope generally allows label combinations that no single global observer can produce. The implementation's robust loss also has a nominal cross-entropy anchor. It should not be described as learning from ambiguous points solely through the envelope overlap term.

### 3.2 Affine speed transport candidate

Let b=log(β/0.1), u=log(F/1 N), and t=log(v/0.1 m/s). We study the affine flow du/dt=a₀+a₁b+ku. Its reference-speed coordinate is

z=exp(−kt)u−(a₀+a₁b)(1−exp(−kt))/k,

with continuous extension z=u−(a₀+a₁b)t at k=0. A six-coefficient quadratic logit in normalized b,z is constrained to be concave in z and fitted jointly with the three flow parameters. Thus the positive force set is a band or empty, and its log-force width scales with exp(kt). The total model has nine parameters. This is an established linear flow construction, not new group or neural-ODE mathematics.

The objective combines mean binary cross-entropy, coefficient regularization and shrinkage of the flow toward uniform force–speed scaling. Two-speed source tasks use five shared position-block folds. Three successful optimization starts are compared only by training objective. Every source fit is frozen before target scoring. The known asymmetric comparator averages two independently fitted source-speed bands, using the same position folds for source-only regularization selection. Its structure differs from the joint candidate, so that comparison alone does not isolate width deformation; the separately executed k=0 ablation addresses this distinction.

At a single reference speed, all three flow parameters have exactly zero likelihood gradient. Two observed speeds improve predictive constraints but do not prove physical identifiability, particularly with censored force ranges. Fitted flow parameters must not be interpreted as uniquely recovered material mechanisms.

The matched ablation fixes only k=0 while retaining the same concave base, beta-dependent translation, normalization, regularization grids and source folds. It uses one zero-k start; the full model uses three. A retrospective optimization audit finds that all selected final full fits' zero-start objectives are within 2.54e−11 of the best start. This supports the optimization comparison for these fits, but is not a general global-convergence guarantee.

### 3.3 Continuous observer extrema

For nonnegative weights w with positive total and fixed probabilities p, define soft Jaccard J(p,y)=Σwpy/[Σwp+Σw(1−p)y]. A zero union has score 1; the case Σwp=0 is handled separately. The fractional-program residual bound assumes Σwp>0. Each distinct observer label map changes only at finitely many feature thresholds. Dinkelbach-style fractional optimization [11] reduces an extremum to a globally weighted quadrant query in periodicity and flyback-ratio space. A sorted sweep and suffix-range tree solve that subproblem in O(N log N), giving O(kN log N) for k iterations. The exact finite-state reduction and residual bounds assume an exactly solved global subproblem and exact arithmetic; the floating-point implementation is not an interval-arithmetic certificate.

Fractional optimization for nondecomposable measures and maximum-weight quadrant algorithms are prior work [8,9]. The relevant contribution here is an audited application and evaluation of their utility, not a generic first algorithm. An independent dense finite-state implementation and randomized tests check the implementation. This oracle optimizes a pooled ratio; it does not directly optimize the mean of panel ratios.

## 4. Executed results

### 4.1 Single-speed cello transfer

Using A at 0.10 m/s as the source, a force–speed quadratic logistic baseline attains panel-average family-minimum IoU 0.6884. Within a separate shared RBF basis, the nominal-IoU, independent-envelope and coherent-family objectives attain 0.6738, 0.6574 and 0.6696, respectively. The coherent candidate fails to surpass the strongest simple baseline at each of the four source choices. The benchmark executes 1,645 fits with no optimizer failures; independent recomputation agrees with all 1,155 panel metric arrays.

The apparent importance of speed covariance is not solely an artifact of comparing a fixed-speed raw 3D model against a lower-dimensional prior: matched 2D comparisons preserve a substantial advantage on cello. However, a fixed speed exponent is prior information when only one source speed is observed. A two-speed learned-exponent model has aggregate family-minimum IoU 0.6946 versus 0.6940 for a raw 3D quadratic; this small difference is not evidence of equivalence or robust superiority.

### 4.2 Published violin-string transfer

For source speed 0.10 m/s, static and uniformly speed-scaled quadratic classifiers attain mean target IoU 0.8311 and 0.7747. Matched static, uniform and known-asymmetric boundary models attain 0.8252, 0.7638 and 0.8607. The asymmetric prior wins the macro comparison for each of four source speeds but loses to the static model for 0.10→0.05 m/s. Thus the favorable cello pattern does not justify a generally favorable violin-string scaling claim.

Across all six two-speed D-source pairs, affine transport attains mean target IoU 0.8613, compared with 0.8445 for the known asymmetric ensemble, 0.8421 for a learned uniform exponent and 0.8134 for a raw 3D quadratic. Its advantage over the asymmetric ensemble occurs in five of six tasks; for the highest-speed source pair it falls to 0.7547 versus 0.8037. On the twelve cello tasks, its aggregate 0.6948 differs little numerically from the strongest simple comparators, and its ranking changes across reasonable aggregations. These are overlapping tasks on few physical specimens, not independent replications suitable for a population-level significance claim.

The exactly matched k=0 model attains 0.8405 on D-string held-speed transfer and 0.6935 on the combined cello endpoint. Full transport beats it on five of six D tasks, but loses on the cello source-specimen held-speed aggregate (0.7136 versus 0.7193). This isolates a conditional benefit of width deformation on the published D-string experiment, without establishing a general benefit across the two corpora.

A subsequent source-diagnostic extension permits k(b)=k0+k1 b, retaining the same basis, normalization, objective, regularization grids, folds and initialization scheme. It adds one parameter without addressing the single-band restriction. Across all eighteen tasks, combined cello performance decreases to 0.6890, while D performance rises only to 0.8623, with two wins and four losses against the constant-k flow. It fails the declared requirement to improve both corpora against all three comparators. The candidate is rejected as a general improvement and is not added to either frozen A2 evaluation. The complete source selections, failed criterion and independent numerical review are preserved in `conditional_transport_assessment.md`.

### 4.3 Observer and numerical audits

Across 56 fixed predictor/source combinations, replacing the 15-rule grid with continuous observer extrema changes minimum pooled IoU by at most 0.000357 and by 0.0000598 on average. No strict within-source ranking reversals of these individually minimized scores occur. On one synthetic N=2,000 input, seven warmed repetitions give median 0.977 ms for the sweep solve and 65.815 ms for the dense solve. Including geometry preparation reduces the relative advantage; the exact record reports both timings. This local experiment is not a universal speed guarantee.

This maximin ranking is different from a shared-observer comparison. For each pair of frozen predictions P,Q, we also enumerate the extrema of IoU(P,S_eta)−IoU(Q,S_eta), using the same eta for both models. Among 364 pooled pairs in the original four-source benchmark, 52 reverse sign across admissible observers; the existing 15-rule grid already detects 51. Among 72 pooled two-speed transport comparisons, the counts are 40 and 36. All five newly detected pooled reversals have weaker-side margins below 0.001. The full audit includes 5,596 pair/scope comparisons and does not support a major computational bottleneck on this dataset.

Aggregation semantics can have a larger effect. The original mean of panelwise minima permits a different adverse observer in every panel. For two empirical model pairs, its preference is opposite to the macro ordering under every single globally shared observer. This is not an arithmetic error: the uncertainty sets and estimands differ. The shared-macro difference interval always contains the difference of shared-macro minima, but need not contain the difference of separately minimized panel means. A twelve-row constructed example and exact finite-state tests verify this distinction. The frozen A2 endpoints were retained unchanged; see `observer_dominance_audit.md` for all pairs and the derivation.

The historical simulator recurrence used g[(1−p)x+py] where its design assumed g(1−p)x+py. The historical calibrated parameters also violate the corrected passive-reflection condition. Their previous agreement scores cannot establish passive physical validity. The corrected simulator passes independent free-decay and implementation-parity checks. Nonetheless, seven of 96 binary memberships change between 400 and 800 kHz for the fixed 1.2-second linear-delay experiment. A first-order allpass delay reduces that count to five but worsens the preceding resolution step. Finite-rate agreement is not continuum certification.

Pure sine and symmetric triangle controls satisfy the historical waveform classifier's Helmholtz rule. End-point inversion experiments also exhibit a common-impedance counterexample: near-perfect two-end consistency can coexist with incorrect inferred contact states. Established endpoint-inversion work already addresses force/velocity reconstruction [10]; our experiments test assumptions rather than claim a new inverse relation.

### 4.4 Boundary identification and acquisition censoring

An exact source-geometry audit preserves observed positive components and finite or censored boundary brackets, without repairing labels. At the nominal cello rule, 366 of 480 position columns contain multiple positive components. Yet only 1,131 of 23,301 labels (4.854%) need disagree with a best contiguous interval, showing why component counts alone overstate the extent of mismatch. Only ten columns have a single band with both endpoints finitely bracketed; no source-speed pair has such a column fully bracketed at both speeds. Across all fifteen rules, minimum mismatch ranges from 4.111% to 8.137%. These are structural model-mismatch diagnostics, not estimates of labeling error.

Published D has only one minimum interval disagreement among 1,041 cells. Nevertheless, five of six speed-pair common-width-rate constraint intersections are empty under the retained exact boundary brackets. The high-speed pair has a nonempty interval but the poorest historical affine prediction and a maximum extrapolation-weight sum of 8.638 toward the lowest speed. Compatibility and predictive quality therefore need not rank together. These statements condition on nominal acquisition slices and retained classifications; unmodeled control, annotation and physical discrepancy prevent interpreting them as a refutation of a physical law. The audit supplies no externally justified curvature bound for out-of-support coverage. Full calculations are in `source_identifiability_audit.md`.

A subsequent [finite width-law diagnostic](../2026-10-14/width_law_assessment.md) quantifies the extra Hamming disagreement imposed by a shared width exponent, beyond independently positioned intervals at the two speeds. It retains free centers, censoring, tied forces and off-grid alternatives, with unrestricted exponent k. The added disagreement is at most twelve source rows (0.336% of the corresponding task) across all fifteen cello rules and at most three published-D rows. This is a nominal-control relaxation, not a certified training-loss lower bound for the implemented cello predictor, which receives slightly varying measured controls within each nominal column. Finite enumeration and an interval sweep give the minimum under the stated mathematical model; floating computation is checked by constructive endpoint witnesses and exhaustive small-instance tests, not interval arithmetic. Small source disagreement neither explains nor rules out a contribution to held-out IoU loss. The diagnostic has no target prediction guarantee and is not claimed as a new optimization theorem.

Restricting the width diagnostic to the predictor's `k ∈ [−2,2]` range raises
the optimum in six of 186 cases by one or three disagreements. All nominal
cello and published-D cases are unchanged. Twenty-three stored unrestricted
witnesses were outside the range, but seventeen admit equally good in-range
alternatives, constructed and checked explicitly. This
[sensitivity analysis](../2026-10-14/width_bound_sensitivity.md) retains the
free-center and nominal-control relaxations; it does not remove the distinction
between source disagreement and predictive performance.

### 4.5 Completed fixed-model A2 evaluation

Both evaluations completed on 14 October 2026 at 06:40 UTC (15:40 KST), preserving their original primary estimands and all declared models. Under the original A/0.10 m/s training protocol, the coherent-family robust candidate scores 0.635452 versus 0.659570 for the source-selected covariant logistic baseline, a difference of −0.024118. All fourteen model results are retained. The baseline's source selection precedes target evaluation; it was not chosen from the A2 ranking.

The separate transport addendum uses two source-A speeds. Its primary pair, (0.05,0.10) m/s, was frozen before decoding. The endpoint remains the mean of three panels' minimum IoU over the fifteen fixed observer rules.

| Source speed pair (m/s) | Full flow | Matched k=0 | Learned exponent | Raw 3D |
|---|---:|---:|---:|---:|
| **0.05,0.10 — primary** | **0.653586** | 0.667870 | 0.662273 | 0.676430 |
| 0.05,0.20 — secondary | 0.657760 | 0.655558 | 0.647782 | 0.555659 |
| 0.10,0.20 — secondary | 0.640014 | 0.602839 | 0.640147 | 0.670740 |

The primary flow differences are −0.014285 against k=0 and −0.008687 against the source-selected simple baseline, learned exponent. Favorable secondary comparisons do not replace these negative primary contrasts. One- and two-speed training budgets are not combined into a target-selected ranking. No conditional-position extension was added later. All nominal, pooled, Brier and individual-panel outcomes are retained alongside the primary results in the [portable A2 evidence release](../2026-10-14/lockbox_A2/README.md).

A [separate arithmetic and provenance audit](../2026-10-14/lockbox_independent_audit.md) independently reconstructs eligibility, observer labels, coordinate transforms and set-based metrics. Across all 26 models, 153,192 probabilities agree within 5.55e−16 and 4,694 recorded metric occurrences match exactly. Frozen standard estimators retain their own prediction routines; raw waveform extraction is not independently repeated. The audit separately recomputes the complete local archive checksum and verifies unchanged frozen specifications and implementation identities. These checks support numerical reproducibility, not physical label validity or additional independent replication.

## 5. Interpretation, limitations and remaining evidence

The two empirical corpora differ simultaneously in string, apparatus, acquisition range and observation procedure. Their reversal cannot be attributed causally to instrument type or to the observer alone. This study predicts operationally defined sets, not musical quality or independently verified stick–slip motion. Multiple observed positive components also limit any single-band model.

The numerical audit concerns the project's legacy solver, not a defect claimed in independent modern published solvers. The diagram extraction is derivative evidence from published results, not a new physical experiment. Recovered gray or missing cells have not been treated as convenient ground truth. All development results, including failures, are retained.

A public A2 archive was designated for fixed-model evaluation before its waveform outcomes were decoded. Model, observation and preprocessing specifications are hashed and unchanged. The original fourteen-model test excludes the later affine-flow candidate. A separate twelve-model addendum was frozen at 2026-10-13 09:02:25 UTC, after development and matched-ablation results but before A2 decoding. It embeds all four model families for all three source-A speed pairs and fixes (.05,.10) m/s as primary. The recorded archive verification followed this freeze, and both evaluations completed without target fitting or selection. This local chronology is not public preregistration or proof of all earlier external exposure. The unchanged addendum's canonical hash is `787d7839871519816f39d710ae1947dd17548c160b9dee5a3b64635a13186377`. A2 has now been examined and cannot serve as untouched confirmation for methods designed after these results.

At present the work supports a measurement-aware transfer study and reproducibility package. It does not justify a general machine-learning breakthrough, universal improvement or submission-ready scientific claim. The matched width ablation's favorable D result, unfavorable cello held-speed result and negative A2 primary contrasts are all retained. The new-specimen evidence weakens a general improvement claim; it cannot be erased by selecting a favorable secondary speed pair. Any subsequent method requires a distinct development and confirmation plan.

## Primary references

1. Schoonderwaldt, Guettler & Askenfelt (2008). *An Empirical Investigation of Bow-Force Limits in the Schelleng Diagram*. [DOI 10.3813/AAA.918070](https://doi.org/10.3813/AAA.918070).
2. Lampis, Chatziioannou & Scavone (2025). *Experimental analysis of cello string types: influence on playability and tonal characteristics using Schelleng diagrams*. Proceedings of Meetings on Acoustics 58, 035013. [DOI 10.1121/2.0002111](https://doi.org/10.1121/2.0002111).
3. Lampis. *A collection of bowed responses of cello strings*, [Part 1](https://doi.org/10.5281/zenodo.17749111), [Part 2](https://doi.org/10.5281/zenodo.17782542).
4. Matusiak, Chatziioannou & van Walstijn (2025). *Numerical modelling of elasto-plastic friction in bow–string interaction with guaranteed passivity*. Frontiers in Signal Processing 5, 1525044. [DOI 10.3389/frsip.2025.1525044](https://doi.org/10.3389/frsip.2025.1525044).
5. van Walstijn, Chatziioannou, Lampis & Matusiak (2026). *A thermal elasto-plastic friction model for bowed-string simulation*. Acta Acustica 10, 47. [DOI 10.1051/aacus/2026042](https://doi.org/10.1051/aacus/2026042).
6. Luan & Scavone (2025). *Physics-Informed Deep Learning for Nonlinear Friction Model of Bow-string Interaction*. [arXiv:2505.18950](https://arxiv.org/abs/2505.18950).
7. Beigi & Conneely (2026). *Learned Bow Control on a Measured Bowed-String Model: a Revised Minimum-Bow-Force Law, a Recurrent Controller, and the Domain of a Supervision Ceiling*. [arXiv:2609.14990v3](https://arxiv.org/abs/2609.14990v3). Preprint status is retained.
8. Narasimhan, Kar & Jain (2015). *Optimizing Non-decomposable Performance Measures: A Tale of Two Classes*. Proceedings of the 32nd International Conference on Machine Learning, PMLR 37, 199–208. [Primary record](https://proceedings.mlr.press/v37/narasimhana15.html).
9. Pérez-Lantero (2010). *Geometric Optimization for Classification Problems*, §2.6.2. [Thesis](https://personal.us.es/dbanez/papers/pablo-thesis.pdf).
10. Woodhouse, Schumacher & Garoff (2000). *Reconstruction of bowing point friction force in a bowed string*. Journal of the Acoustical Society of America 108, 357–368. [DOI 10.1121/1.429529](https://doi.org/10.1121/1.429529).
11. Dinkelbach (1967). *On Nonlinear Fractional Programming*. Management Science 13(7), 492–498. [DOI 10.1287/mnsc.13.7.492](https://doi.org/10.1287/mnsc.13.7.492).
12. Maia Polo, Maity, Yurochkin, Banerjee & Sun (2024). *Weak Supervision Performance Evaluation via Partial Identification*. Advances in Neural Information Processing Systems 37. [Primary conference record](https://proceedings.neurips.cc/paper_files/paper/2024/hash/f4c6bec746b0aeca8c2cd15096f1ad1f-Abstract-Conference.html).

Crossref metadata for the five journal/proceedings DOI records were checked on 2026-10-13 and saved in `reference_metadata.json`; arXiv and PMLR entries were checked against their primary records. The claim-to-source audit is retained in `literature_audit.md`, `continuous_operator_prior_audit.md` and `endpoint_observability_prior_audit.md`. Journal-specific bibliography formatting remains to be applied.
