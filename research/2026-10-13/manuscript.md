# Assessing speed transfer and observation uncertainty in bowed-string playable-set prediction

**Working research draft — 13 October 2026. Not submission-ready.** No author list, acceptance claim, prospective A2 outcome or independently verified physical reference labels are supplied. This draft summarizes executed development experiments; additional results must be inserted with their original protocol and exposure status. The companion [Korean report](README.md) is the main project entry point.

## Abstract

Predicting the bowing conditions that support stable string motion requires assumptions about both physical similarity and the observation used to identify the motion. We examine these assumptions using 23,301 eligible conditions from an existing robotic cello-string dataset and 1,041 systematically classified violin D-string conditions recovered from a published monochord experiment. The datasets differ in apparatus, acquisition range and classification procedure; their scores are consequently interpreted separately. In retrospective cello transfer, force–speed coordinates substantially improve simple classifiers, while a coherent observation-family robust objective does not surpass the strongest simple baseline. On the published violin D string, uniform force–speed scaling instead underperforms static coordinates, while a known asymmetric boundary prior performs better on average. A compact affine log-force transport model attains mean held-speed IoU 0.8613 on six two-speed D-string tasks, compared with 0.8445 for a literature-informed asymmetric ensemble, but provides little aggregate improvement on cello and fails on downward extrapolation from the two highest D-string speeds. Independent audits identify a nonpassive historical simulator calibration, finite-rate classification sensitivity and insufficient specificity of a waveform-shape observer. An exact finite-state evaluation over a continuous observer family changes current pooled IoUs by at most 0.000357 relative to a 15-rule grid and reverses no within-source model rankings. These findings support separating observational, numerical and predictive assumptions; they do not establish a universally improved method or validated performance on assembled violins.

## 1. Research question and prior work

The Schelleng diagram describes bowing regimes as a function of relative bow position and normal bow force, conditional on speed and string properties. Published violin-string experiments already establish that the lower and upper force limits need not obey the same speed dependence [1]. The recent public cello-string collection provides controlled bowing responses across string constructions and speeds, and its associated analysis estimates playable-region borders and discusses acquisition truncation [2,3]. Plotting these maps or fitting alternative power laws therefore cannot alone constitute the present contribution.

Modern bowed-string models already include finite bow width, torsion, hair compliance and elasto-plastic or thermal friction [4,5]. Neural solvers for a bowed oscillator and recent computational violin control also exist [6,7]. This study does not introduce those mechanisms or claim that its inexpensive point-contact simulator is the state of the art. Its question is narrower: **which transfer conclusions survive changes in physical similarity assumptions, numerical implementation and observation definition?**

We examine entire observed positive sets rather than treating a fitted boundary outside the acquisition range as measured truth. We also distinguish a globally shared observer setting from an independent unknown label at every cell. Performance evaluation without ground truth has established partial-identification precedents, including [12], whose assumptions about latent-label marginals differ from our restricted deterministic observer family. These are experimental design choices with statistical and optimization precedents, not by themselves claims of algorithmic novelty.

## 2. Data and estimands

### 2.1 Public cello-string measurements

The preserved tables contain 24,000 trials from four constructions/specimens of cello G strings and three nominal speeds. A positive-force and velocity-plateau eligibility rule retains 23,301 rows. Predictors receive relative bow position, measured normal force and measured speed; no waveform feature enters their input. Supervision is derived from the project's previously computed bridge-force features. It is not the dataset author's independent reference classification.

All four sample1 grids were previously examined. Their comparisons are retrospective development, even when a particular model uses source-only fitting and selection. Classification nuisance settings are fixed from source A; experiments that fit a predictor on B, C or D therefore do not constitute wholly independent refitting of the complete observation pipeline. Full provenance, checksums and exclusions are in `evidence_audit.md` and the compact input manifest.

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

### 4.3 Observer and numerical audits

Across 56 fixed predictor/source combinations, replacing the 15-rule grid with continuous observer extrema changes minimum pooled IoU by at most 0.000357 and by 0.0000598 on average. No strict within-source ranking reversals occur. On one synthetic N=2,000 input, seven warmed repetitions give median 0.977 ms for the sweep solve and 65.815 ms for the dense solve. Including geometry preparation reduces the relative advantage; the exact record reports both timings. This local experiment is not a universal speed guarantee.

The historical simulator recurrence used g[(1−p)x+py] where its design assumed g(1−p)x+py. The historical calibrated parameters also violate the corrected passive-reflection condition. Their previous agreement scores cannot establish passive physical validity. The corrected simulator passes independent free-decay and implementation-parity checks. Nonetheless, seven of 96 binary memberships change between 400 and 800 kHz for the fixed 1.2-second linear-delay experiment. A first-order allpass delay reduces that count to five but worsens the preceding resolution step. Finite-rate agreement is not continuum certification.

Pure sine and symmetric triangle controls satisfy the historical waveform classifier's Helmholtz rule. End-point inversion experiments also exhibit a common-impedance counterexample: near-perfect two-end consistency can coexist with incorrect inferred contact states. Established endpoint-inversion work already addresses force/velocity reconstruction [10]; our experiments test assumptions rather than claim a new inverse relation.

## 5. Interpretation, limitations and remaining evidence

The two empirical corpora differ simultaneously in string, apparatus, acquisition range and observation procedure. Their reversal cannot be attributed causally to instrument type or to the observer alone. This study predicts operationally defined sets, not musical quality or independently verified stick–slip motion. Multiple observed positive components also limit any single-band model.

The numerical audit concerns the project's legacy solver, not a defect claimed in independent modern published solvers. The diagram extraction is derivative evidence from published results, not a new physical experiment. Recovered gray or missing cells have not been treated as convenient ground truth. All development results, including failures, are retained.

A public A2 archive was designated for fixed-model evaluation before its waveform outcomes were decoded. Model, observation and preprocessing specifications are hashed. **Its download and evaluation are pending in this draft.** The original fourteen-model test excludes the later affine-flow candidate and remains unchanged. After the development and matched-ablation results, a separate twelve-model addendum was frozen at 2026-10-13 09:02:25 UTC, while A2 was still an incomplete download with no decoded outputs. It embeds full flow, matched k=0, learned exponent and raw 3D models for all three source-A speed pairs; (.05,.10) m/s is fixed as primary. It uses the same observation definition and records all outcomes, without target fitting or selection. The evaluator requires completion of the original pipeline and an archive-verification timestamp after the addendum freeze. Thus the addendum can test the candidate on the still-unexamined specimen while retaining the original test separately. This is local chronological documentation, not public preregistration or proof of all earlier external exposure. Its canonical specification hash is `787d7839871519816f39d710ae1947dd17548c160b9dee5a3b64635a13186377`.

At present the work supports a measurement-aware transfer study and reproducibility package. It does not justify a general machine-learning breakthrough, universal improvement or submission-ready scientific claim. The matched width ablation is retained with its favorable D-string and unfavorable cello held-speed outcomes. The pending prospective result must also be reported regardless of outcome, and novelty reassessed against the specific supported claim.

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
