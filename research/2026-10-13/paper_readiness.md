# Scientific claim ledger and manuscript readiness

Updated 2026-10-17 against the research README, archived development results,
completed A2 evaluations, source-identification audit and verified primary literature. This is an
independent research-level review, not a prediction of journal acceptance.

**The work is ready to document as a rigorous exploratory research study, but
not ready to claim a generally superior new methodology.** The strongest result
is the difference between what the data, the observation rule and the model
actually support. The negative results are useful evidence. They cannot be
converted into an ICML-level innovation by combining several familiar techniques
or emphasizing the favorable violin subset.

## 1. A defensible central question

> How transferable are operational bowed-string playability maps across bowing
> speeds and string specimens, and how sensitive are the conclusions to the
> assumed speed law and the shared rule used to classify the measurements?

An appropriate working title is:

**Transferability and observation-rule sensitivity of bowed-string playability
maps: a reproducible study with public cello and violin-string experiments.**

This is a technical musical-acoustics and scientific-computing question. Its
meaning comes from distinguishing operational classification agreement from
physical contact-state knowledge, and from testing known physical assumptions
against explicit controls. The current observations do not identify the cause of
every discrepancy; “separating physics, numerics and measurement uncertainty”
should remain the research ambition, not a completed three-way decomposition.

The scope remains compatible with the user's constraints: public data, no own
recording or sensors, no purchases, and reproducible laptop computation. The
violin connection is actual violin strings on a monochord, plus separately
reported assembled-violin aggregate coefficients. It is not experimental
validation on the user's violins or a digital twin of a complete violin.

## 2. Claim ledger

| Candidate statement | Status | Evidence and permissible wording | What it does not establish |
|---|---|---|---|
| The archived empirical labels are physical Helmholtz ground truth. | **Unsupported** | They are project-derived waveform indicators. Published 2008 labels are author classifications with a different observation procedure. | Correct bow-point sticking, raw independent expert labels, or equivalent label semantics across corpora. |
| The project now includes empirical violin-string evidence without new recordings. | **Supported, bounded** | 1,584 published grid slots; 1,171 strict classes; D benchmark 1,041 cells. Exact PDF provenance, 24-cell two-assistant visual check and complete coefficient-table checks are retained. | 1,171 independent instruments, exhaustive human reannotation, or full assembled-violin validation. |
| Existing historical simulator calibration shows an accurate physical model. | **Withdrawn** | Reflection-filter recurrence and passivity defects invalidate that interpretation. Corrected implementation and old results are preserved separately. | Fair historical zero-shot transfer: nuisance quantities were also target-adapted. |
| Force/speed coordinates improve the retained cello speed-transfer comparison. | **Supported on development data** | Matched-dimensional controls retain the advantage; all four source variants are reported. The exponent is a fixed physical prior. | A newly learned law from a single source speed or universal superiority on violin strings. |
| A coherent weak-label minimax training objective improves over strong baselines. | **Negative result** | It loses to the strongest declared simple baseline in all four original source sweeps. Frozen A2 evaluation also loses: .635452 versus the source-selected baseline's .659570. | A promising method can be called successful solely because the uncertainty formulation is elegant. |
| The known asymmetric force-boundary prior performs well on published D data. | **Supported reproduction** | Primary one-speed macro IoU .860716 versus static band .825181; all four source-speed macro comparisons favor the asymmetric prior, with an explicit transfer-direction loss. | Discovery of a new physical law: the prior and experiment come from the same 2008 publication. |
| Affine speed flow is generally better than established prediction methods. | **Unsupported; primary A2 result is negative** | Development cello .694794 is close to learned alpha .694609 and raw 3D .694037. Frozen primary A2 flow .653586 loses to k=0 .667870 and source-selected learned alpha .662273. | Consistent predictive superiority, statistical equivalence, or an identified mechanism. |
| Affine flow gives a modest published-D improvement over a stronger known-prior comparator. | **Supported exploratory observation** | .861265 versus asymmetric two-source ensemble .844512, five source-pair wins and one loss. At the highest training pair, .754746 versus .803740 is a clear loss. | General violin-string superiority, an independent validation, or identification of a physical mechanism. |
| Learned width deformation helps within the tested band model. | **Supported conditionally on D** | Matched k=0 ablation: D .861265 versus .840464, five wins and one loss. Cello aggregate .694794 versus .693502 is small; same-specimen held-speed flow .713565 is worse than k=0 .719325. | A universal width benefit, new flow mathematics, or independent confirmation outside development data. |
| Making the width generator depend on bow position resolves the transfer limitation. | **Negative result** | Same 18 source tasks and selection protocol: cello .688991 versus global flow .694794; D .862281 versus .861265, but only two of six D tasks improve. The declared both-corpus gate fails. | A general improvement, identification of a width mechanism, or novelty from one additional coefficient. |
| A well-fitting affine predictor identifies a common physical width law. | **Unsupported** | Exact retained published-D boundary brackets give empty common-rate intersections in five of six source-speed pairs. Cello nominal single-band mismatch is 4.854%, with extensive censoring. | A physical impossibility theorem after allowing control/annotation error, a calibrated labeling-error rate, or predictive failure whenever exact compatibility fails. |
| A shared width constraint explains the candidate's transfer failures. | **Not established** | Finite nominal-control diagnostic finds at most 12 additional disagreements across all cello rules/tasks and at most 3 on published D, beyond independent-band mismatch. | A lower loss bound for the measured-coordinate cello predictor, a bound on held-out IoU, or a causal explanation for the candidate's failure. |
| The continuous-rule oracle covers every label state in the specified threshold rectangle. | **Supported conditionally** | Exact finite-state representation; established fractional reduction and quadrant sweep; exhaustive/reference tests; residual gap bound under exact arithmetic and a globally solved subproblem. | A certified floating-point enclosure or coverage of the correct mechanical labels. |
| Replacing 15 rules with continuous individual score minima changes the real-data maximin ranking. | **Not observed** | Largest IoU lower-bound difference .000357; mean .0000598; zero within-source maximin rank reversals or ties broken among 56 arrays. | This is not a claim that shared-observer pairwise differences keep one sign. |
| Model ranking is stable under a shared observation rule. | **False for some development comparisons** | Exact pairwise-difference audit: 52/364 original-model pooled pairs reverse, versus 51/364 detected by the 15-rule grid; all 5,596 pair/scope comparisons retained. Two pairs show the original panelwise maximin preference opposed to uniform shared-rule macro dominance. | Confidence intervals, coverage of physical truth, or a reason to change either frozen A2 primary endpoint. |
| The oracle is 77 times faster. | **Superseded measurement** | Seven warm repetitions at synthetic N=2,000 give 67.35× median solve-only ratio, or 47.08× including preparation, on this Mac. | Universal speedup, cold-start performance, or the speed ratio on the actual data's much smaller state matrix. |
| Dual-endpoint reconstruction agreement certifies contact-state accuracy. | **Falsified under allowed misspecification** | A shared 10% impedance error leaves redundant estimates mutually consistent while producing large synthetic stick-indicator error. | A new inverse-physics discovery or a measured error rate on public real recordings. |
| High-rate simulation labels are numerical ground truth. | **Unsupported** | Label changes persist between the tested finite rates and delay methods. | Continuum convergence, structural stability, or a certified physical bifurcation boundary. |
| The additional A2 specimen confirms either candidate's general improvement. | **Negative primary results, completed 14 October** | Original candidate minus source-selected baseline: −.024118. Primary transport flow minus k=0: −.014285; minus source-selected learned exponent: −.008687. Both frozen plans and every secondary result are preserved. | Formal preregistration, population-level confirmation, switching the primary speed pair, or treating now-observed A2 as untouched for a redesigned method. |

All row-wise IoUs are scores on the specified eligible discrete acquisition
grids. They are not uniformly weighted areas of a continuous physical parameter
domain. Pooled IoU, macro panel IoU, one common worst rule across panels and a
mean of separately minimized panel scores answer different questions and must
retain their distinct names.

“Numerically near-equal” is justified for the small cello aggregate differences.
“Statistically equivalent” is not: no equivalence margin and independent
replication design establish that claim. Thousands of cells, CV fits, overlapping
source-pair tasks and optimizer starts do not create thousands of independent
experimental units. The existing explicit per-task results are more informative
than a narrow row-bootstrap interval would be.

## 3. Why the affine candidate is not yet a new-method paper

The model is technically coherent: an autonomous affine ODE moves both the
center and width in log force, its flow composes exactly, and a constrained
quadratic retains an interval-or-empty positive set. Tests and source-only
selection logs substantiate its implementation. These facts are worth retaining.

However, an affine ODE, its exponential solution, a group-composition property,
and a quadratic classifier are established mathematical ingredients. A new
application can be useful without establishing a general methodological advance.
The physically motivated distinction between lower and upper speed dependence
also precedes the present work. The current model's predictive deformation
parameters must not be interpreted as uniquely identified friction mechanisms.

The D-string improvement survives one added strong asymmetric baseline, which
is better evidence than comparison only with a weak raw-speed extrapolator. It
is nevertheless modest and retrospective. Two separately fitted bands averaged
in probability are one way to combine the known prior, not its proven optimal
joint formulation. The flow and that comparator differ in concavity, feature
representation, regularization and parameterization. That comparison alone does
not isolate changing band width. The subsequent matched k=0 ablation provides a
narrower control: D mean IoU improves by .020801 with learned width, in five of
six pairs. Cello aggregate improvement is only .001292, and the flow is worse
on the source specimen's held-out speed. Thus a conditional D width contribution
has evidence; a general width advantage does not. This does not identify a
unique physical mechanism.

On the primary cello comparison, the improvement disappears under several
reasonable breakdowns. Selecting D as the headline now would be outcome-driven
scope narrowing. It is acceptable to report a domain-specific exploratory
finding; it is not acceptable to imply the requested general improvement was
achieved. The current evidence cannot support “universally better,” “state of
the art,” or “ICML outstanding-paper-level novelty.”

The completed A2 test strengthens this limitation: the frozen primary flow
loses to both its matched k=0 ablation and source-selected learned-exponent
baseline. Some secondary source-speed pairs fare better, but their designation
was fixed in advance and cannot be exchanged with the primary pair afterward.

## 4. What the prior work already rules out

These are boundaries on our claims, not assertions that the prior papers answer
every question in this project.

- [Schoonderwaldt, Guettler and Askenfelt (2008)](https://doi.org/10.3813/AAA.918070)
  already investigates asymmetric speed dependence and experimental force
  limits. Learning or imposing those two distinct behaviors is not their first
  discovery here. The same-paper D comparison is a reproduction plus prediction
  experiment.
- [Lampis, Chatziioannou and Scavone (2025)](https://doi.org/10.1121/2.0002111)
  constructs experimental cello playability maps and fits border scaling. Merely
  extracting maps, fitting exponents or comparing string types repeats an
  existing analysis. Retaining holes, censoring and evaluation provenance can
  improve the benchmark, but those design choices alone are not a new universal
  estimator.
- [Luan and Scavone (2025), arXiv:2505.18950](https://arxiv.org/abs/2505.18950)
  applies physics-informed neural and operator learning to a bowed oscillator.
  Adding “PINN” or “DeepONet” cannot itself supply novelty, and that one-degree-
  of-freedom setting does not provide full experimental validation of our maps.
- [Matusiak et al. (2025)](https://doi.org/10.3389/frsip.2025.1525044) and
  [van Walstijn et al. (2026)](https://doi.org/10.1051/aacus/2026042)
  already address passive elasto-plastic and thermal friction with richer
  bow–string physics. Finite contact width, torsion, bow compliance and thermal
  friction are not new ingredients. A point-bow model's failure cannot be
  attributed to one omitted ingredient without controlled evidence.
- [Beigi and Conneely, September 2026 preprint](https://arxiv.org/html/2609.14990v3)
  overlaps computational violin regimes, learned control and scaling, and
  explicitly distinguishes periodic slipping from motion with sticking.
  Claims to first computational four-string violin mapping, first learned bowed
  control, or first recognition of that diagnostic ambiguity are untenable. It
  remains a preprint, not independently verified ground truth for this corpus.
- [Woodhouse, Schumacher and Garoff (2000)](https://doi.org/10.1121/1.429529)
  already reconstructs bowing-point force and velocity; the closely related
  [Lampis et al. (2025) meeting abstract](https://doi.org/10.1121/10.0037894)
  signals continuing work on cello friction-force reconstruction. Our synthetic
  wave-identity check and common-calibration counterexample are useful audits,
  not first-ever endpoint inversion.
- [Dinkelbach (1967)](https://doi.org/10.1287/mnsc.13.7.492),
  [Narasimhan et al. (2015)](https://proceedings.mlr.press/v37/narasimhana15.pdf),
  and [Pérez-Lantero (2010), §2.6.2](https://personal.us.es/dbanez/papers/pablo-thesis.pdf)
  preempt novelty claims for fractional Jaccard optimization and signed quadrant
  sweeps. [Maia Polo et al. (2024)](https://proceedings.neurips.cc/paper_files/paper/2024/file/f4c6bec746b0aeca8c2cd15096f1ad1f-Paper-Conference.pdf)
  preempts the broad idea of evaluating weak supervision through partial
  performance identification. Our globally shared deterministic operator family
  is a specific uncertainty model, not a claim to invent label-uncertain evaluation.

The targeted literature audit does not prove absence of any closer predecessor.
The manuscript should cite these primary works before making a narrow novelty
statement. “Latest technology” is not a substitute for a missing scientific
contribution; a well-tested small model is preferable evidence to an unneeded
foundation model.

## 5. Completed attribution test and remaining evidential gaps

The requested low-cost attribution test is complete. Its result narrows one
gap, while independent validation and target interpretation remain separate.

**Attribution, partially resolved:** the matched k=0 model keeps the concave
quadratic, beta-dependent translation, normalization, source folds,
regularization grids and selection rule. Full flow scores .861265 versus .840464
on published D (five wins, one loss). On cello it scores .694794 versus .693502,
but same-specimen held-speed scores are .713565 versus .719325. Width deformation
therefore has conditional predictive value on D within this model family; it
does not explain a general advantage across both corpora.

**Independence and completed outcome:** the sample1 and published-D corpora were
examined during development. Both separate A2 protocols completed on
**2026-10-14 06:40 UTC (15:40 KST)**, with 5,892 eligible rows from 6,000 raw
trials. The original 14-model evaluation and twelve-model pre-outcome addendum
remain unchanged. The addendum has three A-source speed pairs, each with full
flow, k=0, learned alpha and raw 3D. Its canonical hash in
`transport_lockbox_spec.json` was independently recomputed and matches:

    787d7839871519816f39d710ae1947dd17548c160b9dee5a3b64635a13186377

The recorded freeze time is **2026-10-13T09:02:25.800573Z**. Source speeds
(0.05,0.10) m/s are primary; the other two A-source pairs are secondary. The
primary endpoint is the mean across three A2 speed panels of each panel's
minimum IoU under the fixed 15-rule family. Primary comparisons are full flow
minus k=0 and full flow minus the source-CV-selected simple baseline. All models,
pairs and declared secondary metrics must be reported regardless of outcome.

The recorded local non-exposure check found only partial-download files, the
download lock and a waiting pipeline state, with no decoded A2 outcomes. This is
a recorded **pre-decoding prospective addendum after development**, not formal
preregistration or certification of all possible human/external exposure history.
The archive was fully verified after freezing and before decoding; the separate
evaluator ran after the main evaluation completed. Neither candidate improved
its prespecified primary comparison. Original robust minus baseline is
−.024118; primary flow minus k=0 is −.014285 and minus source-selected learned
alpha is −.008687. The 0.05/0.20 and 0.10/0.20 source pairs remain secondary,
including their favorable outcomes. Full results and numerical replay are in
the [portable A2 release](../2026-10-14/lockbox_A2/README.md).

The [independent A2 audit](../2026-10-14/lockbox_independent_audit.md) checks all
26 models through separate eligibility, observation and metric implementations,
with independently coded prediction transforms. It also recomputes the complete
local archive MD5 and verifies the unchanged freezes. This is independent
arithmetic/provenance reconstruction after waveform feature extraction; it is
not a second physical experiment or an independent raw-waveform reanalysis.

A2 is now exposed. A method designed after these results may use it as
development evidence but cannot call its next A2 evaluation untouched
confirmation. One additional cello specimen also does not establish performance
on a population of instruments or on assembled violins.

**Meaning of the target:** neither the present threshold family nor an internally
consistent simulator proves physical contact-state truth. The narrow operational
map paper can state this limitation. A physical-state certification paper would
need independently justified observation/calibration/model uncertainty bounds;
current metadata and synthetic identities do not supply them.

**Structural diagnostic, completed but limited:** the
[nominal-control width-law calculation](../2026-10-14/width_law_assessment.md)
separates single-band mismatch from the additional disagreement forced by one
shared width exponent. The latter is small: at most twelve cello rows across
all fifteen observers and three published-D rows. Its exact fixed-control
premise is absent in the measured-coordinate cello predictor, so its value is
not a certified loss bound for that predictor. Source Hamming cost under this
relaxation also cannot bound held-out IoU. This result guides interpretation;
it neither establishes a new successful method nor identifies the cause of
the failed A2 transfer.

The [bounded-exponent follow-up](../2026-10-14/width_bound_sensitivity.md) matches
the predictor's `k ∈ [−2,2]` range and raises six of 186 optima by one or three
rows. All nominal cello and published-D optima remain unchanged. Alternative
bounded witnesses attain the original optimum in seventeen cases whose saved
representative lay outside the range. This strengthens the diagnostic's range
sensitivity check, but the other relaxations and novelty limitations remain.

### Completed bounded experiment; no new architecture search recommended

The full-versus-k=0 comparison ran all twelve cello and six D tasks, preserving
the original full-flow outcomes. `affine_width_ablation_summary.json` records
every task, and the updated affine assessment describes the protocol. Full flow
uses three distinct k initializations; k=0 has one unique initialization.
Only one of 810 full-flow CV fits improved its training objective by more than
1e-6 using another start. This detail is disclosed rather than hidden under a
claim of perfectly identical optimization paths.

The ablation resolves the immediate width-attribution question sufficiently to
report a limited D finding. The completed A2 primary evaluation supplies
additional-specimen evidence against general improvement. These completed tests
do not resolve methodological novelty; repeating them as unfinished work would
be stale.

No further architecture search, simulated dataset expansion or task switching
is justified merely to obtain a favorable headline.

## 6. Manuscript framing that fits the present evidence

The strongest current paper is an empirical methods/reproducibility study in
musical acoustics or scientific computing. A reasonable contribution statement
is:

> We provide a reproducible comparison of operational bowed-string playability
> prediction under speed-transfer priors, recover an external published
> violin-string benchmark with explicit missingness, and quantify sensitivity
> to a globally shared observation rule. We find that favorable transfer priors
> depend on the experimental/labeling setting, and that more elaborate robust
> training and affine deformation do not establish a general advantage.

This states substantive work without inventing a new physical law. It is a
plausible manuscript framing, not a claim that any particular journal will
accept it or that the user's desired methodological novelty has been completed.

A focused main text should follow one argument:

1. Define the operational prediction target, acquisition-grid units and the
   difference between weak labels, published author classes and latent physics.
2. Describe reproducible source-only model fitting and the retained missing,
   ambiguous and censored conditions. Make retrospective exposure explicit.
3. Present matched static/uniform/known-asymmetric comparisons across all tasks;
   show that a broad speed-law conclusion fails across the two source corpora.
4. Distinguish the small effect on individually minimized score rankings from
   shared-observer pairwise reversals and the two aggregation counterexamples.
5. Report robust-learning failures, the bounded affine candidate and its matched
   width ablation together. Preserve the D gain, cello near-equality and losses.
6. Report both completed fixed A2 evaluations, including their negative primary
   contrasts, all secondary pairs and the now-exposed status of the specimen.

Detailed simulator bug history, endpoint inverse identities, cache regression
tests and large convergence tables belong in reproducibility material or a
focused supporting appendix unless they are directly needed for this argument.
Giving every prototype equal prominence would create several incomplete papers
inside one manuscript and obscure the central evidence.

## 7. Present decision

The study has a defensible dataset/evaluation foundation and several meaningful
negative controls. It is not an empty proposal. It also has **not satisfied the
original demand for a clearly superior, high-novelty new method**. Preserve that
distinction when describing progress.

The manuscript and Korean report now distinguish development, the original
frozen A2 benchmark and its separately frozen transport addendum. The completed
negative primary results must remain visible when pursuing a new methodology.
Source censoring, structural mismatch and shared-observer comparisons provide
specific technical questions for that next work; they do not yet constitute a
validated novel estimator.
