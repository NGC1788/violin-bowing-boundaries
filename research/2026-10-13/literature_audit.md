# Literature and novelty audit — 2026-10-13

This is a research design audit, not a claim that a new method has already worked. Sources were checked on 2026-10-13. The user's current scope excludes all new violin recordings, sensor construction, and purchases; execution is on the M2 MacBook with 16 GB RAM. Earlier documents that require a final recording session are superseded on that point.

**Method refinement during the audit:** the immediate experiment now focuses on whole-set prediction with a physical coordinate prior and a **globally coherent family of observation operators**. The initially proposed explicit simulator-residual/transition-interval loss below is a possible extension, not a description of completed code. [Formalization, exact bounds, a relaxation counterexample, and the nearest weak-supervision prior](method_formalization.md) explain why independently treating every threshold-disagreement cell as unknown can throw away information. A cheap coherent-family minimax comparison should precede a more elaborate geometric correction.

## Decision

The strongest feasible direction is **learning a transferable correction to a simulator's entire playable set while preserving uncertain labels, disconnected regions, and censored boundaries**. A working title is:

> **Learning playable sets from imperfect physics: censoring-aware discrepancy correction for bowed strings**

The object being learned is the set of bowing conditions that receive a stable Helmholtz classification, not a new sound generator. The existing backed-up stroke features, provisional labels, and paired simulator classifications can support a first experiment without downloading hundreds of gigabytes or training a neural operator. Whether the proposed treatment of the geometry offers anything beyond a conventional residual classifier must be tested. It is not currently defensible to call it an ICML-level methodological advance.

Three tempting directions are weaker:

1. **Adding a PINN or DeepONet:** already studied for bow friction and changes the computational solver rather than addressing the empirically observed simulator discrepancy.
2. **Discovering a different power-law slope:** deviations from Schelleng scaling have long been documented, including on this very dataset.
3. **Adding finite bow width, torsion, or thermal friction:** these are important physical ingredients with strong existing implementations. Implementing them would improve the baseline but would not by itself establish novelty.

## What the nearest primary sources already do

| Primary source | Established contribution relevant here | Consequence for this project |
|---|---|---|
| [Lampis, Chatziioannou & Scavone, POMA 58, 035013 (2025)](https://caml.music.mcgill.ca/lib/exe/fetch.php?media=publications%3Alampis_ismra2025.pdf), DOI [10.1121/2.0002111](https://doi.org/10.1121/2.0002111) | Constructs experimental Schelleng maps for cello string materials/tensions, classifies bridge-force signals, and fits generic power laws to playable-region borders. Uses consecutive Helmholtz points and notes that a high-speed upper border cannot reliably be determined when it approaches the measured force limit. | This is the **main reference for the current public dataset**. Merely plotting the maps, fitting different exponents, comparing materials, or dropping censored borders would repeat existing work. |
| [Lampis, Mayer & Chatziioannou, Acta Acustica 8, 44 (2024)](https://acta-acustica.edpsciences.org/articles/aacus/pdf/2024/01/aacus240063.pdf), DOI 10.1051/aacus/2024034; [data/code](https://zenodo.org/records/10946413) | Studies the transient attack problem using robot-bowed monochord measurements and theoretical playability limits. | A relevant independent experimental resource, but its acceleration/force attack task is different from sustained Helmholtz-set prediction. Do not silently combine the targets. |
| [Luan & Scavone (2025), arXiv:2505.18950](https://arxiv.org/html/2505.18950v1) | Tests PINNs and physics-informed DeepONets on a **one-degree-of-freedom bowed mass–spring oscillator** with a smooth static friction law. High-force operator learning is difficult; observation-assisted training helps. Reported training times in their table are roughly 14–39 hours per configuration on their setup. | It is not a full experimentally validated bowed-string map predictor. A literal replication is an expensive and weak match to the present Mac-scale question. A PINN loss does not automatically correct missing physics. |
| [Matusiak, Chatziioannou & van Walstijn, Frontiers in Signal Processing 5 (2025)](https://pure.qub.ac.uk/en/publications/numerical-modelling-of-elasto-plastic-friction-in-bow-string-inte/), DOI 10.3389/frsip.2025.1525044; [source repository](https://github.com/wasilakis/the_bowed_string); [pinned release](https://zenodo.org/records/15410421) | Derives a passive elasto-plastic friction formulation and an energy-consistent numerical scheme, incorporating finite-width bow contact, hair compliance, and string torsion. The released code is MATLAB. | A serious modern physical baseline exists. The current point-bow waveguide should not be labelled the state of the art or assumed structurally adequate. The concept DOI 15341818 resolves to release 15410421 at this audit. |
| [van Walstijn, Chatziioannou, Lampis & Matusiak, Acta Acustica 10, 47 (2026)](https://pure.qub.ac.uk/en/publications/a-thermal-elasto-plastic-friction-model-for-bowed-string-simulati/), DOI [10.1051/aacus/2026042](https://doi.org/10.1051/aacus/2026042) | Develops thermal elasto-plastic bow friction together with torsion, finite contact width and hair compliance, and compares numerical responses with measurements. | Upper-boundary error cannot presently be attributed uniquely to point contact. Several omitted mechanisms can cause it, and a newer physical treatment already targets these issues. The 2025 code release is not evidence that this complete 2026 implementation is available. |
| [Woodhouse & Galluzzo, Tribology Letters (2025)](https://link.springer.com/article/10.1007/s11249-025-02062-4), DOI 10.1007/s11249-025-02062-4 | Refines rosin tribology and compares predicted transients with experiments. The paper's data-availability statement says data are available from the corresponding author on request. | This is relevant prior physics, but not a currently downloaded open benchmark. No contact with the authors has been initiated. |
| [Beigi & Conneely, arXiv:2609.14990v3 (2026)](https://arxiv.org/html/2609.14990v3) | A September 2026 **preprint** studies a friction-resolved violin simulation, learned controllers, scaling, and a regime diagnostic. It explicitly warns that periodic continuous slipping can be misclassified as Helmholtz without a stick test. Parameters come from published measurements; this is not independent experimental validation of our cello labels. | This recent overlap rules out broad first-use claims for learned bowed control, computational four-string violin maps, or revised minimum-force scaling. Its diagnostic is a concrete adversarial test for our labels, not an unquestionable reference classifier. |
| [Balbi et al., HESS 27, 1089–1109 (2023)](https://hess.copernicus.org/articles/27/1089/2023/hess-27-1089-2023.html) | Calibrates a physical flood model using binary extent observations with spatially correlated Gaussian-process discrepancy; discusses identifiability and overfitting when only binary observations are available. | **Learning a GP discrepancy to correct a physical binary region is already established.** A generic simulator-plus-GP or residual logistic classifier is a baseline, not the novelty claim. |
| [Gotovos et al., IJCAI (2013)](https://www.ijcai.org/Proceedings/13/Papers/202.pdf); [Shekhar & Javidi, AISTATS (2019)](https://proceedings.mlr.press/v89/shekhar19a/shekhar19a.pdf); [Mason et al., AISTATS (2022)](https://proceedings.mlr.press/v151/mason22a.html) | Develop active/GP level-set estimation, including sample-complexity or multiscale results. | Sampling near a decision boundary and stopping on GP uncertainty are established. Theorems with continuous noisy function observations cannot automatically be applied to rule-derived binary labels. |
| [Renganathan, Rao & Navon, CAMERA (2022)](https://arxiv.org/abs/2203.01436) | Uses cost-aware multifidelity modelling for failure-level-set and reliability estimation. | Multifidelity plus active sampling is not a novel combination. A comparison is required if this is later introduced. The existing backup does not itself identify discretization error separately from model error. |
| [Plumlee & Joseph, Orthogonal Gaussian Process Models](https://arxiv.org/abs/1611.00203) | Restricts a GP discrepancy to be orthogonal to a parametric mean to address identifiability. | Orthogonalizing a correction against physical parameters is not new by itself. Any such extension needs a precise reason specific to set geometry and an empirical advantage. |
| [Xie, Liu & Gan, Nature Communications (2022)](https://www.nature.com/articles/s41467-022-35084-w); [Bakarji et al., Nature Computational Science (2022)](https://www.nature.com/articles/s43588-022-00355-5) | Learn dimensionless representations and governing/scaling relations under dimensional constraints. | Using nondimensional force and velocity is sensible inductive bias; it is not a standalone methodological innovation. |

The search is targeted, not proof of absence of any predecessor. Paper titles and dates do not establish correctness; primary-source results must still be checked against reproducible outputs.

## A correction to the current framing

The archived labels are **derived supervision**, not published ground truth. The exact numerical IoUs in the handoff are agreement with that derived classifier, subject to its exclusions and matching rules. Reproducing them is useful but does not establish physical correctness of every cell.

In particular:

- A bridge-force waveform cannot automatically establish the bow-point sticking fraction. The recent preprint makes the missing diagnostic concrete, but its simulation-state test is not available in the original measurements.
- Agreement with a flyback scaling law supports consistency with a physical pattern. If impedance is subsequently estimated from the same labelled measurements, agreement involving that fitted impedance is not an independent calibration test.
- The four labelled “types” A–D in the experimental dataset are **four constructions of cello G strings**, not violin G/D/A/E strings. The [official dataset description](https://zenodo.org/records/17749111) is explicit about core/winding materials and sample/tension groups.
- The 24,000 rows do not provide 24,000 independent instrument replications. There are only four backed-up sample/type grids and three nominal speeds per grid; cross-condition variability must be reported at that level.
- Same-classifier simulation/measurement comparison controls one processing difference, but can make the same systematic classification mistake twice.
- A high-speed boundary that leaves the measured force range is **unknown beyond the grid**, not zero error, absent motion, or a boundary at the largest measured force.

## One concrete method to investigate

### Object and inputs

Let a condition be `x = (log beta, log(F/F_ref), log(v/v_ref), z)` with positive physical force, speed and reference scales. `z` contains only independently available string properties. If impedance is estimated using target-domain bridge-force labels, that must be declared adaptation and made available to every competitor; it cannot silently become a zero-shot input.

The target is the entire set `S = {x: Helmholtz under the stated observation protocol}`. Predict either a signed margin `g(x)` or an occupancy probability. Do not enforce a single force interval per beta: real or classifier-induced holes and multiple components must first be retained and counted.

Construct a fixed simulator margin `m_sim(x)` from its archived occupancy map, preferably a signed distance in declared log-coordinate units. Fit a low-complexity correction:

`g(x) = a * m_sim(x) + delta_phi(x)`.

Freezing the previously calibrated simulator avoids pretending that discrepancy and physical parameters have been jointly identified. A low-rank kernel, RBF features, or a small MLP is adequate for the first study. The residual formulation is standard; the research question is whether the following information-preserving geometry improves transfer and data efficiency.

### Preserve what the grid actually observes

For each adjacent pair of reliable opposite labels along a force line, the transition lies **in an interval**, rather than at its geometric midpoint. Store the orientation and endpoints. If ambiguous cells intervene, the bracket widens; if no opposite class is observed before the acquisition boundary, store only a one-sided constraint. Keep every observed component and report its persistence across the predeclared classification settings.

Train with cell classification loss plus a transition loss that is zero when the corresponding predicted crossing lies inside its observed interval, with an oriented sign penalty at the bracket endpoints. Do not replace a missing crossing by a finite target. Censored edges contribute only the supported one-sided sign constraint. Fixed complexity penalties must be chosen on training/development conditions. A parameterization that forces only two global boundaries would contradict the multi-component motivation.

There are implementation choices to investigate: a differentiable zero-crossing loss, a piecewise-linear force-line margin with a convex interval loss, or a monotone branch per separately identified component. Choosing one is not itself a contribution. The first experiment should use the simplest form whose constraints can be verified on synthetic cases.

### Keep classifier uncertainty separate

For a predeclared set of plausible classifier settings `eta`, compute `S_eta`. Define a stable core `L = intersection_eta S_eta` and a possible region `U = union_eta S_eta`. Prediction sensitivity to this family is measurable. It is **not** a calibrated probability that the physical state lies in `[L,U]`, because all classifiers may share a systematic error.

Archived features may permit changing a subset of downstream thresholds; they do not necessarily permit recomputing cycle folding or hysteresis. The code must label exactly which sensitivities were actually recomputed. Thresholds cannot be selected by which one makes a proposed predictor look best.

For any admissible target set `T` satisfying `L subset T subset U`, a prediction `P` has exact extremal IoU bounds:

`IoU_min(P) = measure(P intersect L) / measure(P union U)`

`IoU_max(P) = measure(P intersect U) / measure(P union L)`.

The lower extreme excludes uncertain points inside `P` and includes uncertain points outside `P`; the upper extreme does the reverse. `measure` can be cell count or fixed acquisition-cell area in log coordinates. State the convention for an empty union. These elementary bounds are an audit tool, not a novel theorem or a distribution-free truth guarantee. They reveal whether a reported gain survives plausible label disagreement. A stricter certified ranking requires one method's lower bound to exceed another's upper bound; failure of that sufficient condition is inconclusive.

### Falsifiable methodological hypothesis

At a matched number of calibration conditions and labels, the interval/censoring-aware correction yields better **held-out whole-set prediction and transition-interval error** than a conventional pointwise residual classifier, especially when acquisition range and label reliability change. The improvement must persist after removing simulator-derived privileged information from all learners and under the declared label sensitivity analysis.

Failure of this hypothesis means the extra method is not justified. A useful benchmark, a negative result about structural mismatch, and a well-calibrated simpler predictor remain valid research outputs; renaming the same algorithm would not create novelty.

## Comparisons that can falsify the proposal

| Method | Why required |
|---|---|
| Existing literature-parameter and calibrated physical maps | Quantifies the original physical baseline. Only comparisons with actually available runs can be claimed. |
| Physics-only scaling/analytic band fitted using the same calibration labels | Tests whether the geometry is explained by a simple law. Do not assume textbook exponents fit. |
| Data-only logistic/polynomial model and RBF kernel model | Tests whether physics helps at all. |
| Conventional residual classifier using `m_sim` plus the same features | Nearest cheap, defensible methodological baseline. |
| Proposed full objective | Candidate contribution. |
| Remove interval treatment; replace brackets with midpoints | Isolates interval censoring value. |
| Discard censored edges versus retain valid one-sided constraints | Tests whether censored information helps rather than distorts. |
| Merge all components / fill holes versus preserve them | Tests the topological motivation; report raw holes before interpretation. |
| Remove simulator margin while matching feature capacity | Separates physics information from regularization/capacity. |
| Retrain across the predeclared classifier family | Distinguishes predictor gains from fragile target choices. |

Use identical training labels, development folds and budget for all learned models. Nominally shared GPU time is not a useful fairness metric on this laptop; count labels, model parameters, wall time, peak memory and any simulator evaluations. Active-learning label-query experiments may later be replayed on a fixed labelled pool, but are not new physical experiments and must not access unrevealed labels during selection.

## Evaluation discipline

1. Preserve the historical A-type, 0.10 m/s calibration condition and report every other backed-up condition individually. Do not call all 24,000 strokes held out. A single calibration condition provides no empirical basis for learning arbitrary speed dependence.
2. Add separately labelled retrospective leave-one-type and leave-one-speed evaluations. Existing physical parameters were calibrated on A; a leave-A-out learned model using that simulator still inherits A information. Report that explicitly or exclude that claim.
3. Reserve an unused sample/setup/tension archive for the strongest future confirmation. Existing conditions were already inspected during prior development. A new split alone does not erase that exposure.
4. Report micro and macro IoU, balanced accuracy, Brier/log score if probabilistic, false-playable and false-unplayable fractions, transition-interval violation, censored fraction, and performance on every condition. A single pooled accuracy can conceal a failed high-force edge.
5. For geometric set metrics, fix quadrature weights from the acquisition grid, not from outcomes. Report equal-cell IoU alongside weighted area IoU so coordinate changes do not manufacture gains.
6. Resample whole beta force-lines or conditions when assessing correlated maps. With only four types, cross-type population confidence intervals will be weak; display all four results instead of asserting broad statistical generalization.
7. Calibrated probabilities on rule labels are probabilities for those labels. They are not probabilities of mechanically correct sticking. Conformal intervals would also need an exchangeable calibration design; arbitrary new-string shifts invalidate the standard guarantee.

## What is feasible now, and what is not yet evidence

**Now, using backups:** align the measured/simulated trial keys, reconstruct exclusions, audit weak-label pathologies from stored features, reproduce historical counts and IoUs, train small baselines, examine interval/censoring and topology, run all listed retrospective comparisons. A 24,000-row tabular problem is compatible with 16 GB RAM; no giant model is needed.

**Needs fresh computation or restored data:** full waveform relabelling, new physical stick-state diagnostics, independent attack validation, empirical discretization convergence, and a genuine prospective test on a previously unused public sample. Restore only the minimum useful archive, with an explicit disk budget. Waveguide point simulations on a sparse grid can provide synthetic diagnostic counterexamples; they cannot by themselves validate real labels.

**Violin inclusion:** a sourced violin parameter sweep and numerical transfer analysis are possible without owned instruments. They must be presented as computational violin cases, separately from the cello measurements. A physical body/bridge admittance model or public violin measurements would strengthen relevance; relabelling four cello string constructions as violin strings is unacceptable. Do not change f0 alone and claim a validated violin digital twin.

**Novelty status:** the exact interval-aware, weak-label-audited set correction is a proposed direction whose nearest-comparison performance is untested. A residual correction, censoring, level-set sampling, dimensionless coordinates, and uncertainty bands each have prior work. A strong paper must show a specific new inference mechanism or clearly documented failure that existing methods cannot resolve under the same information. No search or implementation guarantees acceptance or universal improvement.
