# Protocol-dependent bowed-string boundaries: primary-source and identifiability audit

Prepared 2026-10-18 KST. Scope: a bounded primary-literature screen and an inferential audit, not a new experiment or proof of worldwide novelty. Read handoff §10.13, §10.12, §8 correction, §10.1–10.11, then `COLLAB_LOG.md`. Handoff numerical results are contextual claims until independently reproduced. No pipeline, raw archive, simulator, or repository freeze was modified for this note.

**Decision:** the strongest defensible next direction is measured-path, finite-horizon regime prediction and protocol-support analysis. The current evidence does not identify a universal stationary force law, a particular bifurcation, or a physical hysteresis mechanism. The primary literature already covers finite-window effects, attack versus maintenance, and path-sustained motion. A genuinely new methodology remains to be demonstrated; adding survival analysis or a recurrent network alone would not provide it.

## 1. What the primary sources actually establish

| Primary source inspected | Verified content and location | Consequence for the present work |
|---|---|---|
| [Lampis, Mayer & Chatziioannou, Acta Acustica 8, 44 (2024)](https://pub.mdw.ac.at/media/content_files/aacus240063_EwplPL3.pdf), DOI 10.1051/aacus/2024034 | §2.1, PDF p. 3: “the bow force was not directly controlled.” Bow elevation sets force; increased hair tension and a 6 cm central bow segment were used to keep force approximately constant. §1.1 explicitly distinguishes maintaining sustained Helmholtz motion from achieving a clean attack. | This validates the rig-design concern and rejects a claim that attack/maintenance is a newly discovered distinction. It does **not** by itself quantify force loss in the later collection, prove a mechanical cause, or validate every low-force sensor reading. |
| [Lampis, Chatziioannou & Scavone, POMA 58, 035013 (2025)](https://caml.music.mcgill.ca/lib/exe/fetch.php?media=publications%3Alampis_ismra2025.pdf), DOI 10.1121/2.0002111 | Methods, PDF p. 3: 40 positions × 50 forces × 3 speeds; elevation rather than feedback force control. Analysis uses the constant-force/constant-velocity part. §3A, PDF p. 7: borders require at least three consecutive Helmholtz grid points; a power law is fitted separately at each speed. The 0.20 m/s upper edge is omitted because of force-range clipping. | This is the primary dataset paper, not a neutral source of physical state labels. In the inspected paper I found no joint analysis of elapsed plateau time, measured force history and attack acceleration. That limited absence supports an empirical reanalysis opportunity, not a blanket priority claim. |
| [Schoonderwaldt, Guettler & Askenfelt, Acta Acustica united with Acustica 94, 604–622 (2008)](https://www.knutsacoustics.com/files/schoonderwaldt-et-al.-empirical-schelleng.pdf), DOI 10.3813/AAA.918070 | §2.4/Fig. 1: establish Helmholtz during the initial phase, then smoothly move force to target; typical steady intervals 0.65–4.0 s. §3.2: near the lower edge, alternating regimes make the answer depend on the sampled segment. §4 uses repeated selected intervals and a strict stability rule; separate descending-force experiments examine breakdown. | Their protocol has a maintenance emphasis. Calling every reported grid cell a pure, verified maintenance measurement goes further than the design statement: initial establishment is an aim, and individual raw confirmations are unavailable. Their speed-independence cannot be dismissed as a simple time-window artifact, especially after the supplied censored reanalysis. |
| [Beigi & Conneely, arXiv:2609.14990v3](https://arxiv.org/html/2609.14990v3), §§II-F–H | §II-F: changing to 140 settling periods changes Helmholtz count 118→163; §II-H reports β exponent −1.14→−1.58 across timing conventions and missing G-string boundaries at 7/10 positions under the fixed-duration setting. §II-G explicitly reports motion sustained along a controller path outside held-command playable cells. | Quantitative finite-window and path-dependent-map effects already have direct bowed-string precedent. This is a simulation preprint, not experimental ground truth. The text does not establish 140 periods as an infinite-time convergence certificate. Its own findings therefore motivate finite-horizon scrutiny but do not prove the present stationary β exponent is −2. |
| [van Walstijn et al., Acta Acustica 10, 47 (2026)](https://doi.org/10.1051/aacus/2026042) | Local supplied full PDF/text inspected, especially §3: 1,600 strokes at 0.1 m/s; acceleration chosen to favor Helmholtz during the plateau; robot does not feed back sensed normal force; window-mean applied force handles a normal-load reduction. Model includes torsion and finite bow width in addition to thermal/elasto-plastic contact. Publisher fetch was blocked; local full text was available. | The acceleration and load concerns are explicitly relevant. A simplified point-bow contact-law insertion is not a reproduction or falsification of this model. The one-speed comparison cannot settle the present speed-exponent question. |
| [Meisel, Klaus, Kuehn & Plenz, PLoS Computational Biology 11:e1004097 (2015)](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1004097) | Methods/results derive local recovery scaling for normal forms and measure exponential return after controlled small perturbations; recovery-rate and variance evidence are compared. Their restricted-observation threshold prediction already uses bifurcation scaling. | A critical-slowing extrapolation principle is not new. Their recovery rate is not the first time a window classifier reports a run of Helmholtz-like outcomes. Transporting the normal-form exponent to that event time requires a separate derivation and assumptions. |
| [Yuen & Musta, arXiv:2403.16832v2 (2026 revision)](https://arxiv.org/abs/2403.16832v2) | Develops a test for practically sufficient follow-up in cure-fraction estimation under explicit tail conditions. | An identifiability/follow-up check is already a statistical research topic. “Never locks” cannot be inferred from administrative censoring alone; a fitted cure fraction needs assumptions beyond the finite observations. |

The Beigi findings above summarize about 150 words of source-derived material; the primary paper supports them, but its broader mechanistic/statistical claims were not independently audited here. Likewise, this note does not endorse every conclusion of any cited paper.

## 2. Corrections required before any new scientific headline

### Matched time is a conditional association, not an isolated speed law

Let the measured protocol be

\[
\mathcal P_i=\{F_i(u),v_i(u),\beta_i,a_i(u),s_i(u):0\le u\le C_i\},
\]

where `C_i` is the available follow-up and `s` is bow travel. An operational classification is also indexed by a window/observer rule `O`:

\[
Y_i^O(t)=O\{\text{measured waveform over the declared window}\}.
\]

A fitted minimum boundary at common elapsed time estimates an association under the protocols represented in the dataset. It does not automatically estimate
`F_min(beta, speed | fixed attack, fixed force history, infinite follow-up)`.

During an ideal speed plateau, `s(t)=s(0)+v*t`. Holding time fixed while changing speed changes travel; if measured force decreases with travel, it also changes force history. The attack acceleration appears tied to the commanded speed/force/position design. Adding acceleration as a regression covariate cannot create experimental overlap where that design provides none. In a single-speed stratum, an arbitrary time effect and an arbitrary travel effect are nonidentifiable: replacing `f(t)` by `f(t)+h(t)` and `g(s)` by `g(s)-h((s-s0)/v)` leaves their sum unchanged on the observed path. Across speeds, this ambiguity is reduced only under restrictions and actual support, not simply by adding more rows.

Thus an exponent near +1 after time matching is permissible as **a descriptive, observer- and protocol-conditioned exponent**. It does not confirm Schelleng, resolve the older violin/cello disagreement, or prove the original larger exponent was caused exclusively by settling. Force history, attack, session and specimen remain relevant. The corrected β grouping prevents an implementation artifact; it does not eliminate these scientific ambiguities.

### Evidence for travel-dependent force loss is stronger than a simple drift story, but not full calibration

The handoff reports cross-speed collapse versus travel, stopping of the change when motion ends, dependence on force/position, and differences between sessions. If independently reproduced, these discriminate against a *pure additive time-only* drift. They do not rule out all position-dependent sensor errors, load transmission effects, changing hair contact, or a combination of true load change and offset. The statement “resolved real force” is stronger than the current sensor setup alone warrants. A defensible wording is: **the readings show reproducible travel dependence consistent with open-loop mechanics and inconsistent with the tested time-only drift explanation**.

An estimated hair stiffness or path-mismatch distance from this pattern is model-dependent. The analogy of two series springs is a candidate mechanical explanation, not a separately calibrated parameter estimate. Bridge DC sign arguments also require validated sensor sign, baseline and dynamic/static calibration; they should not independently certify bow normal force.

### A positive history coefficient is not a physical hysteresis test

`log(F_onset/F_current)` contains the same measured `F_current` used as an instantaneous predictor. Error in current force therefore enters two regressors with opposite signs. A positive coefficient can arise from measurement-error correction, support imbalance or mis-specified instantaneous dependence. It is not enough to add an acceleration term and read the remaining coefficient causally.

Physical hysteresis requires distinct outcomes under comparable current controls but different histories, with those histories genuinely observed in overlapping support and alternative explanations constrained. Simulation can demonstrate such a mechanism in that simulator. The measured-data analysis should first report common-support counts, balance, session-specific results and predictive increments; it should not call the available data a randomized or identified natural experiment.

### A fitted θ does not name the bifurcation

For local perturbations of the saddle-node normal form, recovery rate can scale with distance to the transition as a square root; at Hopf it can be linear under the specified conditions. A latency to cross an arbitrary detection criterion is a different functional of the trajectory. It depends on initialization, forcing path, noise, threshold, transient amplitude and observation window. Multiple mechanisms can produce the same latency exponent over a short range. Nonsmooth behavior can also have long transients; failure to detect divergence does not identify grazing.

Accordingly, a comparison of `theta=0.5`, `theta=1`, free theta, and a finite-timescale fit is at most **comparison of phenomenological event-time families**. AIC or a likelihood ratio compares those families under their assumed likelihood; it does not turn them into an exhaustive list of bifurcations. A profile likelihood only quantifies identification *within that family*. It cannot repair an invalid event definition or identify the stationary law nonparametrically.

### Finite follow-up cannot distinguish “never” from “later” without tail assumptions

Suppose all observed event data end by `C_max`. Any fitted event-time distribution on `[0,C_max]` can be extended by assigning its remaining probability either to an event after `C_max` or to infinity. Both extensions have the same observable distribution up to `C_max`. This elementary construction is not a new theorem; it explains why cure fractions and stationary limits need explicit restrictions. Here follow-up also varies with the protocol, and first entry need not be absorbing because labels can switch back.

The previous suffix definition (“first time after which at least 80% of all remaining windows are positive”) is retrospective: adding follow-up can move an earlier event time. It must not be used as if it were a stopping time observable online. A first-K completed-window detection is causal in the observation filtration, but its occurrence is a **detection event**, not proven permanent physical locking. Report recurrence separately and timestamp at the window end.

## 3. The next research object that can be evaluated honestly

Use the finite-horizon target

\[
p_{\Delta}^O(h_t,c)=P\{Y^O(t+\Delta)=1\mid\text{measured prefix }h_t,\text{declared future control }c\}
\]

or, separately, a probability of a **specified first detection** within a finite horizon. `h_t` contains only information available by the forecast origin; `c` may contain a future commanded path only if that is actually known at prediction time. An actual future measured force must not be smuggled into a prospective forecast. If estimating conditional response using realized future force, label that task explicitly as retrospective reconstruction.

An honest boundary is then a level set of a finite-horizon, protocol-conditioned probability. It may have more than one component. A single threshold should not be imposed by deleting inconvenient nonmonotone cells.

This target has practical meaning: it asks whether a requested bowing path will produce the operationally specified regime by a usable deadline and whether that answer transfers to another specimen/session. It does not require pretending to know the infinite-time maintenance boundary.

### Plausible method direction, with no present novelty entitlement

A useful candidate could jointly model **entry and loss of the observed regime**, conditioned on measured control history, with an explicitly fixed observation operator and a declared support/rejection rule. The potentially distinctive contribution would be a verified treatment of finite-window observations, path exposure and limited support that improves calibrated held-specimen prediction without pretending the latent equilibrium is identified.

The ingredients are not new in themselves. Recurrent dynamic survival prediction from longitudinal measurements already exists, for example [Dynamic-DeepHit, Lee et al., IEEE TBME (2019 online / 2020 issue)](https://ieeexplore.ieee.org/document/8681104). In the bowed-string setting Beigi already compares recurrent controllers and state-dependent behavior. A standard HMM, a two-state hazard model, a recurrent network, or a new acronym for their combination is therefore an implementation/baseline, not the methodological advance.

To earn a stronger methods claim, demonstrate a specific nontrivial result, such as an observation-operator correction whose target and bias can be proved under stated assumptions, or sharp protocol-support bounds with attainable witnesses and genuine computational benefit. Then show that result helps on data and survives alternatives. Arbitrary latent error rates and transition rates jointly learned from one weak classifier are generally not separately identified without anchors; do not advertise that as calibrated physical-state inference.

## 4. Minimal falsifiable evaluation before major model work

1. **Lock the operational target.** Name window length, observer, time origin, forecast horizon, recurrence handling and event timestamp. Window length changes the classifier's periodicity statistic, so it changes the target unless a correction is actually validated.
2. **Keep one stroke together.** No random window-level split. Overlapping windows and whole trajectories must stay in their assigned split. Tune on training strokes only, use held β columns or blocks for within-session tests, and hold complete specimens/sessions for transport tests.
3. **Freeze a future specimen now.** A1/A2 and already read diagnostics are development data. A later raw archive can supply prospective validation only if its outcomes are not inspected before model/analysis choices are frozen. Sensor-level eligibility and extraction checks are allowed only as specified; outcome-driven changes end that status.
4. **Use a common available horizon.** Give the at-risk/support count at every horizon and speed. Do not compare a late slow-speed sample with a fast-speed sample that cannot still be observed. For extrapolation beyond support, report failure or a bound rather than a point guarantee.
5. **Disentangle information from architecture.** Compare instantaneous-controls logistic/GAM/tree predictors; the same methods with predeclared path summaries; a small recurrent model; and the candidate, all with the same observable input budget. Include a persistence baseline when past regime labels are provided. A gain from additional force history is an information result unless the candidate beats equally informed baselines.
6. **Forecast rather than classify the input.** Prevent shared raw samples between input features and target windows at the forecast boundary. A model that sees the same bridge waveform used to generate its target tests label imitation, not control-path prediction.
7. **Test probabilistic and decision performance.** Report per-session finite-horizon Brier/log scores, calibration, detection false alarms, recurrence, and regime IoU at frozen thresholds. Use the specified shared observer for each model comparison. A favorable pooled score must not erase failures on a specimen or speed.
8. **Respect the unit count.** Thousands of overlapping windows are not thousands of independent specimens. Stroke/block resampling describes conditional variability; generalization across instruments needs enough independent specimens. A1→A2 and A2→A1 are two directional tests on two physical instances, not a broad population estimate.
9. **Challenge the claimed mechanism.** Run measurement-offset/scale sensitivity and force-history ablations; compare time and travel only where support exists; separate current-force-only, force-history-only, attack and combined features. Report unavailable contrasts.
10. **Stop or downgrade the claim** if the improvement vanishes against a history-informed GAM/tree baseline, requires realized future measurements, is confined to one specimen/window, relies on unsupported counterfactual paths, or requires calling a model-dependent asymptote physical truth. A carefully replicated empirical protocol effect can still be useful acoustics research.

## 5. What may and may not be said now

| Statement | Assessment |
|---|---|
| “Matched-time speed exponent is near +1 in the corrected A2 analysis.” | A development result to be quoted with the exact observer, force variable and available-support definition; not re-estimated by this literature audit. |
| “We isolated settling and recovered the true stationary v/β² law.” | Unsupported: force history/attack/support remain coupled, and the fast-speed asymptote is fitted rather than directly observed. |
| “The force readings have a travel-dependent component consistent with the open-loop rig.” | A reasonable empirical claim after independent reproduction of the block-level checks; the source supports the open-loop mechanism, not every measured magnitude. |
| “The old speed-independent violin result was wrong.” | Unsupported. Different strings, protocols and observers are involved; the supplied reanalysis preserves its result. |
| “The two protocol boundaries can differ in simulations.” | Plausible, model-specific evidence; requires numerical/contact classification checks. The conceptual distinction and history dependence have precedent. |
| “The fitted survival exponent identifies saddle-node/Hopf/grazing.” | Unsupported, even with a narrow parametric confidence interval. |
| “Protocol-conditioned finite-horizon prediction improves across held-out specimens.” | A valuable, testable next claim. It is not yet an established result. |
| “This is a generally new high-novelty methodology.” | Not established. A concrete algorithm/theory advance plus fair performance improvement is still needed. |

## Source-access record

Opened primary full text on the web: Schoonderwaldt 2008 (coauthor-hosted PDF); Lampis 2024 Acta (institution-hosted PDF); Lampis 2025 POMA (coauthor-lab PDF); Beigi & Conneely arXiv v3 HTML; Meisel 2015 PLoS article. Read the supplied van Walstijn 2026 full text locally; DOI publisher fetch failed. Yuen/Musta v2 abstract/version record opened. Dynamic-DeepHit's publisher search result supplied its abstract; direct page was robot-blocked, so only its documented high-level longitudinal-prediction capability is used here. No secondary summary is used as evidence for a physical or mathematical claim.

The earlier censoring/transport search also found direct prior art in [Liu, de Paula & Tamer, arXiv:2501.10117](https://arxiv.org/html/2501.10117v1) (prediction sets under interval censoring), [Horowitz & Lee, CWP14/26, 2026](https://cemmap.ac.uk/publication/binary-classification-with-the-maximum-score-model-and-linear-programming-2/) (partial-identification minimax-regret classification), and [Pfister & Bühlmann, arXiv:2402.09758](https://arxiv.org/html/2402.09758v2) (extrapolation under explicit functional restrictions). These reinforce the same caution: finite-data uncertainty plus an optimization procedure is not, by itself, a new methodological gap.
