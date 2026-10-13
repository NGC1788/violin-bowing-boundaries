# What numerical and observation uncertainty can contribute

This note evaluates a possible next direction after the coherent-family learning objective and the asymmetric-boundary prior failed their main comparisons. Those are negative results. A newly found numerical issue does not retroactively turn either learner into a successful method.

## Decision

The strongest defensible next result is a **crossed audit of whether a reported bowed-string regime boundary survives numerical refinement, settling time, and the observation rule**. It could become a useful computational musical-acoustics contribution if it demonstrates an attributable and reproducible failure of current boundary estimation, followed by a validated remedy. It is not yet a new general machine-learning algorithm, and generic multifidelity or boundary-enclosure language would overstate novelty.

The initial changes, 21/90 labels from 50 to 100 kHz and 19/96 from 100 to 200 kHz, do not establish convergence. The populations also differ because some coarse cases are rejected. Compare on common eligible cases as well as each available pair. A denser discretization is a more expensive calculation, not automatically ground truth.

## Closest primary literature

| Source | Relevant prior and limit on a novelty claim |
|---|---|
| [Marques, Lam & Willcox, CLoVER, NeurIPS 2018](https://proceedings.neurips.cc/paper/2018/file/01a0683665f38d8e5e567b3b15ca98bf-Paper.pdf) | Cost-aware contour estimation already combines biased, noisy information sources without requiring a fidelity ordering. It models a target function with an unbiased source; our highest-rate simulation has not been shown unbiased. Merely adding an observer setting as another source would not be a new principle. |
| [Renganathan, Rao & Navon, CAMERA (2022)](https://arxiv.org/abs/2203.01436) | Adaptive multifidelity GP modelling for failure-level-set and probability estimation already exists. Correct authors are Renganathan, Rao and Navon; an earlier audit row mistakenly said Chaudhuri. |
| [Mell, Rey & Schoefs, IJNME 121, 4566–4583 (2020)](https://doi.org/10.1002/nme.6451); [author-network full text](https://www.weamec.fr/en/wp-content/uploads/sites/2/2020/11/Multifidelity-adaptive-kriging-metamodel-based-on-discretization-error-bounds.pdf) | The closest enclosure precedent couples multifidelity Kriging to actual discretization-error bounds, preserving safe/failure classification and bounding failure probability. Our adjacent-rate differences are empirical sensitivity estimates, not those rigorous bounds. |
| [Arai et al., SIADS 8, 757–789 (2009)](https://eprints.lib.hokudai.ac.jp/repo/huscap/all/39480/JADS8-3_p757-789.pdf) | Rigorous outer approximations, Morse decompositions, continuation and Conley indices already map dynamical behavior over parameter boxes. Certification requires enclosing the dynamics, not checking several trajectories. A high-dimensional nonsmooth delay-line friction system would need substantial new numerical analysis to use this machinery rigorously. |
| [Desvages & Bilbao, Applied Sciences 6, 135 (2016)](https://www.research.ed.ac.uk/files/28188128/Bilbao_open_access_paper.pdf) | Energy-balanced numerical schemes for nonlinear bowed-string contact are established. Numerical energy stability alone does not establish convergence of a regime boundary or correctness of a physical friction model. |
| [Russo, Ducceschi & Bilbao, DAFx 2022](https://www.dafx.de/paper-archive/2022/papers/DAFx20in22_paper_14.pdf); [authors' code and erratum](https://github.com/Nemus-Project/EfficientBowedString) | Efficient bowed-string solvers already compare integrators and high-rate reference solutions. The paper's bowed-mass solver comparison uses a 30×44.1 kHz reference, illustrating that 200 kHz is not a universal accuracy certificate. Their code includes multiple independently formulated solvers; reproducing a small matched comparison could be more informative than only increasing our own solver's rate. |
| [Mores, PLOS ONE 14, e0211217 (2019)](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0211217) | Reports coexisting stable bowing patterns at the same bow location, selected during onset. Thus history dependence has physical precedent. A single-start regime map need not characterize all attainable motions. |
| [Passa, Terrien, Maugeais & Gazengel, JSV 621, 119467 (2026)](https://www.sciencedirect.com/science/article/abs/pii/S0022460X25005401) | Studies basins of periodic regimes in a multistable self-oscillating musical-instrument model. The accessible primary abstract supports the relevance of coexisting regimes; no detailed claim about its algorithm is made here because the full paper was not available in this audit. |
| [Matusiak, Chatziioannou & van Walstijn (2025)](https://doi.org/10.3389/frsip.2025.1525044) and [van Walstijn et al. (2026)](https://doi.org/10.1051/aacus/2026042) | Passive elasto-plastic and thermal elasto-plastic bowed-string formulations are closer modern physical baselines than our elementary point-bow model. Correcting its recurrence does not close that model gap. |

No broad claim of first use, certification, first stable mapping, or a new multifidelity principle survives this prior literature. A specific reproducible insight may still be new; that requires results and a more focused comparison.

## Define the target before reducing uncertainty

Let `x` denote physical controls, `r` a numerical resolution, `T` elapsed settling time, `h` an onset/history protocol, and `eta` an observation rule. The computed outcome is

`Y[r,T,h,eta](x) = O_eta( simulation[r,T,h](x) )`.

These indices do not all mean measurement noise. A change in `h` can select a different physical attractor. A change in `eta` can alter the operational definition without altering the dynamics. A change in `r` can alter both discrete dynamics and the realized numerical resonator. Therefore a pooled probability of Helmholtz motion over all settings has no unique physical meaning until their distribution and the intended estimand are specified.

There are at least three legitimate targets:

1. Outcome for one declared onset/history and observation protocol.
2. Reachability: at least one history in a declared family reaches the target regime.
3. Robust attainability: every history in that family reaches it.

Do not mix them. Even exhaustive testing of three ramps is not exhaustive testing of the state-space basin.

For a finite declared family `A`, the empirical sets `L_A=intersection_a S_a` and `U_A=union_a S_a` summarize agreement. Adding more configurations can only shrink `L_A` and enlarge `U_A`. Consequently their width is not automatically an error bar that shrinks with computation: it can reveal persistent differences between protocols. Replacing obsolete coarse resolutions by a refined tail is a different operation and requires an explicit rule. Neither construction encloses the continuum physical set without additional assumptions.

## A cheap experiment that distinguishes mechanisms

The new `convergence_factorial.py` already provides the correct starting design: a prespecified 96-case subset; rates 50, 100, 200, 400, 800 kHz; durations 0.6 and 1.2 s; a common final 0.4 s observation window; and a common 50 kHz box-averaged output. This is substantially more informative than selecting a few attractive or failing waveforms.

Interpret its results together with the following contrasts:

- **Resolution contrast:** compare adjacent rates at fixed history, duration and observation rule. Report common-case counts and invalid coarse configurations.
- **Settling contrast:** compare final windows at both durations at fixed rate. More elapsed time can reveal a different attractor or remaining transient; it is not the same uncertainty as discretization.
- **Observer contrast:** apply the entire predeclared threshold family to the same saved waveform/features. This costs essentially no new simulation. It tests decision-rule fragility while holding dynamics fixed.
- **Internal-state contrast:** compare stick fraction, slip-event pattern, frequency and broad spectral summaries across rates. A waveform label flip with nearly unchanged state summaries is different evidence from a large physical-state change. Conversely, agreement in a few summaries does not prove identical trajectories.
- **History contrast, only if needed after these results:** use a few declared onset ramps with the same final force/speed, recording after equal post-ramp time. Treat this as protocol dependence, not random uncertainty or a full basin estimate.

The existing new code reports fixed nominal-frequency spectral bands, which avoids making frequency changes disappear by aligning every spectrum to its estimated period. Internal contact states are privileged simulator diagnostics; they cannot be used as if measured on the public cello data. Average slip onsets per nominal period alone also cannot prove one slip in every cycle: irregular timing can have the same average.

The current waveform classifier has a subharmonic rejection gate but no symmetric upper-frequency gate. Thus a highly periodic single-flyback waveform well above nominal frequency can still receive its `helmholtz` shape label. Retain the term **provisional waveform label** until a mechanism-level rule has been separately specified and tested. The September 2026 [Beigi & Conneely preprint](https://arxiv.org/html/2609.14990v3) already warns that periodic continuous slip can mimic Helmholtz under output-only diagnostics; requiring a stick test alone is therefore not new.

## Numerical confounding worth testing

The fractional-delay waveguide realizes a filter, not an exact continuum string. Linear interpolation adds frequency-dependent attenuation that varies with beta and rate. Subtracting a low-frequency group delay does not match the complete phase response at every harmonic. The new `linear_decay_diagnostics` explicitly computes effective damping and phase at nominal frequency, so it can test whether large regime changes track changes in the realized resonator.

One plausible **mechanistic** result would be: apparent boundary shifts attributed to nonlinear friction are partly explained by a changing linear resonator, and matching its measured/analytic decay and frequency response reduces those shifts at fixed cost. This is a hypothesis, not an established result. It needs a matched ablation and an independent solver. Fractional-delay compensation, lossless all-pass interpolation and energy-consistent integration are already established tools; using them is not in itself novel.

Matching only fundamental Q may leave different high-harmonic damping. Friction switching, finite contact width and thermal state can alter sensitivity. A successful correction on one elementary waveguide would need carefully limited claims unless it transfers to another solver/friction law.

## What a real certificate would require

Suppose a declared continuous diagnostic margin `m(x)` has a computed approximation `m_r(x)` and a **proved** bound `|m_r-m| <= e_r`. Then `m_r-e_r>0` certifies a positive sign and `m_r+e_r<0` certifies a negative sign. The unresolved interval is a valid outer region. This is a standard error-bound argument, already exploited in related reliability work, not a new theorem here.

Adjacent-rate differences do not supply `e_r` without convergence order, stability and asymptotic-regime assumptions. Binary flyback counts can jump when an extremum appears; a smooth-state norm bound need not give a useful bound on that diagnostic. Multistability and long transients complicate the limit in both resolution and time. The observation operator must also be included in a proof rather than treated as exact by definition.

Finite agreement can be reported as a stability check. It cannot be advertised as guaranteed physical classification, certified bifurcation, or calibrated uncertainty.

## Route to a method, and stopping conditions

Only after attribution works would an adaptive strategy be justified: spend the next unit of compute on rate refinement, a longer run, or an additional history according to the cause of unresolved classification, while evaluating observer alternatives almost freely. A useful outcome would be fewer **incorrectly declared stable cells** at a matched compute cost, or the same independently checked stable coverage for less computation.

Required comparators are fixed high resolution, uniform refinement, a simple adjacent-rate agreement stop rule, and a conventional cost-aware contour strategy. Evaluate on independent prespecified conditions and a stronger held-out numerical check. A finite 800 kHz reference can support relative agreement claims, not continuum certification. Label ambiguous cells honestly; do not get an apparent accuracy gain merely by abstaining on most of the parameter domain. Report coverage and error together.

Adding these actions to a scheduler is not enough for methodological novelty because CLoVER already handles multiple information sources. A new contribution would require a clearly different estimand, a justified error or decision guarantee, or strong evidence that the proposed attribution changes scientifically important conclusions and transfers across independent formulations. None has been demonstrated yet.

Stop or narrow the claim if refinement differences persist without an attributable mechanism, if the trusted coverage becomes tiny, if the result exists only in the corrected legacy implementation, or if modern independent solvers show no analogous problem. In that case a transparent negative comparison and reproducibility package are useful outcomes, but an ICML-level methodological claim would be unsupported.
