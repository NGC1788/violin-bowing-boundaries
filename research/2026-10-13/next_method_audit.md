# Next-method audit: identification before another predictor

Date: 2026-10-13. Independent literature and feasibility review after commit
`fce8d32`. This note proposes no modification to either frozen A2 protocol and
does not inspect A2 outcomes. Existing sample1 and published-D results are
development evidence. Searches used primary papers and proceedings, including
2024–2026 publications; absence from this bounded search is not evidence of
worldwide novelty.

## Decision

**There is not yet a verified new-method gap that justifies another large model
implementation.** The most useful next question is whether the available
observations identify a transferable boundary law at all. One small, source-only
identification experiment is justified before pursuing this as a method. A
second idea, shared-observer model dominance, is mathematically clean but is
currently better treated as an evaluation diagnostic.

Neither recommendation converts the earlier negative coherent-loss results or
the nearly tied cello affine-flow results into a positive discovery. Published-D
width deformation is an interesting conditional result; it is not evidence of a
general new learning principle.

## 1. Strongest conditional direction: identify the transport law from censored boundaries

**Question:** Can we distinguish genuine changes in the speed dependence of
bow-force boundaries from the apparent changes caused by an incomplete force
range, a finite grid, and a shared observation rule?

The desired output would be a set of source-compatible transport laws and
explicitly unresolved target cells. This is a stronger scientific question than
whether another fitted scalar exponent or flow gets a slightly higher IoU. It
also addresses why a fitted width parameter can help prediction without being
physically identified.

### What is already known

- Set-membership system identification already uses finite noisy data and
  regularity constraints to bound unobserved function values. Thus a feasible
  model set and an envelope are not new: [Milanese and Novara, Automatica
  2004](https://doi.org/10.1016/j.automatica.2004.02.002).
- Derivative-constrained nonparametric extrapolation and partial identification
  of out-of-support function values are explicit in [Pfister and Bühlmann,
  2024 preprint](https://arxiv.org/abs/2402.09758). Our speed values are only
  a few slices, so their assumptions cannot be justified merely by fitting a
  smooth curve through those slices.
- Sharp smoothness-based bounds outside observational overlap, with efficient
  computation, are established for policy evaluation by [Khan, Saveski and
  Ugander, ICML 2024](https://openreview.net/pdf?id=oiY7yhyi6W). Using a linear
  program for extrapolation is not itself a contribution.
- Set-valued prediction for domain generalization is already the explicit
  subject of [Tsibulsky, Nevo and Shalit, ICML
  2025](https://proceedings.mlr.press/v267/tsibulsky25a.html). Abstention,
  small prediction sets, and domain robustness cannot be presented as a new
  paradigm here.
- Boundary asymmetry is already empirical physics in [Schoonderwaldt, Guettler
  and Askenfelt, 2008](https://doi.org/10.3813/AAA.918070). Their violin-string
  experiment and proposed law are not independent evidence streams.

The possible contribution would have to be a **sharp, useful propagation of
joint boundary censoring and a shared observer uncertainty through a physically
specified transport class**, together with an identification/falsification
result. None of those italicized requirements is established yet. Simply
combining the cited ideas and calling the output uncertainty-aware is insufficient.

### A minimal exact calculation, and why it is a baseline

Fix a bridge-position column and one operational class rule. If its observed
Helmholtz cells form one interval, an adjacent negative/positive pair brackets
the lower boundary; an adjacent positive/negative pair brackets the upper
boundary. A missing exterior negative is a one-sided constraint. Internal
negative cells between positive cells contradict the single-band assumption;
they must not be silently filled. Multiple measured outcomes at the same
setting likewise require an explicit observation model, not unannounced voting.

For either boundary write its log force at source log speed t_j as

    r_j ∈ [L_j, U_j],       r(t) = c + α t.

The bracket endpoints may be infinite. The feasible exponent set is exactly

    A = intersection over i<j of
        [(L_j-U_i)/(t_j-t_i), (U_j-L_i)/(t_j-t_i)].

For any α in A the admissible intercept interval is

    [ max_j(L_j-α t_j), min_j(U_j-α t_j) ].

This follows by eliminating c from the intervals: intervals on the real line
have a common intersection precisely when every pair intersects. The formula
is an elementary interval-feasibility calculation, **not a novelty claim**.
Closed brackets conservatively include exact transition endpoints; a final
implementation must keep its classification convention and numeric tolerance
explicit. An all-negative column does not locate the two boundaries.

For a common α across several bridge positions, intersect their constraints,
while retaining separate intercepts. Lower and upper exponents must be handled
separately and their boundaries must remain ordered over the declared speed
range. With uncertain labels, one coherent observer must determine all source
columns. Combining a lower root from one observer with an upper root from
another enlarges the set and can destroy sharpness.

Even without sampling noise, these brackets often identify an interval of
exponents, not a point. This exact calculation is a necessary comparator for
any future learned transport-uncertainty method.

### The indispensable assumptions

1. **A single band for the operational class.** This is plausible for much of
   published D, but is not a universal consequence of bowed-string physics and
   is already questionable for the retained cello indicator labels. Excluding
   disagreeing cells or choosing whichever observer makes a band look clean
   would change the problem. For a strict guarantee, an incompatible observer
   either falsifies the maintained model or leaves its predictions unresolved.
2. **A law valid outside the observed speeds.** Exact power-law boundaries are
   one option. A stated global derivative/curvature bound is another. Two
   observed speeds cannot determine that bound. Smoothness by itself supplies
   no quantitative extrapolation guarantee: adding
   `M * product_j(t-t_j)` preserves all observed boundary values and can change
   an unobserved value arbitrarily. This is an elementary nonidentification
   argument, not a new theorem.
3. **A defensible error model.** Grid intervals are observation constraints,
   not confidence intervals. Allowing a selected number of wrong labels is an
   additional contamination assumption. Its tolerance cannot be tuned on the
   held-out speed and then called coverage. Source cross-validation by itself
   cannot prove a bound on target-domain errors.
4. **An explicit specimen assumption.** A speed law within one specimen is
   not enough to bound another specimen's baseline force limits. Cross-specimen
   guarantees require bounded specimen variation or additional source specimens.
   Such a bound is not presently established. This limitation must not be hidden
   by evaluating only the favorable within-specimen comparison.

Unknown common multiplicative force calibration cancels from a log-force band
width, but only if both boundaries refer to the same calibration. Additive
offsets, speed-dependent gains and unknown missing roots do not cancel. This
simple invariance may make width identification useful; it is not a cure for
boundary censoring or an original invariance principle.

### The actual mathematical/algorithmic hurdle

The nontrivial version retains **one shared observer parameter**, disjunctive
membership constraints, source-derived interval censoring, separate boundary
speed laws, and ordering at all target speeds. It must project this joint
feasible set into target membership without replacing it by unrelated pointwise
intervals. A correct projection must give an explicit compatible source witness
for every endpoint it calls attainable, and an explicit infeasibility witness
when a proposed law cannot explain the source data.

An outer interval from a local optimizer is not a sharp bound. A handful of
fitted near-optimal flows is not an exhaustive identified set. Model agreement
is not coverage of the physical truth. Enumeration and linear programming are
adequate baselines at these small grid sizes; an efficient new solver would
need a demonstrated complexity or practical benefit beyond them.

There may be no useful computational gap here: two/three speeds and eleven or
forty bridge positions are small, and the simple power-law version has the
closed-form elimination above. The more general formulation only becomes
interesting if its extra structure produces narrower *valid* target sets on
both empirical domains. Complexity invented by enlarging the model class is
not an achievement.

### One cheap decision experiment before implementation

Use the already assigned source bracket/topology audit; do not start another
training sweep. On each declared source-speed pair:

1. Preserve actual adjacent-cell brackets and one-sided bounds. Report the
   number of non-band, all-negative and censored columns before fitting.
2. Compute the exact feasible lower/upper exponent intervals under the
   simplest power-law assumption. For cello, report nominal and all fifteen
   coherent source rules separately; published D has one author-classification
   rule. Report empty and unbounded sets, not just successful examples.
3. If these sets are nonempty and useful, freeze them before a single held-speed
   evaluation. Produce target singleton/ambiguous predictions from all
   source-compatible laws. Compare with the exact interval baseline, midpoint
   imputation, and the already frozen point predictors at matched retained-cell
   coverage. Report coverage, false-positive and false-negative rates, and the
   unresolved fraction for every task; include cross-specimen failures.

The falsifiable benefit is **fewer incorrect confident labels at the same
retained coverage**, while propagating acquisition limits honestly. Greater
coverage obtained by declaring uncertainty away is a failure. So is near-total
abstention, an advantage confined to D, or a result that requires selecting the
law/observer after seeing targets. No guarantee that this test will succeed is
available. Because existing data have already been explored, a positive result
would remain development evidence and would not alter the A2 frozen protocols.

If the initial source audit shows predominantly incompatible or unbounded
sets, stop this direction as a prediction-method proposal. The negative
identification result may still be scientifically informative, but should be
written as such rather than as a new robust predictor.

### Refinement: every minimum-disagreement band, and a falsification gap

The useful next object is not one manually chosen repaired band. For each
source column, retain **all** intervals attaining its minimum observed label
disagreement, including censored intervals. Then ask whether a proposed speed
law can realize a compatible choice across all columns and source speeds.
This avoids a real inferential error: rejecting a law merely because an arbitrary
tie-breaking rule selected an inconvenient boundary.

However, all-optimum retention is not a new machine-learning principle.
Minimum-disagreement interval/rectangle algorithms have direct precedent in
[Darnstädt, Ries and Simon, TCS
2018](https://doi.org/10.1016/j.tcs.2017.11.022). Retaining competing near-optimal
predictors is also established model-multiplicity methodology, for example
[Hsu et al., ICLR 2024](https://arxiv.org/abs/2402.00728). The latter is not an
exact algorithm for these intervals; it rules out claiming the general
all-good-models principle as new.

More directly, feasibility-based discrimination with disjunctive and coupled
constraints is studied in [Tractable model discrimination for safety-critical
systems, NAHS 2022](https://www.sciencedirect.com/science/article/pii/S1751570X22000395).
Set-membership outlier analysis using infeasibility certificates and
combinatorial searches predates this in [Borchers and Findeisen, NOLCOS
2013](https://skoge.folk.ntnu.no/prost/proceedings/nolcos-2013/papers/0161.pdf).
Those settings are dynamical systems; the connection is the inferential
principle and optimization structure, not a claim that they already analyze
Schelleng diagrams.

A more interpretable comparison than a yes/no minimum-repair feasibility test
would report the **extra disagreement required by each law**. On the same
source rows and with the same weights, define

    L_band = minimum disagreement over independent interval bands;
    L_M    = minimum disagreement over bands satisfying speed law M;
    G_M    = L_M - L_band ≥ 0.

If every optimum is represented, `G_M=0` means at least one globally compatible
set of independently best bands exists. `G_M>0` means the law needs extra
operational-label disagreements beyond those already forced by the band
assumption. This is an empirical model-invalidation/repair quantity, **not a
p-value or a physical rejection probability**. It usefully separates failure
of band topology from extra failure of transport, without replacing observed
labels by purported ground truth. It is still an established optimization idea,
not a new methodology by definition.

The genuine possible gap is an exact, output-sensitive solver for the **joint**
shared-observer and censored-band problem, with certificates and provable
improvement over exhaustive enumeration/MILP, plus a justified statistical
interpretation of the gap under correlated acquisitions. A pointwise union of
boundary brackets need not be attainable jointly; this must be proved or
tracked with source witnesses. For the current tiny force grids, brute-force
interval enumeration may already be sufficient, which removes the practical
motivation for an elaborate solver. Merely using dynamic programming, interval
unions, or a MILP is not the missing contribution.

This refinement changes the immediate stopping rule: if the source audit
falsifies a constant-width generator, document the failure and compare a small
set of *previously motivated* laws through `G_M`; do not keep widening the model
until one passes and then present the final law as discovered. A force-offset
law is also prior art: [Mores, JASA
2017](https://pubmed.ncbi.nlm.nih.gov/28863554/) reports a nonzero minimum-force
baseline at very low bow speed. This primary abstract does not justify every
possible additive functional form, but it decisively prevents claiming the
baseline-force concept as new.

**Explicit stop criteria for a method-paper direction:** stop if the simple
finite-grid procedure already solves all instances cheaply; if admissible
target sets are mostly unresolved; if the uncertainty/error tolerance is chosen
after target inspection; if positive claims require dropping non-band columns;
or if improvement only occurs in one empirical corpus. The resulting source
falsification study can still be valuable acoustics research, but its contribution
would be the scientific evidence and auditable analysis, not a new general
learning algorithm.

## 2. Reserve diagnostic: shared-observer pairwise dominance

**Question:** Does one model beat another under every *same* admissible
observation rule, rather than merely have a better individually worst score?

Let P and Q be hard predicted sets, Y_θ one shared operational label map, and
a, b, n its weighted set masses |P|, |Q|, |Y_θ|. With positive union denominators,

    IoU(P,Y_θ) ≥ IoU(Q,Y_θ)
    iff TP_P(θ) * [b+n(θ)] ≥ TP_Q(θ) * [a+n(θ)].

Cross-products of the true-positive counts cancel. If a=b>0, dominance is
equivalent to `TP_P(θ)-TP_Q(θ) ≥ 0`. The worst difference in that special case
is one signed weighted-quadrant problem, already covered by the geometric
primitive used in the existing observer audit. Empty unions need the explicitly
declared score convention and separate treatment.

For a≠b, the joint counts retain a coupled quadratic term. A sum/difference of
ratios is not the one-ratio Dinkelbach problem. A claimed exact solver must
optimize the **shared** θ, validate against exhaustive finite states, and avoid
substituting `min IoU(P)-min IoU(Q)` for `min[IoU(P)-IoU(Q)]`.

This could provide a sharper *comparison* than nonoverlap of two individual
score intervals. It cannot improve predictions by itself. The general decision
principle is established in imprecise-probability dominance, for example
[Zaffalon, ISIPTA 1999](https://isipta99.sipta.org/proc/019.html), and label-free
performance identification is already studied by [Maia Polo et al., NeurIPS
2024](https://proceedings.neurips.cc/paper_files/paper/2024/hash/f4c6bec746b0aeca8c2cd15096f1ad1f-Abstract-Conference.html).
Those papers do not by themselves establish an exact solver for this particular
shared-threshold IoU difference; equally, not finding that exact formulation
here is not evidence that it is new.

**Why it is not the primary next method:** the existing continuous observer
audit found tiny grid optimism and no within-source pairwise ranking reversals.
That does not prove pairwise dominance is uninformative, but it supplies weak
motivation for a new solver. More decisively, published D has no waveform-derived
observer family: its author classifications form a singleton. The method
therefore degenerates to ordinary score subtraction on D. Inventing uncertain
D labels or treating unanalysed E cells as an observer family would not be a
legitimate two-domain validation.

A cheap check, if needed for the paper, is to enumerate the existing finite
observer states for the already frozen model predictions and report how many
model pairs have certified same-rule dominance. This is a diagnostic; do not
expand it into a training method unless it first changes a consequential
conclusion.

## Why endpoint inversion is not a third recommendation

The completed endpoint audit found a common-scale failure: shared impedance
error can preserve agreement between reconstructed ends while severely changing
the inferred sticking indicator. A method would need external calibration or
additional independently defensible constitutive constraints to remove that
ambiguity. The 2008 D data contain classifications, not the required endpoint
waveforms. Thus it cannot currently satisfy the proposed two-domain test.
Reconstruction of bow-point friction force also has direct precedent in
[Woodhouse, Schumacher and Garoff, JASA 2000](https://doi.org/10.1121/1.429529).

## Latest bowed-string overlap that any proposal must face

[Van Walstijn et al., Acta Acustica
2026](https://doi.org/10.1051/aacus/2026042) combines thermal and elasto-plastic
friction, with measured cello comparisons at 0.1 m/s. A simple differentiable
friction extension is not a sufficient novelty argument against this work.
[Beigi and Conneely, September 2026
preprint](https://arxiv.org/abs/2609.14990) already reports a changed minimum-force
law in simulation, four violin-string maps, learned controllers, and a failure
of periodicity-only Helmholtz classification. Its claims are preprint evidence,
not independent experimental truth; nevertheless those broad topics cannot be
advertised as new here.

The plausible remaining scientific contribution is a careful statement of
**which boundary-law distinctions these public measurements can and cannot
support**. A strong methodological paper would require more than this statement:
a new, technically nontrivial inferential procedure, verified assumptions, and
nontrivial benefit across both empirical settings. At present that combination
has not been demonstrated.
