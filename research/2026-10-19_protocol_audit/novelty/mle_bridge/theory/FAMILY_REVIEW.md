# Whole-family interpretation review

2026-10-21. This note reviews the explicitly post hoc extension fixed in `../FAMILY_EXTENSION.md` (local commit 45014c5). It does not change the earlier selected-certificate experiment. `family.py` joins saved geometry to saved MLE deletion refits; it runs no new optimizer and changes no labels.

## What was checked

All 7,718 minimum pairs and 24 minimum triples from the earlier complete geometric search have exact ID matches in the corresponding exhaustive deletion-control populations. They form 7,742 certificate memberships across 77 folder–slope cases. These are overlapping subsets of the same measurements, not independent replications. Fifty-two cases are classifier-specific geometric failures; 25 were already force-grid-infeasible. Results and member-level values are retained in `family.json`.

The earlier lexicographic choice is not representative of every valid minimum explanation. In the 52 classifier-specific cases, family sizes range from 1 to 381 (median 38). Thirty-one families contain both deletions that increase the descriptive fixed-law likelihood ratio T and deletions that decrease it. Thus cardinality and uniform geometric validity do not fix the likelihood direction.

The opposite limitation is equally important: the absence of MLE support cannot always be blamed on a poor choice of explanation. In 12 of the 52 cases, every minimum geometric certificate moves the fitted beta exponent farther from the target when removed. In six cases, every member increases T. Both cases requiring exactly three columns belong to this latter group:

| Minimum-triple family | Number of triples | T before deletion | T increase for every member | Beta movement |
|---|---:|---:|---:|---|
| A1, 0.20 m/s, p=-1.2 | 19 | 17.5865 | +0.9993 to +5.9015 | All farther from -1.2 |
| B_T2r1, 0.05 m/s, p=-2 | 5 | 4.9414 | +0.0973 to +0.9583 | All farther from -2 |

This does not contradict the exact geometric theorem. Those triples prevent a common intercept from covering every selected finite bracket throughout the waveform-rule domain. In a likelihood fit, observations that disagree with a hard tolerance can still help locate a slope, support a less extreme alternative, or raise residual scale. Removing them changes the fit and the relative support of both models.

## What a likelihood-aware choice adds, and what it does not

Selecting the minimum geometric certificate with greatest T reduction gives a positive reduction in 46/52 classifier-specific cases. Only 2/52 attain the greatest reduction available among every size-matched eligible deletion. The selected family's best T change can still be strongly negative; the worst case is B_T2r2 at 0.05 m/s, p=-1.2, where even its best geometric pair increases T by approximately 42.82.

The median control rank of the best family member is about .981, compared with .611 for the previously fixed member. This is NOT evidence of a statistically significant improvement or of enrichment of geometric certificates. Taking a maximum over a family containing tens or hundreds of candidates naturally raises its rank. The extension reports the selected maximum descriptively, with family sizes and all member outcomes, and supplies no random-family calibration or selective-inference p-value. A claim of special predictive value would need a separate comparison accounting for the size and structure of the family.

The useful output is an explanation choice constrained to retain the proven geometric property, accompanied by its observed MLE effect. It is not a minimum MLE-reversal set, a causal explanation, a command to delete inconvenient measurements, or a claim that the best member generalizes to another experiment.

## Coherent-rule example reviewed separately

The prespecified no-loss near-cutoff rule changes the ratio ceiling from .5 to approximately .493263927. It improves A4 0.20 m/s's hard required tolerance from about .4092 to .2056. Yet the original 120-row A4 MLE moves beta from -1.41687275 to -1.40459447, farther from -2, and its descriptive T increases from 32.21608 to 34.94402.

The estimands intentionally differ: the hard decision uses originally finite eligible columns in one speed folder, whereas the original MLE uses all three speeds and includes censoring. The rule changes two A4 MLE intervals in total, including a formerly left-censored interval that becomes finite (finite 85→86, left 27→26, right 8→8). Therefore the MLE shift cannot be attributed solely to the one finite target-boundary change. This is a direct example of why a favorable hard-predicate change does not imply a favorable likelihood change; it is not an old confidence-interval reversal or evidence of a new physical law.
