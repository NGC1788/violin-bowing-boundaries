# Projection audit v2 — preflight and execution review

2026-10-30 KST. This is a standard convex numerical solver audit. No novel method or observed-label forecasting improvement is claimed.

Before any actual prediction array was opened, the root reviewer inspected the solver and runner. The requested correction was to remove a hardcoded zero FW gap in the coherent-identity branch: recovering weights with a finite reconstruction residual does not justify substituting an exact zero gradient. The revised implementation computes the objective, gradient and full-simplex FW gap at the actual represented point `w @ bank`, records its residual from q, and returns q bitwise unchanged. A failed gap is still recorded as failure. Inputs outside the clip domain receive no fabricated interior-gradient certificate.

The corrected synthetic suite passed 20 tests, including analytic gradient/Hessian finite differences, a forward-KL closed form, 96 fixed random/near-boundary/near-coherent inputs, exact vertex identity, identity near the clipping threshold, failure fallback, array-key restrictions and identical ID selection to v1. These tests do not establish adequacy on the real prediction inputs.

The root reviewer explicitly authorized freezing and exactly one actual 256-vector audit after that correction and passing tests. The acceptance thresholds, same 64 IDs, four frozen prediction inputs and 600-second limit remain unchanged. The identity case explicitly distinguishes unchanged original q from the feasible represented point used for the numerical gap. No post-outcome tuning or rerun is authorized.

## Post-execution review

The authorized audit completed once. Source/input hashes matched before and after execution, and the result seal was independently rechecked. No code, protocol or thresholds were changed after observing the outcome.

Of 256 inputs, 239 passed the unchanged full-simplex FW-gap criterion. Eleven were coherent identities outside the clipping domain and correctly had no gradient-based gap certificate. Six were actual optimization failures, all with `line_search_failed`, and large gaps from approximately 0.442 to 0.768. These failures are not rounding noise; some failed candidates had substantially worse objective than the existing 64-iteration candidates. The solver therefore remains unsuitable as a universally converged reference.

All 256 candidates passed the numerical coherence check. The all-ten-target score guard passed 235 candidates; 15 gap-passing candidates failed the strict score guard. Consequently 224 passed both tests. Identity outputs or fallbacks preserved the input whenever applicable. The returned forecasts had zero worst-target score deterioration across all four model inputs; 237/256 returned forecasts were coherent, and 19 remained incoherent. These counts must not be conflated with empirical forecasting quality.

The maximum FW gap among passing inputs was 8.63e-12. The maximum identity reconstruction residual was 1.74e-18. The highest positive candidate score difference among gap-passing cases was 1.69e-8, reinforcing the need to check the actual clipped score rather than infer its sign from a small objective gap. This is a numerical diagnostic, not a claim that classical exact Bregman projection theory failed.

The proper stopping action is to preserve all six numerical failures and the 19 incoherent returned cases, close this supplementary branch, and resume the violin future-prediction work. No observed targets were consulted, no empirical model ranking changed, and no further projection audit is justified within this once-only supplement.
