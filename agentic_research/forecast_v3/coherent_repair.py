"""Classical Bregman coherentization with a finite-target numerical safeguard.

This is a strong auxiliary BASELINE, not a newly discovered theorem. For exact
forward Bernoulli-KL projection onto a convex coherent set, the Bregman
Pythagorean inequality gives pointwise log-score dominance (Predd et al., 2009).
We use a bounded approximate solver, then separately check all ten possible
observer outputs with the *actual clipped score*. A failed candidate returns
the original prediction, which can remain incoherent. Floating-point checks
are not interval-arithmetic proofs and permit no positive score tolerance.
"""
from __future__ import annotations
import time
import numpy as np
from .models import PATTERNS, validate_probabilities, signed_pattern_masses, simplex_projection

CLIP=1e-6
COHERENCE_TOL=1e-9


def all_pattern_losses(probabilities):
    """Exactly the evaluator's clipped mean-nine binary loss for all10 targets."""
    p=np.clip(validate_probabilities(probabilities),CLIP,1-CLIP)
    # Use the same elementary operations and reduction as experiment.loss_rows.
    return np.column_stack([-(y*np.log(p)+(1-y)*np.log1p(-p)).mean(axis=1)
                            for y in PATTERNS])


def certify_candidate(original,candidate):
    """Label-free numeric check; accept only coherent, finite, non-worsening rows.

    'Non-worsening' refers to computed float64 scores over the finite rule bank.
    No positive acceptance tolerance is used. On uncertainty/failure use q.
    """
    q=validate_probabilities(original);p=np.asarray(candidate,dtype=np.float64)
    if p.shape!=q.shape:raise ValueError('Candidate probabilities do not align')
    finite=np.isfinite(p).all(1)
    in_range=((p>=-1e-12)&(p<=1+1e-12)).all(1)
    safe=np.where(np.isfinite(p),np.clip(p,0,1),.5)
    masses=signed_pattern_masses(safe)
    coherent=(masses>=-COHERENCE_TOL).all(1)
    delta=all_pattern_losses(safe)-all_pattern_losses(q)
    worst=delta.max(1)
    accepted=finite&in_range&coherent&(worst<=0.)
    repaired=np.where(accepted[:,None],safe,q)
    return repaired,accepted,worst


def _objective(weights,log_q,log_not_q):
    p=np.clip(weights@PATTERNS,1e-12,1-1e-12)
    return (p*(np.log(p)-log_q)+(1-p)*(np.log1p(-p)-log_not_q)).mean(1)


def _solve(q,max_iterations):
    clipped=np.clip(q,CLIP,1-CLIP)
    log_q=np.log(clipped);log_not_q=np.log1p(-clipped)
    initial=simplex_projection(signed_pattern_masses(clipped))
    # Tiny interior initialization stabilizes logs; the final guard is independent of it.
    weights=(1-1e-8)*initial+1e-9
    values=_objective(weights,log_q,log_not_q)
    steps=np.ones(len(q))
    iterations=0
    for iteration in range(max_iterations):
        p=np.clip(weights@PATTERNS,1e-12,1-1e-12)
        gradient=(np.log(p)-np.log1p(-p)-log_q+log_not_q)@PATTERNS.T/9
        proposal=weights.copy();next_values=values.copy();accepted=np.zeros(len(q),bool)
        for _ in range(14):
            pending=np.flatnonzero(~accepted)
            if not len(pending):break
            candidate=simplex_projection(weights[pending]-steps[pending,None]*gradient[pending])
            f=_objective(candidate,log_q[pending],log_not_q[pending])
            slope=np.sum(gradient[pending]*(candidate-weights[pending]),axis=1)
            good=f<=values[pending]+.25*slope+1e-14
            idx=pending[good];proposal[idx]=candidate[good];next_values[idx]=f[good];accepted[idx]=True
            steps[pending[~good]]*=.5
        movement=np.max(np.abs(proposal-weights),axis=1)
        weights=proposal;values=next_values;iterations=iteration+1
        steps=np.minimum(steps*1.2,8.)
        if movement.max()<1e-10:break
    return np.clip(weights@PATTERNS,0,1),np.clip(initial@PATTERNS,0,1),iterations


def repair(probabilities, max_iterations=64, chunk_size=2048):
    """Return repaired probabilities and aggregate audit metadata; no labels used.

    Candidate1 is bounded forward-KL projected gradient. If its finite-target
    check fails, candidate2 is the original latent-simplex initializer. If both
    fail, return the original q. This fixed fallback order is not target fitting.
    """
    q=validate_probabilities(probabilities)
    if isinstance(max_iterations,bool) or not isinstance(max_iterations,int) or not 0<=max_iterations<=128:
        raise ValueError('max_iterations must be an integer from0to128')
    if isinstance(chunk_size,bool) or not isinstance(chunk_size,int) or chunk_size<1:
        raise ValueError('chunk_size must be a positive integer')
    start=time.perf_counter();output=np.empty_like(q);accepted_all=[];init_count=0;iterations=[]
    for start_row in range(0,len(q),chunk_size):
        original=q[start_row:start_row+chunk_size]
        candidate,initializer,steps=_solve(original,max_iterations)
        repaired,accepted,_=certify_candidate(original,candidate)
        alternative,alternative_ok,_=certify_candidate(original,initializer)
        use_initializer=(~accepted)&alternative_ok
        repaired[use_initializer]=alternative[use_initializer]
        accepted|=use_initializer;init_count+=int(use_initializer.sum())
        output[start_row:start_row+len(original)]=repaired
        accepted_all.append(accepted);iterations.append(steps)
    accepted=np.concatenate(accepted_all)
    # Independent final check of the returned values, including original fallbacks.
    delta=all_pattern_losses(output)-all_pattern_losses(q)
    if np.any(delta>0.):raise AssertionError('Returned repair increased a feasible-target clipped score')
    masses=signed_pattern_masses(output)
    coherent=(masses>=-COHERENCE_TOL).all(1)
    changed=np.max(np.abs(output-q),axis=1)>1e-12
    meta={'method':'classical forward Bernoulli-KL coherentization with finite-target numerical guard',
          'prior_art':'Predd et al.2009; known coherentization baseline, not a new general theorem',
          'rows':len(q),'accepted_rows':int(accepted.sum()),'changed_rows':int(changed.sum()),
          'fallback_original_rows':int((~accepted).sum()),'initializer_fallback_rows':init_count,
          'coherent_output_rows':int(coherent.sum()),'incoherent_fallback_rows':int((~coherent).sum()),
          'max_certified_delta':float(delta.max()),'minimum_worst_case_gain':float(-delta.max(axis=1).max()),
          'mean_worst_case_gain':float((-delta.max(axis=1)).mean()),
          'score':'mean nine-rule log loss with probability clipping','probability_clip':CLIP,
          'acceptance_score_tolerance':0.,'coherence_tolerance':COHERENCE_TOL,
          'max_iterations':max_iterations,'max_iterations_used':max(iterations),'chunk_size':chunk_size,
          'repair_seconds':time.perf_counter()-start,
          'numerical_scope':'Finite-target float64 check, not rigorous interval arithmetic; no positive score tolerance.',
          'guarantee_scope':'All10 valid observer outputs under the audited shared-guard rules; fallback may be incoherent.'}
    return output,meta
