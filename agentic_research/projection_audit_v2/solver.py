"""Classical simplex active-set Newton projection, audited in float64.

No observed targets are consumed. The gap is a numerical convex error bound,
not an exact-arithmetic certificate and not a new projection theorem.
"""
from __future__ import annotations
import time
import numpy as np
from scipy.special import rel_entr
from agentic_research.projection_audit_v1.solver import patterns, validate, all_losses

CLIP = 1e-6
GAP_TOL = 1e-8
INTERNAL_GAP_TOL = 1e-12
MAX_ITERATIONS = 1000


def derivatives(w, q, bank):
    p = w @ bank
    if not np.isfinite(p).all() or np.any((p <= 0) | (p >= 1)):
        raise ArithmeticError('Actual marginals must be interior')
    qc = np.clip(q, CLIP, 1-CLIP)
    value = float(np.mean(rel_entr(p,qc)+rel_entr(1-p,1-qc)))
    residual = (np.log(p)-np.log1p(-p)-np.log(qc)+np.log1p(-qc))/len(q)
    gradient = bank @ residual
    hessian = (bank/(len(q)*p*(1-p))) @ bank.T
    return value, gradient, hessian


def fw_gap(w, gradient):
    return float(w @ gradient-gradient.min())


def _line_search(w, direction, q, bank, f, g, max_step=1.):
    negative = direction < 0
    boundary = float(np.min(-w[negative]/direction[negative])) if negative.any() else np.inf
    step = min(max_step,boundary)
    descent = float(g @ direction)
    if not np.isfinite(descent) or descent >= 0:
        return None
    for _ in range(70):
        candidate = w+step*direction
        if step == boundary:
            blockers = np.flatnonzero(negative)[np.argmin(-w[negative]/direction[negative])]
            candidate[blockers] = 0.
        # Only arithmetic-sized negatives are canonicalized; recompute all tests.
        if candidate.min() < -1e-14:
            step *= .5
            continue
        candidate = np.maximum(candidate,0.)
        candidate /= candidate.sum()
        if np.array_equal(candidate,w):
            return None
        try:
            new_f,new_g,_ = derivatives(candidate,q,bank)
        except ArithmeticError:
            step *= .5
            continue
        # Objective loses tiny differences near optimum. Accept only the
        # Armijo inequality or numerical equality with a smaller full gap.
        old_gap = fw_gap(w,g)
        if new_f <= f+1e-4*step*descent or (
            abs(new_f-f) <= 8*np.finfo(float).eps*max(1.,abs(f))
            and fw_gap(candidate,new_g) < old_gap):
            return candidate
        step *= .5
    return None


def _optimize(q, bank):
    w = np.full(len(bank),1/len(bank),dtype=float)
    stop = 'iteration_limit'
    counts = {'newton_steps':0,'vertex_steps':0,'line_search_failures':0}
    for iteration in range(MAX_ITERATIONS):
        f,g,h = derivatives(w,q,bank)
        gap = fw_gap(w,g)
        if gap <= INTERNAL_GAP_TOL:
            stop='internal_gap';break
        support = np.flatnonzero(w > 0)
        # Eliminate the affine constraint using the largest active weight.
        anchor = int(support[np.argmax(w[support])])
        free = support[support != anchor]
        candidate = None
        if len(free) and np.max(np.abs(g[free]-g[anchor])) > max(1e-14,gap*1e-4):
            basis = np.eye(len(w))[:,free]-np.eye(len(w))[:,[anchor]]
            reduced_g = basis.T@g
            reduced_h = basis.T@h@basis
            try:
                direction = basis @ np.linalg.solve(reduced_h,-reduced_g)
            except np.linalg.LinAlgError:
                direction = basis @ np.linalg.lstsq(reduced_h,-reduced_g,rcond=1e-14)[0]
            candidate = _line_search(w,direction,q,bank,f,g)
        if candidate is not None:
            w=candidate;counts['newton_steps']+=1
            continue
        # A vertex step admits an inactive vertex and is a global descent
        # direction whenever the full-simplex gap is positive.
        direction=-w.copy();direction[int(np.argmin(g))]+=1.
        candidate = _line_search(w,direction,q,bank,f,g)
        if candidate is None:
            counts['line_search_failures']+=1;stop='line_search_failed';break
        w=candidate;counts['vertex_steps']+=1
    f,g,_ = derivatives(w,q,bank)
    return w,{'iterations':iteration+1,'stop':stop,'objective':f,
              'fw_gap':fw_gap(w,g),**counts}


def _coherent_weights(q, bank):
    affine=np.vstack((np.ones(len(bank)),bank.T))
    matching=np.flatnonzero(np.all(bank==q,axis=1))
    if len(matching):
        weights=np.zeros(len(bank));weights[matching[0]]=1.
    elif affine.shape[0]==affine.shape[1]:
        weights=np.linalg.solve(affine,np.r_[1.,q])
    else:
        weights=np.linalg.lstsq(affine,np.r_[1.,q],rcond=None)[0]
    residual=float(np.max(np.abs(affine@weights-np.r_[1.,q])))
    # No negative-mass tolerance for identity: numerical doubt takes the solver.
    if residual <= 1e-12 and weights.min() >= 0:
        return weights,residual
    return None,residual


def reference_project(q, bank=None):
    started=time.perf_counter()
    bank=patterns() if bank is None else np.asarray(bank,dtype=float)
    q=validate(q,bank)
    weights,residual=_coherent_weights(q,bank)
    if weights is not None:
        # Returning an already coherent forecast unchanged is exact identity.
        # An endpoint/outside-clip identity does not receive a fake FW bound.
        inside=bool(np.all((q >= CLIP)&(q <= 1-CLIP)))
        weights=weights/weights.sum()
        represented=weights@bank
        residual=float(max(np.max(np.abs(represented-q)),abs(weights.sum()-1)))
        objective=float(np.mean(rel_entr(represented,np.clip(q,CLIP,1-CLIP))+
                                rel_entr(1-represented,1-np.clip(q,CLIP,1-CLIP))))
        actual_gap=None
        if inside:
            objective,gradient,_=derivatives(weights,q,bank)
            actual_gap=fw_gap(weights,gradient)
        gap_pass=bool(actual_gap is not None and 0 <= actual_gap <= GAP_TOL)
        return {'probabilities':q.copy(),'returned_probabilities':q.copy(),
                'weights':weights,'objective':objective,'fw_gap':actual_gap,
                'gap_pass':gap_pass,'feasibility_error':residual,'identity':True,
                'identity_reconstruction_residual':residual,
                'gap_basis':'represented_feasible_weights; original q returned unchanged',
                'identity_outside_clip':not inside,'accepted':gap_pass and residual<=1e-10,
                'worst_feasible_target_delta':0.,'iterations':0,
                'stop':'coherent_identity','newton_steps':0,'vertex_steps':0,
                'line_search_failures':0,'seconds':time.perf_counter()-started}
    w,record=_optimize(q,bank)
    p=w@bank
    worst=float((all_losses(p,bank)-all_losses(q,bank)).max())
    feasibility=max(float(max(0.,-w.min())),abs(float(w.sum()-1)),
                    float(max(0.,-p.min(),p.max()-1)))
    gap_pass=bool(np.isfinite(record['fw_gap']) and 0 <= record['fw_gap'] <= GAP_TOL)
    accepted=gap_pass and feasibility <= 1e-10 and worst <= 0
    return {'probabilities':p,'returned_probabilities':p if accepted else q.copy(),
            'weights':w,**record,'gap_pass':gap_pass,'identity':False,
            'identity_reconstruction_residual':None,'gap_basis':'candidate_weights',
            'identity_outside_clip':False,'feasibility_error':feasibility,
            'worst_feasible_target_delta':worst,'accepted':bool(accepted),
            'seconds':time.perf_counter()-started}
