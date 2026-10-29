"""Numerical reference for a classical convex Bernoulli-KL projection.

Not a new theorem or an exact-arithmetic solver. The Frank-Wolfe gap is a
float64 estimate of a convex suboptimality upper bound, not an interval proof.
No observed target labels are used anywhere in this module.
"""
from __future__ import annotations
import time
import numpy as np
from scipy.optimize import minimize

CLIP = 1e-6
MASS_FLOOR = 1e-12
GAP_TOL = 1e-8


def patterns():
    # Independently enumerate the empty set and every prefix-row/suffix-column
    # rectangle; sorted order matches the frozen observer's ten outcomes.
    bank = {tuple([0.] * 9)}
    for rows in range(1, 4):
        for cols in range(1, 4):
            bank.add(tuple(float(r < rows and c >= 3 - cols)
                           for r in range(3) for c in range(3)))
    return np.array(sorted(bank), dtype=np.float64)


def validate(q, bank):
    q = np.asarray(q, dtype=np.float64)
    if q.ndim != 1 or q.shape != (bank.shape[1],):
        raise ValueError('A single probability vector must match the rule bank')
    if not np.isfinite(q).all() or np.any((q < 0) | (q > 1)):
        raise ValueError('Invalid probabilities')
    return q


def objective_and_gradient(w, q, bank):
    p = w @ bank
    if not np.isfinite(p).all() or np.any((p <= 0) | (p >= 1)):
        raise ArithmeticError("Objective gradient requires actual interior marginals")
    qc = np.clip(q, CLIP, 1 - CLIP)
    value = np.mean(p * (np.log(p) - np.log(qc)) +
                    (1-p) * (np.log1p(-p) - np.log1p(-qc)))
    gp = (np.log(p) - np.log1p(-p) - np.log(qc) + np.log1p(-qc)) / len(q)
    return float(value), bank @ gp


def gap(w, q, bank):
    _, g = objective_and_gradient(w, q, bank)
    # Over the full simplex, including its boundary (not the tiny-floor set).
    return float(w @ g - g.min())


def all_losses(p, bank):
    p = np.clip(np.asarray(p, dtype=np.float64), CLIP, 1-CLIP)
    return -(bank * np.log(p) + (1-bank) * np.log1p(-p)).mean(axis=1)


def simplex(v):
    u = np.sort(v)[::-1]
    s = np.cumsum(u) - 1
    idx = np.arange(1, len(v)+1)
    rho = np.flatnonzero(u - s/idx > 0)[-1]
    return np.maximum(v - s[rho]/(rho+1), 0)


def reference_project(q, bank=None):
    bank = patterns() if bank is None else np.asarray(bank, dtype=np.float64)
    q = validate(q, bank)
    k = len(bank)
    # Two fixed starts; neither depends on labels or any performance score.
    affine = np.vstack((np.ones(k), bank.T))
    raw = np.linalg.lstsq(affine, np.r_[1., np.clip(q, CLIP, 1-CLIP)], rcond=None)[0]
    initial = simplex(raw)
    starts = [np.full(k, 1/k), (1-k*MASS_FLOOR)*initial + MASS_FLOOR]
    records = []
    start_time = time.perf_counter()
    for start in starts:
        try:
            result = minimize(objective_and_gradient, start, args=(q, bank), jac=True,
                              method='SLSQP', bounds=[(MASS_FLOOR, 1.)]*k,
                              constraints={'type':'eq', 'fun':lambda w: w.sum()-1,
                                           'jac':lambda w: np.ones(k)},
                              options={'maxiter':1000, 'ftol':1e-13, 'disp':False})
        except (ArithmeticError, ValueError) as exc:
            records.append({'usable':False, 'exception':type(exc).__name__, 'message':str(exc)})
            continue
        if not np.isfinite(result.x).all():
            records.append({'usable':False, 'status':int(result.status)})
            continue
        w = np.maximum(result.x, MASS_FLOOR)
        w /= w.sum()
        records.append({'usable':True, 'weights':w,
                        'raw_minimum_weight':float(result.x.min()),
                        'raw_simplex_error':abs(float(result.x.sum()-1)),
                        'canonicalization_max_change':float(np.abs(w-result.x).max()),
                        'objective':objective_and_gradient(w,q,bank)[0],
                        'fw_gap':gap(w,q,bank), 'success':bool(result.success),
                        'status':int(result.status), 'iterations':int(result.nit)})
    usable = [r for r in records if r['usable']]
    if not usable:
        raise ArithmeticError('Both reference optimizations were unusable: '+repr(records))
    # Selection is solely convex objective, not observed or feasible-target loss.
    best = min(usable, key=lambda r:r['objective'])
    w = best['weights']; p = w @ bank
    worst = float((all_losses(p,bank) - all_losses(q,bank)).max())
    feasibility = max(float(max(0., -w.min())), abs(float(w.sum()-1)),
                      float(max(0., -p.min(), p.max()-1)))
    accepted = best['fw_gap'] <= GAP_TOL and feasibility <= 1e-10 and worst <= 0.
    return {'probabilities':p, 'weights':w,
            'returned_probabilities':p if accepted else q.copy(),
            'objective':best['objective'], 'fw_gap':best['fw_gap'],
            'feasibility_error':feasibility,
            'worst_feasible_target_delta':worst, 'accepted':bool(accepted),
            'seconds':time.perf_counter()-start_time,
            'starts':[{key:val for key,val in r.items() if key != 'weights'} for r in records]}
