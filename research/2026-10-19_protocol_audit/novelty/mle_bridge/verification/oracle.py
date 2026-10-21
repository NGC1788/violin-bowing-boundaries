"""Independent Gaussian interval-likelihood oracle; no project fitter imports.

The empirical target is a full-rank, all-speed design. A selected two-column
geometric obstruction by itself is not a valid regression significance test.
"""
from __future__ import annotations

import numpy as np
from scipy.special import log_ndtr, ndtr

LOG2PI = np.log(2.0 * np.pi)


def log_mass(lower, upper):
    """log P(lower < Z <= upper), with strict invalid-input rejection.

Negative and positive intervals use different tails; crossing-zero intervals
use subtraction of the two excluded tails. Infinite endpoints are supported.
No clipping of an impossible/underflowed interval to a positive mass occurs.
"""
    lower, upper = np.broadcast_arrays(np.asarray(lower, float), np.asarray(upper, float))
    if np.any(np.isnan(lower) | np.isnan(upper) | (lower >= upper)):
        raise ValueError('interval endpoints must satisfy lower < upper')
    ans = np.empty(lower.shape)
    negative = upper <= 0
    positive = lower >= 0
    cross = ~(negative | positive)
    for mask, outer, inner in ((negative, upper, lower), (positive, -lower, -upper)):
        outer_log = log_ndtr(outer[mask])
        inner_log = log_ndtr(inner[mask])
        ans[mask] = outer_log + np.log(-np.expm1(inner_log - outer_log))
    # For very narrow intervals crossing zero the CDF difference is more
    # accurate than 1 - two tails, while both branches are far from tail loss.
    width = upper[cross] - lower[cross]
    vals = np.empty_like(width)
    narrow = width < .1
    if np.any(narrow):
        from scipy.special import erf
        vals[narrow] = np.log(.5 * (erf(upper[cross][narrow] / np.sqrt(2)) -
                                   erf(lower[cross][narrow] / np.sqrt(2))))
    vals[~narrow] = np.log1p(-ndtr(lower[cross][~narrow]) - ndtr(-upper[cross][~narrow]))
    ans[cross] = vals
    return ans


def validate_design(X, lower, upper, scale=None, scale_bounds=(-9., 3.5)):
    X, lower, upper = np.asarray(X, float), np.asarray(lower, float), np.asarray(upper, float)
    issues = []
    if X.ndim != 2 or lower.shape != (len(X),) or upper.shape != (len(X),):
        return {'valid': False, 'issues': ['shape'], 'rank': None}
    if not np.isfinite(X).all(): issues.append('nonfinite_design')
    if np.isnan(lower).any() or np.isnan(upper).any(): issues.append('nan_interval')
    if (lower == upper).any(): issues.append('zero_width_interval')
    if (lower > upper).any(): issues.append('reversed_interval')
    rank = int(np.linalg.matrix_rank(X)) if np.isfinite(X).all() else None
    if rank is not None and rank < X.shape[1]: issues.append('rank_deficient')
    if len(X) <= X.shape[1]: issues.append('no_residual_degrees_of_freedom')
    informative = ~(np.isneginf(lower) & np.isposinf(upper))
    if not informative.any(): issues.append('no_interval_information')
    if informative.any() and np.all(np.isneginf(lower[informative])):
        issues.append('all_informative_intervals_left_censored')
    if informative.any() and np.all(np.isposinf(upper[informative])):
        issues.append('all_informative_intervals_right_censored')
    if scale is not None:
        if not np.isfinite(scale): issues.append('nonfinite_log_scale')
        elif scale <= scale_bounds[0] + 1e-6 or scale >= scale_bounds[1] - 1e-6:
            issues.append('log_scale_at_or_outside_bound')
    return {'valid': not issues, 'issues': issues, 'rank': rank,
            'n': len(X), 'p': X.shape[1]}


def negative_log_likelihood(params, X, lower, upper):
    params, X = np.asarray(params, float), np.asarray(X, float)
    mean = np.dot(X, params[:-1]); sd = np.exp(params[-1])
    return float(-np.sum(log_mass((np.asarray(lower) - mean) / sd,
                                 (np.asarray(upper) - mean) / sd)))


def gradient(params, X, lower, upper):
    """Derivative from conditional endpoint density (checked numerically)."""
    params, X = np.asarray(params, float), np.asarray(X, float)
    mean = np.dot(X, params[:-1]); sd = np.exp(params[-1])
    a, b = (np.asarray(lower) - mean) / sd, (np.asarray(upper) - mean) / sd
    mass = log_mass(a, b)
    da, db = np.exp(-.5 * a * a - .5 * LOG2PI - mass), np.exp(-.5 * b * b - .5 * LOG2PI - mass)
    scaled_a, scaled_b = np.zeros(len(X)), np.zeros(len(X))
    np.multiply(a, da, out=scaled_a, where=np.isfinite(a))
    np.multiply(b, db, out=scaled_b, where=np.isfinite(b))
    return np.r_[X.T @ ((db - da) / sd), np.sum(scaled_b - scaled_a)]


def numerical_gradient(params, X, lower, upper, step=1e-5):
    """Five-point stencil, independent of the analytic score formula."""
    out = []; p = np.asarray(params, float)
    for i in range(len(p)):
        h = step * max(1., abs(p[i])); e = np.eye(len(p))[i] * h
        fs = [negative_log_likelihood(p + j * e, X, lower, upper) for j in (-2, -1, 1, 2)]
        out.append((fs[0] - 8 * fs[1] + 8 * fs[2] - fs[3]) / (12 * h))
    return np.asarray(out)


def high_precision_log_mass(a, b):
    """100-decimal independent erfc oracle used only by tests."""
    import mpmath as mp
    with mp.workdps(100):
        a, b = mp.mpf(float(a)), mp.mpf(float(b)); root2 = mp.sqrt(2)
        if b <= 0:
            mass = (mp.erfc(-b / root2) - mp.erfc(-a / root2)) / 2
        elif a >= 0:
            mass = (mp.erfc(a / root2) - mp.erfc(b / root2)) / 2
        else:
            mass = (mp.erf(b / root2) - mp.erf(a / root2)) / 2
        return float(mp.log(mass))


def separated_pair_nll_bound(gap, sigma):
    """Sharp half-line lower bound for a separated pair at fixed sigma.

After the fixed covariate offset is subtracted, two intervals separated by
gap g>0 share a scalar intercept. Their joint NLL is at least
-2 log Phi(-g/(2 sigma)). Enlarging the intervals to opposing half-lines
gives this bound, sharp when the intervals already are those half-lines.
This deterministic bound is not a likelihood-ratio significance calibration.
"""
    gap, sigma = np.asarray(gap, float), np.asarray(sigma, float)
    if np.any(~np.isfinite(gap) | (gap <= 0)) or np.any(~np.isfinite(sigma) | (sigma <= 0)):
        raise ValueError('finite positive gap and sigma required')
    return -2 * log_ndtr(-gap / (2 * sigma))


def independent_fit(X, lower, upper, fixed_beta=None):
    """Independent bounded canonical fit, numerical derivatives only.

Input design must be [1, log speed, log beta]. Coefficients are standardized
for numerical optimization; fixed-beta fits subtract the known beta offset.
Optimize eta=theta/sigma and tau=1/sigma. Interval bounds are affine
functions in these variables. No production result is used as a start.
Both starts derive from the intervals.
"""
    from scipy.optimize import minimize
    X, lo, hi = np.asarray(X,float), np.asarray(lower,float), np.asarray(upper,float)
    check = validate_design(X,lo,hi)
    if not check['valid']: return {'success':False,'reason':'design','diagnostics':check}
    if fixed_beta is None:
        xx = X.copy(); ll,hh = lo.copy(),hi.copy()
    else:
        xx = X[:,:2].copy(); ll,hh = lo-fixed_beta*X[:,2],hi-fixed_beta*X[:,2]
    center = xx[:,1:].mean(axis=0); scale = xx[:,1:].std(axis=0)
    zz = np.c_[np.ones(len(xx)),(xx[:,1:]-center)/scale]
    finite = np.isfinite(ll)&np.isfinite(hh)
    yy = np.where(finite,(np.where(np.isfinite(ll),ll,0)+np.where(np.isfinite(hh),hh,0))/2,
                  np.where(np.isfinite(hh),hh-.2,ll+.2))
    initial = np.linalg.lstsq(zz,yy,rcond=None)[0]
    spread = max(.05,float(np.std(yy-zz@initial)))
    def canonical_nll(eta_tau):
        eta,tau=eta_tau[:-1],eta_tau[-1]
        if tau<=0: return np.inf
        return float(-np.sum(log_mass(tau*ll-zz@eta,tau*hh-zz@eta)))
    def canonical_numgrad(at):
        g=[]
        for i in range(len(at)):
            h=1e-5*max(1.,abs(at[i]));e=np.eye(len(at))[i]*h
            fs=[canonical_nll(at+j*e) for j in (-2,-1,1,2)]
            g.append((fs[0]-8*fs[1]+8*fs[2]-fs[3])/(12*h))
        return np.array(g)
    bounds=[(None,None)]*zz.shape[1]+[(np.exp(-3.5),np.exp(9.))]
    candidates=[]
    for ss in (spread,max(.3,spread*2)):
        start=np.r_[initial/ss,1/ss]
        opt=minimize(canonical_nll,start,jac='3-point',
                     method='L-BFGS-B',bounds=bounds,
                     options={'maxiter':1200,'ftol':1e-14,'gtol':1e-8,'maxls':40})
        candidates.append(opt)
    best=min(candidates,key=lambda o:o.fun)
    canonical_grad=float(max(abs(canonical_numgrad(best.x))))
    if not best.success or canonical_grad>5e-4:
        # A derivative-free independent refinement; scale bounds remain active.
        alt=minimize(canonical_nll,best.x,method='Nelder-Mead',
                     bounds=bounds,options={'maxiter':4000,'xatol':1e-10,'fatol':1e-11})
        if alt.fun<=best.fun+1e-9: best=alt
        canonical_grad=float(max(abs(canonical_numgrad(best.x))))
    q=np.r_[best.x[:-1]/best.x[-1],-np.log(best.x[-1])]
    grad=float(max(abs(numerical_gradient(q,zz,ll,hh))))
    coeff=np.r_[q[0]-center@(q[1:-1]/scale),q[1:-1]/scale]
    full=np.r_[coeff,q[-1]] if fixed_beta is None else np.r_[coeff,fixed_beta,q[-1]]
    edge=validate_design(X,lo,hi,float(q[-1]))
    return {'success':bool(best.success and grad<2e-3 and canonical_grad<2e-3 and edge['valid']),
            'params':full.tolist(),'nll':float(best.fun),'grad_max':grad,
            'diagnostics':edge,'optimizer':str(best.message),'fixed_beta':fixed_beta,
            'starts':len(candidates),'canonical_grad_max':canonical_grad,
            'parameterization':'eta=theta/sigma,tau=1/sigma'}


def original_rows_from_columns(columns):
    """Literal first two-true detection, including all censoring cases."""
    rows=[]
    for c in columns:
        forces=np.maximum(np.asarray(c['force'],float),.001)
        if np.any(np.diff(forces)<0): raise ValueError('unsorted forces')
        h=np.asarray(c['baseline'],bool); start=None
        for i in range(len(h)-1):
            if bool(h[i]) and bool(h[i+1]): start=i;break
        if start is None: lo,hi=float(np.log(forces[-1])),float('inf')
        elif start==0: lo,hi=float('-inf'),float(np.log(forces[0]))
        else: lo,hi=float(np.log(forces[start-1])),float(np.log(forces[start]))
        rows.append({'id':c['id'],'session':c['session'],'group':c['group'],
                     'column':c['column'],'eligible':c['eligible'],
                     'x':[1.,float(np.log(c['speed'])),float(np.log(c['beta']))],
                     'lower':lo,'upper':hi,'start':start})
    return rows
