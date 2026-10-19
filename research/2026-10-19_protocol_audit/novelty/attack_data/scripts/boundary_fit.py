"""Two-boundary likelihood model for Guettler diagrams.
P(success | F, a) = pmax * sigmoid((lnF - L(a))/sL) * sigmoid((U(a) - lnF)/sU)
L(a) = l0 + l1*x + l2*x^2,  x = ln(a / 1 m s^-2);  U(a) = u0 + u1*x
Lower-boundary log-log slope s(a) = dL/dlna = l1 + 2*l2*x.
"""
import numpy as np
from scipy.optimize import minimize

def _unpack(p, quad):
    if quad:
        l0, l1, l2, u0, u1, lsL, lsU, lpm = p
    else:
        l0, l1, u0, u1, lsL, lsU, lpm = p; l2 = 0.0
    return l0, l1, l2, u0, u1, np.exp(lsL), np.exp(lsU), 1 / (1 + np.exp(-lpm))

def nll(p, lnF, x, y, quad):
    l0, l1, l2, u0, u1, sL, sU, pm = _unpack(p, quad)
    L = l0 + l1 * x + l2 * x * x
    U = u0 + u1 * x
    pr = pm / (1 + np.exp(-(lnF - L) / sL)) / (1 + np.exp(-(U - lnF) / sU))
    pr = np.clip(pr, 1e-9, 1 - 1e-9)
    return -np.sum(y * np.log(pr) + (1 - y) * np.log(1 - pr))

def fit(F, a, y, quad=False, start=None):
    lnF, x, y = np.log(F), np.log(a), np.asarray(y, float)
    if start is None:
        start = [np.log(1.4), 0.4, -0.2, np.log(1.0), 0.5, 1.0, np.log(0.08), np.log(0.1), 2.0] if quad else \
                [np.log(1.4), 0.4, np.log(1.0), 0.5, 1.0, np.log(0.08), np.log(0.1), 2.0]
        if quad:
            start = [np.log(1.4), 0.4, 0.0, np.log(10.0), 0.9, np.log(0.08), np.log(0.1), 2.0]
        else:
            start = [np.log(1.4), 0.4, np.log(10.0), 0.9, np.log(0.08), np.log(0.1), 2.0]
    best = None
    # multi-start to avoid local minima (upper boundary sometimes outside data)
    for du in (0.0, -1.0, 1.0):
        for dl in (0.0, -0.3, 0.3):
            s = list(start)
            s[0] += dl
            s[3 if quad else 2] += du
            r = minimize(nll, s, args=(lnF, x, y, quad), method='Nelder-Mead',
                         options=dict(maxiter=20000, maxfev=20000, xatol=1e-6, fatol=1e-8))
            r = minimize(nll, r.x, args=(lnF, x, y, quad), method='BFGS')
            if best is None or r.fun < best.fun:
                best = r
    return best

def describe(r, quad):
    l0, l1, l2, u0, u1, sL, sU, pm = _unpack(r.x, quad)
    return dict(l0=l0, l1=l1, l2=l2, u0=u0, u1=u1, sL=sL, sU=sU, pmax=pm, nll=r.fun)

def lower_force(r, quad, a):
    l0, l1, l2, *_ = _unpack(r.x, quad)
    x = np.log(a)
    return np.exp(l0 + l1 * x + l2 * x * x)

def lower_slope(r, quad, a):
    l0, l1, l2, *_ = _unpack(r.x, quad)
    return l1 + 2 * l2 * np.log(a)
