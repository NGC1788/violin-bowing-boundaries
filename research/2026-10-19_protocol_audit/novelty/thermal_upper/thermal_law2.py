"""Post hoc extensions of thermal_law.py (which stays unchanged because FROZEN_v1 hashes it). Claude, 2026-10-19.

General result used here: if the contrast depends on bow force and speed only through the power u = F v
(any function dmu(u), constant mu_s or mu_s(u)), Schelleng's condition F dmu(F v) = 2 Z v / beta^m becomes
    u dmu(u) = 2 Z v^2 / beta^m,
so F_max v is a function of the single 'power similarity variable' Z v^2 / beta^m. With h = d ln u / d ln(Z v^2/beta^m) in (0, 1]:
    e_v = 2h - 1,  e_beta = -m h,  e_Z = h   ->  e_beta = -m (1 + e_v) / 2   (m = 1: Schelleng's 1/beta).
Models added: thermal with a free beta exponent m in the Schelleng skeleton (k and m free, mu_s fixed)."""
import numpy as np

import thermal_law as TL
from thermal_law import MODELS as _M1

MODELS = dict(_M1)
MODELS["T2km_peak"] = (2, [0.0, 1.0], lambda p, lv, lb, lz: TL._th(TL.MU_PEAK, p[0], lv, lb, lz, m=p[1]))
MODELS["T2km_break"] = (2, [0.0, 1.0], lambda p, lv, lb, lz: TL._th(TL.MU_BREAK, p[0], lv, lb, lz, m=p[1]))
# plain Schelleng skeleton with a free beta exponent (constant contrast), 2 parameters
MODELS["S2m"] = (2, [1.0, 1.0], lambda p, lv, lb, lz: p[0] + lz + lv - p[1] * lb)


def mu_s_of(name, theta):
    if name == "T2km_peak":
        return TL.MU_PEAK
    if name == "T2km_break":
        return TL.MU_BREAK
    return TL.mu_s_of(name, theta)


def predict(name, theta, lv, lb, lz):
    return MODELS[name][2](np.asarray(theta, float), np.asarray(lv, float), np.asarray(lb, float), np.asarray(lz, float))


def design_exponents(name, theta, lv, lb, lz):
    y = predict(name, theta, lv, lb, lz)
    X = np.column_stack([np.ones_like(lv), lv, lb])
    c = np.linalg.lstsq(X, y, rcond=None)[0]
    return dict(c=float(c[0]), e_v=float(c[1]), e_beta=float(c[2]))
