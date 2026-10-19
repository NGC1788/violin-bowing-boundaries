"""Simulate the IWK collection protocol and ask whether the measured A1/A2 difference in the matched-time
speed exponent (+0.6 vs +1.0) can come from the different force-decrease-with-travel of the two sessions.
(Claude, 2026-10-18)

Protocol per stroke (as measured on A1/A2, see handoff §10.11-10.13):
  force applied before the bow moves (onset force F0), bow accelerates with
  a = max(v^2 / (2 * 0.048 m), exp(c_v) * F0^1.05 * beta^q_v)  (A2 fit: c 6.18/5.52/5.01, q 2.64/2.33/2.04),
  then holds v for PLATEAU s; the force follows the measured travel curve F(x) = F0 - D(x) (>= 2 mN):
    D_A2(x) from A2 ranks 1-20: 0, 0.044, 0.088, 0.125, 0.153, 0.174 N at x = 0, 2, 5, 8, 11, 14 cm
    D_A1(x) from A1 ranks 1-20: 0, 0.020, 0.045, 0.050, 0.050, 0.050 N
    D_0(x) = 0 (ideal constant force).
Labels: exact contact states, 0.2 s windows starting 0.0/0.1/0.2/0.3 s after the plateau starts (same
contact_label as sim_contact2); boundary per (v, beta) = first K=2 consecutive Helmholtz in *current window
force* order, interval-censored normal fit log F* = c + a log v + b log beta (as in Astra's audit).
Usage: python sim_iwk_protocol.py [friction: mores|matusiak]
"""
import math
import sys
import time
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm

REPO = Path.home() / "violin-bowing-boundaries"
sys.path.insert(0, str(REPO / "scripts"))
from bowed_string import FrictionParams, StringParams  # noqa: E402
from bowed_string_cpu import simulate_cpu  # noqa: E402

RATE, PR = 100_000, 1000.0
F0_HZ = StringParams().f0
SPEEDS = [0.05, 0.10, 0.20]
BETAS = [0.04, 0.06, 0.08, 0.10, 0.13, 0.17]
FORCES = np.geomspace(0.05, 4.0, 40)
ACC = {0.05: (6.18, 2.64), 0.10: (5.52, 2.33), 0.20: (5.01, 2.04)}
PLATEAU = 0.6
XS = np.array([0, 0.02, 0.05, 0.08, 0.11, 0.14, 0.30])
DECAY = {"A2-like": np.array([0, 0.044, 0.088, 0.125, 0.153, 0.174, 0.25]),
         "A1-like": np.array([0, 0.020, 0.045, 0.050, 0.050, 0.050, 0.05]),
         "constant": np.zeros(7)}
FRICTION = {"mores": FrictionParams(), "matusiak": FrictionParams(mu_s=1.0207, mu_d=0.5071, v0=0.19)}
TIMES = (0.0, 0.1, 0.2, 0.3)


def accel(v, beta, f0):
    c, q = ACC[v]
    return max(v * v / (2 * 0.048), math.exp(c) * f0 ** 1.05 * beta ** q)


def profiles(v, beta, decay):
    vel, frc, starts = [], [], []
    ramps = [v / accel(v, beta, f) for f in FORCES]
    total = max(ramps) + PLATEAU + 0.01
    t = (np.arange(int(total * PR)) + 0.5) / PR
    for f, ramp in zip(FORCES, ramps):
        a = v / ramp
        vv = np.minimum(a * t, v)
        x = np.cumsum(vv) / PR
        ff = np.maximum(f - np.interp(x, XS, decay), 0.002)
        vel.append(vv); frc.append(ff); starts.append(ramp)
    return np.stack(vel, 1), np.stack(frc, 1), np.array(starts), total


def contact_label(slip, beta):
    s = (np.asarray(slip) > 0.5).astype(np.int8)
    starts = int(np.sum((s[1:] == 1) & (s[:-1] == 0)))
    per = starts / (len(s) / RATE * F0_HZ)
    frac = float(s.mean())
    if starts == 0:
        return False
    return 0.9 <= per <= 1.1 and 0.3 * beta <= frac <= 3 * beta


def run(decay, friction):
    rows = {t: [] for t in TIMES}  # t -> list of (log v, log beta, lo, hi)
    for v in SPEEDS:
        for beta in BETAS:
            pv, pf, starts, total = profiles(v, beta, decay)
            _, diag = simulate_cpu(StringParams(), friction, [beta] * len(FORCES), pv, pf, PR, RATE, 0.0, int(total * RATE),
                                   output_every=1, record_from=0, threads=2, return_diagnostics=True)
            slip = diag["slip_fraction_by_output"]
            for t in TIMES:
                cells = []
                for k in range(len(FORCES)):
                    a = int((starts[k] + t) * RATE)
                    b = a + int(0.2 * RATE)
                    fw = float(np.mean(pf[int(a / RATE * PR):int(b / RATE * PR), k]))
                    cells.append((fw, contact_label(slip[a:b, k], beta)))
                cells.sort()
                h = [c[1] for c in cells]
                lf = np.log([c[0] for c in cells])
                s = next((i for i in range(len(h) - 1) if h[i] and h[i + 1]), None)
                lo, hi = (lf[-1], np.inf) if s is None else ((-np.inf, lf[0]) if s == 0 else (lf[s - 1], lf[s]))
                rows[t].append((math.log(v), math.log(beta), lo, hi))
    return {t: np.array(r) for t, r in rows.items()}


def fit(D):
    lv, lb, lo, hi = D.T
    def nll(x):
        mu = x[0] + x[1] * lv + x[2] * lb
        s = np.exp(x[3])
        p = np.where(np.isfinite(hi), norm.cdf((hi - mu) / s), 1.0) - np.where(np.isfinite(lo), norm.cdf((lo - mu) / s), 0.0)
        return -np.sum(np.log(np.clip(p, 1e-300, None)))
    return min((minimize(nll, [0, a, -1, np.log(0.5)], method="Nelder-Mead", options={"maxiter": 20000, "xatol": 1e-8, "fatol": 1e-10})
                for a in (0.0, 1.0, 2.0)), key=lambda r: r.fun).x


def main():
    name = sys.argv[1] if len(sys.argv) > 1 else "mores"
    print(f"마찰 {name}, β {BETAS}, 힘 {len(FORCES)}개 0.05~4 N, 평탄 {PLATEAU} s")
    for label, decay in DECAY.items():
        t0 = time.time()
        res = run(decay, FRICTION[name])
        line = []
        for t, D in res.items():
            x = fit(D)
            cens = f"검열 L{int(np.sum(~np.isfinite(D[:, 2])))}/R{int(np.sum(~np.isfinite(D[:, 3])))}"
            line.append(f"t={t:.1f}: 속도 {x[1]:+.2f} β {x[2]:+.2f} ({cens})")
        print(f"  {label:9s} " + " | ".join(line) + f"  ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
