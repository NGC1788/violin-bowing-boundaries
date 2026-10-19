"""Critical slowing down at the establishment boundary, measured on exact contact states.

For a few (speed, beta), strokes start from rest at a fixed force on a fine force grid around the
establishment boundary and are held for 4 s. Lock-in time = first time from which >= 90 % of the
remaining 10-period windows are Helmholtz (one slip per period, slip fraction ~ beta).
Fit: tau = A (ln F - ln F*)^(-theta) over locked strokes, with F* and theta free; compare
theta = 1/2 (saddle-node), theta = 1, and a non-divergent fit (tau = A exp(-k d)).
"""

import math
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

REPO = Path.home() / "violin-bowing-boundaries"
sys.path.insert(0, str(REPO / "scripts"))
from bowed_string import FrictionParams, StringParams  # noqa: E402
from bowed_string_cpu import simulate_cpu  # noqa: E402

RATE, PROFILE_RATE, F0 = 100_000, 1000.0, StringParams().f0
T_RAMP, T_HOLD = 0.2, 4.0
CASES = [(0.10, 0.08, 0.347), (0.10, 0.12, 0.347), (0.05, 0.12, 0.136), (0.05, 0.15, 0.113)]  # (v, beta, rough F_E)


def windows(slip, beta, n_per=10):
    s = (np.asarray(slip) > 0.5).astype(np.int8)
    period = RATE / F0
    width = int(n_per * period)
    out = []
    for start in range(0, len(s) - width, width):
        w = s[start:start + width]
        starts = int(np.sum((w[1:] == 1) & (w[:-1] == 0)))
        per, frac = starts / n_per, float(w.mean())
        out.append(0.9 <= per <= 1.1 and 0.3 * beta <= frac <= 3 * beta)
    return np.array(out), width / RATE


def lock_time(flags, dt):
    for i in range(len(flags)):
        if flags[i] and flags[i:].mean() >= 0.9:
            return i * dt
    return None


def fit(F, tau, form):
    lf = np.log(F)
    def resid(x):
        lfs = x[0]
        d = np.maximum(lf - lfs, 1e-4)
        if form == "free":
            pred = x[1] - x[2] * np.log(d)
        elif form == "half":
            pred = x[1] - 0.5 * np.log(d)
        elif form == "one":
            pred = x[1] - 1.0 * np.log(d)
        else:
            pred = x[1] - x[2] * d
        return pred - np.log(tau)
    x0 = [lf.min() - 0.1, np.log(np.median(tau)), 0.7]
    if form in ("half", "one"):
        x0 = x0[:2]
    lo = [lf.min() - 3, -20] + ([-5] if form in ("free", "none") else [])
    hi = [lf.min() - 1e-3, 20] + ([10] if form in ("free", "none") else [])
    r = least_squares(resid, x0, bounds=(lo, hi))
    n, k = len(tau), len(r.x)
    rss = float(np.sum(r.fun ** 2))
    return r.x, n * math.log(rss / n) + 2 * k, rss


def main():
    out_lines = []
    for v, beta, fe in CASES:
        forces = np.geomspace(0.6 * fe, 4 * fe, 36)
        total = T_RAMP + T_HOLD
        t = (np.arange(int(total * PROFILE_RATE)) + 0.5) / PROFILE_RATE
        vel = np.clip(t / T_RAMP, 0, 1) * v
        pv = np.repeat(vel[:, None], len(forces), 1)
        pf = np.repeat(forces[None, :], len(t), 0)
        _, diag = simulate_cpu(StringParams(), FrictionParams(), [beta] * len(forces), pv, pf, PROFILE_RATE, RATE, 0.0,
                               int(total * RATE), output_every=1, record_from=int(T_RAMP * RATE), threads=2,
                               return_diagnostics=True)
        slip = diag["slip_fraction_by_output"]
        lock = []
        for k, f in enumerate(forces):
            flags, dt = windows(slip[:, k], beta)
            lock.append(lock_time(flags, dt))
        locked = [(f, tl) for f, tl in zip(forces, lock) if tl is not None and tl > 0]
        never = [f for f, tl in zip(forces, lock) if tl is None]
        line = f"v={v} β={beta}: 잠김 {sum(tl is not None for tl in lock)}/{len(forces)}, 미잠금 힘 최대 {max(never):.3f} N" if never else f"v={v} β={beta}: 모두 잠김"
        print(line); out_lines.append(line)
        if len(locked) >= 6:
            F = np.array([x[0] for x in locked]); tau = np.array([x[1] for x in locked])
            res = {form: fit(F, tau, form) for form in ("free", "half", "one", "none")}
            for form, (x, aic, rss) in res.items():
                extra = f" θ={x[2]:.2f}" if form == "free" else (f" k={x[2]:.2f}" if form == "none" else "")
                s = f"    {form:5} F*={math.exp(x[0]):.3f} N  AIC {aic:7.1f}{extra}"
                print(s); out_lines.append(s)
            show = "  ".join(f"{f:.3f}:{tl:.2f}s" for f, tl in locked[:8])
            print("    가장 낮은 힘들의 잠금 시간:", show); out_lines.append("    " + show)
    Path(__file__).with_name("sim_csd.log").write_text("\n".join(out_lines))


if __name__ == "__main__":
    main()
