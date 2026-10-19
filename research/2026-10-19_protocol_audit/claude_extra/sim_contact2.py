"""Establishment vs maintenance boundary, second version: reliable establishment, denser grid.

Changes from sim_contact.py: the establishment phase lasts 1.5 s (establishment from rest itself
needs > 1 s near the boundary), five speeds and six beta values where Helmholtz motion can be
started from rest, forces down to 0.002 N, one simulator batch per speed (memory), and the safe
force for M is the E force with the earliest stable Helmholtz motion within that (v, beta) band.
Usage: python sim_contact2.py [rate_hz]
"""

import math
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path.home() / "violin-bowing-boundaries"
sys.path.insert(0, str(REPO / "scripts"))
from bowed_string import FrictionParams, StringParams  # noqa: E402
import os
if os.environ.get('FRICTION_CURVE') == 'gauss':
    sys.path.insert(0, str(Path(__file__).parent))
    from bowed_string_gauss import simulate_cpu  # noqa: E402
elif os.environ.get('FRICTION_CURVE') == 'ep':  # elasto-plastic bristle model, argv: rate vS muS muC
    sys.path.insert(0, str(Path(__file__).parent))
    import functools
    import elastoplastic  # noqa: E402
    simulate_cpu = functools.partial(elastoplastic.simulate_cpu,
                                     slip_threshold=float(os.environ.get('SLIP_THR', elastoplastic.SLIP_THRESHOLD)))
elif os.environ.get('FRICTION_CURVE') == 'thermal':  # van Walstijn 2026 thermal elasto-plastic, Table 1 values
    sys.path.insert(0, str(Path(__file__).parent))
    import functools
    import thermal_ep  # noqa: E402
    simulate_cpu = functools.partial(thermal_ep.simulate_cpu,
                                     slip_threshold=float(os.environ.get('SLIP_THR', thermal_ep.SLIP_THRESHOLD)))
else:
    from bowed_string_cpu import simulate_cpu  # noqa: E402

RATE = int(sys.argv[1]) if len(sys.argv) > 1 else 100_000
FRIC = dict(zip(("v0", "mu_s", "mu_d"), map(float, sys.argv[2:5]))) if len(sys.argv) > 4 else {}
PROFILE_RATE = 1000.0
BETAS = [float(b) for b in os.environ['BETAS'].split(',')] if os.environ.get('BETAS') else [0.06, 0.08, 0.10, 0.12, 0.15, 0.18]
SPEEDS = [0.05, 0.075, 0.10, 0.15, 0.20]
FORCES = np.geomspace(0.002, 3.0, 40)
T_RAMP, T_EST, T_DOWN, T_HOLD = 0.2, 1.5, 0.3, 1.0
RAMP_ACC = float(os.environ.get('RAMP_ACC', 0))  # > 0: constant bow acceleration (m/s^2) as in the IWK rig (1.2 m/s^2 measured)
RAMP_DIST = float(os.environ.get('RAMP_DIST', 0))  # > 0: fixed ramp distance d (m), a = v^2 / (2 d); IWK low-force strokes: d = 0.048
# STRING=violinD: violin D4 on a monochord, as in Schoonderwaldt et al. 2008 (Z0 ~ 0.25 kg/s from their E-string ratio;
# corner_hz scaled with f0 to keep the cello corner/f0 ratio ~61, q1 kept at 2000)
STRING = StringParams(f0=293.66, impedance=0.25, length=0.325, corner_hz=18000.0) if os.environ.get('STRING') == 'violinD' else StringParams()
F0 = STRING.f0


def profile(protocol, v, f_target, f_safe):
    total = T_RAMP + T_EST + T_DOWN + T_HOLD
    t = (np.arange(int(total * PROFILE_RATE)) + 0.5) / PROFILE_RATE
    acc = RAMP_ACC if RAMP_ACC > 0 else (v * v / (2 * RAMP_DIST) if RAMP_DIST > 0 else 0)
    vel = np.minimum(acc * t, v) if acc > 0 else np.clip(t / T_RAMP, 0, 1) * v
    if protocol == "E":
        force = np.full_like(t, f_target)
    else:
        frac = np.clip((t - (T_RAMP + T_EST)) / T_DOWN, 0, 1)
        force = np.exp(np.log(f_safe) + frac * (np.log(f_target) - np.log(f_safe)))
    return vel, force, total


def contact_label(slip, beta):
    s = (np.asarray(slip) > 0.5).astype(np.int8)
    starts = int(np.sum((s[1:] == 1) & (s[:-1] == 0)))
    per = starts / (len(s) / RATE * F0)
    frac = float(s.mean())
    if starts == 0:
        return "stuck" if frac < 0.5 else "sliding"
    if 0.9 <= per <= 1.1 and 0.3 * beta <= frac <= 3 * beta:
        return "helmholtz"
    return "multiple" if per > 1.3 else "other"


def simulate(protocol, v, forces_by_beta, safe):
    cases = [(b, f) for b in BETAS for f in forces_by_beta.get(b, [])]
    if not cases:
        return {}, {}
    vel, frc, total = zip(*[profile(protocol, v, f, safe.get(b, 1.0)) for b, f in cases])
    total = total[0]
    pre = int((T_RAMP + T_EST - 0.25) * RATE)
    _, diag = simulate_cpu(STRING, FrictionParams(**FRIC), [b for b, _ in cases], np.stack(vel, 1), np.stack(frc, 1),
                           PROFILE_RATE, RATE, 0.0, int(total * RATE), output_every=1, record_from=pre, threads=2,
                           return_diagnostics=True)
    slip = diag["slip_fraction_by_output"]
    n_last, n_pre = int(0.4 * RATE), int(0.2 * RATE)
    final = {(v, b, f): contact_label(slip[-n_last:, k], b) for k, (b, f) in enumerate(cases)}
    before = {(v, b, f): contact_label(slip[:n_pre, k], b) for k, (b, f) in enumerate(cases)}
    return final, before


def boundary(labels, v, b):
    fs = sorted(f for (vv, bb, f) in labels if vv == v and bb == b)
    if len(fs) < 4:
        return None
    h = [labels[(v, b, f)] == "helmholtz" for f in fs]
    s = next((i for i in range(len(h) - 1) if h[i] and h[i + 1]), None)
    if s is None:
        return "none"
    return ("<=", fs[0]) if s == 0 else ("=", math.sqrt(fs[s - 1] * fs[s]))


def fit(labels):
    pts = [(math.log(v), math.log(b), math.log(r[1])) for v in SPEEDS for b in BETAS
           if isinstance(r := boundary(labels, v, b), tuple) and r[0] == "="]
    if len(pts) < 5:
        return None, pts
    X = np.column_stack([np.ones(len(pts)), [p[0] for p in pts], [p[1] for p in pts]])
    y = np.array([p[2] for p in pts])
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    r = y - X @ coef
    se = np.sqrt(np.diag(np.linalg.inv(X.T @ X) * (r @ r) / max(len(pts) - 3, 1)))
    return (coef, se), pts


def table(name, labels):
    res, pts = fit(labels)
    line = f"[{name}] 경계 점 {len(pts)}/{len(SPEEDS) * len(BETAS)}"
    if res:
        line += f"   F_min ∝ v^{res[0][1]:+.2f}±{res[1][1]:.2f} · β^{res[0][2]:+.2f}±{res[1][2]:.2f}"
        rng = np.random.default_rng(0)
        P = np.array(pts); draws = []
        for _ in range(2000):
            sample = P[rng.integers(0, len(P), len(P))]
            if len(set(sample[:, 0])) < 2 or len(set(sample[:, 1])) < 2:
                continue
            X = np.column_stack([np.ones(len(sample)), sample[:, 0], sample[:, 1]])
            draws.append(np.linalg.lstsq(X, sample[:, 2], rcond=None)[0][1])
        lo, hi = np.percentile(draws, [2.5, 97.5])
        line += f"   (속도지수 부트스트랩 95% [{lo:+.2f}, {hi:+.2f}])"
    print(line)
    for v in SPEEDS:
        cells = []
        for b in BETAS:
            r = boundary(labels, v, b)
            cells.append("-" if r is None else "없음" if r == "none" else (f"≤{r[1]:.3f}" if r[0] == "<=" else f"{r[1]:.3f}"))
        print(f"   v={v:<5}: " + " ".join(f"{c:>7}" for c in cells))
    return res


def main():
    t0 = time.time()
    print(f"rate {RATE} Hz · 마찰 {FRIC or '문헌값(Mores)'} · 곡선 {os.environ.get('FRICTION_CURVE', 'hyperbolic')} · slip_thr {os.environ.get('SLIP_THR', 'default')} · ramp {f'{RAMP_ACC} m/s²' if RAMP_ACC > 0 else (f'distance {RAMP_DIST} m' if RAMP_DIST > 0 else f'{T_RAMP} s')} · 현 {os.environ.get('STRING', 'celloG')} β {BETAS}")
    lab_e, lab_m, before_m = {}, {}, {}
    for v in SPEEDS:
        final, _ = simulate("E", v, {b: list(FORCES) for b in BETAS}, {})
        lab_e.update(final)
        safe = {}
        for b in BETAS:
            hs = [f for f in FORCES if final[(v, b, f)] == "helmholtz"]
            if len(hs) >= 2:
                safe[b] = float(np.exp(np.mean(np.log(hs))))
        final_m, before = simulate("M", v, {b: list(FORCES) for b in safe}, safe)
        lab_m.update(final_m); before_m.update(before)
        print(f"  v={v}: E 헬름홀츠 {sum(final[k] == 'helmholtz' for k in final)}/{len(final)}, "
              f"M 확립 성공 {sum(before[k] == 'helmholtz' for k in before)}/{len(before)}  ({time.time() - t0:.0f}s)", flush=True)
    valid = {k: l for k, l in lab_m.items() if before_m[k] == "helmholtz"}
    print()
    table("E 수립 (정지에서 출발)", lab_e)
    table("M 유지 (확립 후 힘 낮춤)", valid)
    only_m = sum(1 for k in valid if valid[k] == "helmholtz" and lab_e[k] != "helmholtz")
    only_e = sum(1 for k in valid if lab_e[k] == "helmholtz" and valid[k] != "helmholtz")
    print(f"\n같은 조건 비교 {len(valid)}칸: 유지에서만 헬름홀츠 {only_m}, 수립에서만 {only_e}")


if __name__ == "__main__":
    main()
