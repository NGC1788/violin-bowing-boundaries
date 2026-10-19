"""Establishment vs maintenance minimum bow force, labelled from exact contact states.

Same experiment as sim_hysteresis.py, but each stroke is labelled from the simulator's own
slip indicator (simulate_cpu(..., return_diagnostics=True), output_every=1), not from the
bridge-force waveform classifier (which calls weak noise-free sinusoids Helmholtz):
  Helmholtz  = one slip interval per period (0.9-1.1 slip starts per period over the last 0.4 s),
               with a slip fraction of the order of beta (0.3 beta .. 3 beta)
  multiple   = > 1.3 slip starts per period;  stuck = no slip;  other = anything else
E: force fixed at the target from rest. M: establish at a safe force (middle of E's Helmholtz band
for that speed and beta), lower smoothly to the target, hold 1 s; strokes not in Helmholtz motion
just before the descent are excluded from M.
"""

import math
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path.home() / "violin-bowing-boundaries"
sys.path.insert(0, str(REPO / "scripts"))
from bowed_string import FrictionParams, StringParams  # noqa: E402
from bowed_string_cpu import simulate_cpu  # noqa: E402

RATE, PROFILE_RATE = 100_000, 1000.0
BETAS = [0.03, 0.05, 0.08, 0.12]
SPEEDS = [0.05, 0.10, 0.20]
FORCES = np.geomspace(0.005, 3.0, 32)
T_RAMP, T_EST, T_DOWN, T_HOLD = 0.2, 0.4, 0.3, 1.0
F0 = StringParams().f0


def profiles(protocol, v, f_target, f_safe):
    total = T_RAMP + T_EST + T_DOWN + T_HOLD
    t = (np.arange(int(total * PROFILE_RATE)) + 0.5) / PROFILE_RATE
    vel = np.clip(t / T_RAMP, 0, 1) * v
    if protocol == "E":
        force = np.full_like(t, f_target)
    else:
        frac = np.clip((t - (T_RAMP + T_EST)) / T_DOWN, 0, 1)
        force = np.exp(np.log(f_safe) + frac * (np.log(f_target) - np.log(f_safe)))
    return vel, force, total


def contact_label(slip, beta):
    """slip: 0/1 indicator per step over an interval."""
    s = (np.asarray(slip) > 0.5).astype(np.int8)
    starts = int(np.sum((s[1:] == 1) & (s[:-1] == 0)))
    periods = len(s) / RATE * F0
    per = starts / periods
    frac = float(s.mean())
    if starts == 0:
        return "stuck" if frac < 0.5 else "sliding"
    if 0.9 <= per <= 1.1 and 0.3 * beta <= frac <= 3 * beta:
        return "helmholtz"
    if per > 1.3:
        return "multiple"
    return "other"


def run(protocol, safe=None):
    string, friction = StringParams(), FrictionParams()
    cases = [(v, b, f) for v in SPEEDS for b in BETAS for f in FORCES]
    if protocol == "M":
        cases = [c for c in cases if (c[0], c[1]) in safe]
    vel, frc, total = zip(*[profiles(protocol, v, f, (safe or {}).get((v, b), 1.0)) for v, b, f in cases])
    total = total[0]
    pv, pf = np.stack(vel, axis=1), np.stack(frc, axis=1)
    steps = int(total * RATE)
    pre = int((T_RAMP + T_EST - 0.25) * RATE)
    t0 = time.time()
    _, diag = simulate_cpu(string, friction, [b for _, b, _ in cases], pv, pf, PROFILE_RATE, RATE, 0.0, steps,
                           output_every=1, record_from=pre, threads=2, return_diagnostics=True)
    slip = diag["slip_fraction_by_output"]
    print(f"[{protocol}] {len(cases)} 긋기 {time.time() - t0:.0f}s, 기록 {slip.shape}", flush=True)
    final, before = {}, {}
    n_last, n_pre = int(0.4 * RATE), int(0.2 * RATE)
    for k, (v, b, f) in enumerate(cases):
        final[(v, b, f)] = contact_label(slip[-n_last:, k], b)
        before[(v, b, f)] = contact_label(slip[:n_pre, k], b)
    return final, before


def boundary(labels, v, b):
    if not all((v, b, f) in labels for f in FORCES):
        return None
    h = [labels[(v, b, f)] == "helmholtz" for f in FORCES]
    s = next((i for i in range(len(h) - 1) if h[i] and h[i + 1]), None)
    if s is None:
        return "none"
    return ("<=", FORCES[0]) if s == 0 else ("=", math.sqrt(FORCES[s - 1] * FORCES[s]))


def fit(labels):
    pts = []
    for v in SPEEDS:
        for b in BETAS:
            r = boundary(labels, v, b)
            if isinstance(r, tuple) and r[0] == "=":
                pts.append((math.log(v), math.log(b), math.log(r[1])))
    if len(pts) < 5:
        return None, len(pts)
    X = np.column_stack([np.ones(len(pts)), [p[0] for p in pts], [p[1] for p in pts]])
    y = np.array([p[2] for p in pts])
    coef, res, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ coef
    cov = np.linalg.inv(X.T @ X) * (resid @ resid) / max(len(pts) - 3, 1)
    return (coef, np.sqrt(np.diag(cov))), len(pts)


def show(name, labels):
    res, n = fit(labels)
    head = f"[{name}] 경계 점 {n}/{len(SPEEDS) * len(BETAS)}"
    if res:
        (c, se) = res
        head += f"  F_min ∝ v^{c[1]:+.2f}±{se[1]:.2f} · β^{c[2]:+.2f}±{se[2]:.2f}"
    print(head)
    for v in SPEEDS:
        cells = []
        for b in BETAS:
            r = boundary(labels, v, b)
            cells.append("-" if r is None else "없음" if r == "none" else (f"≤{r[1]:.3f}" if r[0] == "<=" else f"{r[1]:.3f}"))
        print(f"    v={v:.2f}: " + "  ".join(f"β{b}:{c}" for b, c in zip(BETAS, cells)))


def main():
    lab_e, _ = run("E")
    counts = {}
    for k, l in lab_e.items():
        counts[l] = counts.get(l, 0) + 1
    print("E 라벨 분포:", counts)
    safe = {}
    for v in SPEEDS:
        for b in BETAS:
            hs = [f for f in FORCES if lab_e[(v, b, f)] == "helmholtz"]
            if len(hs) >= 2:
                safe[(v, b)] = float(np.exp(np.mean(np.log(hs))))
    print(f"안전 힘 확보 (v, β): {len(safe)}/{len(SPEEDS) * len(BETAS)}")
    lab_m, before = run("M", safe)
    valid = {k: l for k, l in lab_m.items() if before[k] == "helmholtz"}
    print(f"M: 힘 낮추기 직전 헬름홀츠 {len(valid)}/{len(lab_m)}")
    show("E 수립", lab_e)
    show("M 유지", valid)
    common = [k for k in valid]
    only_m = sum(1 for k in common if valid[k] == "helmholtz" and lab_e[k] != "helmholtz")
    only_e = sum(1 for k in common if lab_e[k] == "helmholtz" and valid[k] != "helmholtz")
    print(f"\n같은 조건에서 유지(M)에서만 헬름홀츠 {only_m}칸, 수립(E)에서만 {only_e}칸 (비교 {len(common)}칸)")


if __name__ == "__main__":
    main()
