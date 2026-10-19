"""Establishment vs maintenance minimum bow force in the corrected passive waveguide.

Same string, same friction law, same classifier; only the force history differs:
  E (establishment, IWK-like):   force fixed at the target from the start, bow accelerates from rest.
  M (maintenance, Schoonderwaldt-like): Helmholtz motion is first established at a safe force,
                                        then the force is lowered smoothly to the target and held.
The lower boundary is read from the last part of the hold, with the dataset classifier, and the
law F_min = c v^a beta^p is fitted per protocol. If a(E) ~ 1 and a(M) ~ 0, the long-standing
theory/measurement disagreement on the speed dependence would be a protocol (hysteresis) effect.
Simulator: Codex's corrected passive waveguide (scripts/bowed_string_cpu.py), literature friction.
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
import regime_map as rm  # noqa: E402

RATE, OUT_EVERY, PROFILE_RATE = 100_000, 2, 1000.0      # simulate at 100 kHz, record at 50 kHz
BETAS = [0.03, 0.05, 0.08, 0.12]
SPEEDS = [0.05, 0.10, 0.20]
FORCES = np.geomspace(0.03, 3.0, 28)
T_RAMP, T_EST, T_DOWN, T_HOLD = 0.2, 0.4, 0.3, 1.0         # seconds


def profiles(protocol, v, f_target, f_safe):
    """Bow velocity and force at 1 kHz for one stroke."""
    total = T_RAMP + T_EST + T_DOWN + T_HOLD
    t = (np.arange(int(total * PROFILE_RATE)) + 0.5) / PROFILE_RATE
    vel = np.clip(t / T_RAMP, 0, 1) * v
    if protocol == "E":
        force = np.full_like(t, f_target)
    else:
        start_down = T_RAMP + T_EST
        frac = np.clip((t - start_down) / T_DOWN, 0, 1)
        force = np.exp(np.log(f_safe) + frac * (np.log(f_target) - np.log(f_safe)))
    return vel, force, total


def label_window(x, ref_f0=98.08):
    periodicity, f0 = rm.dominant_period(x)
    shape = rm.flyback_structure(x, rm.RATE_HZ / f0 if math.isfinite(f0) else math.nan, 0.5, 3.0)
    thresholds = {**rm.DEFAULT_THRESHOLDS, "min_amplitude": 0.02}
    return rm.classify(float(np.std(x)), periodicity, f0, shape["flybacks_per_period"], ref_f0, thresholds)


def run(protocol, safe=None):
    string, friction = StringParams(), FrictionParams()
    cases = [(v, b, f) for v in SPEEDS for b in BETAS for f in FORCES]
    if protocol == "M":
        cases = [c for c in cases if (c[0], c[1]) in safe]
    vel, frc, total = zip(*[profiles(protocol, v, f, f_safe=(safe or {}).get((v, b), 1.0)) for v, b, f in cases])
    total = total[0]
    pv, pf = np.stack(vel, axis=1), np.stack(frc, axis=1)
    steps = int(total * RATE)
    record_from = int((T_RAMP + T_EST - 0.25) * RATE)    # from just before the force descent to the end
    t0 = time.time()
    bridge = simulate_cpu(string, friction, [b for _, b, _ in cases], pv, pf, PROFILE_RATE, RATE, 0.0,
                          steps, output_every=OUT_EVERY, record_from=record_from, threads=2)
    print(f"[{protocol}] {len(cases)} 긋기 시뮬레이션 {time.time() - t0:.0f}s", flush=True)
    labels, established = {}, {}
    width = int(0.2 * rm.RATE_HZ)
    for k, (v, b, f) in enumerate(cases):
        x = np.asarray(bridge[:, k], dtype=float)
        labels[(v, b, f)] = label_window(x[-width:])          # final 0.2 s: fully settled hold
        established[(v, b, f)] = label_window(x[:width])       # 0.2 s just before the descent (M check)
    return labels, established


def lower_boundary(labels, v, b):
    if not all((v, b, f) in labels for f in FORCES):
        return None
    h = [labels[(v, b, f)] == "helmholtz" for f in FORCES]
    s = next((i for i in range(len(h) - 1) if h[i] and h[i + 1]), None)
    if s is None:
        return None
    if s == 0:
        return (-np.inf, np.log(FORCES[0]))
    return (np.log(FORCES[s - 1]), np.log(FORCES[s]))


def fit_law(labels):
    pts = []
    for v in SPEEDS:
        for b in BETAS:
            iv = lower_boundary(labels, v, b)
            if iv and np.isfinite(iv[0]):
                pts.append((np.log(v), np.log(b), 0.5 * (iv[0] + iv[1])))
    if len(pts) < 5:
        return None, pts
    X = np.column_stack([np.ones(len(pts)), [p[0] for p in pts], [p[1] for p in pts]])
    coef, *_ = np.linalg.lstsq(X, np.array([p[2] for p in pts]), rcond=None)
    return coef, pts


def main():
    results = {}
    lab_e, _ = run("E")
    safe = {}
    for v in SPEEDS:
        for b in BETAS:
            hs = [f for f in FORCES if lab_e[(v, b, f)] == "helmholtz"]
            if len(hs) >= 2:
                safe[(v, b)] = float(np.exp(np.mean(np.log(hs))))   # middle of the establishment band
    print(f"유지 절차에 쓸 안전 힘을 찾은 (속도, β): {len(safe)}/{len(SPEEDS) * len(BETAS)}", flush=True)
    lab_m, est_m = run("M", safe)
    ok = sum(1 for k in est_m if est_m[k] == "helmholtz")
    print(f"[M] 힘을 낮추기 직전 헬름홀츠였던 긋기: {ok}/{len(est_m)}", flush=True)
    for protocol, labels in (("E", lab_e), ("M", lab_m)):
        coef, pts = fit_law(labels)
        results[protocol] = (labels, coef, pts)
        print(f"[{protocol}] 경계 점 {len(pts)}/{len(SPEEDS) * len(BETAS)}", end="")
        if coef is not None:
            print(f"  F_min ∝ v^{coef[1]:+.2f} · β^{coef[2]:+.2f}")
        else:
            print("  (적합 불가)")
        for v in SPEEDS:
            row = []
            for b in BETAS:
                iv = lower_boundary(labels, v, b) if all((v, b, f) in labels for f in FORCES) else None
                row.append("없음" if iv is None else ("≤%.3f" % math.exp(iv[1]) if not np.isfinite(iv[0]) else "%.3f" % math.exp(0.5 * (iv[0] + iv[1]))))
            print(f"     v={v:.2f}: " + "  ".join(f"β{b}:{r}" for b, r in zip(BETAS, row)))
    # hysteresis: cells Helmholtz under M but not under E
    common = [k for k in lab_m]
    gap = sum(1 for k in common if lab_m[k] == "helmholtz" and lab_e[k] != "helmholtz")
    rev = sum(1 for k in common if lab_e[k] == "helmholtz" and lab_m[k] != "helmholtz")
    print(f"\n유지(M)에서만 헬름홀츠: {gap}칸 · 수립(E)에서만 헬름홀츠: {rev}칸 (공통 {len(common)}칸)")


if __name__ == "__main__":
    main()
