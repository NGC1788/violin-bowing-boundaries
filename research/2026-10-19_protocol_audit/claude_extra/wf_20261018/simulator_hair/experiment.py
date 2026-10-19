"""Lock-in time and lower boundary of the extended point-bow model under the IWK attack (Claude, 2026-10-18).

Cello G (repo StringParams: f0 98.08 Hz, Z 0.946 kg/s), beta 0.1, bow speeds 0.05 / 0.10 / 0.20 m/s,
40 constant forces 0.03-4 N (geometric). Attack "iwk": force applied at rest, constant acceleration
a = max(v^2 / (2 * 0.048 m), exp(c_v) F^1.05 beta^q_v) (A2 fit: c 6.18/5.52/5.01, q 2.64/2.33/2.04), then a
1.5 s plateau; attack "ramp": linear 0.2 s ramp. Optional "decay": the A2 travel-dependent force decrease
D(x) (0, .044, .088, .125, .153, .174 N at 0, 2, 5, 8, 11, 14 cm) as in claude_extra/sim_iwk_protocol.py.

Per stroke (exact contact states, slip flag per sample):
  * label of the 0.2 s window starting at the plateau start (contact rule of sim_contact2: slip onsets per
    period 0.9-1.1 and slip fraction 0.3-3 beta) and, independently, the measured-data waveform classifier
    (regime_map features + A2 thresholds) on the simulated bridge force at 50 kHz;
  * lock-in time from plateau start: 0.1 s (10-period) windows every 0.05 s, first window from which >= 90 % of
    the remaining windows are Helmholtz (>= 4 remaining windows, else 'never'); same rule as measured_targets.py;
  * label of a late window (1.2-1.4 s after plateau start).
Boundaries: first K=2 consecutive Helmholtz forces (in force order), geometric midpoint.
Usage: python -B experiment.py CONFIG [CONFIG ...] [--attack iwk|ramp] [--decay] [--beta 0.1]
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import hairsim as H  # noqa: E402
from bowed_string import StringParams  # noqa: E402

REPO = Path.home() / "violin-bowing-boundaries"
sys.path.insert(0, str(REPO / "scripts"))
import regime_map as rm  # noqa: E402

SUMMARY = next((REPO / "reports/research_20261013/lockbox_A2_preprocess/2024-04-19_TypeA_sample2/regime_map").glob("*/summary.json"))
APPLIED = json.loads(SUMMARY.read_text())["thresholds_applied"]

RATE, PR = 100_000, 1000.0
STRING = StringParams()
F0 = STRING.f0
SPEEDS = [0.05, 0.10, 0.20]
FORCES = np.geomspace(0.03, 4.0, 40)
ACC = {0.05: (6.18, 2.64), 0.10: (5.52, 2.33), 0.20: (5.01, 2.04)}
PLATEAU = 1.5
XS = np.array([0, 0.02, 0.05, 0.08, 0.11, 0.14, 0.30])
DECAY_A2 = np.array([0, 0.044, 0.088, 0.125, 0.153, 0.174, 0.25])

# measured comparators (A2, beta ~ 0.1; measured_targets.log and per_speed_levels.py)
TARGET_FIT = {0.05: 0.29, 0.10: 0.64, 0.20: 1.23}
TARGET_COL = {0.05: 0.653, 0.10: 0.892, 0.20: 0.928}

FR = {"M": ("hyperbolic", H.FRICTION_MORES), "H": ("hyperbolic", H.FRICTION_MAT_HYP), "E": ("ep", H.FRICTION_MAT_EP)}


def config(name):
    """name = <elements>-<friction>; elements from: base | D (damping) S (dispersion) T (torsion, Matusiak FIR)
    t (torsion, constant-gain mode) K (hair KV, Matusiak) W (hair MSD, van Walstijn); friction M/H/E.
    Optional suffixes after ':' scale parameters, e.g. 'D-H:r=0.5' (damping r0,r1 x0.5), ':q=60' (torsion Q),
    ':z=3' (torsion Z_T x3), ':hk=0.1' (hair stiffness x0.1), ':hd=0.1' (hair damper x0.1)."""
    base, _, mods = name.partition(":")
    elems, fr = base.rsplit("-", 1)
    kw = dict(damping="onepole", torsion=None, hair=None, dispersion=False)
    scale = dict(kv.split("=") for kv in mods.split(",") if kv) if mods else {}
    scale = {k: float(v) for k, v in scale.items()}
    if elems != "base":
        for ch in elems:
            if ch == "D":
                kw["damping"] = "physical"
                if "r" in scale:
                    kw["damping_params"] = dict(r0=H.MAT_STRING["r0"] * scale["r"], r1=H.MAT_STRING["r1"] * scale["r"])
            elif ch == "S":
                kw["dispersion"] = True
            elif ch in "Tt":
                tor = dict(H.MAT_TORSION)
                if "q" in scale:
                    tor["q"] = scale["q"]
                if "z" in scale:
                    tor["z"] = tor["z"] * scale["z"]
                kw["torsion"] = tor
                kw["torsion_mode"] = "physical" if ch == "T" else "gain"
            elif ch in "KW":
                hair = dict(H.HAIR_KV_MAT if ch == "K" else H.HAIR_MSD_VW)
                if "hk" in scale:
                    hair["k"] = hair["k"] * scale["hk"]
                if "hd" in scale:
                    key = "d" if ch == "K" else "r"
                    hair[key] = hair[key] * scale["hd"]
                kw["hair"] = hair
            else:
                raise ValueError(f"unknown element {ch}")
    kind, fp = FR[fr]
    kw["friction_kind"] = kind
    return fp, kw


def accel(v, beta, f):
    c, q = ACC[v]
    return max(v * v / (2 * 0.048), math.exp(c) * f ** 1.05 * beta ** q)


def profiles(v, beta, attack, decay):
    vel, frc, starts = [], [], []
    ramps = [v / accel(v, beta, f) if attack == "iwk" else 0.2 for f in FORCES]
    total = max(ramps) + PLATEAU + 0.01
    t = (np.arange(int(total * PR)) + 0.5) / PR
    for f, ramp in zip(FORCES, ramps):
        vv = np.minimum(v / ramp * t, v)
        x = np.cumsum(vv) / PR
        ff = np.maximum(f - np.interp(x, XS, DECAY_A2), 0.002) if decay else np.full_like(t, f)
        vel.append(vv); frc.append(ff); starts.append(ramp)
    return np.stack(vel, 1), np.stack(frc, 1), np.array(starts), total


def contact_label(s, beta):
    s = s.astype(np.int8)
    starts = int(np.sum((s[1:] == 1) & (s[:-1] == 0)))
    per = starts / (len(s) / RATE * F0)
    frac = float(s.mean())
    return starts > 0 and 0.9 <= per <= 1.1 and 0.3 * beta <= frac <= 3 * beta


def wave_label(x):
    rate = RATE / 2
    period, f0 = rm.dominant_period(x, rate)
    shape = rm.flyback_structure(x, rate / f0 if math.isfinite(f0) else math.nan, APPLIED["flyback_frac"],
                                 APPLIED["hysteresis_sigma"])
    fpp = shape["flybacks_per_period"]
    lab = rm.classify(float(np.std(x)), period, f0, int(fpp) if fpp == fpp else None, 98.08, APPLIED)
    return lab == "helmholtz"


NOISE_N = 0.020  # N, white Gaussian sensor noise added before waveform labelling (measured pre-motion bridge
                 # p2p 0.077 N over a 0.1 s window ~ 7.6 sigma); without it the classifier calls sub-mN periodic
                 # residue of a sliding string 'helmholtz'


def window_labels(flags, x, start, beta, end):
    """Contact and waveform labels of 0.1 s windows every 0.05 s from start (samples at RATE; x at RATE/2)."""
    win, step = int(0.1 * RATE), int(0.05 * RATE)
    lc, lw = [], []
    a = start
    while a + win <= end:
        lc.append(contact_label(flags[a:a + win], beta))
        lw.append(wave_label(x[a // 2:(a + win) // 2]))
        a += step
    return np.array(lc, bool), np.array(lw, bool)


def lock_from(labs, step=0.05):
    for i in range(len(labs) - 3):
        if labs[i] and labs[i:].mean() >= 0.9:
            return i * step
    return math.inf


def boundary(forces, helm, K=2):
    order = np.argsort(forces)
    f = np.asarray(forces)[order]
    h = np.asarray(helm)[order]
    s = next((i for i in range(len(h) - K + 1) if all(h[i:i + K])), None)
    if s is None:
        return math.inf
    if s == 0:
        return -f[0]  # left-censored: below the lowest force
    return float(math.sqrt(f[s - 1] * f[s]))


def run(name, beta=0.1, attack="iwk", decay=False):
    fp, kw = config(name)
    res = {"config": name, "beta": beta, "attack": attack, "decay": decay, "speeds": {}}
    t0 = time.time()
    for v in SPEEDS:
        pv, pf, starts, total = profiles(v, beta, attack, decay)
        steps = int(total * RATE)
        out, diag = H.simulate(STRING, fp, [beta] * len(FORCES), pv, pf, PR, RATE, 0.0, steps, 2, 0, threads=2, **kw)
        flags = diag["slip_flags"]
        rng = np.random.default_rng(12345)
        noisy = out.astype(float) + rng.normal(0.0, NOISE_N, out.shape)
        rows = []
        for k, f in enumerate(FORCES):
            a = int(starts[k] * RATE)
            b = a + int(0.2 * RATE)
            fw = float(np.mean(pf[int(a / RATE * PR):int(b / RATE * PR), k]))
            h0 = contact_label(flags[a:b, k], beta)
            w0 = wave_label(noisy[a // 2:b // 2, k])
            la, lb = a + int(1.2 * RATE), a + int(1.4 * RATE)
            hl = contact_label(flags[la:lb, k], beta)
            wl = wave_label(noisy[la // 2:lb // 2, k])
            lc, lw = window_labels(flags[:, k], noisy[:, k], a, beta, min(steps, a + int(PLATEAU * RATE)))
            rows.append(dict(force=float(f), window_force=fw, ramp_s=float(starts[k]), helm_t0=bool(h0), wave_t0=bool(w0),
                             helm_late=bool(hl), wave_late=bool(wl), lock_contact_s=lock_from(lc), lock_wave_s=lock_from(lw),
                             contact_windows="".join("H" if x else "." for x in lc),
                             wave_windows="".join("H" if x else "." for x in lw)))
        fw = [r["window_force"] for r in rows]
        b0 = boundary(fw, [r["helm_t0"] for r in rows])
        bw = boundary(fw, [r["wave_t0"] for r in rows])
        bl = boundary(FORCES, [r["helm_late"] for r in rows])
        bwl = boundary(FORCES, [r["wave_late"] for r in rows])

        def band_stats(ref, key):
            band = [r[key] for r in rows if math.isfinite(ref) and ref > 0 and 1.3 * ref <= r["window_force"] <= 3 * ref]
            L = np.array(band) if band else np.array([math.inf])
            fin = L[np.isfinite(L)]
            return dict(n=len(band), by0=float(np.mean(L <= 0.0)), by02=float(np.mean(L <= 0.2)),
                        by04=float(np.mean(L <= 0.4)), never=float(np.mean(~np.isfinite(L))),
                        median=float(np.median(fin)) if len(fin) else math.nan)
        ref_w = bw if math.isfinite(bw) and bw > 0 else bwl
        ref_c = b0 if math.isfinite(b0) and b0 > 0 else bl
        summ = dict(boundary_t0=b0, boundary_wave_t0=bw, boundary_late=bl, boundary_wave_late=bwl,
                    lock_wave=band_stats(ref_w, "lock_wave_s"), lock_contact=band_stats(ref_c, "lock_contact_s"),
                    elapsed_s=diag["elapsed_s"], contact_admittance=diag["contact_admittance"],
                    min_friction_power=float(np.min(diag["min_friction_dissipation_watts"])))
        res["speeds"][str(v)] = dict(summary=summ, rows=rows)
    res["wall_s"] = time.time() - t0
    return res


def fmt(x):
    if not math.isfinite(x):
        return "  none"
    return f"<{-x:.3f}" if x < 0 else f"{x:6.3f}"


def line(res):
    parts = []
    for v in SPEEDS:
        s = res["speeds"][str(v)]["summary"]
        lw, lc = s["lock_wave"], s["lock_contact"]
        parts.append(f"v{v:.2f}: wave F*0 {fmt(s['boundary_wave_t0'])} late {fmt(s['boundary_wave_late'])} lock<=.2 {lw['by02']:.2f} never {lw['never']:.2f} (n{lw['n']}) "
                     f"| contact F*0 {fmt(s['boundary_t0'])} late {fmt(s['boundary_late'])} lock<=.2 {lc['by02']:.2f} <=.4 {lc['by04']:.2f} (n{lc['n']})")
    return f"{res['config']:<14s}\n   " + "\n   ".join(parts) + f"  [{res['wall_s']:.0f}s]"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("configs", nargs="+")
    ap.add_argument("--attack", default="iwk")
    ap.add_argument("--decay", action="store_true")
    ap.add_argument("--beta", type=float, default=0.1)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    outdir = HERE / "results"
    outdir.mkdir(exist_ok=True)
    for name in args.configs:
        res = run(name, args.beta, args.attack, args.decay)
        tag = f"{name.replace(':', '_').replace('=', '').replace(',', '_')}_{args.attack}{'_decay' if args.decay else ''}_b{args.beta}{args.tag}"
        (outdir / f"{tag}.json").write_text(json.dumps(res, indent=1, default=lambda o: None if not math.isfinite(o) else o))
        print(line(res), flush=True)


if __name__ == "__main__":
    main()
