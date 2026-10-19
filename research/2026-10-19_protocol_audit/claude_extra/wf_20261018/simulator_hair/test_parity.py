"""Parity, passivity and damping-realisation tests for hairsim.py (Claude, 2026-10-18).

1. all new elements off  vs scripts/bowed_string_cpu.simulate_cpu (feedback_delay linear): bridge force + contact states
2. torsion "gain" mode   vs claude_extra/bowed_string_torsion.simulate_cpu (its default TORSION)
3. friction "ep"         vs claude_extra/elastoplastic.simulate_cpu (Matusiak S-model)
4. FIR passivity (max |H| <= 1) and design error; realised mode decay of the transverse loop vs the target
   sigma_k = r0 + r1 k^2 (Matusiak Table I) and of the torsional loop vs Q_T
5. hair energy balance residual (trapezoidal rule), friction dissipation >= 0 (hyperbolic), bow work >= dissipation
6. time-domain free decay: bow briefly, lift (force 0), fit harmonic decay rates of the bridge force
Usage: python -B test_parity.py
"""
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(Path.home() / "violin-rne/claude_extra"))
import hairsim as H  # noqa: E402
from bowed_string import StringParams  # noqa: E402
from bowed_string_cpu import simulate_cpu as cpu_ref  # noqa: E402
import bowed_string_torsion as TOR  # noqa: E402
import elastoplastic as EPREF  # noqa: E402

RATE, PR = 100_000, 1000.0
S = StringParams()


def profiles(betas, forces, v=0.1, ramp=0.2, total=0.6):
    t = (np.arange(int(total * PR)) + 0.5) / PR
    vel = np.stack([np.clip(t / ramp, 0, 1) * v for _ in betas], 1)
    frc = np.stack([np.full_like(t, f) for f in forces], 1)
    return vel, frc, int(total * RATE)


def report(name, ok, detail):
    print(f"[{'PASS' if ok else 'FAIL'}] {name}: {detail}")
    return ok


def main():
    allok = True
    betas = [0.06, 0.10, 0.10, 0.15, 0.10, 0.08]
    forces = [0.2, 0.5, 1.5, 0.8, 0.05, 3.0]
    vel, frc, steps = profiles(betas, forces)

    # 1 -------------------------------------------------------------- baseline parity
    ref, rd = cpu_ref(S, H.FRICTION_MORES, betas, vel, frc, PR, RATE, 0.0, steps, 1, 0, threads=2, return_diagnostics=True)
    out, d = H.simulate(S, H.FRICTION_MORES, betas, vel, frc, PR, RATE, 0.0, steps, 1, 0)
    diff = float(np.max(np.abs(out - ref)))
    sdiff = int(np.sum(d["slip_flags"].astype(float) != rd["slip_fraction_by_output"]))
    allok &= report("baseline vs bowed_string_cpu", diff == 0.0 and sdiff == 0,
                    f"max |bridge diff| {diff:.3g} N, slip-flag mismatches {sdiff} of {d['slip_flags'].size}")
    ref2, rd2 = cpu_ref(S, H.FRICTION_MAT_HYP, betas, vel, frc, PR, RATE, 0.0, steps, 1, 0, threads=2, return_diagnostics=True)
    out2, d2 = H.simulate(S, H.FRICTION_MAT_HYP, betas, vel, frc, PR, RATE, 0.0, steps, 1, 0)
    diff2 = float(np.max(np.abs(out2 - ref2)))
    allok &= report("baseline (Matusiak-equivalent hyperbola) vs bowed_string_cpu", diff2 == 0.0,
                    f"max |bridge diff| {diff2:.3g} N")

    # 2 -------------------------------------------------------------- torsion parity
    tref, td = TOR.simulate_cpu(S, H.FRICTION_MORES, betas, vel, frc, PR, RATE, 0.0, steps, 1, 0, threads=2, return_diagnostics=True)
    tout, tdd = H.simulate(S, H.FRICTION_MORES, betas, vel, frc, PR, RATE, 0.0, steps, 1, 0, torsion=TOR.TORSION, torsion_mode="gain")
    diff = float(np.max(np.abs(tout - tref)))
    sdiff = int(np.sum(tdd["slip_flags"].astype(float) != td["slip_fraction_by_output"]))
    allok &= report("torsion(gain) vs bowed_string_torsion", diff == 0.0 and sdiff == 0,
                    f"max |bridge diff| {diff:.3g} N, slip mismatches {sdiff}")

    # 3 -------------------------------------------------------------- EP parity (shorter, it is slow)
    vel3, frc3, steps3 = profiles(betas[:4], forces[:4], total=0.3)
    eref, ed = EPREF.simulate_cpu(S, H.FRICTION_MAT_EP, betas[:4], vel3, frc3, PR, RATE, 0.0, steps3, 1, 0, threads=2, return_diagnostics=True)
    eout, edd = H.simulate(S, H.FRICTION_MAT_EP, betas[:4], vel3, frc3, PR, RATE, 0.0, steps3, 1, 0, friction_kind="ep")
    diff = float(np.max(np.abs(eout - eref)))
    sdiff = int(np.sum(edd["slip_flags"].astype(float) != ed["slip_fraction_by_output"]))
    allok &= report("ep vs elastoplastic.py", diff == 0.0 and sdiff == 0, f"max |bridge diff| {diff:.3g} N, slip mismatches {sdiff}")

    # 4 -------------------------------------------------------------- FIR passivity and realised damping
    f0 = S.f0
    c = 2 * S.length * f0
    p = H.MAT_STRING
    for disp in (False, True):
        firs = H.transverse_firs([0.1], RATE, f0, p["r0"], p["r1"], c, p["kappa2"] if disp else 0.0)
        (hb, n0b, pkb, eb), (hn, n0n, pkn, en) = firs[0]
        allok &= report(f"transverse FIR passivity (dispersion={disp})", pkb <= 1.0 and pkn <= 1.0,
                        f"max|H| bridge {pkb:.5f} (len {len(hb)}, rms err {eb:.2e}), nut {pkn:.5f} (len {len(hn)}, rms err {en:.2e})")
        # loop response: modes where the loop phase is a multiple of 2 pi
        nf = 1 << 20
        Hb = np.fft.rfft(hb, nf) * np.exp(-2j * np.pi * np.arange(nf // 2 + 1) / nf * n0b)
        Hn = np.fft.rfft(hn, nf) * np.exp(-2j * np.pi * np.arange(nf // 2 + 1) / nf * n0n)
        Lp = Hb * Hn
        fr = np.arange(nf // 2 + 1) / nf * RATE
        ph = np.unwrap(np.angle(Lp))
        rows = []
        for kk in (1, 2, 5, 10, 20, 30, 40):
            i = int(np.argmin(np.abs(ph + 2 * np.pi * kk)))
            sig_real = -math.log(abs(Lp[i])) * f0
            kw = 2 * np.pi * fr[i]
            sig_tgt = p["r0"] + p["r1"] * H._k2(np.array([kw]), c, p["kappa2"] if disp else 0.0)[0]
            rows.append(f"k{kk} {fr[i]:.0f}Hz Q {kw / (2 * sig_real):.0f}/{kw / (2 * sig_tgt):.0f}")
        print("     realised/target Q: " + ", ".join(rows))
    # one-pole baseline for comparison
    cfg = H.prepare_waveguide(S, [0.1], RATE)
    rows = []
    for kk in (1, 2, 5, 10, 20, 30, 40):
        w = 2 * np.pi * kk * f0 / RATE
        hp = cfg["gain"][0] * (1 - cfg["pole"][0]) / (1 - cfg["pole"][0] * np.exp(-1j * w))
        loop = abs(hp) ** 2
        for key in ("delay_bridge", "delay_nut"):
            fr_ = cfg[key][0] - math.floor(cfg[key][0])
            loop *= abs((1 - fr_) + fr_ * np.exp(-1j * w))
        sig = -math.log(loop) * f0
        rows.append(f"k{kk} Q {2 * np.pi * kk * f0 / (2 * sig):.0f}")
    print("     baseline one-pole loop Q (Q1 2000, corner 6 kHz, incl. linear-interp loss): " + ", ".join(rows))
    tf = H.torsion_firs([0.1], RATE, H.MAT_TORSION["f"], H.MAT_TORSION["q"])
    (hb, n0b, pkb, eb), (hn, n0n, pkn, en) = tf[0]
    nf = 1 << 18
    Lp = (np.fft.rfft(hb, nf) * np.exp(-2j * np.pi * np.arange(nf // 2 + 1) / nf * n0b)) * \
         (np.fft.rfft(hn, nf) * np.exp(-2j * np.pi * np.arange(nf // 2 + 1) / nf * n0n))
    fr = np.arange(nf // 2 + 1) / nf * RATE
    ph = np.unwrap(np.angle(Lp))
    rows = []
    for kk in (1, 2, 4, 8):
        i = int(np.argmin(np.abs(ph + 2 * np.pi * kk)))
        sig_real = -math.log(abs(Lp[i])) * H.MAT_TORSION["f"]
        rows.append(f"k{kk} {fr[i]:.0f}Hz Q {2 * np.pi * fr[i] / (2 * sig_real):.1f}")
    allok &= report("torsion FIR passivity", pkb <= 1.0 and pkn <= 1.0,
                    f"max|H| {pkb:.5f}/{pkn:.5f}, lengths {len(hb)}/{len(hn)}; realised Q (target {H.MAT_TORSION['q']}): " + ", ".join(rows))

    # 5 -------------------------------------------------------------- hair energy balance, dissipation
    for hair in (H.HAIR_KV_MAT, H.HAIR_MSD_VW):
        o, dd = H.simulate(S, H.FRICTION_MORES, betas, vel, frc, PR, RATE, 0.0, steps, 1, 0, damping="physical",
                           torsion=H.MAT_TORSION, hair=hair)
        rel = float(np.max(dd["max_hair_energy_residual_J"] / np.maximum(dd["max_hair_energy_J"], 1e-30)))
        md = float(np.min(dd["min_friction_dissipation_watts"]))
        wk = dd["bow_work_J"] - dd["friction_dissipation_J"]
        allok &= report(f"hair {hair['kind']} energy balance / dissipation", rel < 1e-9 and md >= -1e-12 and np.all(wk >= 0),
                        f"max |dH - dt(f u - r u^2)| / max H = {rel:.2e}; min friction power {md:.2e} W; "
                        f"bow work - friction dissipation (energy into string+hair+torsion) min {wk.min():.3e} J; Y_contact {dd['contact_admittance']:.4f}")

    # 6 -------------------------------------------------------------- free decay in the time domain
    total = 1.5
    t = (np.arange(int(total * PR)) + 0.5) / PR
    v = np.where(t < 0.25, np.clip(t / 0.05, 0, 1) * 0.1, 0.1)
    f = np.where(t < 0.25, 1.0, 0.0)
    vel6 = np.stack([v, v], 1); frc6 = np.stack([f, f], 1)
    for label, kw in (("baseline one-pole", {}), ("physical damping", dict(damping="physical"))):
        o, _ = H.simulate(S, H.FRICTION_MORES, [0.1, 0.1], vel6, frc6, PR, RATE, 0.0, int(total * RATE), 1, 0, **kw)
        x = o[int(0.3 * RATE):, 0].astype(float)
        per = RATE / f0
        seg = int(round(4 * per))  # Hann window over 4 periods: other harmonics fall outside the main lobe
        win = np.hanning(seg)
        rows = []
        for kk in (1, 3, 5, 9, 15, 25):  # not multiples of 1/beta = 10 (absent harmonics at beta 0.1)
            n_idx = np.arange(len(x))
            demod = x * np.exp(-2j * np.pi * kk * f0 / RATE * n_idx)
            amps = np.array([abs((demod[s0:s0 + seg] * win).sum()) for s0 in range(0, len(x) - seg, seg // 4)])
            tt = np.arange(len(amps)) * (seg // 4) / RATE
            good = (amps > amps[0] * 1e-2) & (np.arange(len(amps)) < 200)
            sl = np.polyfit(tt[good], np.log(amps[good]), 1)[0] if good.sum() >= 3 else np.nan
            tgt = p["r0"] + p["r1"] * (2 * np.pi * kk * f0 / c) ** 2
            rows.append(f"k{kk}: sigma {-sl:.2f} (Matusiak {tgt:.2f}) 1/s")
        print(f"     free decay, {label}: " + "; ".join(rows))
    print("ALL PASS" if allok else "SOME TESTS FAILED")


if __name__ == "__main__":
    main()
