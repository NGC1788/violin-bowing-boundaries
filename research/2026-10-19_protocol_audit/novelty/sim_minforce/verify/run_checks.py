"""Checks 1-3 + energy, using the builder's code read-only and the verifier's own code."""
import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
os.environ.setdefault("NUMBA_CACHE_DIR", os.path.join(HERE, "nbcache"))
sys.dont_write_bytecode = True
sys.path.insert(0, HERE)
sys.path.insert(1, os.path.dirname(HERE))
import numpy as np
import vsim as V
import sim  # builder (read-only)
import diag_c0  # builder (read-only)

out = {}
# ---- 1. normalisation ----
c = dict(Z=(V.Z, sim.Z), L=(V.L, sim.L), mode_mass=(V.M, sim.MODE_MASS), zeta1=(V.ZETA1, sim.ZETA1),
         powerlaw_p=(float(V._P), sim.DP), f0_from_L=(V.C / (2 * V.L), sim.F0))
out["constants_mine_vs_builder"] = c
for cfg in ("C0", "C1"):
    for b in (0.05, 0.1, 0.12):
        mine = V.modal_coeffs(cfg, b, 96000.0, 300)
        bm = sim.build_modes(cfg, b)
        dA = max(np.max(np.abs(mine["A"][:, 0, 0] - bm["a11"])), np.max(np.abs(mine["A"][:, 1, 1] - bm["a22"])),
                 np.max(np.abs(mine["A"][:, 0, 1] - bm["a12"]) / np.abs(bm["a12"])), np.max(np.abs(mine["A"][:, 1, 0] - bm["a21"]) / np.abs(bm["a21"])))
        dB = max(np.max(np.abs(mine["B"][:, 1] - bm["b2"]) / np.abs(bm["b2"])), np.max(np.abs(mine["B"][:, 0] - bm["b1"]) / np.abs(bm["b1"])))
        # free-decay rate from the eigenvalues of my transition matrices (independent of the builder's formula)
        ev = np.array([np.linalg.eigvals(mine["A"][j]) for j in (0, 9, 99, 299)])
        sig = -np.log(np.abs(ev[:, 0])) * 96000.0
        fz = np.abs(np.angle(ev[:, 0])) * 96000.0 / (2 * np.pi)
        out[f"modal|{cfg}|{b}"] = dict(N_builder=bm["N"], Y_mine_x2Z=mine["Y"] * 2 * V.Z, Y_builder_x2Z=(bm["Ymod"] + bm["dY"]) * 2 * V.Z,
                                       max_rel_diff_A=float(dA), max_rel_diff_B=float(dB),
                                       tau_s_modes_1_10_100_300=[float(1 / x) for x in sig],
                                       zeta_from_eig=[float(s_ / (2 * np.pi * f_)) for s_, f_ in zip(sig, fz)],
                                       zeta_spec=[float(mine["z"][j]) for j in (0, 9, 99, 299)],
                                       f_from_eig=[float(x) for x in fz])
        print("MODAL", cfg, b, out[f"modal|{cfg}|{b}"], flush=True)

# ---- 1b. Gibbs precursor of the band-limited corner in the builder's C0 modal basis (free motion, no bow) ----
for b in (0.05, 0.1, 0.12):
    bm = sim.build_modes("C0", b)
    n = bm["n"]; vb = 0.1
    p = (-2 * vb / (math.pi * b) / n) * (-1.0) ** n
    q = np.zeros_like(p)
    T0n = int(round(96000 / V.F0))
    vel = np.empty(2 * T0n)
    for k in range(vel.size):
        q, p = bm["a11"] * q + bm["a12"] * p, bm["a21"] * q + bm["a22"] * p
        vel[k] = np.sum(bm["phi"] * p)
    t = (np.arange(vel.size) + 1) / 96000 * V.F0          # in periods; corner at nut at t=0
    ph = (t + 0.5) % 1.0                                    # slip occupies ph in [1-b, 1)  (corner at bow 0.5 T0 later)
    stick = (ph < 1 - b - 0.002) & (ph > 0.002)             # exclude 2 samples at the edges
    dev = np.abs(vel[stick] - vb)
    jump = vb / b
    gF = 2 * V.Z * dev.max() / (V.MUS - V.MUD)
    out[f"gibbs|{b}"] = dict(jump_mps=jump, max_stick_dev_mps=float(dev.max()), rel_overshoot=float(dev.max() / jump),
                             rms_stick_dev=float(np.sqrt(np.mean((vel[stick] - vb) ** 2))),
                             force_equiv_2Zdev_N=float(2 * V.Z * dev.max()), F_needed_to_hold_N=float(gF))
    print("GIBBS", b, out[f"gibbs|{b}"], flush=True)

# ---- 3. raw waveforms of the builder's C0 ramp run (beta 0.1) ----
wz = np.load(os.path.join(os.path.dirname(HERE), "wave_C0_v0.1_b0.1.npz"))
dvs, Fw, fsw = wz["dv"], wz["forces"], float(wz["fs"])
rows = []
for i in range(len(Fw)):
    dv = dvs[i].astype(float)
    slip = np.abs(dv) > 0.001
    e = np.flatnonzero(np.diff(np.r_[0, slip.astype(int), 0]))
    runs = e[1::2] - e[0::2]
    lab1, inf1 = V.label(dv, fsw)
    lab5, inf5 = V.label(dv, fsw, min_slip=5)
    labb, infb = sim.classify(dv, fsw, 0.1)
    rows.append(dict(F=float(Fw[i]), slip_frac=float(slip.mean()), n_slip_runs=int(runs.size),
                     median_slip_run_samples=None if runs.size == 0 else float(np.median(runs)),
                     frac_runs_le3=None if runs.size == 0 else float(np.mean(runs <= 3)),
                     longest_slip=None if runs.size == 0 else int(runs.max()),
                     mean_dv=float(dv.mean()), min_dv=float(dv.min()), max_dv=float(dv.max()),
                     mine=lab1, mine_flicker5=lab5, mine_flicker5_frac=inf5.get("frac"), builder=bool(labb)))
out["builder_C0_ramp_b0.1_waves"] = rows
for r in rows[14:]:
    print("WAVE", r, flush=True)

# ---- 2. energy sanity in the builder's own solver (ideal start, C0 beta 0.1) ----
Fe = np.array([0.335, 0.61, 2.03])
odv, of, md = diag_c0.run("C0", 0.1, "ideal", True, F=Fe)
en = []
for i, F in enumerate(Fe):
    vstr = odv[i] + 0.1
    Pstr = float(np.mean(of[i] * vstr)); Pbow = float(np.mean(of[i] * 0.1)); heat = float(-np.mean(of[i] * odv[i]))
    stick = odv[i] == 0
    en.append(dict(F=float(F), P_into_string_W=Pstr, P_bow_W=Pbow, friction_heat_W=heat,
                   max_abs_stick_force_over_musF=float(np.max(np.abs(of[i][stick])) / (V.MUS * F)) if stick.any() else None,
                   slip_force_over_F=float(np.median(np.abs(of[i][~stick])) / F) if (~stick).any() else None,
                   label_builder=bool(sim.classify(odv[i], 96000.0, 0.1)[0]), label_mine=V.label(odv[i], 96000.0)[0]))
    print("ENERGY", en[-1], flush=True)
# expected string dissipation in ideal Helmholtz: 2*sigma*E, E = sum_n (1/2) M p_n^2 amplitude-averaged
n = np.arange(1, 301); sigma = V.ZETA1 * 2 * math.pi * V.F0
E = np.sum(0.5 * V.M * (2 * 0.1 / (math.pi * 0.1) / n) ** 2)   # kinetic peak per mode = total mode energy
out["energy_builder_ideal_C0_b0.1"] = dict(rows=en, expected_loss_2sigmaE_W=float(2 * sigma * E))
print("EXPECTED 2 sigma E", 2 * sigma * E)
json.dump(out, open(os.path.join(HERE, "checks.json"), "w"), indent=1)
print("saved checks.json")
