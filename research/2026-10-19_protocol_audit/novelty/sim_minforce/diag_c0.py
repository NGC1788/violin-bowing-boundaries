"""C0 failure diagnostics (NOT the pre-registered protocol; nothing here is used for a reading).
Reproduces debug attempts 2 and 3 and the maintenance (ideal Helmholtz start) scan.
Writes diag_c0.json and fig_sim.png.  Run: python diag_c0.py"""
import json
import math
import os

import numpy as np
from numba import njit, prange

import sim

OUT = os.path.join(sim.HERE, "diag_c0.json")


@njit(cache=True)
def k1(F, q0, p0, phi, a11, a12, a21, a22, b1, b2, dt, vb, acc, nsteps, i0, odv, of, fric, hyst):
    """Same solver as sim.sim_one, plus: initial state, records the friction force, optional
    no-hysteresis rule (stick whenever |f_stick| <= mu_s F)."""
    N = phi.size
    q = q0.copy(); p = p0.copy(); Y = 0.0
    for j in range(N):
        Y += phi[j] * phi[j] * b2[j]
    YF = Y * F; slipping = False; sdir = 1.0
    for k in range(nsteps):
        vbow = acc * (k + 1) * dt
        if vbow > vb:
            vbow = vb
        vfree = 0.0
        for j in range(N):
            vfree += phi[j] * (a21[j] * q[j] + a22[j] * p[j])
        w = vfree - vbow; f = 0.0; dv = 0.0; done = False
        if slipping and hyst:
            u = sim._slip_u(sdir * w, YF, fric, sim.MUD, sim.MUS, sim.MUINF, sim.V0)
            if u > 0.0:
                f = -sdir * F * sim._mu(u, fric, sim.MUD, sim.MUS, sim.MUINF, sim.V0); dv = sdir * u; done = True
        if not done:
            fst = -w / Y
            if abs(fst) <= sim.MUS * F:
                f = fst; slipping = False
            else:
                sdir = 1.0 if w > 0.0 else -1.0
                u = sim._slip_u(abs(w), YF, fric, sim.MUD, sim.MUS, sim.MUINF, sim.V0)
                if u < 0.0:
                    u = 0.0
                f = -sdir * F * sim._mu(u, fric, sim.MUD, sim.MUS, sim.MUINF, sim.V0); dv = sdir * u; slipping = True
        for j in range(N):
            g = phi[j] * f
            qn = a11[j] * q[j] + a12[j] * p[j] + b1[j] * g
            p[j] = a21[j] * q[j] + a22[j] * p[j] + b2[j] * g
            q[j] = qn
        r = k - i0
        if r >= 0 and r < odv.size:
            odv[r] = dv; of[r] = f


@njit(parallel=True, cache=True)
def kb(Fs, q0s, p0s, phi, a11, a12, a21, a22, b1, b2, dt, vb, acc, nsteps, i0, odv, of, fric, hyst):
    for i in prange(Fs.size):
        k1(Fs[i], q0s[i], p0s[i], phi, a11, a12, a21, a22, b1, b2, dt, vb, acc, nsteps, i0, odv[i], of[i], fric, hyst)


def run(cfg, beta, start, hyst=True, fs=sim.FS_BASE, F=sim.EXT_FORCES, vb=0.1):
    md = sim.build_modes(cfg, beta, fs=fs)
    n = md["n"]; w = 2 * np.pi * md["f"]; m = sim.MODE_MASS
    if start == "ideal":   # ideal Helmholtz state (mid-stick), constant bow speed from t = 0
        p0 = (-2 * vb / (math.pi * beta) / n) * (-1.0) ** n
        q0s = np.array([md["phi"] * (sim.MUD * Fi) / (m * w ** 2) for Fi in F]); acc = 1e12; t_pl = 0.0
    else:                  # PLAN protocol: rest, linear ramp
        p0 = np.zeros(md["N"]); q0s = np.zeros((len(F), md["N"])); acc = sim.ACC[vb]; t_pl = vb / acc
    p0s = np.array([p0] * len(F))
    i0 = int(round((t_pl + 0.2) * fs)) - 1; i1 = int(round((t_pl + 0.4) * fs)) - 1
    odv = np.zeros((len(F), i1 - i0)); of = np.zeros_like(odv)
    kb(np.asarray(F, float), q0s, p0s, md["phi"], md["a11"], md["a12"], md["a21"], md["a22"], md["b1"], md["b2"],
       md["dt"], vb, acc, i1, i0, odv, of, 0 if cfg in ("C0", "C1") else 1, hyst)
    return odv, of, md


def fmin(F, odv, fs, eps=0.01, vb=0.1):
    labs = [sim.classify(odv[i], fs, vb, eps_frac=eps)[0] for i in range(len(F))]
    fe, _ = sim.fmin_K(F, labs)
    fp, _ = sim.fmin_K(F[sim.N_EXT_BELOW:], labs[sim.N_EXT_BELOW:])
    return fp, fe, "".join("H" if x else "." for x in labs)


def main():
    out = {}
    F = sim.EXT_FORCES
    bs = sim.BETAS + [sim.BETA_EXTRA]
    # 1. maintenance scan: ideal Helmholtz start, both friction rules
    for hyst in (True, False):
        rows = {}
        for b in bs:
            odv, of, md = run("C0", b, "ideal", hyst)
            fp, fe, lab = fmin(F, odv, sim.FS_BASE)
            rows[f"{b}"] = dict(Fmin_plan=fp, Fmin_ext=fe, F_S=sim.F_S(b, 0.1, "C0"), Q=None if fe is None else fe / sim.F_S(b, 0.1, "C0"), labels=lab)
            print("IDEAL hyst", hyst, b, rows[f"{b}"], flush=True)
        x = [math.log(b) for b in sim.BETAS if rows[f"{b}"]["Fmin_ext"] is not None]
        y = [math.log(rows[f"{b}"]["Fmin_ext"]) for b in sim.BETAS if rows[f"{b}"]["Fmin_ext"] is not None]
        out[f"ideal_start_scan|hyst={hyst}"] = dict(rows=rows, b_beta=float(np.polyfit(x, y, 1)[0]) if len(x) > 1 else None)
    # 2. band limit (debug 2), beta 0.1
    for fs in (96000.0, 192000.0, 384000.0):
        for start in ("ramp", "ideal"):
            odv, of, md = run("C0", 0.1, start, True, fs=fs)
            fp, fe, lab = fmin(F, odv, fs)
            out[f"bandlimit|{start}|fs={int(fs)}"] = dict(N=md["N"], Fmin_plan=fp, Fmin_ext=fe, labels=lab)
            print("BAND", start, fs, out[f"bandlimit|{start}|fs={int(fs)}"], flush=True)
    # 3. friction rule x label threshold (debug 3), beta 0.1, fs 96 k
    keep = {}
    for start in ("ramp", "ideal"):
        for hyst in (True, False):
            odv, of, md = run("C0", 0.1, start, hyst)
            keep[(start, hyst)] = (odv, of)
            for eps in (0.01, 0.05, 0.2):
                fp, fe, lab = fmin(F, odv, sim.FS_BASE, eps=eps)
                out[f"rule|{start}|hyst={hyst}|eps={eps}"] = dict(Fmin_plan=fp, Fmin_ext=fe, labels=lab)
                print("RULE", start, hyst, eps, fp, fe, lab, flush=True)
    # 4. stick-phase friction force in clean (ideal-start) Helmholtz vs Schelleng's assumed variation
    odv, of = keep[("ideal", True)]
    T0n = sim.FS_BASE / sim.F0
    for fi in (0.335, 0.611, 2.03):
        i = int(np.argmin(abs(F - fi))); dv = odv[i]; f = of[i]; st = (dv == 0)
        ed = np.flatnonzero(np.diff(st.astype(int)))
        segs = [(a + 1, b + 1) for a, b in zip(ed[:-1], ed[1:]) if st[a + 1] and (b - a) > 0.5 * T0n]
        rng = [f[a:b].max() - f[a:b].min() for a, b in segs]
        rip = [np.std(f[a:b] - np.polyval(np.polyfit(np.arange(b - a), f[a:b], 1), np.arange(b - a))) for a, b in segs]
        out[f"stickforce|F={fi}"] = dict(F=float(F[i]), n_stick=len(segs), range_median=float(np.median(rng)), ripple_rms_median=float(np.median(rip)),
                                         schelleng_stick_variation=sim.F_S(0.1, 0.1, "C0") * sim.DMU, dmu_F=sim.DMU * float(F[i]))
        print("STICKFORCE", out[f"stickforce|F={fi}"], flush=True)
    json.dump(out, open(OUT, "w"), indent=1)
    # figure
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.6))
    bb = np.geomspace(0.018, 0.22, 50)
    ax[0].plot(bb, [sim.F_S(b, 0.1, "C0") for b in bb], "k--", label="F_S (C0: Z/R = pi zeta1)")
    ax[0].plot(bb, [sim.F_S(b, 0.1, "C1") for b in bb], "k:", label="F_S (C1-C3: (6/pi) S10)")
    for hyst, c in ((True, "#1f77b4"), (False, "#d62728")):
        R = out[f"ideal_start_scan|hyst={hyst}"]
        xs = sorted(bs); ys = [R["rows"][f"{b}"]["Fmin_ext"] for b in xs]
        ax[0].plot([x for x, y in zip(xs, ys) if y], [y for y in ys if y], "-o", color=c,
                   label=f"C0 maintenance (ideal start), {'hysteresis' if hyst else 'stick-if-possible'} rule, b={R['b_beta']:.2f}")
    ax[0].plot([0.1], [sim.F_MEAS_ONSET], "*", color="purple", ms=13, label="measured A T1 onset 0.40 N")
    ax[0].plot([0.1], [sim.F_MEAS_WINDOW], "*", color="orange", ms=13, label="measured A T1 window 0.27 N")
    ax[0].axhline(0.005, color="0.6", lw=0.8)
    ax[0].text(0.02, 0.97, "PLAN ramp protocol: no Helmholtz (K=2) at any force\n0.00045-5 N for any beta -> C0 has no F_min", transform=ax[0].transAxes, va="top", fontsize=7, color="r")
    ax[0].set_xscale("log"); ax[0].set_yscale("log"); ax[0].set_xlabel("beta"); ax[0].set_ylabel("F_min (N), v = 0.1 m/s")
    from matplotlib.ticker import NullFormatter, FixedLocator, FixedFormatter
    ax[0].xaxis.set_minor_formatter(NullFormatter()); ax[0].xaxis.set_major_locator(FixedLocator(sim.BETAS)); ax[0].xaxis.set_major_formatter(FixedFormatter([str(b) for b in sim.BETAS]))
    ax[0].set_title("C0 validation FAILED (v = 0.1 m/s)", fontsize=9); ax[0].legend(fontsize=6); ax[0].grid(True, which="both", alpha=0.3)
    i = int(np.argmin(abs(F - 2.03))); n3 = int(3 * T0n); t = np.arange(n3) / sim.FS_BASE * 1e3
    ax[1].plot(t, keep[("ramp", True)][0][i][:n3], lw=0.7, label="PLAN ramp protocol (0.2 s after plateau)")
    ax[1].plot(t, keep[("ideal", True)][0][i][:n3], lw=0.7, label="ideal Helmholtz start (0.2 s)")
    ax[1].set_xlabel("ms"); ax[1].set_ylabel("bow-point relative velocity (m/s)"); ax[1].set_title("C0, beta 0.1, F = 2.03 N", fontsize=9); ax[1].legend(fontsize=7)
    ax[2].plot(t, keep[("ideal", True)][1][i][:n3] / F[i], lw=0.7, label="friction force / F, ideal start")
    ax[2].plot(t, keep[("ramp", True)][1][i][:n3] / F[i], lw=0.5, alpha=0.6, label="friction force / F, ramp")
    for yv in (sim.MUS, sim.MUD):
        ax[2].axhline(yv, color="k", lw=0.6, ls="--")
    ax[2].set_xlabel("ms"); ax[2].set_title("stick-phase force ripple (C0, beta 0.1, F = 2.03 N)", fontsize=9); ax[2].legend(fontsize=7)
    fig.tight_layout(); fig.savefig(os.path.join(sim.HERE, "fig_sim.png"), dpi=140)
    print("saved")


if __name__ == "__main__":
    main()
