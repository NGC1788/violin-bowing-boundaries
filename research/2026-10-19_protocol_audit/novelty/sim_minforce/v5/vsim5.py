"""Simulation v5 (PLAN_v5.md): v4 string and filter, but the measurement's start: string at rest, linear bow-speed
ramp, label 0.2-0.4 s after the plateau starts. Claude, 2026-10-21."""
import json, math, os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
os.environ.setdefault("NUMBA_CACHE_DIR", os.path.join(HERE, "nbcache"))
for d in ("verify", "v2", "v3", "v4"):
    sys.path.insert(0, os.path.join(HERE, "..", d))
import numpy as np
from numba import njit, prange
import vsim as VS
import vsim3 as V3
import vsim4 as V4

Z, F0 = VS.Z, VS.F0


@njit(cache=True)
def run_one(F, db2, dn2, h, L, ldel, vb, acc_b, fs, nsteps, i0, Lx, odv):
    Y = 1.0 / (2.0 * Z)
    xb = np.zeros(Lx); on = np.zeros(dn2)
    slipping = False; s = 1.0
    for k in range(nsteps):
        vbow = acc_b * (k + 1) / fs
        if vbow > vb:
            vbow = vb
        acc = 0.0
        for j in range(L):
            acc += h[j] * xb[(k - ldel - j) % Lx]
        vib = -acc
        inn = k % dn2
        vin = -on[inn]
        w = vib + vin - vbow
        f, dv, slipping, s = VS.fric(w, Y, F, slipping, s, False)
        xb[k % Lx] = vin + Y * f
        on[inn] = vib + Y * f
        r = k - i0
        if r >= 0 and r < odv.size:
            odv[r] = dv


@njit(parallel=True, cache=True)
def run_batch(Fs, db2, dn2, h, L, ldel, vb, acc_b, fs, nsteps, i0, Lx, odv):
    for i in prange(Fs.size):
        run_one(Fs[i], db2, dn2, h, L, ldel, vb, acc_b, fs, nsteps, i0, Lx, odv[i])


def fmin(beta, h, tau, acc_b, D=4000, vb=0.1, win=(0.2, 0.4)):
    fs = F0 * D; db2 = int(round(beta * D)); dn2 = D - db2
    assert abs(db2 - beta * D) < 1e-9
    L = len(h); ldel = db2 - int(round(tau)); assert ldel >= 1
    Fs = VS.EXT; n = Fs.size; Lx = ldel + L + 2
    tpl = vb / acc_b
    i0 = int(round((tpl + win[0]) * fs)); i1 = int(round((tpl + win[1]) * fs))
    odv = np.zeros((n, i1 - i0))
    run_batch(Fs, db2, dn2, np.asarray(h, float), L, ldel, vb, acc_b, fs, i1, i0, Lx, odv)
    labs = [V3.label3(odv[i], fs, beta, vb) for i in range(n)]
    H = [l[0] for l in labs]
    return VS.fminK(Fs, H, K=2), "".join("H" if l[0] else ("c" if l[1]["classifier"] else ".") for l in labs)


def main():
    betas = [0.02, 0.03, 0.05, 0.08, 0.12, 0.2]
    a, p = V4.powerlaw(); a_lo, p_lo = V4.powerlaw(11)
    h, q = V4.design_minphase(4000, 1024, a, p); assert q["gate"]
    h_lo, q_lo = V4.design_minphase(4000, 1024, a_lo, p_lo); assert q_lo["gate"]
    g = math.exp(-2 * math.pi * math.exp(a))
    res = {}

    def scan(name, hh, tau, acc_b, bs, vb=0.1):
        rows = {}
        for b in bs:
            fm, pat = fmin(b, hh, tau, acc_b, vb=vb)
            rows[str(b)] = dict(fmin=fm, labels=pat)
            print(f"{name} v {vb} β {b}: F_min {fm}  {pat}", flush=True)
        return rows

    for name, hh, tau, acc_b in (("R1", h, q["phase_delay_f0"], 0.104), ("R1-fast", h, q["phase_delay_f0"], 1.0),
                                 ("R1-low", h_lo, q_lo["phase_delay_f0"], 0.104), ("R0", np.array([g]), 0.0, 0.104)):
        rows = scan(name, hh, tau, acc_b, betas + [0.1])
        res[name] = dict(rows=rows, b_beta=V4.slope(betas, [rows[str(b)]["fmin"] for b in betas]),
                         n_beta_defined=sum(1 for b in betas if rows[str(b)]["fmin"]))
    sp = {"0.05": scan("R1", h, q["phase_delay_f0"], 0.088, [0.1], vb=0.05)["0.1"]["fmin"], "0.1": res["R1"]["rows"]["0.1"]["fmin"],
          "0.2": scan("R1", h, q["phase_delay_f0"], 0.417, [0.1], vb=0.2)["0.1"]["fmin"]}
    okv = [float(v) for v in sp if sp[v]]
    res["R1_speed"] = dict(fmin=sp, b_v=float(np.polyfit(np.log(okv), np.log([sp[str(v)] for v in okv]), 1)[0]) if len(okv) >= 2 else None)
    h8, q8 = V4.design_minphase(8000, 2048, a, p)
    res["R1_conv_D8000"] = {str(b): fmin(b, h8, q8["phase_delay_f0"], 0.104, D=8000)[0] for b in (0.03, 0.12)}
    bb = res["R1"]["b_beta"]; f1 = res["R1"]["rows"]["0.1"]["fmin"]
    res["readings"] = dict(
        b_beta_R1=bb,
        b_reading=None if bb is None else ("ramp restores the measured flattening (b >= -2.0)" if bb >= -2.0 else ("does not (b <= -3.0)" if bb <= -3.0 else "partial")),
        F_min_R1_beta0p1=f1, level_reading=None if f1 is None else ("matches (0.2-0.55 N)" if 0.2 <= f1 <= 0.55 else f"ratio to 0.33 N = {f1 / 0.33:.2f}"),
        first_beta_sustained=min([b for b in betas if res["R1"]["rows"][str(b)]["fmin"]], default=None),
        b_v=res["R1_speed"]["b_v"], v4_maintenance=dict(b_beta=-4.12, F_min_beta0p1=0.335),
        measured=dict(b_onset=-1.23, b_window=-1.54, F_onset=0.40, F_window=0.27))
    json.dump(res, open(os.path.join(HERE, "results_v5.json"), "w"), indent=1, default=float)
    print(json.dumps(res["readings"], indent=1, default=float)); print("conv", res["R1_conv_D8000"], "speed", res["R1_speed"])
    print({k: (res[k]["b_beta"], res[k]["n_beta_defined"]) for k in ("R1", "R1-fast", "R1-low", "R0")})


if __name__ == "__main__":
    main()
