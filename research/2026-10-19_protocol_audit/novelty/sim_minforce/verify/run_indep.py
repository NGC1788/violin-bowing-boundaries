"""Check 4/5: independent re-implementation runs.  python run_indep.py"""
import json
import math
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
os.environ.setdefault("NUMBA_CACHE_DIR", os.path.join(HERE, "nbcache"))
sys.dont_write_bytecode = True
sys.path.insert(0, HERE)
sys.path.insert(1, os.path.dirname(HERE))
import numpy as np
import vsim as V
import sim  # builder, read-only, only for the side-by-side C1 comparison and its classifier

OUT = os.path.join(HERE, "indep.json")
out = json.load(open(OUT)) if os.path.exists(OUT) else {}
BETAS = [0.02, 0.03, 0.05, 0.08, 0.12, 0.2]
S10 = float(np.sum(V.zeta_c1(np.arange(1, 11)) / np.arange(1, 11)))
ZR_EFF = 6 / math.pi * S10


def FS_C1(b, v=0.1):
    return V.Z * ZR_EFF * v / (2 * b ** 2 * (V.MUS - V.MUD))


def runstats(dv, fs):
    slip = np.abs(dv) > 0.001
    e = np.flatnonzero(np.diff(np.r_[0, slip.astype(int), 0]))
    r = e[1::2] - e[0::2]
    return dict(n_slip=int(r.size), med_slip=None if r.size == 0 else float(np.median(r)),
                slip_frac=float(slip.mean()), exp_slip_samples=None)


def summarize(name, Fs, odv, fs, beta, FSfun, extra=None):
    labs, labs_b, fr = [], [], []
    for i in range(len(Fs)):
        ok, inf = V.label(odv[i], fs)
        labs.append(ok); fr.append(inf.get("frac"))
        labs_b.append(bool(sim.classify(odv[i], fs, 0.1)[0]))
    fm = V.fminK(Fs, labs); fmb = V.fminK(Fs, labs_b)
    d = dict(beta=beta, forces=[float(x) for x in Fs], labels="".join("H" if x else "." for x in labs),
             labels_builder_classifier="".join("H" if x else "." for x in labs_b),
             Fmin=fm, Fmin_builder_classifier=fmb, F_S=FSfun(beta), Q=None if fm is None else fm / FSfun(beta))
    if extra:
        d.update(extra)
    print(f"{name} b={beta}: Fmin={fm} (bcls {fmb}) F_S={FSfun(beta):.4g} Q={d['Q']}  {d['labels']}", flush=True)
    return d


t0 = time.time()
# ---- Gibbs precursor, builder's C0 basis, free motion of the ideal Helmholtz state (fixed mask) ----
g = {}
for b in (0.05, 0.1, 0.12):
    bm = sim.build_modes("C0", b)
    n = bm["n"]
    p = (-2 * 0.1 / (math.pi * b) / n) * (-1.0) ** n; q = np.zeros_like(p)
    vel = np.empty(2 * 982)
    for k in range(vel.size):
        q, p = bm["a11"] * q + bm["a12"] * p, bm["a21"] * q + bm["a22"] * p
        vel[k] = np.sum(bm["phi"] * p)
    sl = vel < 0
    dil = sl.copy()
    for s_ in (1, 2):
        dil |= np.roll(sl, s_) | np.roll(sl, -s_)
    st = ~dil
    under = float(0.1 - vel[st].min()); over = float(vel[st].max() - 0.1)
    g[f"{b}"] = dict(jump=0.1 / b, undershoot_mps=under, overshoot_mps=over, under_rel_jump=under / (0.1 / b),
                     slip_samples=int(sl.sum() / 2), F_needed_N=2 * V.Z * under / (V.MUS - V.MUD))
    print("GIBBS", b, g[f"{b}"], flush=True)
out["gibbs_builder_C0_basis"] = g

# ---- WG C0, PLAN ramp protocol, all betas + 0.1 ----
for loss in ("dist", "bridge"):
    rows = {}
    for b in BETAS + [0.1]:
        odv, of, fs = V.wg_run(b, V.EXT, loss=loss)
        rows[f"{b}"] = summarize(f"WG-C0-ramp-{loss}", V.EXT, odv, fs, b, V.F_S)
        if b == 0.1:
            np.savez_compressed(os.path.join(HERE, f"wg_C0_{loss}_b0.1.npz"), dv=odv.astype(np.float32), f=of.astype(np.float32), forces=V.EXT, fs=fs)
    Fm = [rows[f"{b}"]["Fmin"] for b in BETAS]
    out[f"WG_C0_ramp_{loss}"] = dict(rows=rows, b_beta=V.slope(BETAS, Fm), Q01=rows["0.1"]["Q"])
    print(loss, "b_beta", out[f"WG_C0_ramp_{loss}"]["b_beta"], "Q01", rows["0.1"]["Q"], flush=True)
# ---- WG C0 ideal start (maintenance), stick-if-possible variant, finer D ----
for hy in (True, False):
    for loss in ("dist", "bridge"):
        rows = {}
        for b in BETAS + [0.1]:
            odv, of, fs = V.wg_run(b, V.EXT, start="ideal", hyst=hy, loss=loss)
            rows[f"{b}"] = summarize(f"WG-C0-ideal-{loss}-hyst{hy}", V.EXT, odv, fs, b, V.F_S)
        out[f"WG_C0_ideal_{loss}_hyst{hy}"] = dict(rows=rows, b_beta=V.slope(BETAS, [rows[f"{b}"]["Fmin"] for b in BETAS]), Q01=rows["0.1"]["Q"])
        print("   b_beta", out[f"WG_C0_ideal_{loss}_hyst{hy}"]["b_beta"], flush=True)
# creeping-slip pathology of Coulomb + hysteresis: per-period exact-stick / creep samples, ideal start, beta 0.1, F 0.2 N
odv, of, fs = V.wg_run(0.1, [0.2], start="ideal", hyst=True, loss="dist", win=(0.0, 0.1))
cr = []
for k in range(9):
    seg = odv[0][k * 1000:(k + 1) * 1000]
    cr.append(dict(period=k, exact_stick=int(np.sum(seg == 0)), creep_0_to_1mm=int(np.sum((seg != 0) & (np.abs(seg) < 1e-3))),
                   slip_gt_1mm=int(np.sum(np.abs(seg) >= 1e-3)), max_creep=float(np.max(np.abs(seg[(seg != 0) & (np.abs(seg) < 1e-3)]), initial=0))))
    print("CREEP", cr[-1], flush=True)
out["creep_WG_ideal_dist_hyst_b0.1_F0.2"] = cr
for b in (0.05, 0.12):
    odv, of, fs = V.wg_run(b, V.EXT, hyst=False)
    out[f"WG_C0_ramp_dist_nohyst|{b}"] = summarize("WG-C0-ramp-nohyst", V.EXT, odv, fs, b, V.F_S)
    odv, of, fs = V.wg_run(b, V.EXT, D=2000)
    out[f"WG_C0_ramp_dist_D2000|{b}"] = summarize("WG-C0-ramp-D2000", V.EXT, odv, fs, b, V.F_S)
json.dump(out, open(OUT, "w"), indent=1)
print(f"[WG done {time.time() - t0:.1f}s]", flush=True)

# ---- own modal C0 (reproduce builder failure) and C1, ramp protocol ----
PF = V.EXT[8:]
for b in (0.05, 0.12):
    odv, of, md = V.mod_run("C0", b, V.EXT[::2])
    out[f"MOD_C0_ramp|{b}"] = summarize("MOD-C0-ramp", V.EXT[::2], odv, 96000.0, b, V.F_S, dict(Y_x2Z=md["Y"] * 2 * V.Z))
for b in (0.05, 0.1, 0.12):
    odv, of, md = V.mod_run("C1", b, PF)
    r = summarize("MOD-C1-ramp-96k", PF, odv, 96000.0, b, FS_C1, dict(Y_x2Z=md["Y"] * 2 * V.Z))
    r["stats"] = [dict(F=float(PF[i]), **runstats(odv[i], 96000.0), **{("lab_" + k): v for k, v in V.label(odv[i], 96000.0)[1].items()}) for i in range(len(PF))]
    out[f"MOD_C1_ramp_96k|{b}"] = r
    np.savez_compressed(os.path.join(HERE, f"mod_C1_b{b}.npz"), dv=odv.astype(np.float32), forces=PF, fs=96000.0)
    # builder's own simulator, same forces, for side-by-side
    bdv, bmd = sim.simulate("C1", 0.1, b, PF)
    out[f"BUILDER_C1_ramp|{b}"] = summarize("BUILDER-C1-ramp", PF, bdv, 96000.0, b, FS_C1)
    out[f"BUILDER_C1_ramp|{b}"]["max_abs_dv_diff_vs_mine"] = float(np.max(np.abs(bdv - odv)))
json.dump(out, open(OUT, "w"), indent=1)
print(f"[MOD 96k done {time.time() - t0:.1f}s]", flush=True)
for b in (0.05, 0.12):
    odv, of, md = V.mod_run("C1", b, PF, fs=192000.0, N=601)
    out[f"MOD_C1_ramp_192k_N601|{b}"] = summarize("MOD-C1-ramp-192k", PF, odv, 192000.0, b, FS_C1, dict(Y_x2Z=md["Y"] * 2 * V.Z))
json.dump(out, open(OUT, "w"), indent=1)
print(f"[all done {time.time() - t0:.1f}s]  S10={S10:.6g} ZR_eff={ZR_EFF:.6g} F_S_C1(0.1)={FS_C1(0.1):.5g}", flush=True)
