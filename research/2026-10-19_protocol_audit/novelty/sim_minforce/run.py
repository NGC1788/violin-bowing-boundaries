"""Stages: checks | c0 | conv | all | summary | fig   (python run.py <stage>)
Results accumulate in results_sim.json; each stage appends a short block to RESULT.md."""
import json
import math
import os
import sys
import time

import numpy as np

import sim

OUT = os.path.join(sim.HERE, "results_sim.json")
RES_MD = os.path.join(sim.HERE, "RESULT.md")


def load():
    return json.load(open(OUT)) if os.path.exists(OUT) else {}


def save(d):
    json.dump(d, open(OUT, "w"), indent=1)


def md_append(text):
    with open(RES_MD, "a") as fh:
        fh.write(text.rstrip() + "\n\n")


def key(cfg, v):
    return f"{cfg}|v={v}"


def run_beta(cfg, v, beta, fs=sim.FS_BASE, nmodes=None, forces=None, wave_path=None):
    forces = sim.EXT_FORCES if forces is None else np.asarray(forces, float)
    t0 = time.time()
    dv, md = sim.simulate(cfg, v, beta, forces, fs=fs, nmodes=nmodes)
    wall = time.time() - t0
    labs, labs0, fr, mp, nt = [], [], [], [], []
    for i in range(len(forces)):
        ok, info = sim.classify(dv[i], fs, v)
        ok0, _ = sim.classify(dv[i], fs, v, eps_frac=0.0)
        labs.append(ok); labs0.append(ok0)
        fr.append(round(info["frac"], 3)); mp.append(None if info["mean_period_rel"] is None else round(info["mean_period_rel"], 4))
        nt.append(info["n_trans"])
    out = dict(beta=beta, fs=fs, N=md["N"], Y_modal_x2Z=md["Ymod"] * 2 * sim.Z, Y_eff_x2Z=(md["Ymod"] + md["dY"]) * 2 * sim.Z, Y_target_x2Z=md["Ytarget"] * 2 * sim.Z,
               forces=[float(x) for x in forces], labels=labs, labels_eps0=labs0, frac=fr, mean_period_rel=mp,
               n_trans=nt, wall_s=round(wall, 3), F_S=sim.F_S(beta, v, cfg))
    if wave_path is not None:
        np.savez_compressed(wave_path, dv=dv.astype(np.float32), forces=forces, fs=fs, labels=np.array(labs))
    return out


def fmins(r):
    f = np.array(r["forces"]); lab = r["labels"]; lab0 = r["labels_eps0"]
    ext, iext = sim.fmin_K(f, lab)
    pf = f[sim.N_EXT_BELOW:]
    plan, ip = sim.fmin_K(pf, lab[sim.N_EXT_BELOW:])
    e0, _ = sim.fmin_K(pf, lab0[sim.N_EXT_BELOW:])
    r["Fmin_plan"] = plan
    r["censored_plan"] = (ip == 0)
    r["Fmin_ext"] = ext
    r["censored_ext"] = (iext == 0)
    r["Fmin_plan_eps0"] = e0
    r["Q_plan"] = None if plan is None else plan / r["F_S"]
    r["Q_ext"] = None if ext is None else ext / r["F_S"]
    return r


def run_set(cfg, v, betas, store, wave=False):
    k = key(cfg, v)
    store.setdefault("runs", {}).setdefault(k, {})
    for b in betas:
        wp = os.path.join(sim.HERE, f"wave_{cfg}_v{v}_b{b}.npz") if (wave and b == sim.BETA_EXTRA) else None
        r = fmins(run_beta(cfg, v, b, wave_path=wp))
        store["runs"][k][f"{b}"] = r
        print(f"{k} beta={b}: N={r['N']} Fmin_plan={r['Fmin_plan']} (cens {r['censored_plan']}) Fmin_ext={r['Fmin_ext']} "
              f"F_S={r['F_S']:.4g} Q_plan={r['Q_plan']} eps0:{r['Fmin_plan_eps0']} wall={r['wall_s']}s", flush=True)
        print("   labels:", "".join("H" if x else "." for x in r["labels"]), flush=True)
    save(store)


def slope(betas, F):
    x = [math.log(b) for b, f in zip(betas, F) if f is not None]
    y = [math.log(f) for f in F if f is not None]
    if len(x) < 2:
        return None
    return float(np.polyfit(x, y, 1)[0])


def cfg_summary(store, cfg, v):
    R = store["runs"][key(cfg, v)]
    bs = sim.BETAS
    Fp = [R[f"{b}"]["Fmin_plan"] for b in bs]
    Fe = [R[f"{b}"]["Fmin_ext"] for b in bs]
    cens = [R[f"{b}"]["censored_plan"] for b in bs]
    r01 = R.get(f"{sim.BETA_EXTRA}")
    F08, F12 = R["0.08"]["Fmin_plan"], R["0.12"]["Fmin_plan"]
    Fint = None if (F08 is None or F12 is None) else math.exp(np.interp(math.log(0.1), [math.log(0.08), math.log(0.12)], [math.log(F08), math.log(F12)]))
    FS01 = sim.F_S(0.1, v, cfg)
    return dict(Fmin_plan=Fp, Fmin_ext=Fe, censored_plan=cens,
                Q_plan=[None if f is None else f / sim.F_S(b, v, cfg) for b, f in zip(bs, Fp)],
                b_beta_plan=slope(bs, Fp), b_beta_ext=slope(bs, Fe),
                b_beta_plan_uncensored=slope([b for b, c in zip(bs, cens) if not c], [f for f, c in zip(Fp, cens) if not c]),
                Fmin_01=None if r01 is None else r01["Fmin_plan"], Fmin_01_ext=None if r01 is None else r01["Fmin_ext"],
                Q_01=None if (r01 is None or r01["Fmin_plan"] is None) else r01["Fmin_plan"] / FS01,
                Q_01_interp=None if Fint is None else Fint / FS01, F_S_01=FS01)


def stage_checks():
    st = load()
    ck = {}
    ck["constants"] = dict(Z=sim.Z, L=sim.L, c=sim.CW, B_inharm=sim.B_INH, mode_mass=sim.MODE_MASS, zeta1=sim.ZETA1,
                           damping_powerlaw_a=sim.DA, damping_powerlaw_p=sim.DP, S10_results_json=sim.S10,
                           S10_recomputed=float(np.sum(sim.zeta_measured(np.arange(1, 11)) / np.arange(1, 11))),
                           ZR_eff=sim.ZR_EFF, ZR_C0=sim.ZR_C0, F_meas_onset=sim.F_MEAS_ONSET, F_meas_window=sim.F_MEAS_WINDOW)
    print(json.dumps(ck["constants"], indent=1))
    # admittance check
    adm = {}
    for cfg in ("C0", "C1", "C3"):
        for b in sim.BETAS + [sim.BETA_EXTRA]:
            md = sim.build_modes(cfg, b)
            adm[f"{cfg}|{b}"] = dict(N=md["N"], fmax=float(md["f"][-1]), Ymod_x2Z=md["Ymod"] * 2 * sim.Z, Yeff_x2Z=(md["Ymod"] + md["dY"]) * 2 * sim.Z, Ytarget_x2Z=md["Ytarget"] * 2 * sim.Z)
    # numerical infinite-sum check of the flexible-string ZOH admittance (should be 1/(2Z))
    nn = np.arange(1, 2_000_001, dtype=float)
    th = 2 * np.pi * sim.F0 / sim.FS_BASE
    for b in (0.02, 0.1, 0.2):
        terms = np.sin(np.pi * nn * b) ** 2 * np.sin(nn * th) / (nn * 2 * np.pi * sim.F0 * sim.MODE_MASS)
        cs = np.cumsum(terms)
        adm[f"flex_infinite_sum|{b}"] = float(np.mean(cs[-200000:]) * 2 * sim.Z)
    ck["admittance"] = adm
    for k_, v_ in adm.items():
        print("ADM", k_, v_)
    # free decay check
    fd = {}
    for cfg in ("C1", "C3"):
        md = sim.build_modes(cfg, 0.0731)
        q0 = 1e-3 * np.sin(np.pi * md["n"] * 0.137) / md["n"] ** 2
        y = np.zeros(int(1.2 * sim.FS_BASE))
        sim.free_decay(q0, md["phi"], md["a11"], md["a12"], md["a21"], md["a22"], y)
        t = np.arange(y.size) / sim.FS_BASE
        Lw = int(round(10 / sim.F0 * sim.FS_BASE))
        win = np.hanning(Lw); win /= win.sum()
        for n in (1, 2, 3, 5, 10, 17, 30, 60):
            fn, zn = md["f"][n - 1], md["zeta"][n - 1]
            zc = y * np.exp(-2j * np.pi * fn * t)
            zf = np.convolve(zc, win, mode="valid")
            tf = (np.arange(zf.size) + (Lw - 1) / 2) / sim.FS_BASE
            sig = zn * 2 * np.pi * fn
            ta = Lw / sim.FS_BASE
            tb = min(ta + 3 / sig, tf[-1])
            sel = (tf >= ta) & (tf <= tb)
            A = np.polyfit(tf[sel], np.log(np.abs(zf[sel])), 1)[0]
            ph = np.polyfit(tf[sel], np.unwrap(np.angle(zf[sel])), 1)[0]
            fm = fn + ph / (2 * np.pi)
            fd[f"{cfg}|n={n}"] = dict(f_model=float(fn), f_meas=float(fm), zeta_model=float(zn), zeta_meas=float(-A / (2 * np.pi * fm)))
            print("DECAY", cfg, n, fd[f"{cfg}|n={n}"], flush=True)
    ck["free_decay"] = fd
    # timing + one Helmholtz sanity run
    for cfg in ("C0", "C2"):
        r = run_beta(cfg, 0.1, 0.1, forces=[0.05, 0.1, 0.2, 0.4, 0.8, 1.6, 3.2, 0.05])   # first call includes JIT
        r = run_beta(cfg, 0.1, 0.1, forces=[0.05, 0.1, 0.2, 0.4, 0.8, 1.6, 3.2, 0.05])
        ck[f"sanity_{cfg}"] = dict(forces=r["forces"], labels=r["labels"], frac=r["frac"], mean_period_rel=r["mean_period_rel"], wall_s_8runs=r["wall_s"])
        print("SANITY", cfg, ck[f"sanity_{cfg}"], flush=True)
    st["checks"] = ck
    save(st)
    a = adm
    md_append("## Stage 1 — unit checks (auto)\n"
              f"- L = {sim.L:.4f} m, Z = {sim.Z:.4f} kg/s, B = {sim.B_INH:.3e}, zeta1 = {sim.ZETA1:.3e}; damping power law p = {sim.DP:.3f} (a = {sim.DA:.3f}); "
              f"S10 recomputed {ck['constants']['S10_recomputed']:.6g} vs results.json {sim.S10:.6g}\n"
              f"- fs = 96 kHz, modes f_n <= 0.3066 fs = 29.4 kHz (no residual term): N = {a['C1|0.1']['N']} (C0-C2), {a['C3|0.1']['N']} (C3). "
              f"Modal-sum bow admittance x 2Z (beta 0.02/0.1/0.2): C1 {a['C1|0.02']['Ymod_x2Z']:.3f}/{a['C1|0.1']['Ymod_x2Z']:.3f}/{a['C1|0.2']['Ymod_x2Z']:.3f}; "
              f"C3 modal {a['C3|0.02']['Ymod_x2Z']:.3f}/{a['C3|0.1']['Ymod_x2Z']:.3f}/{a['C3|0.2']['Ymod_x2Z']:.3f} vs continuous stiff-string ZOH value {a['C3|0.02']['Ytarget_x2Z']:.3f}/{a['C3|0.1']['Ytarget_x2Z']:.3f}/{a['C3|0.2']['Ytarget_x2Z']:.3f}; "
              f"numerical infinite ZOH sum x 2Z (flexible) {a['flex_infinite_sum|0.02']:.4f}/{a['flex_infinite_sum|0.1']:.4f}/{a['flex_infinite_sum|0.2']:.4f}\n"
              "- free decay (C1): " + "; ".join(f"n={k.split('=')[1]}: f {v['f_meas']:.2f}/{v['f_model']:.2f} Hz, zeta {v['zeta_meas']:.3e}/{v['zeta_model']:.3e}" for k, v in fd.items() if k.startswith("C1")) + "\n"
              "- free decay (C3): " + "; ".join(f"n={k.split('=')[1]}: f {v['f_meas']:.2f}/{v['f_model']:.2f}, zeta {v['zeta_meas']:.3e}/{v['zeta_model']:.3e}" for k, v in fd.items() if k.startswith("C3")) + "\n"
              f"- sanity beta 0.1, v 0.1, F = 0.05..3.2 N: C0 labels {ck['sanity_C0']['labels'][:7]}, C2 labels {ck['sanity_C2']['labels'][:7]}; wall for 8 parallel runs {ck['sanity_C0']['wall_s_8runs']} s")


def stage_c0():
    st = load()
    run_set("C0", 0.1, sim.BETAS + [sim.BETA_EXTRA], st, wave=True)
    s = cfg_summary(st, "C0", 0.1)
    Qp = s["Q_01"]
    bb = s["b_beta_plan"]
    bbe = s["b_beta_ext"]
    passQ = Qp is not None and 0.5 <= Qp <= 2.0
    passb = bb is not None and abs(bb + 2) <= 0.3
    passbe = bbe is not None and abs(bbe + 2) <= 0.3
    s.update(pass_Q=passQ, pass_b_plan=passb, pass_b_ext=passbe)
    st.setdefault("summary", {})["C0|v=0.1"] = s
    save(st)
    print(json.dumps(s, indent=1))
    md_append("## Stage 2 — C0 validation (auto)\n"
              f"- F_min (plan grid) per beta {sim.BETAS}: {[None if f is None else round(f, 5) for f in s['Fmin_plan']]} (censored at grid floor: {s['censored_plan']}); extended grid: {[None if f is None else round(f, 5) for f in s['Fmin_ext']]}\n"
              f"- Q_plan per beta: {[None if q is None else round(q, 2) for q in s['Q_plan']]}\n"
              f"- beta = 0.1 direct: F_min {s['Fmin_01']}, F_S {s['F_S_01']:.5f}, Q_sim(0.1) = {Qp} (interp 0.08/0.12: {s['Q_01_interp']})\n"
              f"- b_beta plan grid {bb}, extended grid {bbe}, plan uncensored only {s['b_beta_plan_uncensored']}\n"
              f"- pass: Q within x2 {passQ}; b_beta (plan) {passb}; b_beta (ext) {passbe}")


def stage_conv():
    st = load()
    cv = {}
    for b in (0.03, 0.12):
        base = fmins(run_beta("C1", 0.1, b))
        N0 = base["N"]
        Fb = base["Fmin_ext"]
        fine = Fb * 1.025 ** np.arange(-14, 15)
        res = {}
        for name, fs, nm in (("base", 96000.0, None), ("half_dt", 192000.0, N0), ("half_dt_2N", 192000.0, 2 * N0)):
            g = base if name == "base" else fmins(run_beta("C1", 0.1, b, fs=fs, nmodes=nm))
            fr = run_beta("C1", 0.1, b, fs=fs, nmodes=nm, forces=fine)
            ff, _ = sim.fmin_K(np.array(fr["forces"]), fr["labels"])
            res[name] = dict(fs=fs, N=g["N"], Fmin_grid=g["Fmin_ext"], Fmin_fine=ff, fine_labels="".join("H" if x else "." for x in fr["labels"]),
                             grid_labels="".join("H" if x else "." for x in g["labels"]), wall_grid=g["wall_s"])
            print("CONV", b, name, res[name], flush=True)
        f0_ = res["base"]["Fmin_fine"]
        for name in ("half_dt", "half_dt_2N"):
            f1 = res[name]["Fmin_fine"]
            res[name]["rel_change_fine"] = None if (f0_ is None or f1 is None) else f1 / f0_ - 1
            g0, g1 = res["base"]["Fmin_grid"], res[name]["Fmin_grid"]
            res[name]["rel_change_grid"] = None if (g0 is None or g1 is None) else g1 / g0 - 1
        cv[f"{b}"] = res
    worst = max(abs(cv[b][n]["rel_change_fine"]) for b in cv for n in ("half_dt", "half_dt_2N") if cv[b][n]["rel_change_fine"] is not None)
    st["convergence"] = dict(results=cv, worst_rel_change_fine=worst, pass_=worst < 0.10)
    save(st)
    md_append("## Stage 3 — convergence, C1, v = 0.1 (auto)\n" + "\n".join(
        f"- beta {b}: " + "; ".join(f"{n} (fs {int(cv[b][n]['fs'])}, N {cv[b][n]['N']}): grid F_min {cv[b][n]['Fmin_grid']}, fine F_min {None if cv[b][n]['Fmin_fine'] is None else round(cv[b][n]['Fmin_fine'], 5)}"
                            + ("" if n == "base" else f", change fine {cv[b][n]['rel_change_fine']}, grid {cv[b][n]['rel_change_grid']}") for n in cv[b]) for b in cv)
        + f"\n- worst |change| (fine grid, 2.5 % steps) = {worst:.3f} -> pass {worst < 0.10}")


def stage_all():
    st = load()
    for cfg, v in (("C1", 0.1), ("C2", 0.1), ("C3", 0.1), ("C1", 0.05), ("C1", 0.2), ("C2", 0.05), ("C2", 0.2)):
        run_set(cfg, v, sim.BETAS + [sim.BETA_EXTRA], st, wave=(v == 0.1))
        st.setdefault("summary", {})[key(cfg, v)] = cfg_summary(st, cfg, v)
        save(st)
        s = st["summary"][key(cfg, v)]
        md_append(f"## Stage 4 — {cfg}, v = {v} (auto)\n"
                  f"- F_min plan per beta {sim.BETAS}: {[None if f is None else round(f, 5) for f in s['Fmin_plan']]} (censored {s['censored_plan']}); ext {[None if f is None else round(f, 5) for f in s['Fmin_ext']]}\n"
                  f"- Q_plan: {[None if q is None else round(q, 2) for q in s['Q_plan']]}; Q_sim(0.1) direct {s['Q_01']} (F_min {s['Fmin_01']}), interp {s['Q_01_interp']}\n"
                  f"- b_beta plan {s['b_beta_plan']}, ext {s['b_beta_ext']}")


def stage_summary():
    st = load()
    sm = st["summary"]
    bv = {}
    for cfg in ("C1", "C2"):
        per = {}
        for b in sim.BETAS + [sim.BETA_EXTRA]:
            F = [st["runs"][key(cfg, v)][f"{b}"]["Fmin_plan"] for v in (0.05, 0.1, 0.2)]
            per[f"{b}"] = slope([0.05, 0.1, 0.2], F) if all(f is not None for f in F) else None
        vals = [x for k_, x in per.items() if x is not None and k_ != f"{sim.BETA_EXTRA}"]
        bv[cfg] = dict(per_beta=per, mean_six_beta=float(np.mean(vals)) if vals else None, median_six_beta=float(np.median(vals)) if vals else None)
    st["b_v"] = bv
    q1 = sm["C1|v=0.1"]["Q_01"]
    reading = "H1" if (q1 is not None and q1 >= 5) else ("H0" if (q1 is not None and q1 <= 2) else "partial")
    st["reading"] = dict(C1_Q01=q1, verdict=reading,
                         b_beta={c: sm[f"{c}|v=0.1"]["b_beta_plan"] for c in ("C0", "C1", "C2", "C3")},
                         Q01={c: sm[f"{c}|v=0.1"]["Q_01"] for c in ("C0", "C1", "C2", "C3")},
                         Fmin01={c: sm[f"{c}|v=0.1"]["Fmin_01"] for c in ("C0", "C1", "C2", "C3")})
    save(st)
    print(json.dumps(dict(b_v=bv, reading=st["reading"]), indent=1))


def stage_fig():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    st = load()
    fig, ax = plt.subplots(figsize=(7, 5))
    bs = sim.BETAS + [sim.BETA_EXTRA]
    cols = {"C0": "#444444", "C1": "#1f77b4", "C2": "#d62728", "C3": "#2ca02c"}
    lab = {"C0": "C0 Coulomb, zeta1/n", "C1": "C1 measured zeta_n", "C2": "C2 + hyperbolic friction", "C3": "C3 + bending stiffness"}
    for cfg in ("C0", "C1", "C2", "C3"):
        R = st["runs"][key(cfg, 0.1)]
        b_ = sorted(bs)
        F = [R[f"{b}"]["Fmin_plan"] for b in b_]
        c = [R[f"{b}"]["censored_plan"] for b in b_]
        xs = [b for b, f in zip(b_, F) if f is not None]
        ys = [f for f in F if f is not None]
        ax.plot(xs, ys, "-o", color=cols[cfg], label=f"{lab[cfg]} (b={st['summary'][key(cfg, 0.1)]['b_beta_plan']:.2f})" if st['summary'][key(cfg, 0.1)]['b_beta_plan'] is not None else lab[cfg], ms=5)
        for b, f, cc in zip(b_, F, c):
            if f is not None and cc:
                ax.plot([b], [f], "v", color=cols[cfg], ms=9, mfc="none")
    bb = np.geomspace(0.018, 0.22, 50)
    ax.plot(bb, [sim.F_S(b, 0.1, "C0") for b in bb], "--", color=cols["C0"], lw=1, label="F_S, Z/R = pi zeta1 (C0)")
    ax.plot(bb, [sim.F_S(b, 0.1, "C1") for b in bb], "--", color="k", lw=1.5, label="F_S, Z/R_eff = (6/pi) S10 (C1-C3)")
    ax.plot([0.1], [sim.F_MEAS_ONSET], "*", color="purple", ms=14, label=f"measured A T1 onset {sim.F_MEAS_ONSET:.2f} N (b=-1.23)")
    ax.plot([0.1], [sim.F_MEAS_WINDOW], "*", color="orange", ms=14, label=f"measured A T1 window {sim.F_MEAS_WINDOW:.2f} N (b=-1.54)")
    ax.plot(bb, sim.F_MEAS_ONSET * (bb / 0.1) ** -1.23, ":", color="purple", lw=1)
    ax.plot(bb, sim.F_MEAS_WINDOW * (bb / 0.1) ** -1.54, ":", color="orange", lw=1)
    ax.axhline(0.005, color="0.6", lw=0.8)
    ax.text(0.019, 0.0053, "plan force-grid floor", fontsize=7, color="0.4")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("beta"); ax.set_ylabel("F_min (N)  [v = 0.1 m/s]")
    ax.set_title("Simulated minimum bow force, string A T1 (open v = censored at grid floor)", fontsize=9)
    ax.legend(fontsize=7, loc="lower left")
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(sim.HERE, "fig_sim.png"), dpi=150)
    print("fig saved")


if __name__ == "__main__":
    for stg in sys.argv[1:]:
        t0 = time.time()
        globals()[f"stage_{stg}"]()
        print(f"[stage {stg} done in {time.time() - t0:.1f} s]", flush=True)
