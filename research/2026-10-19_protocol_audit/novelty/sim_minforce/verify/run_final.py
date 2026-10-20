import json, math, os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); sys.path.insert(1, os.path.dirname(HERE)); sys.dont_write_bytecode = True
import numpy as np, vsim as V, sim
out = json.load(open(os.path.join(HERE, "indep.json")))
BETAS = [0.02, 0.03, 0.05, 0.08, 0.12, 0.2]
S10 = float(np.sum(V.zeta_c1(np.arange(1, 11)) / np.arange(1, 11))); FS1 = lambda b: V.Z * (6 / math.pi * S10) * 0.1 / (2 * b ** 2 * 0.6)
def labs(odv, fs): return [V.label(odv[i], fs)[0] for i in range(odv.shape[0])]
def st(dv):
    s = np.abs(dv) > 1e-3; e = np.flatnonzero(np.diff(np.r_[0, s.astype(int), 0])); r = e[1::2] - e[0::2]
    ok, inf = V.label(dv, 96000.0 if dv.size < 19500 else 97800.0)
    return dict(slip_frac=round(float(s.mean()), 3), n_slip=int(r.size), med_slip=None if r.size == 0 else float(np.median(r)), max_slip=None if r.size == 0 else int(r.max()), frac=inf.get("frac"), relP=inf.get("relP"), H=ok)
# A. WG C0 ramp, stick-if-possible, all betas, both losses; and long plateau (label 3.0-3.2 s after plateau)
for loss in ("dist", "bridge"):
    for win, tag in (((0.2, 0.4), "plan"), ((3.0, 3.2), "late3s")):
        Fm = {}; L = {}
        for b in BETAS + [0.1]:
            odv, of, fs = V.wg_run(b, V.EXT, loss=loss, hyst=False, win=win)
            lb = labs(odv, fs); Fm[f"{b}"] = V.fminK(V.EXT, lb); L[f"{b}"] = "".join("H" if x else "." for x in lb)
        d = dict(Fmin=Fm, labels=L, Q={k: (None if v is None else v / V.F_S(float(k))) for k, v in Fm.items()}, b_beta=V.slope(BETAS, [Fm[f"{b}"] for b in BETAS]))
        out[f"WG_C0_ramp_{loss}_nohyst_{tag}"] = d
        print("WG ramp nohyst", loss, tag, "Q", {k: (None if v is None else round(v, 2)) for k, v in d["Q"].items()}, "b", d["b_beta"], flush=True)
        print("   ", L["0.1"], flush=True)
odv, of, fs = V.wg_run(0.1, V.EXT, loss="dist", hyst=False)
out["WG_C0_ramp_dist_nohyst_b0.1_stats"] = [dict(F=float(V.EXT[i]), **st(odv[i])) for i in range(8, 32, 3)]
for r in out["WG_C0_ramp_dist_nohyst_b0.1_stats"]: print("WGSTAT", r, flush=True)
# B. own modal C1 ramp with stick-if-possible rule (rule sensitivity), 96k
PF = V.EXT[8:]
for b in (0.05, 0.1, 0.12):
    odv, of, md = V.mod_run("C1", b, PF, hyst=False)
    lb = labs(odv, 96000.0); fm = V.fminK(PF, lb)
    out[f"MOD_C1_ramp_96k_nohyst|{b}"] = dict(Fmin=fm, Q=None if fm is None else fm / FS1(b), labels="".join("H" if x else "." for x in lb))
    print("C1 nohyst", b, out[f"MOD_C1_ramp_96k_nohyst|{b}"], flush=True)
# C. raw-waveform check of C1 (hysteresis) labels around F_min at beta 0.1 and 0.12
for b in (0.1, 0.12):
    z = np.load(os.path.join(HERE, f"mod_C1_b{b}.npz"))
    for i in range(13, 21):
        r = dict(F=round(float(z["forces"][i]), 4), **st(z["dv"][i].astype(float)), builder=bool(sim.classify(z["dv"][i].astype(float), 96000.0, 0.1)[0]))
        out.setdefault(f"C1_wave_stats|{b}", []).append(r); print("C1WAVE", b, r, flush=True)
print("C1 builder-vs-mine max|dv diff|:", {b: out[f"BUILDER_C1_ramp|{b}"]["max_abs_dv_diff_vs_mine"] for b in (0.05, 0.1, 0.12)})
json.dump(out, open(os.path.join(HERE, "indep.json"), "w"), indent=1)
