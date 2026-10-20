import json, math, os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); sys.dont_write_bytecode = True
import numpy as np, vsim as V
out = json.load(open(os.path.join(HERE, "indep.json")))
S10 = float(np.sum(V.zeta_c1(np.arange(1, 11)) / np.arange(1, 11))); FS1 = lambda b: V.Z * (6 / math.pi * S10) * 0.1 / (2 * b ** 2 * 0.6)
PF = V.EXT[8:]
for b in (0.1, 0.12):
    odv, of, md = V.mod_run("C1", b, PF, hyst=False)
    rows = []
    for i in range(10, 20):
        s = np.abs(odv[i]) > 1e-4; ok, inf = V.label(odv[i], 96000.0)
        rows.append(dict(F=round(float(PF[i]), 4), H=ok, slip_frac=round(float(s.mean()), 3), frac=inf.get("frac"), relP=inf.get("relP")))
    out[f"C1_nohyst_wave_stats|{b}"] = rows
    print("C1nohyst", b, rows, flush=True)
for b in (0.05, 0.12):
    odv, of, md = V.mod_run("C1", b, PF, fs=192000.0, N=601, hyst=False)
    lb = [V.label(odv[i], 192000.0)[0] for i in range(len(PF))]; fm = V.fminK(PF, lb)
    out[f"MOD_C1_ramp_192k_nohyst|{b}"] = dict(Fmin=fm, Q=None if fm is None else fm / FS1(b), labels="".join("H" if x else "." for x in lb))
    print("C1 nohyst 192k", b, out[f"MOD_C1_ramp_192k_nohyst|{b}"], flush=True)
json.dump(out, open(os.path.join(HERE, "indep.json"), "w"), indent=1)
