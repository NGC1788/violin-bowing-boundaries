"""v3b (ADDENDUM_v3b.md): v3 with K = 300 at beta >= 0.08. Claude, 2026-10-21."""
import json, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import vsim3 as V3
VS, V2 = V3.VS, V3.V2
B = [0.08, 0.1, 0.12, 0.2]
res = {"C0b_from_v3": {str(b): json.load(open(os.path.join(HERE, "results_v3.json")))["C0b"]["rows"][str(b)]["fmin"] for b in B}}
for name, zf in (("C1b", VS.zeta_c1), ("C1b-flat", V2.zeta_flat)):
    h, q = V2.design_fir(4000, 300, zf)
    gate = bool(q["max_log_err_n_le_30"] <= 0.1 and abs(q["dc_gain"] - 1) <= 0.01)
    rows = {str(b): V3.fmin(b, h, 300) for b in B}
    fm = {k: v[0] for k, v in rows.items()}
    ok = [b for b in B if fm[str(b)]]
    bb = float(np.polyfit(np.log(ok), np.log([fm[str(b)] for b in ok]), 1)[0]) if len(ok) >= 3 else None
    res[name] = dict(fir=q, gate=gate, fmin=fm, diag={k: v[1] for k, v in rows.items()}, b_beta_large=bb)
    print(name, "gate", gate, q, "F_min", fm, "b_beta(0.08-0.2)", bb, flush=True)
h, q = V2.design_fir(4000, 300, VS.zeta_c1)
sp = {str(v): V3.fmin(0.1, h, 300, vb=v)[0] for v in (0.05, 0.1, 0.2)}
okv = [v for v in (0.05, 0.1, 0.2) if sp[str(v)]]
res["speed_C1b"] = dict(fmin=sp, b_v=float(np.polyfit(np.log(okv), np.log([sp[str(v)] for v in okv]), 1)[0]) if len(okv) >= 2 else None)
h8, q8 = V2.design_fir(8000, 600, VS.zeta_c1)
res["conv_C1b_D8000_b0.12"] = dict(fmin=V3.fmin(0.12, h8, 600, D=8000)[0], fir=q8)
c0 = res["C0b_from_v3"]; okc = [b for b in B if c0[str(b)]]
res["C0b_b_beta_large"] = float(np.polyfit(np.log(okc), np.log([c0[str(b)] for b in okc]), 1)[0])
f1, f0 = res["C1b"]["fmin"]["0.1"], c0["0.1"]
res["readings"] = dict(gate=res["C1b"]["gate"], r=(f1 / f0) if (f1 and f0) else None, Q=(f1 / V3.F_S_C1) if f1 else None,
                       b_beta_large_C1b=res["C1b"]["b_beta_large"], b_beta_large_C0b=res["C0b_b_beta_large"], b_v=res["speed_C1b"]["b_v"])
json.dump(res, open(os.path.join(HERE, "results_v3b.json"), "w"), indent=1, default=float)
print(json.dumps({k: res[k] for k in ("readings", "speed_C1b", "conv_C1b_D8000_b0.12")}, indent=1, default=float))
