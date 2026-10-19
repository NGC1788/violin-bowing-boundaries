"""In-sample decomposition on all 11 exposed diagrams (post hoc, descriptive): how much of the session-level
lower-boundary variation is absorbed by label-free protocol covariates vs published string properties, the
within-diagram (Mundlak) vs between-diagram dec5 coefficients, and the floor-limited share of observed intervals."""
import json
import numpy as np
import common as C
import model as M

DATA = {t: (C.load_exposed(t), C.GROUP[t]) for t in C.TAGS}
out = {"variance_components": {}, "coefficients": {}, "floor_share": {}}
for cs in ("mean", "props", "protocol", "protocol_dec5", "full"):
    P = M.PACE(cs, 1, True, shape="quad_s").fit(DATA).params()
    out["variance_components"][cs] = dict(tau=P["tau"], omega=P["omega"], sigma=P["sigma"],
                                          between_var=P["tau"] ** 2 + P["omega"] ** 2,
                                          beta=dict(zip(["c_v1", "c_v2", "c_v3"] + P["keys"], P["beta"])),
                                          se=dict(zip(["c_v1", "c_v2", "c_v3"] + P["keys"], np.sqrt(np.diag(P["cov"])).tolist())))
base = out["variance_components"]["mean"]["between_var"]
for cs, v in out["variance_components"].items():
    v["share_of_mean_model_between_var_removed"] = 1 - v["between_var"] / base
# diagram-cluster bootstrap of PACE* and full coefficients
rng = np.random.default_rng(20261019)
for cs in ("protocol", "full"):
    draws = []
    for _ in range(300):
        pick = rng.integers(0, 11, 11)
        dd = {}
        for j, i in enumerate(pick):
            t = C.TAGS[i]
            dd[f"{t}#{j}"] = DATA[t]
        try:
            P = M.PACE(cs, 1, True, shape="quad_s").fit(dd).params()
            draws.append(P["beta"][3:])
        except Exception:
            pass
    draws = np.array(draws)
    keys = M.SETS[cs]
    out["coefficients"][cs] = {k: dict(median=float(np.median(draws[:, i])), p2_5=float(np.percentile(draws[:, i], 2.5)),
                                       p97_5=float(np.percentile(draws[:, i], 97.5)), share_positive=float((draws[:, i] > 0).mean()))
                               for i, k in enumerate(keys)}
    out["coefficients"][cs]["n_boot_ok"] = len(draws)
for t in C.TAGS:
    rows = C.intervals(DATA[t][0], "F_ev", "H2", "delta")
    band = [r for r in rows if r["band"]]
    out["floor_share"][t] = dict(n=len(rows), runs_start_at_lowest=sum(r["s"] == 0 for r in band),
                                 runs_start_2nd_3rd=sum(r["s"] in (1, 2) for r in band),
                                 no_band=sum(1 for r in rows if r["band"] is False))
print(json.dumps(out, indent=1, default=float)[:4000])
json.dump(out, open(C.HERE / "fit_all.json", "w"), indent=1, default=float)
