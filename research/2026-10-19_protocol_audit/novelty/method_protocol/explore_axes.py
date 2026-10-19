"""Exploration (post hoc, exposed 11): sharpness and between-diagram spread of the lower boundary on
alternative label-free force axes."""
import numpy as np, json
import common as C
D = {t: C.load_exposed(t) for t in C.TAGS}
res = {}
for axis in ("F_ev", "F_01", "F_on", "F_nom"):
    rows = {t: C.intervals(D[t], axis, "H2", "delta") for t in C.TAGS}
    s1 = C.fit_sessions(rows)
    a = np.array(s1["alpha"]).reshape(11, 3)
    sp = a.mean(0)
    lev = (a - sp).mean(1)
    within = (a - sp - lev[:, None])
    lz = np.log([C.PROPS[C.GROUP[t]][0] for t in C.TAGS]); le = np.log([C.PROPS[C.GROUP[t]][1] for t in C.TAGS])
    dec = np.array([C.session_covariates(D[t])["diagram"]["dec5"] for t in C.TAGS])
    cal = np.array([C.session_covariates(D[t])["diagram"]["cal"] for t in C.TAGS])
    res[axis] = dict(sigma_within_session=s1["sigma"], b_beta=s1["b"], speed_intercepts=sp.tolist(),
                     sd_diagram_levels=float(lev.std(ddof=1)), sd_session_within=float(within.std(ddof=1)),
                     r_dec5=float(np.corrcoef(dec, lev)[0, 1]), r_logZ=float(np.corrcoef(lz, lev)[0, 1]),
                     r_logEI=float(np.corrcoef(le, lev)[0, 1]), r_cal=float(np.corrcoef(cal, lev)[0, 1]),
                     levels=dict(zip(C.TAGS, np.round(lev, 3).tolist())))
    print(axis, json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in res[axis].items() if k != 'levels'}))
    print("   ", res[axis]["levels"])
json.dump(res, open(C.HERE / "explore_axes.json", "w"), indent=1)
