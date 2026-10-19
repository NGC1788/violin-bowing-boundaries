"""Build per-stroke caches for the 11 exposed diagrams and check reproduction of earlier boundaries/levels."""
import json, time
import numpy as np
import common as C

REF = np.array([-1.451783207275219, 0.8061412547703237, -1.0324842250632422])  # A1+A2 frozen reference (D_T2 plan)
out = {}
for tag in C.TAGS:
    t0 = time.time()
    d = C.load_exposed(tag)
    rows = C.intervals(d, "F_ev", "H2", "none")
    lo, hi, ok, bad = C.check_rows(rows)
    mu0 = np.array([REF[0] + REF[1] * r["lv"] + REF[2] * r["lb"] for r in rows])
    lev = C.fit_offset(rows, mu0)
    levd = C.fit_offset(C.intervals(d, "F_ev", "H2", "delta"), mu0)
    cov = C.session_covariates(d)
    out[tag] = dict(meta=d["meta"], n_zero_width=int(bad.sum()), level_legacy=lev.get("level"), level_reason=lev.get("reason"),
                    level_delta_fix=levd.get("level"), dec5_pl=cov["diagram"]["dec5"], cal=cov["diagram"]["cal"],
                    dec_ev=cov["diagram"]["dec_ev"], H_share=float(d["H2"].mean()),
                    n_no_band=int(sum(1 for r in rows if r["band"] is False)))
    print(tag, f"{time.time()-t0:.1f}s", json.dumps({k: v for k, v in out[tag].items() if k != 'meta'}), d["meta"]["rank_reversed"], flush=True)
json.dump(out, open(C.HERE / "build_cache.json", "w"), indent=1)
