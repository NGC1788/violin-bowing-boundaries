"""Verifier: does the flyback trend continue past the plan's f = 0.7 end? f in {0.8, 0.9, 1.01 (= any >=1 flyback)}.
Also summarises audit_check.json (within-column corr, amplitude scaling, v/beta-floor removal)."""
import json, sys
from pathlib import Path
import numpy as np
LSD = Path.home() / "violin-rne/novelty/label_sensitivity"; sys.path.insert(0, str(LSD))
import label_sweep as LS
A = json.loads((Path(__file__).resolve().parent / "audit_check.json").read_text())
wc = A["within_column_corr"]; am = A["amp_scaling"]
print("within-column corr range", min(wc.values()), max(wc.values()), {k: round(v, 2) for k, v in wc.items()})
print("amp p_beta", min(v["p_beta"] for v in am.values()), max(v["p_beta"] for v in am.values()),
      "p_v", min(v["p_v"] for v in am.values()), max(v["p_v"] for v in am.values()))
for r, d in A["vbeta_removal"].items():
    t = np.array(list(d.values())); print(r, "removal by beta tercile min", t.min(0).round(3), "max", t.max(0).round(3))
print("vb summary", json.dumps(A["vbeta_summary"], default=float))
R = json.loads((LSD / "results.json").read_text())["sessions"]
res = {}
for sess, st in LS.STEM.items():
    stem, rep = st if isinstance(st, tuple) else (st, None); raw = LS.read(stem, rep)
    b0 = LS.build(raw, LS.VARIANTS["base"])["H"]
    for f in (0.8, 0.9, 1.01):
        d = LS.build(raw, (0.8, f, 0)); e = dict(window=LS.fit(d)[1], changed=float(np.mean(d["H"] != b0)))
        if sess in LS.ONSET: e["onset"] = LS.fit(LS.to_onset(d, sess))[1]
        res.setdefault(f"fly{f}", {})[sess] = e
for ax, ss in (("window", list(LS.STEM)), ("onset", LS.ONSET)):
    for k, v in res.items():
        x = np.array([v[s][ax] for s in ss]); b = np.array([R[s]["base"][f"{ax}_b_beta"] for s in ss])
        m, sd, ci = LS.tci(x); dm, _, dci = LS.tci(x - b)
        print(f"{ax} {k}: mean {m:+.3f} CI [{ci[0]:+.3f},{ci[1]:+.3f}] shift {dm:+.3f} [{dci[0]:+.3f},{dci[1]:+.3f}] changed {np.mean([v[s]['changed'] for s in ss]):.3f} min sess {x.min():+.2f}")
    for k in ("fly0.7",):
        x = np.array([R[s][k][f"{ax}_b_beta"] for s in ss]); m, sd, ci = LS.tci(x); print(f"{ax} {k} (parent) CI [{ci[0]:+.3f},{ci[1]:+.3f}]")
(Path(__file__).resolve().parent / "extend_flyback.json").write_text(json.dumps(res, indent=1, default=float))
