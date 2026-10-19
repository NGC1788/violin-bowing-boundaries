"""A3 detail (post hoc, after seeing A3): overlap-force H fractions split at beta 0.04, Fisher exact p per cell.
Appends results.json["additions_after_results"]["A3_split"]."""
import json

import numpy as np
from scipy.stats import fisher_exact

import posthoc_hf as P

HERE = P.HERE


def main():
    res = json.loads((HERE / "results.json").read_text())
    hf = P.hf_data()
    out = {}
    for tgt, s1 in P.PAIR.items():
        d = P.R.load(s1)
        ov = res["additions_after_results"]["A3"][tgt]["overlap_force_N"]
        m1 = np.isclose(d["t"], P.T) & (d["beta"] <= 0.07) & (d["force"] >= ov[0]) & (d["force"] <= ov[1])
        keys = [k for k in hf[tgt]["win"] if P.T in hf[tgt]["win"][k] and hf[tgt]["beta"][k] <= 0.07]
        f = np.array([hf[tgt]["win"][k][P.T][0] for k in keys]); h = np.array([hf[tgt]["win"][k][P.T][1] for k in keys], bool)
        v = np.array([hf[tgt]["speed"][k[0]] for k in keys]); b = np.array([hf[tgt]["beta"][k] for k in keys])
        m2 = (f >= ov[0]) & (f <= ov[1])
        cells = {}
        for sp in (0.05, 0.10, 0.20):
            for name, lo, hi in (("beta<0.04", 0, 0.04), ("beta>=0.04", 0.04, 0.0701)):
                a = m1 & np.isclose(d["speed"], sp, atol=0.01) & (d["beta"] >= lo) & (d["beta"] < hi)
                c = m2 & np.isclose(v, sp, atol=0.01) & (b >= lo) & (b < hi)
                n1, k1, n2, k2 = int(a.sum()), int(d["H"][a].sum()), int(c.sum()), int(h[c].sum())
                p = float(fisher_exact([[k1, n1 - k1], [k2, n2 - k2]])[1]) if n1 and n2 else None
                cells[f"{sp:.2f} {name}"] = dict(s1=f"{k1}/{n1}", hf=f"{k2}/{n2}",
                                                 s1_frac=k1 / n1 if n1 else None, hf_frac=k2 / n2 if n2 else None, fisher_p=p)
        out[tgt] = cells
        print(tgt, json.dumps(cells), flush=True)
    res["additions_after_results"]["A3_split"] = out
    (HERE / "results.json").write_text(json.dumps(res, indent=1, default=float))


if __name__ == "__main__":
    main()
