"""Additions decided AFTER seeing Q1-Q3 (README, 22:41). POST HOC. Adds results.json["additions_after_results"].
A1 contradiction robustness, A2 one delta sensitivity (top-stroke dropout -> right-censored), A3 model-free overlap."""
import json

import numpy as np

import posthoc_hf as P

HERE = P.HERE


def s1_cols(tag, d, RW):
    """sample-1 upper rows (beta <= 0.075) with the number of strokes above the last band end."""
    out = []
    for r in RW[tag]:
        if np.exp(r["lb"]) > 0.075:
            continue
        idx = r["idx"]
        h = d["H"][idx].astype(bool)
        e = P.last_band_end(h)
        out.append(dict(r, n_above=len(h) - 1 - e, col=int(d["column"][idx[0]]),
                        f_end=float(np.log(max(d["force"][idx[e]], 1e-3)))))
    return out


def main():
    res = json.loads((HERE / "results.json").read_text())
    RW = P.R.all_rows()
    hf = P.hf_data()
    add = dict(note="decided after seeing Q1-Q3; post hoc", A1={}, A2={}, A3={})
    rng = np.random.default_rng(7)
    for tgt, s1 in P.PAIR.items():
        d = P.R.load(s1)
        S1 = s1_cols(s1, d, RW)
        rows = P.S.column_table(P.hf_columns_with_trials(hf[tgt]))
        H = P.upper_dict(rows, lambda r: r["column"])
        A = dict(lv=np.array([r["lv"] for r in S1]), lb=np.array([r["lb"] for r in S1]),
                 lo=np.array([r["upper"][0] for r in S1], float), hi=np.array([r["upper"][1] for r in S1], float),
                 col=np.array([r["col"] for r in S1]), n_above=np.array([r["n_above"] for r in S1]))
        # A1
        recs = []
        for j in range(len(H["lv"])):
            m = np.flatnonzero(np.isclose(A["lv"], H["lv"][j], atol=0.05))
            if not len(m):
                continue
            k = m[np.argmin(np.abs(A["lb"][m] - H["lb"][j]))]
            dist = abs(A["lb"][k] - H["lb"][j])
            if np.isfinite(A["hi"][k]) and np.isfinite(H["lo"][j]):
                recs.append(dict(dist=dist, contra=bool(H["lo"][j] > A["hi"][k]), n_above=int(A["n_above"][k]),
                                 v=float(np.exp(H["lv"][j]))))
            if np.isfinite(A["lo"][k]) and np.isfinite(H["hi"][j]) and H["hi"][j] < A["lo"][k]:
                recs.append(dict(dist=dist, contra_below=True))
        up = [x for x in recs if "contra" in x]
        add["A1"][tgt] = dict(
            hf_higher_contradictions=sum(x["contra"] for x in up), denom=len(up),
            n_above_of_contradictions=[x["n_above"] for x in up if x["contra"]],
            speeds_of_contradictions=[x["v"] for x in up if x["contra"]],
            with_dist_le_0p06=dict(contradictions=sum(x["contra"] for x in up if x["dist"] <= 0.06),
                                   denom=sum(1 for x in up if x["dist"] <= 0.06)),
            excluding_single_top_dropout=dict(contradictions=sum(x["contra"] and x["n_above"] >= 2 for x in up)),
            hf_lower_contradictions=sum(1 for x in recs if x.get("contra_below")),
            sample1_finite_n_above=np.bincount(A["n_above"][np.isfinite(A["hi"])], minlength=4).tolist())
        # A2: top-stroke dropout -> right-censored at band end
        lo2, hi2 = A["lo"].copy(), A["hi"].copy()
        recode = np.isfinite(hi2) & (A["n_above"] == 1)
        hi2[recode] = np.inf
        lv = np.r_[A["lv"], H["lv"]]; lb = np.r_[A["lb"], H["lb"]]
        lo = np.r_[lo2, H["lo"]]; hi = np.r_[hi2, H["hi"]]
        ses = np.r_[np.zeros(len(A["lv"])), np.ones(len(H["lv"]))]
        clus = np.r_[A["col"], 1000 + np.asarray(H["col"])]
        p, _ = P.q2_fit(lv, lb, ses, lo, hi)
        ids0 = np.unique(clus[ses == 0]); ids1 = np.unique(clus[ses == 1])
        members = {c: np.flatnonzero(clus == c) for c in np.unique(clus)}
        rng = np.random.default_rng(7)
        bs = []
        for _ in range(1000):
            pick = np.r_[rng.choice(ids0, len(ids0)), rng.choice(ids1, len(ids1))]
            ii = np.concatenate([members[c] for c in pick])
            q, _ = P.q2_fit(lv[ii], lb[ii], ses[ii], lo[ii], hi[ii], x0=p, polish=False)
            bs.append(q[3])
        add["A2"][tgt] = dict(n_recoded=int(recode.sum()), n_sample1_finite_after=int(np.sum(np.isfinite(hi2))),
                              delta=float(p[3]), delta_ci=np.percentile(bs, [2.5, 97.5]).tolist(), e_v=float(p[1]),
                              e_b=float(p[2]), bootstrap="B=1000, seed 7, beta-column clusters within session")
        # A3: model-free overlap at the same measured force
        m1 = np.isclose(d["t"], P.T) & (d["beta"] <= 0.07)
        hf_keys = [k for k in hf[tgt]["win"] if P.T in hf[tgt]["win"][k] and hf[tgt]["beta"][k] <= 0.07]
        hf_f = np.array([hf[tgt]["win"][k][P.T][0] for k in hf_keys])
        hf_h = np.array([hf[tgt]["win"][k][P.T][1] for k in hf_keys], bool)
        hf_v = np.array([hf[tgt]["speed"][k[0]] for k in hf_keys])
        lo_ov, hi_ov = float(hf_f.min()), float(d["force"][m1].max())
        per = {}
        for v in (0.05, 0.10, 0.20):
            a = m1 & np.isclose(d["speed"], v, atol=0.01) & (d["force"] >= lo_ov) & (d["force"] <= hi_ov)
            b = np.isclose(hf_v, v, atol=0.01) & (hf_f >= lo_ov) & (hf_f <= hi_ov)
            per[f"{v:.2f}"] = dict(sample1_n=int(a.sum()), sample1_H=int(d["H"][a].sum()),
                                   sample1_H_frac=float(d["H"][a].mean()) if a.any() else None,
                                   hf_n=int(b.sum()), hf_H=int(hf_h[b].sum()),
                                   hf_H_frac=float(hf_h[b].mean()) if b.any() else None,
                                   sample1_median_force=float(np.median(d["force"][a])) if a.any() else None,
                                   hf_median_force=float(np.median(hf_f[b])) if b.any() else None)
        add["A3"][tgt] = dict(sample1=s1, overlap_force_N=[lo_ov, hi_ov], by_speed=per)
        print(tgt, json.dumps({k: add[k][tgt] for k in ("A1", "A2", "A3")}, default=float), flush=True)
    res["additions_after_results"] = add
    (HERE / "results.json").write_text(json.dumps(res, indent=1, default=float))


if __name__ == "__main__":
    main()
