"""Robustness variants of the lower-boundary level (t = 0.2 s, offset from the A1+A2 P-free law without Z).
(Claude, 2026-10-19; output levels_variants.json)

(1) common detection limit c: every force reading below c is treated as unknown, i.e. a column whose lower
    interval reaches below log c becomes left-censored at max(hi, log c) -> (-inf, max(hi, log c)) if lo < log c.
    The reference law is refitted on A1+A2 with the same rule (so A1/A2 are treated identically to the test strings).
(2) zero-offset correction: subtract each folder's post-stop floor (ranks 1-5) from all its window forces
    (7 diagrams with blocks); law refitted on corrected A1+A2.
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.argv = sys.argv[:1]
sys.path.insert(0, str(Path(__file__).parent))
import descriptors as DS  # noqa: E402

K, D = DS.K, DS.D
TAGS = DS.TAGS
OUT = Path(__file__).parent


def recensor(rows, c):
    lc = np.log(c)
    out = []
    for r in rows:
        r2 = dict(r)
        lo, hi = r["lower"]
        if np.isfinite(hi) and lo < lc:
            r2["lower"] = (-np.inf, max(hi, lc))
        out.append(r2)
    return out


def levels(rows_of, train_tags=("A1", "A2")):
    tr = [r for t in train_tags for r in rows_of[t]]
    noz = K.C.physics_models(tr, {t: 1.0 for t in TAGS})
    base = lambda r: noz["P-free"][0](r, "A1")[0]
    res = {}
    for t, rs in rows_of.items():
        res[t] = dict(all=K.offset_fit(rs, base, "lower"),
                      **{fo: K.offset_fit([r for r in rs if r["fold"] == fo], base, "lower") for fo in ("v1", "v2", "v3")},
                      n_left=int(sum(not np.isfinite(r["lower"][0]) for r in rs)))
    pl = noz["P-free"][1][0]
    return res, [float(x) for x in pl]


def main():
    base_rows = {t: DS.rows_by_folder(t)[1] for t in TAGS}
    out = {}
    res, law = levels(base_rows)
    out["c=none"] = dict(levels=res, law=law)
    print("law (no recensoring): const, v, beta, log sigma =", np.round(law, 3))
    for c in (0.15, 0.2, 0.25, 0.3, 0.4):
        rows_c = {t: recensor(rs, c) for t, rs in base_rows.items()}
        res, law = levels(rows_c)
        out[f"c={c}"] = dict(levels=res, law=law)
        print(f"c = {c:.2f} N  law {np.round(law, 2)} | " + " ".join(f"{t} {res[t]['all']:+.2f}({res[t]['n_left']})" for t in TAGS), flush=True)
    # zero-offset correction
    desc = json.load(open(OUT / "descriptors.json"))
    rows_z = {}
    for t in DS.BLOCKS:
        d = dict(D.load(t))
        f = d["force"].copy()
        for k, name in enumerate(d["folders"]):
            fl = desc["folder"][f"{t}|{DS.speed_of(name)}"]["floor"]
            f[d["folder"] == k] -= fl
        d["force"] = f
        rs = K.C.intervals(d, DS.T)
        for r in rs:
            r["tag"] = t
            r["fold"] = DS.speed_of(d["folders"][d["folder"][r["idx"][0]]])
        rows_z[t] = rs
    res, law = levels(rows_z)
    out["zero_corrected"] = dict(levels=res, law=law)
    print("zero-offset corrected: " + " ".join(f"{t} {res[t]['all']:+.3f}" for t in rows_z))
    for c in (0.2, 0.3):
        res, law = levels({t: recensor(rs, c) for t, rs in rows_z.items()})
        out[f"zero_corrected_c={c}"] = dict(levels=res, law=law)
        print(f"zero-offset corrected, c = {c}: " + " ".join(f"{t} {res[t]['all']:+.3f}" for t in rows_z))
    json.dump(out, open(OUT / "levels_variants.json", "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
