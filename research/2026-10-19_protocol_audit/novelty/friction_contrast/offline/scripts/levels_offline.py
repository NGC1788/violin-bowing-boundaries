"""Boundary levels of all locally processed diagrams (now including B_T2 sample 1 and C_T2) and the Schelleng
friction-factor test (Claude, 2026-10-19, offline; post hoc).

Level = offset of a diagram's lower / upper boundary from the A1+A2 P-free law without Z (t = 0.2 s, K = 2),
exactly as ~/violin-rne/final/tension_pairs.py (offset_fit of confirm_A_T2.py). CIs: 200 bootstrap resamples of the
(beta, v) columns of the diagram (paired: the same resample for lower and upper).
Free law per diagram: log F = a + b log v + c log beta (interval-censored), bootstrap CIs (100 resamples).

Schelleng: F_max = 2 Z v / (beta dmu), F_min = Z^2 v / (2 R beta^2 dmu).
With z = log(Z/Z_A) (published) and d = log(dmu/dmu_A), r = log(R/R_A):
   L_up = z - d,   L_lo = 2 z - d - r.
 (i) required contrast ratio from the upper level: d = z - L_up (R not involved).
 (ii) if only the friction contrast differed between strings (r = 0): L_lo - L_up = z (> 0 for T2, 0 for T1).
     A per-string friction factor moves both boundaries by the same amount; it cannot widen the band.
Run: python -B levels_offline.py <out_dir>   (writes cache npz for the new tags to ~/violin-rne/final/cache)
"""
import json
import sys
from pathlib import Path

import numpy as np

FINAL = Path.home() / "violin-rne/final"
sys.path.insert(0, str(FINAL))
OUT = Path(sys.argv[1])
sys.argv = sys.argv[:1]  # compare.py parses argv at import
import confirm_A_T2 as K  # noqa: E402

D, C = K.D, K.C
D.STEMS["A_T2"] = "2024-06-19_TypeA_T2_sample1"
D.STEMS["B_T2s1"] = "2024-06-13_TypeB_T2_sample1"
D.STEMS["C_T2"] = "2024-06-11_TypeC_T2_sample1"
POMA = {  # T (N), Z (kg/s), EI (1e-4 N m^2), d (mm), Zto
    "A1": (145.31, 1.0592, 3.032, 0.947, 0.039), "A2": (145.31, 1.0592, 3.032, 0.947, 0.039),
    "B1": (150.87, 1.0998, 4.397, 0.927, 0.055), "C1": (144.40, 1.0525, 3.142, 1.077, 0.039),
    "D1": (142.67, 1.0398, 5.246, 1.123, 0.064), "A_T2": (163.03, 1.1884, 3.969, 1.001, 0.027),
    "B_T2s1": (163.23, 1.1899, 7.295, 0.973, 0.060), "C_T2": (153.04, 1.1154, 3.969, 1.083, 0.040),
    "B_T2r1": (163.23, 1.1899, 7.295, 0.973, 0.060), "B_T2r2": (163.23, 1.1899, 7.295, 0.973, 0.060)}
TAGS = list(POMA)
NBOOT, NBOOT_LAW = 200, 100
rng = np.random.default_rng(7)


def design(r):
    return np.array([1.0, r["lv"], r["lb"]]), 0.0


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows = {t: K.rows_for(t) for t in TAGS}
    for t in ("A_T2", "B_T2s1", "C_T2"):
        d = D.load(t)
        print(f"{t}: windows {len(d['t'])}, folders {list(d['folders'])}, speeds {sorted(set(np.round(d['speed'], 3)))}, "
              f"H share {d['H'].mean():.2f}, columns {len(rows[t])}")
    tr = rows["A1"] + rows["A2"]
    noz = C.physics_models(tr, {t: 1.0 for t in TAGS})
    base = {j: (lambda r, j=j: noz["P-free"][0](r, "A1")[j]) for j in (0, 1)}  # z_of = 1 for every tag -> tag-independent
    res = dict(levels={}, free_laws={})
    print("\nlevels vs the A1+A2 P-free law (log units), 95 % column-bootstrap CI")
    for t in TAGS:
        rs = rows[t]
        lo = K.offset_fit(rs, base[0], "lower"); up = K.offset_fit(rs, base[1], "upper")
        bl, bu, bd = [], [], []
        for _ in range(NBOOT):
            s = [rs[i] for i in rng.integers(0, len(rs), len(rs))]
            a, b = K.offset_fit(s, base[0], "lower"), K.offset_fit(s, base[1], "upper")
            bl.append(a); bu.append(b); bd.append(a - b)
        ci = lambda x: [float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))]
        res["levels"][t] = dict(lower=float(lo), upper=float(up), lower_ci=ci(bl), upper_ci=ci(bu), diff=float(lo - up), diff_ci=ci(bd),
                                boot_lower=[float(x) for x in bl], boot_upper=[float(x) for x in bu])
        print(f"  {t:7s} lower {lo:+.3f} [{ci(bl)[0]:+.3f}, {ci(bl)[1]:+.3f}]  upper {up:+.3f} [{ci(bu)[0]:+.3f}, {ci(bu)[1]:+.3f}]  "
              f"lower-upper {lo - up:+.3f} [{ci(bd)[0]:+.3f}, {ci(bd)[1]:+.3f}]")
    print("\nfree censored law per diagram (t = 0.2 s): exponents with 95 % CI (100 column resamples)")
    for t in TAGS:
        rs = rows[t]
        law = {}
        for which in ("lower", "upper"):
            th = C.censored_fit(rs, design, which)
            bs = []
            for _ in range(NBOOT_LAW):
                s = [rs[i] for i in rng.integers(0, len(rs), len(rs))]
                bs.append(C.censored_fit(s, design, which)[1:3])
            bs = np.array(bs)
            law[which] = dict(speed=float(th[1]), beta=float(th[2]), sigma=float(np.exp(th[3])),
                              speed_ci=np.percentile(bs[:, 0], [2.5, 97.5]).tolist(), beta_ci=np.percentile(bs[:, 1], [2.5, 97.5]).tolist())
        res["free_laws"][t] = law
        print(f"  {t:7s} lower v^{law['lower']['speed']:+.2f} [{law['lower']['speed_ci'][0]:+.2f}, {law['lower']['speed_ci'][1]:+.2f}] "
              f"beta^{law['lower']['beta']:+.2f} [{law['lower']['beta_ci'][0]:+.2f}, {law['lower']['beta_ci'][1]:+.2f}] | "
              f"upper v^{law['upper']['speed']:+.2f} [{law['upper']['speed_ci'][0]:+.2f}, {law['upper']['speed_ci'][1]:+.2f}] "
              f"beta^{law['upper']['beta']:+.2f} [{law['upper']['beta_ci'][0]:+.2f}, {law['upper']['beta_ci'][1]:+.2f}]")

    # Schelleng friction-factor tests
    print("\nSchelleng friction-factor test (z = log Z/Z_A, published Z)")
    LV = res["levels"]
    test = {}
    for t in TAGS:
        z = np.log(POMA[t][1] / POMA["A1"][1])
        bl, bu = np.array(LV[t]["boot_lower"]), np.array(LV[t]["boot_upper"])
        d_req = z - bu  # required log contrast ratio from the upper level
        widen = (bl - bu) - z  # departure from 'friction only' (should be 0)
        r_req = z + bu - bl  # = 2z - d - L_lo with d from the upper level : required log R ratio
        test[t] = dict(z=float(z), d_required=float(z - LV[t]["upper"]), d_required_ci=np.percentile(d_req, [2.5, 97.5]).tolist(),
                       friction_only_residual=float(LV[t]["diff"] - z), friction_only_residual_ci=np.percentile(widen, [2.5, 97.5]).tolist(),
                       r_required=float(z + LV[t]["upper"] - LV[t]["lower"]), r_required_ci=np.percentile(r_req, [2.5, 97.5]).tolist())
        T = test[t]
        print(f"  {t:7s} z {z:+.3f} | dmu_X/dmu_A required by upper {np.exp(T['d_required']):.2f} "
              f"[{np.exp(T['d_required_ci'][0]):.2f}, {np.exp(T['d_required_ci'][1]):.2f}] | friction-only residual (L_lo-L_up-z) "
              f"{T['friction_only_residual']:+.3f} [{T['friction_only_residual_ci'][0]:+.3f}, {T['friction_only_residual_ci'][1]:+.3f}] | "
              f"R_X/R_A then required {np.exp(T['r_required']):.2f} [{np.exp(T['r_required_ci'][0]):.2f}, {np.exp(T['r_required_ci'][1]):.2f}]")
    nonA = [t for t in TAGS if not t.startswith("A")]
    neg = [t for t in nonA if test[t]["friction_only_residual_ci"][1] < 0]
    print(f"  non-A diagrams whose band is significantly WIDER than a pure friction-contrast change allows: {len(neg)}/{len(nonA)} {neg}")
    a_t2 = test["A_T2"]
    print(f"  A_T2: residual {a_t2['friction_only_residual']:+.3f} {np.round(a_t2['friction_only_residual_ci'], 3).tolist()}")
    yard = abs(LV["A1"]["diff"] - LV["A2"]["diff"]); yard_b = abs(LV["B_T2r1"]["diff"] - LV["B_T2r2"]["diff"])
    print(f"  yardsticks for (L_lo - L_up): A1 vs A2 {yard:.3f}; B_T2 rep1 vs rep2 {yard_b:.3f}; "
          f"B_T2 s1 vs reps {LV['B_T2s1']['diff'] - 0.5 * (LV['B_T2r1']['diff'] + LV['B_T2r2']['diff']):+.3f}")
    res["schelleng_test"] = test

    print("\ntension pairs T1 -> T2 (level differences, log; CI from independent bootstraps)")
    pairs = []
    for t1, t2 in (("A1", "A_T2"), ("A2", "A_T2"), ("B1", "B_T2s1"), ("B1", "B_T2r1"), ("B1", "B_T2r2"), ("C1", "C_T2")):
        zr = np.log(POMA[t2][1] / POMA[t1][1])
        out = {}
        for w in ("lower", "upper"):
            d = LV[t2][w] - LV[t1][w]
            bd = np.array(LV[t2]["boot_" + w]) - np.array(LV[t1]["boot_" + w])
            out[w] = (float(d), np.percentile(bd, [2.5, 97.5]).tolist(), float(d / zr))
        pairs.append(dict(t1=t1, t2=t2, z=float(zr), **out))
        print(f"  {t1:3s}->{t2:7s} Z ratio {np.exp(zr):.3f} | lower {out['lower'][0]:+.3f} [{out['lower'][1][0]:+.3f}, {out['lower'][1][1]:+.3f}] "
              f"(Z exponent {out['lower'][2]:+.1f}) | upper {out['upper'][0]:+.3f} [{out['upper'][1][0]:+.3f}, {out['upper'][1][1]:+.3f}] "
              f"(Z exponent {out['upper'][2]:+.1f})")
    res["tension_pairs"] = pairs

    print("\ncorrelations of levels with log properties (10 diagrams; B_T2 counted 3x, A 2x -> descriptive only)")
    Lo = np.array([LV[t]["lower"] for t in TAGS]); Up = np.array([LV[t]["upper"] for t in TAGS])
    cor = {}
    for i, nm in enumerate(("T", "Z", "EI", "d", "Zto")):
        x = np.log([POMA[t][i] for t in TAGS])
        cor[nm] = (float(np.corrcoef(x, Lo)[0, 1]), float(np.corrcoef(x, Up)[0, 1]))
        print(f"  log {nm:3s}: lower {cor[nm][0]:+.2f}, upper {cor[nm][1]:+.2f}")
    # one value per string (average replicate diagrams) to avoid pseudo-replication
    groups = {"A": ["A1", "A2"], "B1": ["B1"], "C1": ["C1"], "D1": ["D1"], "A_T2": ["A_T2"], "B_T2": ["B_T2s1", "B_T2r1", "B_T2r2"], "C_T2": ["C_T2"]}
    gL = np.array([np.mean([LV[t]["lower"] for t in g]) for g in groups.values()])
    gU = np.array([np.mean([LV[t]["upper"] for t in g]) for g in groups.values()])
    gP = {nm: np.log([POMA[g[0]][i] for g in groups.values()]) for i, nm in enumerate(("T", "Z", "EI", "d", "Zto"))}
    print("  per string (7 strings, replicates averaged): " + "; ".join(
        f"{nm} lower {np.corrcoef(x, gL)[0, 1]:+.2f} upper {np.corrcoef(x, gU)[0, 1]:+.2f}" for nm, x in gP.items()))
    res["correlations"] = cor
    res["per_string"] = {g: dict(lower=float(a), upper=float(b)) for g, a, b in zip(groups, gL, gU)}
    for t in TAGS:
        LV[t].pop("boot_lower"); LV[t].pop("boot_upper")
    (OUT / "levels.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
