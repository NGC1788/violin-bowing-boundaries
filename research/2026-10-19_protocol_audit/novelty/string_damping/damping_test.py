"""Zero-parameter Schelleng minimum force from pluck damping vs measured minimum-force boundaries.
Plan (fixed before digitizing): README.md. POST HOC with respect to the boundary data. Claude, 2026-10-20."""
import json, sys
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr, t as tdist

H = Path.home(); HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(H / "violin-rne/novelty/lower_law"))
import onset_axis as OA  # noqa: E402
LL, SV, BT = OA.LL, OA.SV, OA.BT
TL = SV.TL

OA.RAMP["B1"] = H / "violin-rne/astra_20261018/archive_companion/2024-04-10_TypeB_sample1/ramp_meta_2024-04-10_TypeB_sample1.csv"
Z = {"A T1": 1.0592, "B T1": 1.0998, "C T1": 1.0525, "D T1": 1.0398, "A T2": 1.1884, "B T2": 1.1899, "C T2": 1.1154, "D T2": 1.2021}  # POMA Table 1, kg/s
TYPE = {"A1": "A T1", "A2": "A T1", "A3": "A T1", "A4": "A T1", "B1": "B T1", "C1": "C T1", "D1": "D T1", "D2": "D T1",
        "A_T2": "A T2", "B_T2s1": "B T2", "B_T2s2": "B T2", "B_T2p6": "B T2", "C_T2": "C T2", "D_T2": "D T2"}
DMU, CONST, NPRI = 0.6, 0.5, 10           # primary constants (README)
V0, B0 = 0.1, 0.1
ONSET_EXP = json.loads((H / "violin-rne/novelty/lower_law/onset_with_B1.json").read_text())
OAJ = json.loads((H / "violin-rne/novelty/lower_law/onset_axis.json").read_text())
EXP = {"onset": (ONSET_EXP["re_v"]["mu"], ONSET_EXP["re_beta"]["mu"]),
       "window": (OAJ["re_measured"]["v"]["mu"], OAJ["re_measured"]["beta"]["mu"])}


# ---------------- damping side ----------------
def zeta_table():
    dj = json.loads((HERE / "damping_digitized.json").read_text())
    return {k: {p["mode"]: p["zeta"] for p in v["points"]} for k, v in dj["panels"].items()}


def zeta_n(z, n):
    if n in z:
        return z[n], False
    lo = [m for m in z if m < n]; hi = [m for m in z if m > n]
    if not lo or not hi:
        return None, True
    a, b = max(lo), min(hi)
    w = (n - a) / (b - a)
    return float(np.exp((1 - w) * np.log(z[a]) + w * np.log(z[b]))), True


def S(z, N):
    s, n_interp, last = 0.0, 0, 0
    for n in range(1, N + 1):
        v, ip = zeta_n(z, n)
        if v is None:
            break
        s += v / n; n_interp += ip; last = n
    return s, n_interp, last


def F_S(Zt, s, v=V0, b=B0, const=CONST, dmu=DMU):
    return const * Zt * (6 / np.pi) * s * v / (b ** 2 * dmu)


def pct(a, b):
    return abs(a - b) / ((a + b) / 2) * 100


def validation(zt):
    mats, out = "ABCD", {}
    def pairs_mat(modes=None):
        vals = []
        for ten in ("T1", "T2"):
            for m1, m2 in combinations(mats, 2):
                z1, z2 = zt[f"{m1} {ten}"], zt[f"{m2} {ten}"]
                for n in set(z1) & set(z2):
                    if modes is None or n in modes:
                        vals.append(pct(z1[n], z2[n]))
        return float(np.mean(vals)), len(vals)
    def pairs_ten(modes=None):
        vals = []
        for m in mats:
            z1, z2 = zt[f"{m} T1"], zt[f"{m} T2"]
            for n in set(z1) & set(z2):
                if modes is None or n in modes:
                    vals.append(pct(z1[n], z2[n]))
        return float(np.mean(vals)), len(vals)
    out["material_all_modes_pct"], out["tension_all_modes_pct"] = pairs_mat(), pairs_ten()
    out["per_mode_material_pct"] = {n: pairs_mat({n}) for n in range(1, 20)}
    out["mode1_material_pct"], out["mode1_tension_pct"] = pairs_mat({1}), pairs_ten({1})
    out["poma_text"] = "material ~30% vs tension ~20%; modes 4 and 9 largest across materials (50-60%); mode 1: tension 27% vs material 17%"
    return out


# ---------------- boundary side ----------------
def onset_rows(s):
    if s == "B_T2s2":
        return OA.bt2s2_onset()
    return OA.onset_rows(s)


def window_rows(s):
    if s == "B_T2s2":
        return LL.bt2s2_rows(BT.STD_SPEEDS)
    return SV.rows_of({"A3": "TGT_2024-04-22_TypeA_sample3", "A4": "TGT_2024-04-24_TypeA_sample4", "D2": "TGT_2024-04-26_TypeD_sample2"}.get(s, s))


def level(rows, bv, bb):
    keep = [r for r in rows if np.isfinite(r["lower"][0]) or np.isfinite(r["lower"][1])]
    lo, hi, _ = SV.widen([r["lower"][0] for r in keep], [r["lower"][1] for r in keep])
    mu = np.array([bv * (r["lv"] - np.log(V0)) + bb * (r["lb"] - np.log(B0)) for r in keep])
    return float(TL.offset_fit(mu, lo, hi)[0]), keep


def session_level(rows, bv, bb, nb=200, seed=7):
    y, keep = level(rows, bv, bb)
    cl = defaultdict(list)
    for r in LL.with_bcol([dict(r) for r in keep]):
        cl[r["bcol"]].append(r)
    keys = sorted(cl); rng = np.random.default_rng(seed); bs = []
    for _ in range(nb):
        kk = rng.choice(len(keys), len(keys))
        bs.append(level([r for k in kk for r in cl[keys[k]]], bv, bb)[0])
    lb = np.array([r["lb"] for r in keep])
    return dict(level=y, se=float(np.std(bs, ddof=1)), n=len(keep), beta_range=[float(np.exp(lb.min())), float(np.exp(lb.max()))])


def ols(x, y):
    x, y = np.asarray(x), np.asarray(y); n = len(x); X = np.c_[np.ones(n), x]
    b, *_ = np.linalg.lstsq(X, y, rcond=None); r = y - X @ b; s2 = r @ r / (n - 2)
    se = np.sqrt(s2 * np.linalg.inv(X.T @ X)[1, 1]); tc = tdist.ppf(0.975, n - 2)
    return dict(slope=float(b[1]), ci=[float(b[1] - tc * se), float(b[1] + tc * se)], intercept=float(b[0]), n=n)


def verdict_T2(ci):
    if ci[0] > 1 or ci[1] < 1:
        return "contradicted (CI excludes 1)"
    if ci[0] > 0:
        return "supported (CI excludes 0, includes 1)"
    return "not decided"


def band(q):
    return "below maintenance (contradiction)" if q < 0.5 else ("matches" if q <= 2 else ("establishment-sized excess" if q <= 8 else "fails by ~10x or more"))


def main():
    zt = zeta_table()
    res = {"constants": dict(dmu=DMU, const=CONST, N=NPRI, v=V0, beta=B0, exponents=EXP), "damping": {}, "validation": validation(zt)}
    for t, z in zt.items():
        d = {}
        for N in (5, 10, 15):
            s, ni, last = S(z, N); d[f"S{N}"] = s; d[f"S{N}_interp"] = ni; d[f"S{N}_last_mode"] = last
        d["zeta1"] = z.get(1); d["Z_over_R_eff_N10"] = 6 / np.pi * d["S10"]; d["F_S_primary"] = F_S(Z[t], d["S10"])
        lo = min(F_S(Z[t], d[f"S{N}"], const=c, dmu=m) for N in (5, 15, 10) for c in (0.5, 2 / 3) for m in (0.6, 1.0))
        hi = max(F_S(Z[t], d[f"S{N}"], const=c, dmu=m) for N in (5, 15, 10) for c in (0.5, 2 / 3) for m in (0.6, 1.0))
        d["F_S_band"] = [lo, hi]
        res["damping"][t] = d
    sessions = {"onset": ["A1", "A2", "A3", "A4", "B1", "C1", "D1", "D2", "A_T2", "B_T2s1", "B_T2s2", "C_T2"],
                "window": ["A1", "A2", "A3", "A4", "B1", "C1", "D1", "D2", "A_T2", "B_T2s1", "B_T2s2", "B_T2p6", "C_T2", "D_T2"]}
    for ax, ss in sessions.items():
        bv, bb = EXP[ax]; lev = {}
        for s in ss:
            if ax == "window" and s == "B_T2p6":
                parts = [session_level(window_rows(p), bv, bb) for p in ("B_T2r1", "B_T2r2")]
                lev[s] = dict(level=float(np.mean([p["level"] for p in parts])), se=float(np.sqrt(sum(p["se"] ** 2 for p in parts)) / 2),
                              n=sum(p["n"] for p in parts), beta_range=parts[0]["beta_range"])
            else:
                lev[s] = session_level(onset_rows(s) if ax == "onset" else window_rows(s), bv, bb)
            print(f"{ax} {s:7s} level {lev[s]['level']:+.3f} ± {lev[s]['se']:.3f}  F={np.exp(lev[s]['level']):.3f} N  β {lev[s]['beta_range'][0]:.3f}-{lev[s]['beta_range'][1]:.3f}", flush=True)
        by = defaultdict(list)
        for s, v in lev.items():
            by[TYPE[s]].append(v["level"])
        types = sorted(by)
        L = {t: float(np.mean(by[t])) for t in types}
        T1 = {}
        bmin = min(v["beta_range"][0] for v in lev.values()); bmax = max(v["beta_range"][1] for v in lev.values())
        for t in types:
            fm = np.exp(L[t]); fs = res["damping"][t]["F_S_primary"]; q = fm / fs
            qb = [fm / res["damping"][t]["F_S_band"][1], fm / res["damping"][t]["F_S_band"][0]]
            # Q at the ends of the measured beta range: measured scales as beta^bb, F_S as beta^-2
            q_ends = [q * (bmin / B0) ** (bb + 2), q * (bmax / B0) ** (bb + 2)]
            T1[t] = dict(F_meas=fm, F_S=fs, Q=q, Q_band=qb, band=band(q), band_range=[band(qb[0]), band(qb[1])], Q_at_beta_ends=q_ends)
        Ls = np.array([L[t] for t in types])
        T2 = {}
        for key, getp in (("S10", lambda t: np.log(Z[t] * res["damping"][t]["S10"])), ("S5", lambda t: np.log(Z[t] * res["damping"][t]["S5"])),
                          ("S15", lambda t: np.log(Z[t] * res["damping"][t]["S15"])), ("zeta1", lambda t: np.log(Z[t] * res["damping"][t]["zeta1"]))):
            P = np.array([getp(t) for t in types]); o = ols(P, Ls)
            T2[key] = dict(ols=o, verdict=verdict_T2(o["ci"]), sd_L=float(np.std(Ls, ddof=1)), sd_L_minus_P=float(np.std(Ls - P, ddof=1)),
                           spearman=float(spearmanr(P, Ls)[0]), P={t: float(p) for t, p in zip(types, P)})
        res[ax] = dict(sessions=lev, type_level=L, beta_range=[bmin, bmax], T1=T1, T2=T2)
    (HERE / "results.json").write_text(json.dumps(res, indent=1, default=float))

    v = res["validation"]
    print(f"\nvalidation: material {v['material_all_modes_pct'][0]:.0f}% vs tension {v['tension_all_modes_pct'][0]:.0f}% | mode1 material {v['mode1_material_pct'][0]:.0f}% tension {v['mode1_tension_pct'][0]:.0f}%")
    top = sorted(((k, x[0]) for k, x in v["per_mode_material_pct"].items() if x[1] >= 6), key=lambda kv: -kv[1])[:4]
    print("  largest material variation modes:", ", ".join(f"{k}:{x:.0f}%" for k, x in top))
    for t, d in res["damping"].items():
        print(f"  {t}: zeta1 {d['zeta1']*1e4:.2f}e-4, S10 {d['S10']*1e4:.2f}e-4 (interp {d['S10_interp']}), Z/R_eff {d['Z_over_R_eff_N10']:.4f}, F_S {d['F_S_primary']:.4f} N band {d['F_S_band'][0]:.4f}-{d['F_S_band'][1]:.4f}")
    for ax in ("onset", "window"):
        r = res[ax]; print(f"\n== {ax} axis (exponents v {EXP[ax][0]:+.3f}, β {EXP[ax][1]:+.3f}); β range {r['beta_range'][0]:.3f}-{r['beta_range'][1]:.3f}")
        for t, x in r["T1"].items():
            print(f"  T1 {t}: F_meas {x['F_meas']:.3f} N, F_S {x['F_S']:.4f} N, Q {x['Q']:.1f} [{x['Q_band'][0]:.1f}, {x['Q_band'][1]:.1f}] -> {x['band']} (band {x['band_range']}); Q at β ends {x['Q_at_beta_ends'][0]:.1f}..{x['Q_at_beta_ends'][1]:.1f}")
        for k, x in r["T2"].items():
            o = x["ols"]
            print(f"  T2[{k}] slope {o['slope']:+.2f} [{o['ci'][0]:+.2f}, {o['ci'][1]:+.2f}] (n {o['n']}) -> {x['verdict']}; SD L {x['sd_L']:.3f} vs SD(L-P) {x['sd_L_minus_P']:.3f}; Spearman {x['spearman']:+.2f}")


if __name__ == "__main__":
    main()
