"""Do session/protocol descriptors predict the lower-boundary level across sessions better than published string
properties?  (Claude, 2026-10-19; reads descriptors.json, levels_variants.json; prints the tables; analysis.json)

Units: speed folders (= sessions), level relative to the A1+A2 law; speed fixed effects in every folder model.
  A  diagram-level (mean over the 3 speeds): single-predictor Pearson r, leave-one-diagram-out (LODO) CV R^2
  B  folder-level, speed FE + 1 predictor: LODO CV RMSE vs speed-only
  C  within-diagram: diagram FE + speed FE + descriptor (only session variation; string properties constant);
     permutation test shuffling the descriptor across the 3 speeds inside each diagram
  D  same-property replicates (A1/A2; B_T2s1/r1/r2) vs between-property spread; tension pairs
"""
import itertools
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
desc = json.load(open(HERE / "descriptors.json"))
var = json.load(open(HERE / "levels_variants.json"))
TAGS = desc["tags"]
POMA = {  # Lampis et al. POMA 58, 035013 (2025) Table 1: T (N), Z (kg/s), EI (1e-4 N m^2), d (mm), Zto
    "A1": (145.31, 1.0592, 3.032, 0.947, 0.039), "A2": (145.31, 1.0592, 3.032, 0.947, 0.039),
    "B1": (150.87, 1.0998, 4.397, 0.927, 0.055), "C1": (144.40, 1.0525, 3.142, 1.077, 0.039),
    "D1": (142.67, 1.0398, 5.246, 1.123, 0.064), "A_T2": (163.03, 1.1884, 3.969, 1.001, 0.027),
    "B_T2s1": (163.23, 1.1899, 7.295, 0.973, 0.060), "B_T2r1": (163.23, 1.1899, 7.295, 0.973, 0.060),
    "B_T2r2": (163.23, 1.1899, 7.295, 0.973, 0.060), "C_T2": (153.04, 1.1154, 3.969, 1.083, 0.040)}
PROPS = ["T", "Z", "EI", "d", "Zto"]
GROUP = {"A1": "A_T1", "A2": "A_T1", "B1": "B_T1", "C1": "C_T1", "D1": "D_T1", "A_T2": "A_T2",
         "B_T2s1": "B_T2", "B_T2r1": "B_T2", "B_T2r2": "B_T2", "C_T2": "C_T2"}
FOLDS = ("v1", "v2", "v3")
LOGX = {"cal_low", "cal_high", "cal_pl", "fmin", "g"}
PROTO7 = ["cal_low", "d0", "dec10", "desc_b", "floor", "acc_b", "fmin", "cens"]
PROTO10 = ["cal_pl", "dec5_pl", "fmin", "cens"]
rng = np.random.default_rng(0)
report = {}


def xval(tag, fo, name):
    if name in PROPS:
        v = POMA[tag][PROPS.index(name)]
        return np.log(v)
    rec = desc["folder"][f"{tag}|{fo}"] if fo else desc["diagram"][tag]
    v = rec.get(name, np.nan)
    return np.log(v) if name in LOGX else v


def yval(tag, fo, resp):
    if resp == "base":
        return desc["folder"][f"{tag}|{fo}"]["lower"] if fo else desc["diagram"][tag]["lower"]
    lv = var[resp]["levels"][tag]
    return lv[fo] if fo else lv["all"]


def ols(X, y):
    b, *_ = np.linalg.lstsq(X, y, rcond=None)
    return b


def diag_level(tags, preds, resp):
    out = {}
    y = np.array([yval(t, None, resp) for t in tags])
    sst = np.sum((y - y.mean()) ** 2)
    for p in preds:
        x = np.array([xval(t, None, p) for t in tags])
        if np.ptp(x) == 0 or not np.all(np.isfinite(x)):
            continue
        r = np.corrcoef(x, y)[0, 1]
        # LODO CV
        press = 0.0
        for i in range(len(tags)):
            m = np.arange(len(tags)) != i
            if np.ptp(x[m]) == 0:
                press += (y[i] - y[m].mean()) ** 2
                continue
            b = ols(np.column_stack([np.ones(m.sum()), x[m]]), y[m])
            press += (y[i] - b[0] - b[1] * x[i]) ** 2
        # permutation p for |r|
        perm = [abs(np.corrcoef(rng.permutation(x), y)[0, 1]) for _ in range(5000)]
        out[p] = dict(r=float(r), cv_r2=float(1 - press / sst), p_perm=float(np.mean(np.array(perm) >= abs(r))),
                      slope=float(ols(np.column_stack([np.ones(len(x)), x]), y)[1]))
    return out


def folder_level(tags, preds, resp):
    rows = [(t, fo) for t in tags for fo in FOLDS]
    y = np.array([yval(t, fo, resp) for t, fo in rows])
    S = np.array([[fo == f for f in FOLDS] for _, fo in rows], float)
    diag = np.array([t for t, _ in rows])
    def cv(Xextra):
        err = []
        for t in tags:
            m = diag != t
            X = np.column_stack([S] + ([Xextra] if Xextra is not None else []))
            b = ols(X[m], y[m])
            err += list(y[~m] - X[~m] @ b)
        return float(np.sqrt(np.mean(np.square(err))))
    out = {"speed_only": dict(cv_rmse=cv(None))}
    for p in preds:
        x = np.array([xval(t, fo, p) for t, fo in rows])
        if not np.all(np.isfinite(x)):
            continue
        X = np.column_stack([S, x])
        b = ols(X, y)
        out[p] = dict(cv_rmse=cv(x), slope=float(b[-1]))
    return out


def within(tags, preds, resp, nperm=4000):
    rows = [(t, fo) for t in tags for fo in FOLDS]
    y = np.array([yval(t, fo, resp) for t, fo in rows])
    S = np.array([[fo == f for f in FOLDS[1:]] for _, fo in rows], float)
    G = np.array([[t == u for u in tags] for t, _ in rows], float)
    base = np.column_stack([G, S])
    out = {}
    for p in preds:
        x = np.array([xval(t, fo, p) for t, fo in rows])
        if not np.all(np.isfinite(x)):
            continue
        def coef(xx):
            X = np.column_stack([base, xx])
            b = ols(X, y)
            res = y - X @ b
            return b[-1], res
        b, res = coef(x)
        dof = len(y) - base.shape[1] - 1
        X = np.column_stack([base, x])
        cov = np.sum(res ** 2) / dof * np.linalg.pinv(X.T @ X)
        se = float(np.sqrt(cov[-1, -1]))
        # residual SD of x within diagrams (after speed FE) -> how much session variation there is
        bx = ols(base, x)
        sdx = float(np.std(x - base @ bx))
        perm = []
        for _ in range(nperm):
            xp = x.copy()
            for k in range(len(tags)):
                idx = np.arange(3 * k, 3 * k + 3)
                xp[idx] = x[rng.permutation(idx)]
            perm.append(coef(xp)[0])
        out[p] = dict(slope=float(b), se=se, dof=int(dof), sd_x_within=sdx, p_perm=float(np.mean(np.abs(perm) >= abs(b))))
    return out


def show_diag(title, res):
    print(title)
    for p, r in sorted(res.items(), key=lambda kv: -kv[1]["cv_r2"]):
        kind = "물성" if p in PROPS else "세션"
        print(f"   [{kind}] {p:8s} r {r['r']:+.2f}  LODO-CV R² {r['cv_r2']:+.2f}  perm p {r['p_perm']:.3f}  slope {r['slope']:+.3f}")


def show_folder(title, res):
    print(title)
    b0 = res["speed_only"]["cv_rmse"]
    print(f"   속도만: CV RMSE {b0:.3f}")
    for p, r in sorted(((k, v) for k, v in res.items() if k != "speed_only"), key=lambda kv: kv[1]["cv_rmse"]):
        kind = "물성" if p in PROPS else "세션"
        print(f"   [{kind}] {p:8s} CV RMSE {r['cv_rmse']:.3f} ({r['cv_rmse'] - b0:+.3f})  slope {r['slope']:+.3f}")


def main():
    T7 = [t for t in TAGS if np.isfinite(desc["diagram"][t].get("cal_low", np.nan))]
    for resp in ("base", "c=0.2", "c=0.3"):
        print(f"\n================ 반응변수: 최소 활 힘 수준 ({resp})")
        rd7 = diag_level(T7, PROTO7 + PROPS, resp)
        show_diag(f" A. 도표 단위, 메타데이터 있는 {len(T7)}장 ({', '.join(T7)})", rd7)
        rd10 = diag_level(TAGS, PROTO10 + PROPS, resp)
        show_diag(f" A'. 도표 단위, 10장 전부 (특징 기반 세션 기술자)", rd10)
        rf7 = folder_level(T7, PROTO7 + PROPS, resp)
        show_folder(f" B. 폴더 단위 {3 * len(T7)}개 + 속도 고정효과, 도표 하나씩 빼는 교차검증", rf7)
        rf10 = folder_level(TAGS, PROTO10 + PROPS, resp)
        show_folder(f" B'. 폴더 단위 30개", rf10)
        report[resp] = dict(diag7=rd7, diag10=rd10, folder7=rf7, folder10=rf10)
    print("\n================ C. 같은 현 안의 세션 변동 (도표 고정효과 + 속도 고정효과)")
    for resp in ("base", "c=0.2"):
        w7 = within(T7, PROTO7, resp)
        w10 = within(TAGS, PROTO10, resp)
        report[resp]["within7"] = w7; report[resp]["within10"] = w10
        print(f" 반응 {resp}")
        for p, r in list(w7.items()) + [(f"{k} (10장)", v) for k, v in w10.items()]:
            print(f"   {p:16s} 기울기 {r['slope']:+.3f} ± {r['se']:.3f} (자유도 {r['dof']}), 순열 p {r['p_perm']:.3f}, 도표 안 x 표준편차 {r['sd_x_within']:.3f}")
    # combined: best protocol + property at diagram level (7)
    print("\n================ 결합 모델 (도표 7장, base): 수준 = a + b·세션 + c·물성")
    y = np.array([yval(t, None, "base") for t in T7])
    for p in ("cal_low", "dec10", "desc_b", "fmin"):
        for q in PROPS:
            X = np.column_stack([np.ones(len(T7)), [xval(t, None, p) for t in T7], [xval(t, None, q) for t in T7]])
            b = ols(X, y)
            press = 0.0
            for i in range(len(T7)):
                m = np.arange(len(T7)) != i
                bb = ols(X[m], y[m]); press += (y[i] - X[i] @ bb) ** 2
            print(f"   {p:8s} + {q:4s}: b {b[1]:+.3f}, c {b[2]:+.3f}, LODO-CV R² {1 - press / np.sum((y - y.mean()) ** 2):+.2f}")
    # D. replicates vs between
    print("\n================ D. 같은 물성 반복 vs 다른 물성")
    for resp in ("base", "c=0.2", "c=0.3", "zero_corrected"):
        tags = [t for t in TAGS if resp != "zero_corrected" or t in var["zero_corrected"]["levels"]]
        lv = {t: yval(t, None, resp) for t in tags}
        groups = {}
        for t in tags:
            groups.setdefault(GROUP[t], []).append(lv[t])
        ssw = sum(np.sum((np.array(v) - np.mean(v)) ** 2) for v in groups.values())
        dfw = sum(len(v) - 1 for v in groups.values())
        gm = np.mean([lv[t] for t in tags])
        ssb = sum(len(v) * (np.mean(v) - gm) ** 2 for v in groups.values())
        dfb = len(groups) - 1
        F = (ssb / dfb) / (ssw / dfw) if dfw else np.nan
        # permutation of tags over group labels
        labs = [GROUP[t] for t in tags]; vals = np.array([lv[t] for t in tags])
        def Fstat(lbl):
            g = {}
            for l, v in zip(lbl, vals):
                g.setdefault(l, []).append(v)
            sw = sum(np.sum((np.array(v) - np.mean(v)) ** 2) for v in g.values())
            sb = sum(len(v) * (np.mean(v) - vals.mean()) ** 2 for v in g.values())
            return (sb / dfb) / (sw / dfw) if sw > 0 else np.inf
        perms = [Fstat(rng.permutation(labs)) for _ in range(5000)]
        pF = float(np.mean(np.array(perms) >= F))
        rep = {g: v for g, v in groups.items() if len(v) > 1}
        print(f" {resp:15s}: 반복 내 SD {np.sqrt(ssw / dfw):.3f} (자유도 {dfw}; " + ", ".join(f"{g} {np.round(v, 3).tolist()}" for g, v in rep.items())
              + f") | 물성 그룹 평균의 SD {np.std([np.mean(v) for v in groups.values()], ddof=1):.3f} | F {F:.2f}, 순열 p {pF:.3f}")
        report.setdefault("anova", {})[resp] = dict(sd_within=float(np.sqrt(ssw / dfw)), F=float(F), p_perm=pF)
    print("\n================ 장력 쌍 (T1 → T2), 최소 활 힘 수준 차이와 세션 기술자 차이")
    pairs = (("A1", "A_T2"), ("A2", "A_T2"), ("B1", "B_T2s1"), ("B1", "B_T2r1"), ("B1", "B_T2r2"), ("C1", "C_T2"))
    report["pairs"] = {}
    for a, b in pairs:
        zr = np.log(POMA[b][1] / POMA[a][1])
        line = f" {a:3s}→{b:7s} Z비 {np.exp(zr):.3f}:"
        rec = {}
        for resp in ("base", "c=0.2", "c=0.3"):
            dl = yval(b, None, resp) - yval(a, None, resp)
            rec[resp] = dict(dlevel=float(dl), z_exp=float(dl / zr))
            line += f" {resp} Δ{dl:+.3f} (Z지수 {dl / zr:+.1f});"
        if a in var["zero_corrected"]["levels"] and b in var["zero_corrected"]["levels"]:
            dl = yval(b, None, "zero_corrected") - yval(a, None, "zero_corrected")
            line += f" 영점보정 Δ{dl:+.3f};"
            rec["zero_corrected"] = dict(dlevel=float(dl), z_exp=float(dl / zr))
        dd = {k: xval(b, None, k) - xval(a, None, k) for k in ("cal_low", "dec10", "desc_b", "cal_pl", "dec5_pl", "fmin")}
        line += " | Δlog cal_low {cal_low:+.2f}, Δdec10 {dec10:+.3f}, Δdesc_b {desc_b:+.3f}, Δlog cal_pl {cal_pl:+.2f}, Δdec5_pl {dec5_pl:+.3f}, Δlog fmin {fmin:+.2f}".format(**dd)
        rec["d_desc"] = {k: float(v) for k, v in dd.items()}
        report["pairs"][f"{a}->{b}"] = rec
        print(line)
    # descriptor-property collinearity
    print("\n================ 세션 기술자와 물성의 상관 (도표 7장 / 10장)")
    for p in PROTO7:
        x = np.array([xval(t, None, p) for t in T7])
        print(f"   {p:8s} " + " ".join(f"{q} {np.corrcoef(x, [xval(t, None, q) for t in T7])[0, 1]:+.2f}" for q in PROPS))
    for p in PROTO10:
        x = np.array([xval(t, None, p) for t in TAGS])
        print(f"   {p:8s}(10) " + " ".join(f"{q} {np.corrcoef(x, [xval(t, None, q) for t in TAGS])[0, 1]:+.2f}" for q in PROPS))
    json.dump(report, open(HERE / "analysis.json", "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
