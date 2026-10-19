"""Last checks (Claude, 2026-10-19; output analysis3.json).
 1. label-free attack descriptor: median log10 acceleration of force ranks 11-30 (does not use the boundary)
 2. upper boundary contrast: same predictors for the upper level
 3. per-diagram leave-one-diagram-out predictions of the lower level: dec5_pl model vs Z model vs 'same as type A' (0)
 4. tension pairs after the dec5_pl adjustment (LODO residuals)
"""
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.argv = sys.argv[:1]
HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
import analysis as AN  # noqa: E402
import descriptors as DS  # noqa: E402

desc = AN.desc
TAGS = AN.TAGS
rng = np.random.default_rng(2)
out = {}


def corr(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    r = np.corrcoef(x, y)[0, 1]
    perm = [abs(np.corrcoef(rng.permutation(x), y)[0, 1]) for _ in range(5000)]
    return float(r), float(np.mean(np.array(perm) >= abs(r)))


def lodo(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    pred = np.empty_like(y)
    for i in range(len(y)):
        m = np.arange(len(y)) != i
        if np.ptp(x[m]) == 0:
            pred[i] = y[m].mean(); continue
        b = np.polyfit(x[m], y[m], 1)
        pred[i] = np.polyval(b, x[i])
    return pred


def main():
    T7 = [t for t in TAGS if t in DS.RAMP]
    print("1. 경계와 무관한 출발 가속 기술자: 힘 순위 11~30의 median log10 가속도 (m/s²)")
    acc = {}
    for t in T7:
        vals = defaultdict(list)
        for r in csv.DictReader(open(DS.RAMP[t])):
            if r["error"] or r["accel"] in ("", "nan"):
                continue
            rk = int(DS.rank_of(t, int(r["trial"])))
            if 11 <= rk <= 30:
                vals[DS.speed_of(r["folder"])].append(np.log10(float(r["accel"])))
        acc[t] = {fo: float(np.median(v)) for fo, v in vals.items()}
        print(f"   {t:7s} " + " ".join(f"{fo} {acc[t][fo]:+.2f}" for fo in ("v1", "v2", "v3")) + f" | lower level {desc['diagram'][t]['lower']:+.3f}")
    x = [np.mean(list(acc[t].values())) for t in T7]
    r, p = corr(x, [desc["diagram"][t]["lower"] for t in T7])
    print(f"   도표 평균 가속 vs 최소 수준: r {r:+.2f} (p {p:.3f})")
    out["acc_rank11_30"] = dict(values=acc, r=r, p=p)

    print("\n2. 최대 활 힘 수준과의 대조 (같은 예측변수)")
    up = {t: desc["diagram"][t]["upper"] for t in TAGS}
    for name, tags, preds in (("7장", T7, ["dec10", "cal_low", "desc_b", "T", "Z", "EI", "d", "Zto"]),
                              ("10장", TAGS, ["dec5_pl", "cal_pl", "fmin", "T", "Z", "EI", "d", "Zto"])):
        line = f"   {name}: "
        for q in preds:
            xx = [AN.xval(t, None, q) for t in tags]
            r, p = corr(xx, [up[t] for t in tags])
            line += f"{q} {r:+.2f}(p {p:.2f})  "
            out.setdefault("upper", {})[f"{name}|{q}"] = dict(r=r, p=p)
        print(line)

    print("\n3. 도표 하나씩 빼고 예측한 최소 활 힘 수준 (기본 수준; 10장)")
    y = np.array([desc["diagram"][t]["lower"] for t in TAGS])
    models = {"A형과 같다(0)": np.zeros(len(TAGS))}
    for q in ("dec5_pl", "Z", "EI", "Zto", "d"):
        models[q] = lodo([AN.xval(t, None, q) for t in TAGS], y)
    print("   " + "도표".ljust(8) + "실측".rjust(8) + "".join(k.rjust(14) for k in models))
    for i, t in enumerate(TAGS):
        print("   " + t.ljust(8) + f"{y[i]:+8.3f}" + "".join(f"{models[k][i]:+14.3f}" for k in models))
    for k, pr in models.items():
        print(f"   {k:14s} 평균 |오차| {np.mean(np.abs(pr - y)):.3f}, RMSE {np.sqrt(np.mean((pr - y) ** 2)):.3f}")
    out["lodo"] = {k: dict(pred=[float(v) for v in pr], mae=float(np.mean(np.abs(pr - y)))) for k, pr in models.items()}

    print("\n4. 장력 쌍: dec5_pl로 보정한 LODO 잔차의 차이 (T2 − T1)")
    res = y - models["dec5_pl"]
    rd = dict(zip(TAGS, res))
    for a, b in (("A1", "A_T2"), ("A2", "A_T2"), ("B1", "B_T2s1"), ("B1", "B_T2r1"), ("B1", "B_T2r2"), ("C1", "C_T2")):
        zr = np.log(AN.POMA[b][1] / AN.POMA[a][1])
        dl = rd[b] - rd[a]
        print(f"   {a}→{b}: 원래 Δ {y[TAGS.index(b)] - y[TAGS.index(a)]:+.3f} → 보정 Δ {dl:+.3f} (Z 지수 {dl / zr:+.1f})")
        out.setdefault("pairs_adj", {})[f"{a}->{b}"] = dict(d_raw=float(y[TAGS.index(b)] - y[TAGS.index(a)]), d_adj=float(dl), z_exp=float(dl / zr))
    pooled = np.mean([rd["A_T2"] - 0.5 * (rd["A1"] + rd["A2"]), np.mean([rd[t] for t in ("B_T2s1", "B_T2r1", "B_T2r2")]) - rd["B1"], rd["C_T2"] - rd["C1"]])
    zbar = np.mean([np.log(1.1884 / 1.0592), np.log(1.1899 / 1.0998), np.log(1.1154 / 1.0525)])
    print(f"   세 쌍 평균 보정 Δ {pooled:+.3f} (평균 log Z비 {zbar:.3f} → Z 지수 {pooled / zbar:+.1f}); 같은 물성 반복 잔차 SD ≈ 0.08~0.1")
    out["pairs_adj_pooled"] = dict(d=float(pooled), z_exp=float(pooled / zbar))
    json.dump(out, open(HERE / "analysis3.json", "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
