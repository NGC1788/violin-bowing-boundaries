"""Follow-up checks (Claude, 2026-10-19; output analysis2.json).
 1. label-free history descriptor: median log(F_onset / F(t=0.2 s)) of strokes whose t = 0.2 s reading is
    0.10-0.30 N (does not use the Helmholtz labels, unlike desc_b)
 2. force-axis swap: lower level when the boundary is expressed in the force before the bow moves (F_onset, 7 diagrams)
    or in the plateau-start force (0.2 s window at t = 0, all 10), Helmholtz label still at t = 0.2 s
 3. free-slope fit per diagram: level at the reference point (v 0.1 m/s, beta 0.1) and beta/v exponents
 4. leverage: correlations without A1/A2 (they define the reference law)
 5. residual string differences after the label-free decrease descriptor (dec5_pl, all 10 diagrams)
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
import descriptors as DS  # noqa: E402
import analysis as AN  # noqa: E402

K, D = DS.K, DS.D
TAGS = DS.TAGS
desc = json.load(open(HERE / "descriptors.json"))
rng = np.random.default_rng(1)
out = {}


def ramp_table(tag):
    rm = {}
    for r in csv.DictReader(open(DS.RAMP[tag])):
        if r["error"] or r["force_at_onset"] in ("", "nan"):
            continue
        rm[(DS.speed_of(r["folder"]), int(r["trial"]))] = float(r["force_at_onset"])
    return rm


def with_force_axis(tag, kind):
    """copy of the cached windows where the t = 0.2 s window force is replaced by another force of the same stroke."""
    d = dict(D.load(tag))
    f = d["force"].copy()
    fold = np.array([DS.speed_of(d["folders"][k]) for k in d["folder"]])
    if kind == "onset":
        rm = ramp_table(tag)
        f = np.array([rm.get((fo, int(tr)), np.nan) for fo, tr in zip(fold, d["trial"])])
    elif kind == "plateau0":
        key0 = {}
        m0 = np.isclose(d["t"], 0.0)
        for i in np.flatnonzero(m0):
            key0[(int(d["folder"][i]), int(d["trial"][i]))] = d["force"][i]
        f = np.array([key0.get((int(a), int(b)), np.nan) for a, b in zip(d["folder"], d["trial"])])
    keep = np.isfinite(f)
    for k in list(d):
        if k in ("folders", "Z", "f0"):
            continue
        d[k] = d[k][keep] if k != "force" else f[keep]
    rows = K.C.intervals(d, DS.T)
    for r in rows:
        r["tag"] = tag
        r["fold"] = DS.speed_of(d["folders"][d["folder"][r["idx"][0]]])
    return rows


def levels(rows_of):
    tr = rows_of["A1"] + rows_of["A2"]
    noz = K.C.physics_models(tr, {t: 1.0 for t in TAGS})
    base = lambda r: noz["P-free"][0](r, "A1")[0]
    return {t: float(K.offset_fit(rs, base, "lower")) for t, rs in rows_of.items()}


def corr(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    r = np.corrcoef(x, y)[0, 1]
    perm = [abs(np.corrcoef(rng.permutation(x), y)[0, 1]) for _ in range(5000)]
    return float(r), float(np.mean(np.array(perm) >= abs(r)))


def main():
    base_rows = {t: DS.rows_by_folder(t)[1] for t in TAGS}
    lv0 = {t: desc["diagram"][t]["lower"] for t in TAGS}
    T7 = [t for t in TAGS if t in DS.RAMP]
    # 1. label-free history descriptor
    print("1. 라벨 없이 계산한 하강 이력: t=0.2 s 판독이 0.10~0.30 N인 긋기의 median log(F_onset/F(0.2 s))")
    hist = {}
    for t in T7:
        rm = ramp_table(t)
        d = D.load(t)
        m = np.isclose(d["t"], 0.2) & (d["force"] >= 0.10) & (d["force"] <= 0.30)
        vals = defaultdict(list)
        for i in np.flatnonzero(m):
            fo = DS.speed_of(d["folders"][d["folder"][i]])
            fon = rm.get((fo, int(d["trial"][i])), np.nan)
            vals[fo].append(np.log(fon / d["force"][i]))
        hist[t] = float(np.nanmean([np.nanmedian(vals[fo]) for fo in ("v1", "v2", "v3")]))
        print(f"   {t:7s} hist {hist[t]:+.3f}  (n {sum(len(v) for v in vals.values())})  level {lv0[t]:+.3f}")
    r, p = corr([hist[t] for t in T7], [lv0[t] for t in T7])
    print(f"   상관 (7장): r {r:+.2f}, 순열 p {p:.3f}")
    out["hist_fix"] = dict(values=hist, r=r, p=p)
    # 2. force-axis swap
    print("\n2. 힘 축 바꾸기 (라벨은 t=0.2 s 그대로)")
    rows_on = {t: with_force_axis(t, "onset") for t in T7}
    lv_on = levels(rows_on)
    rows_p0 = {t: with_force_axis(t, "plateau0") for t in TAGS}
    lv_p0 = levels(rows_p0)
    for t in TAGS:
        print(f"   {t:7s} 현재 힘 {lv0[t]:+.3f} | 평탄 시작 힘 {lv_p0[t]:+.3f} | 출발 힘 " + (f"{lv_on[t]:+.3f}" if t in lv_on else "  n/a"))
    out["axis"] = dict(current=lv0, plateau0=lv_p0, onset=lv_on)
    for name, lv, tags in (("현재 힘", lv0, TAGS), ("평탄 시작 힘", lv_p0, TAGS), ("출발 힘", lv_on, T7), ("현재 힘 (7장)", lv0, T7)):
        v = np.array([lv[t] for t in tags])
        print(f"   {name:12s}: 표본 간 SD {np.std(v, ddof=1):.3f}, 범위 {np.ptp(v):.3f}")
    # 3. free-slope fit per diagram
    print("\n3. 표본별 자유 지수 적합 (log F = a + b_v log v + b_β log β, 구간검열): 기준점 v 0.1, β 0.1의 수준")
    free = {}
    tr = base_rows["A1"] + base_rows["A2"]
    pA = K.C.censored_fit(tr, lambda r: (np.array([1.0, r["lv"], r["lb"]]), 0.0), "lower")
    ref = lambda p: p[0] + p[1] * np.log(0.1) + p[2] * np.log(0.1)
    for t in TAGS:
        p = K.C.censored_fit(base_rows[t], lambda r: (np.array([1.0, r["lv"], r["lb"]]), 0.0), "lower")
        bt = []
        for _ in range(100):
            rs = base_rows[t]
            bt.append(ref(K.C.censored_fit([rs[i] for i in rng.integers(0, len(rs), len(rs))],
                                           lambda r: (np.array([1.0, r["lv"], r["lb"]]), 0.0), "lower")) - ref(pA))
        free[t] = dict(level_ref=float(ref(p) - ref(pA)), ci=[float(np.percentile(bt, 2.5)), float(np.percentile(bt, 97.5))],
                       b_v=float(p[1]), b_beta=float(p[2]), F_ref=float(np.exp(ref(p))))
        print(f"   {t:7s} 수준(기준점) {free[t]['level_ref']:+.3f} [{free[t]['ci'][0]:+.3f},{free[t]['ci'][1]:+.3f}]  F(0.1,0.1) {free[t]['F_ref']:.3f} N"
              f"  v 지수 {p[1]:+.2f}  β 지수 {p[2]:+.2f}")
    out["free"] = free
    # 4. leverage
    print("\n4. 지렛대 점검: A1·A2(기준 법칙을 만든 두 장) 빼고 상관")
    T5 = [t for t in T7 if t not in ("A1", "A2")]
    T8 = [t for t in TAGS if t not in ("A1", "A2")]
    lvf = {t: free[t]["level_ref"] for t in TAGS}
    for name, tags, preds in (("5장", T5, ["dec10", "cal_low", "fmin", "floor", "T", "Z", "EI", "d", "Zto"]),
                              ("8장", T8, ["dec5_pl", "cal_pl", "fmin", "T", "Z", "EI", "d", "Zto"])):
        for resp_name, lv in (("기본 수준", lv0), ("자유지수 기준점 수준", lvf)):
            line = f"   {name} {resp_name}: "
            for q in preds:
                x = [AN.xval(t, None, q) for t in tags]
                if np.ptp(x) == 0:
                    continue
                r, p = corr(x, [lv[t] for t in tags])
                line += f"{q} {r:+.2f}(p {p:.2f})  "
                out.setdefault("leverage", {})[f"{name}|{resp_name}|{q}"] = dict(r=r, p=p)
            print(line)
    print("   전체 10장, 자유지수 기준점 수준: " + "  ".join(
        f"{q} {corr([AN.xval(t, None, q) for t in TAGS], [lvf[t] for t in TAGS])[0]:+.2f}" for q in ("dec5_pl", "cal_pl", "fmin", "T", "Z", "EI", "d", "Zto")))
    # 5. residual string differences after dec5_pl
    print("\n5. 라벨 없는 감소 기술자(dec5_pl)로 보정한 뒤 남는 현 차이 (10장)")
    for resp_name, lv in (("기본 수준", lv0), ("자유지수 기준점 수준", lvf), ("c=0.2", {t: AN.yval(t, None, 'c=0.2') for t in TAGS})):
        x = np.array([AN.xval(t, None, "dec5_pl") for t in TAGS]); y = np.array([lv[t] for t in TAGS])
        X = np.column_stack([np.ones(len(x)), x]); b = np.linalg.lstsq(X, y, rcond=None)[0]
        res = y - X @ b
        groups = defaultdict(list)
        for t, e in zip(TAGS, res):
            groups[AN.GROUP[t]].append(e)
        ssw = sum(np.sum((np.array(v) - np.mean(v)) ** 2) for v in groups.values()); dfw = sum(len(v) - 1 for v in groups.values())
        line = "   " + resp_name + f": 기울기 {b[1]:+.2f}/N, 잔차 " + " ".join(f"{t} {e:+.2f}" for t, e in zip(TAGS, res))
        print(line)
        gm = {g: float(np.mean(v)) for g, v in groups.items()}
        print(f"      그룹 평균 잔차 " + " ".join(f"{g} {m:+.2f}" for g, m in gm.items()) + f" | 반복 내 SD {np.sqrt(ssw / dfw):.3f} (자유도 {dfw}), 그룹 평균 SD {np.std(list(gm.values()), ddof=1):.3f}")
        # type effect on residual: C vs others
        isC = np.array([t.startswith("C") for t in TAGS])
        diff = res[isC].mean() - res[~isC].mean()
        perm = []
        for _ in range(5000):
            pm = rng.permutation(isC)
            perm.append(res[pm].mean() - res[~pm].mean())
        pC = float(np.mean(np.abs(perm) >= abs(diff)))
        print(f"      C형 두 장 잔차 평균 − 나머지 {diff:+.3f} (순열 p {pC:.3f}; 10장 중 2장을 고르는 45가지)")
        out.setdefault("residual", {})[resp_name] = dict(slope=float(b[1]), res={t: float(e) for t, e in zip(TAGS, res)},
                                                       group_mean=gm, sd_within=float(np.sqrt(ssw / dfw)), C_minus_rest=float(diff), p_C=pC)
    json.dump(out, open(HERE / "analysis2.json", "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
