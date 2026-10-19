"""Post hoc baselines for compare.py (not part of the frozen plan; Claude, 2026-10-18):
(a) repeat baseline: train on B_T2 rep_1 -> test rep_2 and vice versa (same string, same conditions);
(b) self-fit: the P-free form fitted on the test specimen itself (form + boundary-estimation floor);
(c) leave-one-specimen-out with all other specimens as training (does diversity help?)."""
import numpy as np
import compare as C
from data import load

T = 0.2
ALL = ("A1", "A2", "B1", "C1", "D1", "B_T2r1", "B_T2r2")
dat = {t: load(t) for t in ALL}
z_of = {t: float(dat[t]["Z"][0]) for t in ALL}
rows = {}
for t in ALL:
    rows[t] = C.intervals(dat[t], T)
    for r in rows[t]:
        r["tag"] = t


def run(train_tags, test_tag, label):
    tr_rows = sum((rows[t] for t in train_tags), [])
    phys = C.physics_models(tr_rows, z_of)
    dm = C.data_models([dat[t] for t in train_tags])
    out = []
    for name, pf in (("P-free", phys["P-free"][0]), ("P-Sch", phys["P-Sch"][0])):
        sb = C.score_boundaries(rows[test_tag], [pf(r, test_tag) for r in rows[test_tag]])
        out.append(f"{name} 아래 {sb['lower'][0]:.3f} 위 {sb['upper'][0]:.3f}")
    sb = C.score_boundaries(rows[test_tag], [C.band_from_prob(dm["D-gbm"], r, T) for r in rows[test_tag]])
    out.append(f"D-gbm 아래 {sb['lower'][0]:.3f} 위 {sb['upper'][0]:.3f}")
    print(f"{label:28s} → {test_tag:7s}: " + " | ".join(out), flush=True)


print("(a) 반복 기준선 (같은 현을 두 번 잼)")
run(("B_T2r1",), "B_T2r2", "B_T2r1로 학습")
run(("B_T2r2",), "B_T2r1", "B_T2r2로 학습")
print("(b) 자기 적합 (시험 표본 자신으로 학습 = 식의 한계 + 경계 추정 잡음)")
for t in ("B1", "C1", "D1", "B_T2r1", "B_T2r2"):
    run((t,), t, "자기 자신")
print("(c) 나머지 표본 전부로 학습 (A1·A2 + 다른 종류)")
for t in ("B1", "C1", "D1", "B_T2r1"):
    others = tuple(x for x in ALL if x != t and not (t.startswith("B_T2") and x.startswith("B_T2")))
    run(others, t, "나머지 " + "+".join(others))
