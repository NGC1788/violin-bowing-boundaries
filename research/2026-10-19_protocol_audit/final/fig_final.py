"""Final figure: zero-shot boundary error on unseen strings per model, against the repeat baseline and self-fit floor."""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
font_manager.fontManager.addfont("/System/Library/Fonts/AppleSDGothicNeo.ttc")
plt.rcParams["font.family"] = "Apple SD Gothic Neo"; plt.rcParams["axes.unicode_minus"] = False
R = json.load(open("compare_t0.2.json"))["results"]
tests = ["B1", "C1", "D1", "B_T2r1", "B_T2r2"]
models = ["P-Sch", "P-Beigi", "P-free", "D-logit", "D-gbm"]
names = {"P-Sch": "물리: Schelleng\n(v/β², Z²)", "P-Beigi": "물리: Beigi\n(v/β, Z)", "P-free": "물리: 지수 자유\n(3개 상수)", "D-logit": "데이터: 로지스틱", "D-gbm": "데이터: 부스팅"}
fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), sharey=False)
for ax, which, rep, self_ in ((axes[0], "lower", (0.200, 0.266), (0.159, 0.245)), (axes[1], "upper", (0.088, 0.138), (0.095, 0.148))):
    for j, m in enumerate(models):
        vals = [R[t][m]["zero_shot"][f"{which}_med_abs_log"] for t in tests]
        ax.bar(j, np.mean(vals), color="#3b6fb6" if m.startswith("P") else "#c06030", alpha=0.85)
        ax.plot([j] * len(vals), vals, "k.", ms=5)
    ax.axhspan(*rep, color="#2a9d55", alpha=0.18, label="같은 현을 다시 잴 때의 오차 (Part 6 반복)")
    ax.axhspan(*self_, color="#888", alpha=0.15, label="같은 현으로 맞춘 3상수 물리식의 오차 범위")
    ax.set_xticks(range(len(models)), [names[m] for m in models], fontsize=8)
    ax.set_ylabel("경계 예측 오차 |log(예측/실측)| 중앙값")
    ax.set_title({"lower": "아래 경계(최소 활 힘)", "upper": "위 경계(최대 활 힘)"}[which] + " — 처음 보는 현 5개", fontsize=11)
    ax.legend(fontsize=8, loc="upper right")
fig.suptitle("A1·A2(A형 현)로 학습 → B·C·D형, B형 고장력 현 예측 (t = 0.2 s 창). 막대 = 평균, 점 = 현별", fontsize=11)
fig.tight_layout(); fig.savefig("fig_final_prediction.png", dpi=160); print("saved")
