import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
font_manager.fontManager.addfont("/System/Library/Fonts/AppleSDGothicNeo.ttc")
plt.rcParams["font.family"] = "Apple SD Gothic Neo"; plt.rcParams["axes.unicode_minus"] = False
names = ["단순 기준선\n(A형 경계)", "물리: 3상수\n+원논문 Z", "물리: 3상수\nZ 없음", "물리: Beigi\n+Z", "물리: Schelleng\n+Z", "데이터:\n부스팅"]
low = [0.250, 0.520, 0.347, 0.455, 0.939, 0.271]; up = [0.167, 0.121, 0.189, 0.215, 0.215, 0.176]
col = ["#888", "#3b6fb6", "#3b6fb6", "#3b6fb6", "#3b6fb6", "#c06030"]
fig, ax = plt.subplots(1, 3, figsize=(14, 4.4), gridspec_kw={"width_ratios": [3, 3, 2]})
for a, vals, t in ((ax[0], low, "최소 활 힘 예측 오차"), (ax[1], up, "최대 활 힘 예측 오차")):
    a.bar(range(6), vals, color=col); a.set_xticks(range(6), names, fontsize=8); a.set_title(t, fontsize=11)
    a.set_ylabel("|log(예측/실측)| 중앙값")
    for i, v in enumerate(vals): a.text(i, v + 0.01, f"{v:.2f}", ha="center", fontsize=8)
a = ax[2]
a.errorbar([0], [1.71], yerr=[[1.71 - 0.92], [2.49 - 1.71]], fmt="o", color="#3b6fb6", capsize=5, label="최대 활 힘")
a.errorbar([1], [-4.69], yerr=[[-4.69 + 6.27], [-3.32 + 4.69]], fmt="o", color="#c06030", capsize=5, label="최소 활 힘")
a.axhline(1, ls=":", color="#3b6fb6"); a.axhline(2, ls="--", color="#c06030"); a.axhline(0, color="#999", lw=0.8)
a.text(1.25, 2.1, "Schelleng 최소 (Z²)", fontsize=8, color="#c06030"); a.text(-0.35, 1.1, "Schelleng 최대 (Z¹)", fontsize=8, color="#3b6fb6")
a.set_xticks([0, 1], ["최대 활 힘", "최소 활 힘"]); a.set_xlim(-0.5, 1.9); a.set_ylabel("Z 지수 (경계 이동 ÷ log Z비)")
a.set_title("장력이 높아진 현(Z +12%)에서의 Z 지수", fontsize=11)
fig.suptitle("사전 등록 확인 실험: 아직 안 본 A형 고장력 현(A_T2), A1·A2로 학습, t = 0.2 s", fontsize=11)
fig.tight_layout(); fig.savefig("fig_confirm_A_T2.png", dpi=160); print("saved")
