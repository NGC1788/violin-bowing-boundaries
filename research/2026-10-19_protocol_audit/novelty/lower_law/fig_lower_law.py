import json
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
on = json.load(open("onset_axis.json")); me = json.load(open("results.json"))
ses = list(on["onset"].keys())
fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
for i, s in enumerate(ses):
    o = on["onset"][s]; m = me[s]["measured"]
    ax[0].plot(m["b_beta"], i, "o", color="#d38325", ms=6, label="window force (t = 0.2 s)" if i == 0 else None)
    ax[0].errorbar(o["b_beta"], i, xerr=[[o["b_beta"] - o["b_beta_ci"][0]], [o["b_beta_ci"][1] - o["b_beta"]]], fmt="s", color="#2563a0", ms=6, capsize=2, label="onset force" if i == 0 else None)
    ax[0].plot([m["b_beta"], o["b_beta"]], [i, i], color="0.75", lw=1, zorder=0)
for x, lab in ((-2, "Schelleng v/β²"), (-1, "v/β (Beigi sim.)")):
    ax[0].axvline(x, color="k", ls=":", lw=1); ax[0].text(x, -1.2, lab, ha="center", fontsize=8)
r = on["re_onset"]["beta"]; ax[0].axvspan(r["ci_hksj"][0], r["ci_hksj"][1], color="#2563a0", alpha=.12)
ax[0].set_ylim(-1.6, len(ses) - 0.4); ax[0].set_yticks(range(len(ses))); ax[0].set_yticklabels([s.replace("_std3", "") for s in ses], fontsize=8)
ax[0].set_xlabel("β exponent of the minimum bow force"); ax[0].set_title("(a) per-session β exponents (post hoc)", fontsize=10); ax[0].legend(fontsize=8, frameon=False, loc="upper left")
for i, s in enumerate(ses):
    o = on["onset"][s]; ax[1].plot(o["b_v"], o["b_beta"], "s", color="#2563a0"); ax[1].annotate(s.replace("_std3", ""), (o["b_v"], o["b_beta"]), xytext=(4, 3), textcoords="offset points", fontsize=7)
ax[1].plot([1], [-2], "k*", ms=12); ax[1].annotate("Schelleng", (1, -2), xytext=(6, -2), textcoords="offset points", fontsize=8)
ax[1].plot([1], [-1], "k^", ms=9); ax[1].annotate("v/β", (1, -1), xytext=(6, -2), textcoords="offset points", fontsize=8)
ax[1].plot([0], [-2.18], "kx", ms=9); ax[1].annotate("our refit of Schoonderwaldt 2008\nviolin D4: v^-0.09 β^-2.18 (window axis)", (0, -2.18), xytext=(6, -10), textcoords="offset points", fontsize=7)
ax[1].set_xlabel("speed exponent (onset axis)"); ax[1].set_ylabel("β exponent (onset axis)"); ax[1].set_title("(b) F_min ∝ v^a β^b, 11 sessions, onset axis", fontsize=10)
for a in ax:
    for k in ("top", "right"): a.spines[k].set_visible(False)
fig.tight_layout(); fig.savefig("fig_lower_law.png", dpi=170)
