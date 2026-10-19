"""Fig. 4: high-force frozen test (Claude, 2026-10-19). Reads highforce_confirm/score_result.json only."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

R = json.loads((Path.home() / "violin-rne/highforce_confirm/score_result.json").read_text())
T = ["HF_C", "HF_A", "HF_D", "HF_B"]
LAB = {"HF_C": "C_T2 s2", "HF_A": "A_T2 s2", "HF_D": "D_T2 s2", "HF_B": "B_T2 s2"}
MODELS = ["S1_strict", "T1k_break", "T1mu", "T1k_peak", "E3_free_Z", "S2m"]
NAMES = {"S1_strict": "strict cZv/β", "T1k_break": "thermal (break)", "T1mu": "thermal (μ_s)", "T1k_peak": "thermal (peak)",
         "E3_free_Z": "empirical shape + Z", "S2m": "Schelleng, free β exp."}
fig, ax = plt.subplots(1, 2, figsize=(10, 3.8), gridspec_kw=dict(width_ratios=[1.35, 1]))
a = ax[0]
w = 0.13
for k, m in enumerate(MODELS):
    y = [R["targets"][t]["t0.2"]["P2"][m]["level_error"] for t in T]
    a.bar(np.arange(4) + (k - 2.5) * w, y, w, label=NAMES[m])
a.axhspan(-0.148, 0.148, color="0.85", zorder=0, label="frozen 90 % interval (strict)")
a.axhline(0, color="k", lw=0.8)
a.set_xticks(range(4)); a.set_xticklabels([LAB[t] for t in T])
a.set_ylabel("level error, observed − predicted (ln)")
a.set_title("(a) frozen level predictions, 4–12 N", fontsize=10)
a.legend(fontsize=7, ncol=2, loc="upper center", frameon=False, bbox_to_anchor=(0.5, -0.12))
b = ax[1]
for i, t in enumerate(T):
    p = R["targets"][t]["t0.2"]["P1"]
    b.errorbar(i, p["e_beta"], yerr=[[p["e_beta"] - p["e_beta_ci"][0]], [p["e_beta_ci"][1] - p["e_beta"]]], fmt="o", color="k", capsize=3)
    s = R["targets"][t]["t0.2"]["P2"]
    b.plot(i + 0.18, s["T1k_break"]["e_beta_pred_own"], "v", color="tab:orange", label="thermal model" if i == 0 else None)
    b.plot(i + 0.18, s["S1_strict"]["e_beta_pred_own"], "s", color="tab:blue", label="strict Schelleng" if i == 0 else None)
b.axhspan(-1.05, -0.95, color="tab:blue", alpha=0.12, label="'plain' zone")
b.axhline(-0.95, color="0.5", ls=":", lw=0.8)
b.set_xticks(range(4)); b.set_xticklabels([LAB[t] for t in T])
b.set_ylabel("β exponent of F_max (95 % CI)")
b.set_title("(b) free β exponent vs frozen models", fontsize=10)
b.legend(fontsize=7, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=3)
fig.tight_layout()
out = Path.home() / "violin-rne/paper/fig_highforce.png"
fig.savefig(out, dpi=200, bbox_inches="tight")
print(out)
