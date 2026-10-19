"""Figure for posthoc_hf (reads results.json only). python fig_posthoc_hf.py -> fig_posthoc_hf.png"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
r = json.loads((HERE / "results.json").read_text())
TG = ["HF_C", "HF_A", "HF_D", "HF_B"]
LAB = ["C", "A", "D", "B"]
BLUE, ORANGE, AQUA, YELLOW, INK, MUTED = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#222222", "#8a8a85"
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": MUTED,
                     "axes.grid": True, "grid.color": "#e6e6e3", "grid.linewidth": 0.6, "axes.axisbelow": True})
fig, ax = plt.subplots(2, 3, figsize=(13, 7.6))
x = np.arange(4)
q1 = r["Q1"]["targets"]

for k, (key, title) in enumerate((("e_beta", "(a) Q1 beta exponent on HF columns"), ("e_v", "(b) Q1 speed exponent on HF columns"))):
    a = ax[0, k]
    obs = [q1[t]["observed"][key] for t in TG]
    ci = np.array([q1[t]["observed"][key + "_ci"] for t in TG])
    a.errorbar(x, obs, yerr=[np.array(obs) - ci[:, 0], ci[:, 1] - np.array(obs)], fmt="o", color=INK, ms=6, capsize=3,
               lw=1.5, label="observed free fit (95% CI)")
    a.plot(x + 0.15, [q1[t]["b2_local"][key] for t in TG], "s", color=ORANGE, ms=8, label="b2 quadratic surface (pre-HF)")
    a.plot(x - 0.15, [q1[t]["linear_ref_local"][key] for t in TG], "D", color=BLUE, ms=7, label="linear reference")
    a.axhline(-1 if key == "e_beta" else 1, color=MUTED, lw=1, ls="--")
    a.set_xticks(x, LAB); a.set_xlabel("string type (HF session = sample 2)" if k else ""); a.set_title(title, loc="left")
    a.set_ylabel("d ln Fmax / d ln " + ("beta" if key == "e_beta" else "v"))
    if k == 0:
        a.legend(frameon=False, fontsize=7.5, loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=2)

a = ax[0, 2]
w = 0.26
for j, (key, name, col) in enumerate((("level_error_S1_strict_frozen", "S1_strict (frozen)", MUTED),
                                      ("level_error_b2", "b2 surface + sample-1 offset", ORANGE),
                                      ("level_error_linear_ref", "linear ref + sample-1 offset", BLUE))):
    a.bar(x + (j - 1) * w, [q1[t][key] for t in TG], w * 0.92, color=col, label=name)
a.axhspan(-0.15, 0.15, color="#f0efec", zorder=0)
a.axhline(0, color=INK, lw=0.8)
a.set_xticks(x, LAB); a.set_ylabel("level error, observed - predicted (ln)")
a.set_title("(c) Q1 level error (grey band +/-0.15)", loc="left"); a.legend(frameon=False, fontsize=8)

a = ax[1, 0]
q2 = r["Q2"]; a2 = r["additions_after_results"]["A2"]
d = [q2[t]["delta"] for t in TG]; ci = np.array([q2[t]["delta_ci"] for t in TG])
a.errorbar(x - 0.1, d, yerr=[np.array(d) - ci[:, 0], ci[:, 1] - np.array(d)], fmt="o", color=INK, capsize=3, lw=1.5,
           label="planned fit (B 1000, seed 7)")
d2 = [a2[t]["delta"] for t in TG]; ci2 = np.array([a2[t]["delta_ci"] for t in TG])
a.errorbar(x + 0.1, d2, yerr=[np.array(d2) - ci2[:, 0], ci2[:, 1] - np.array(d2)], fmt="s", color=AQUA, capsize=3, lw=1.5,
           label="post-plan: top-stroke dropout censored")
a.set_xlim(-0.5, 3.5)
a.axhline(0, color=INK, lw=0.8)
for i, t in enumerate(TG):
    a.annotate(f"s1 finite\n{q2[t]['sample1_finite']}/{q2[t]['n_columns_sample1']}", (x[i], -0.12), ha="center",
               fontsize=7, color=MUTED)
a.set_ylim(-0.2, 1.0)
a.set_xticks(x, LAB); a.set_ylabel("session offset delta (HF - sample 1, ln)")
a.set_title("(d) Q2 common free shape + session offset", loc="left"); a.legend(frameon=False, fontsize=7, loc="upper left")

a = ax[1, 1]
q3 = r["Q3"]
cols = dict(zip(TG, [BLUE, ORANGE, AQUA, YELLOW]))
for t, s1 in zip(TG, ["C_T2", "A_T2", "D_T2", "B_T2s1"]):
    for src, ls in ((q3["sample1"][s1]["bands"], "-"), (q3["hf"][t]["bands"], "-")):
        bb = [b for b in src if b["speed"] == "all" and b["n"] >= 20]
        xm = [np.sqrt(max(b["band_N"][0], 0.15) * b["band_N"][1]) for b in bb]
        a.plot(xm, [b["r_on_median"] for b in bb], ls, marker="o", ms=4, color=cols[t], lw=2,
               label=t.replace("HF_", "") if src is q3["hf"][t]["bands"] else None)
a.axhline(0, color=INK, lw=0.8)
a.axvspan(3.7, 4.9, color="#f0efec", zorder=0)
a.text(4.25, -1.0, "overlap", ha="center", fontsize=7, color=MUTED)
a.set_xscale("log"); a.set_xlabel("onset (pre-motion) force band, N  [left: sample 1, right: HF]")
a.set_ylabel("median ln(F window t=0.2 / F onset)")
a.set_title("(e) Q3 force reading: window vs onset force", loc="left")
a.legend(frameon=False, fontsize=8, title="string", title_fontsize=8, loc="lower right")
dr = q3["delta_r"]
a.text(0.02, 0.97, "boundary strokes, HF - sample 1:\n" + "  ".join(f"{t[-1]} {dr[t]['delta_r']:+.3f}" for t in TG),
       transform=a.transAxes, va="top", fontsize=7.5, color=INK)

a = ax[1, 2]
sp = r["additions_after_results"]["A3_split"]
cells = [("0.05 beta>=0.04", "v .05, beta>=.04"), ("0.20 beta<0.04", "v .20, beta<.04")]
w = 0.2
for ci_, (cell, name) in enumerate(cells):
    for j, (who, col) in enumerate((("s1_frac", MUTED), ("hf_frac", ORANGE))):
        vals = [sp[t][cell][who] if sp[t][cell][who] is not None else np.nan for t in TG]
        pos = x + (ci_ * 2 + j - 1.5) * w
        a.bar(pos, vals, w * 0.9, color=col if ci_ == 0 else col, alpha=1.0 if ci_ == 0 else 0.55,
              label=(("sample 1" if j == 0 else "HF") + ", " + name))
a.set_xticks(x, LAB); a.set_ylim(0, 1.05); a.set_ylabel("Helmholtz fraction at 3.8-4.6 N")
a.set_title("(f) post-plan: same force, both sessions", loc="left")
a.legend(frameon=False, fontsize=7, loc="upper center", ncol=2, bbox_to_anchor=(0.5, -0.1))
a.text(0.02, 0.98, "D: overlap too narrow (n<20)", transform=a.transAxes, va="top", fontsize=7, color=MUTED)
fig.suptitle("POST HOC diagnosis of the high-force under-prediction (HF labels already seen; t = 0.2 s, K = 2, natural log)",
             fontsize=10.5, x=0.01, ha="left")
fig.tight_layout(rect=(0, 0, 1, 0.96))
fig.savefig(HERE / "fig_posthoc_hf.png", dpi=160)
print("saved")
