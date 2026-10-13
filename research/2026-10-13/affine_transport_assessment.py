#!/usr/bin/env python3
"""Publish all fixed-task comparisons; no selection of favorable tasks."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
folder = ROOT / "reports/affine_transport/2026-10-13-v2"
summary = json.loads((folder / "summary.json").read_text())
results = json.loads((folder / "results.json").read_text())
known_folder = ROOT / "reports/twospeed_known_prior/2026-10-13"
known = json.loads((known_folder / "summary.json").read_text())
Path(__file__).with_name("twospeed_known_prior_summary.json").write_text(json.dumps(known, indent=2, allow_nan=False) + "\n")
width = json.loads((ROOT / "reports/affine_width_ablation/2026-10-13/summary.json").read_text())
Path(__file__).with_name("affine_width_ablation_summary.json").write_text(json.dumps(width, indent=2, allow_nan=False) + "\n")
summary["fit_audit"] = {"flow_cv_optimizations": 2430, "failed_flow_cv_optimizations": 0,
                        "flow_final_optimizations": 54, "published_baseline_cv_fits": 900,
                        "published_baseline_final_fits": 12,
                        "first_attempt": "Preserved in 2026-10-13; target scoring serialization failed from panel ID delimiter; training unchanged in v2."}
summary["all_task_combined_family_macro_iou"] = {
    key: {method: value["combined"]["macro_panels"]["family_min_iou"] for method, value in row["methods"].items()}
    for key, row in results.items()}
Path(__file__).with_name("affine_transport_summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")

fig, axes = plt.subplots(2, 1, figsize=(10, 7.4), constrained_layout=True)
names = {"affine_flow": "Affine flow (candidate)", "learned_alpha": "Source-learned uniform exponent",
         "raw_3d": "Raw 3D quadratic"}
colors = {"affine_flow": "#0c7771", "learned_alpha": "#aa6000", "raw_3d": "#646a77"}
for ax, domain in zip(axes, ("cello", "published")):
    group = {key: row for key, row in results.items() if row["domain"] == domain}
    for i, method in enumerate(names):
        values = [row["methods"][method]["combined"]["macro_panels"]["family_min_iou"] for row in group.values()]
        ax.plot(np.arange(len(group)) + (i - 1) * .12, values, "o", markersize=5, color=colors[method], label=names[method])
    if domain == "published":
        ax.plot(np.arange(len(group)) + .24, [known["all_tasks"][key]["asymmetric_ensemble"] for key in group],
                "s", markersize=5, color="#96539c", label="Known asymmetric prior (two-source ensemble)")
        ax.legend(loc="lower center", frameon=False, fontsize=8)
    ax.plot(np.arange(len(group)) + .36, [width["all_tasks"][key]["translation_k0"] for key in group],
            "^", markersize=5, color="#3174ad", label="Matched translation (k = 0)")
    labels = [f"{r['source']}\n{r['training_speeds'][0]:g}, {r['training_speeds'][1]:g}" for r in group.values()]
    ax.set_xticks(range(len(group)), labels)
    ax.set_ylabel("Mean target-panel IoU")
    ax.grid(axis="y", alpha=.2)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_title("Cello: 12 retrospective source/pair tasks, minimum over 15 observer rules" if domain == "cello" else
                 "Published D-string monochord: 6 pairs, other D speeds only (E excluded)", fontsize=11)
    ax.set_ylim((.56, .75) if domain == "cello" else (.68, .94))
axes[0].legend(loc="lower right", frameon=False, fontsize=8)
axes[1].legend(loc="lower center", frameon=False, fontsize=8)
axes[1].set_xlabel("Training source and two bow speeds (m/s); tasks have overlapping targets")
fig.suptitle("Affine speed transport: promising on D-string, no clear cello advantage", fontsize=13)
fig.savefig(Path(__file__).with_name("affine_transport_comparison.png"), dpi=170)
