"""Plot every held-speed cell for three fixed published-violin baselines.

Run benchmark_published.py first; no selection, fitting or label interpolation here.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, default=ROOT / "reports/published_violin/2026-10-13/predictions.csv")
    parser.add_argument("--out", type=Path, default=HERE / "figures/published_violin_predictions")
    args = parser.parse_args()
    with args.predictions.open() as stream:
        predictions = list(csv.DictReader(stream))
    with (HERE / "schoonderwaldt2008/published_cells.csv").open() as stream:
        cells = [r for r in csv.DictReader(stream) if r["figure"] == "4"]
    geometry = {f"fig4{r['panel']}_b{r['beta_column_0based']}_f{r['force_rank_0based']}": r for r in cells}
    models = [("boundary_static", "Static boundary"), ("boundary_uniform", "Uniform speed scaling"), ("boundary_asymmetric", "Known asymmetric scaling")]
    styles = [(0, 0, "#d1d5db", ".", "Correct non-H"), (1, 1, "#147d64", "o", "Correct H"), (0, 1, "#d97706", "x", "False H"), (1, 0, "#9d38a7", "+", "Missed H")]
    fig, axes = plt.subplots(3, 3, figsize=(11, 9.5), sharex=True, sharey=True)
    values = []
    for row, speed in enumerate((.05, .15, .20)):
        for col, (model, title) in enumerate(models):
            ax = axes[row, col]
            rows = [r for r in predictions if float(r["source_speed"]) == .1 and float(r["target_speed"]) == speed and r["model"] == model]
            assert rows and len({r["cell_id"] for r in rows}) == len(rows)
            x = np.array([float(geometry[r["cell_id"]]["beta_from_reported_rounded_distance"]) for r in rows])
            y = np.array([float(r["force_N"]) for r in rows])
            truth = np.array([int(r["author_H"]) for r in rows])
            pred = np.array([int(r["prediction_H"]) for r in rows])
            for t, p, color, marker, _ in styles:
                select = (truth == t) & (pred == p)
                ax.scatter(x[select], y[select], c=color, marker=marker, s=16 if marker == "." else 22, linewidths=.9)
            unknown = [r for r in cells if float(r["bow_speed_nominal_m_s"]) == speed and r["strict_systematically_analyzed"] != "True"]
            if unknown:
                ax.scatter([float(r["beta_from_reported_rounded_distance"]) for r in unknown], [float(r["bow_force_nominal_N"]) for r in unknown], marker="s", s=22, facecolors="none", edgecolors="#777777", linewidths=.6)
            intersection = int(np.sum((truth == 1) & (pred == 1)))
            union = int(np.sum((truth == 1) | (pred == 1)))
            score = intersection / union
            ax.set_title(f"{title if row == 0 else ''}\nIoU {score:.3f}  |  n = {len(rows)}", fontsize=10, pad=7)
            ax.set_yscale("log")
            ax.set_xlim(.024, .179)
            ax.set_ylim(.037, 3.3)
            ax.set_xticks([.04, .08, .12, .16])
            ax.grid(alpha=.12)
            ax.set_axisbelow(True)
            if col == 0:
                ax.set_ylabel(f"v = {speed:.2f} m/s\nBow force (N)", fontsize=10)
            if row == 2:
                ax.set_xlabel("Relative bow position beta", fontsize=10)
            values.append({"model": model, "speed": speed, "n": len(rows), "unknown": len(unknown), "iou": score})
    handles = [Line2D([], [], color=color, marker=marker, linestyle="none", label=label, markersize=6) for _, _, color, marker, label in styles]
    handles.append(Line2D([], [], color="#777777", marker="s", markerfacecolor="none", linestyle="none", label="Unplotted / excluded", markersize=5))
    fig.suptitle("Published violin D-string: each held-speed prediction\nSource speed 0.10 m/s; rigid monochord; author-classified cells", fontsize=14, y=.99)
    fig.legend(handles=handles, loc="lower center", ncol=5, bbox_to_anchor=(.5, .035), frameon=False, fontsize=9)
    fig.text(.5, .014, "Schoonderwaldt et al. (2008), Fig. 4. Fixed literature priors; retrospective comparison. Unknown cells are not errors.", ha="center", fontsize=8)
    fig.tight_layout(rect=(0, .085, 1, .94))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out.with_suffix(".png"), dpi=180)
    fig.savefig(args.out.with_suffix(".pdf"))
    args.out.with_suffix(".json").write_text(json.dumps(values, indent=2) + "\n")
    print(args.out)


if __name__ == "__main__":
    main()
