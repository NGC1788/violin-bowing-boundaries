#!/usr/bin/env python3
"""Publish the complete retrospective audit and a fixed, non-selected contrast figure."""
import argparse
import csv
import gzip
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]


def build(source, destination):
    summary = json.loads((source / "summary.json").read_text())
    rows = list(csv.DictReader((source / "all_pairs.csv").open()))
    contradictions = []
    for row in rows:
        if row["scope"] != "macro_shared":
            continue
        old = float(row["original_grid_mean_panel_min_difference"])
        lower, upper = float(row["continuous_difference_min"]), float(row["continuous_difference_max"])
        if (old > 1e-12 and upper < -1e-12) or (old < -1e-12 and lower > 1e-12):
            contradictions.append({key: row[key] for key in ("task", "model_a", "model_b",
                "original_grid_mean_panel_min_difference", "continuous_difference_min", "continuous_difference_max")})
    summary["mean_panel_min_preference_opposed_by_every_shared_macro_rule"] = contradictions
    (destination / "observer_dominance_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    raw = (source / "all_pairs.csv").read_bytes()
    (destination / "observer_dominance_all_pairs.csv.gz").write_bytes(gzip.compress(raw, mtime=0))
    figure, axes = plt.subplots(1, 2, figsize=(11., 6.4), layout="constrained")
    def interval(task, first, second):
        row = next(r for r in rows if r["task"] == task and r["scope"] == "pooled"
                   and {r["model_a"], r["model_b"]} == {first, second})
        sign = 1 if row["model_a"] == first else -1
        lower, upper = float(row["continuous_difference_min"]), float(row["continuous_difference_max"])
        return (lower, upper, float(row["continuous_independent_min_difference"])) if sign == 1 else (
            -upper, -lower, -float(row["continuous_independent_min_difference"]))
    specifications = [
        (axes[0], [(f"benchmark_{s}", f"Source {s}", "covariant_family_robust", "covariant_logistic") for s in "ABCD"],
         "Family-robust minus covariant logistic\nAll four original source tasks"),
        (axes[1], [(f"cello__{s}__{a:g}_{b:g}", f"{s}  {a:g}, {b:g}", "affine_flow", "translation_k0")
                   for s in "ABCD" for a, b in ((.05, .1), (.05, .2), (.1, .2))],
         "Affine flow minus matched k=0\nAll twelve two-speed source tasks")]
    for axis, entries, title in specifications:
        for index, (task, label, first, second) in enumerate(entries):
            lower, upper, maximin = interval(task, first, second)
            color = "#c97921" if lower < 0 < upper else "#28658c"
            axis.plot([lower, upper], [index, index], color=color, lw=3, solid_capstyle="round")
            axis.scatter([lower, upper], [index, index], color=color, s=12)
            axis.scatter(maximin, index, color="black", s=17, zorder=3)
        axis.axvline(0, color="0.45", lw=1, ls="--")
        axis.set_yticks(range(len(entries)), [row[1] for row in entries])
        axis.invert_yaxis()
        axis.set_title(title, fontsize=11)
        axis.set_xlabel("Pooled IoU difference under one shared observer")
        axis.grid(axis="x", alpha=.2)
        axis.spines[["top", "right"]].set_visible(False)
    figure.suptitle("Observer choice can reverse a model comparison", fontsize=14)
    figure.supxlabel("Bars: exact continuous-observer range, not confidence intervals.  Dots: difference of individual minimum IoUs.", fontsize=9)
    figure.savefig(destination / "observer_dominance.png", dpi=170)
    figure.savefig(destination / "observer_dominance.pdf")
    plt.close(figure)
    print(json.dumps({"all_comparisons": len(rows), "macro_semantic_preference_reversals": contradictions}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", type=Path, default=ROOT / "reports/observer_dominance/2026-10-13")
    parser.add_argument("--out", type=Path, default=Path(__file__).parent)
    args = parser.parse_args()
    build(args.audit, args.out)
