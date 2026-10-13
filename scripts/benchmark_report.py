#!/usr/bin/env python3
"""Aggregate the fixed A–D source sweeps without re-selecting models on targets."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import benchmark_data as bd


def build(base: Path, out: Path):
    out.mkdir(parents=True, exist_ok=True)
    results = {source: json.loads((base/f"benchmark_{source}"/"results.json").read_text()) for source in "ABCD"}
    rows = []
    for source, table in results.items():
        for name, r in table.items():
            if r["historical_diagnostic"] or r["selection"] == "nominal_cv" or r.get("prior_sensitivity"):
                continue
            rows.append({"source":source,"model":name, **r["macro_panels"],
                         "global_same_rule_min_iou":r["global_same_rule"]["family_min_iou"]})
    with (out/"benchmark_aggregate.csv").open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    plt.rcParams.update({"font.size":10,"axes.spines.top":False,"axes.spines.right":False})
    methods=[("nominal_bce","Nominal BCE"),("family_mean_bce","Family-mean BCE"),
             ("consensus_bce","Consensus BCE"),("nominal_iou","Nominal soft IoU"),
             ("box_robust","Rectangular robust"),("family_robust","Coherent-family robust"),
             ("logistic","Quadratic logistic"),("svc","RBF SVM")]
    values=np.array([[results[s][f"covariant_{method}"]["macro_panels"]["family_min_iou"] for s in "ABCD"] for method,_ in methods])
    fig,ax=plt.subplots(figsize=(7.8,5.6),layout="constrained")
    im=ax.imshow(values,cmap="viridis",vmin=.60,vmax=.71,aspect="auto")
    for (i,j),v in np.ndenumerate(values): ax.text(j,i,f"{v:.3f}",ha="center",va="center",color="white" if v<.661 else "black",fontsize=11)
    ax.set_xticks(range(4),[f"Source {s}" for s in "ABCD"]);ax.set_yticks(range(len(methods)),[label for _,label in methods])
    ax.axhline(5.5,color="white",lw=2)
    ax.set_title("Retrospective transfer: objective comparison\nMacro of 11 panel-wise coherent-family minimum IoUs",pad=15)
    ax.set_xlabel("Models selected only by blocked source-panel cross-validation")
    fig.colorbar(im,ax=ax,label="Higher is better")
    fig.text(.5,-.015,"All models use force–speed covariance. B–D sweeps retain the fixed A-derived label reference.",ha="center",fontsize=9)
    fig.savefig(out/"benchmark_objectives.png",dpi=180,bbox_inches="tight")
    fig.savefig(out/"benchmark_objectives.pdf",bbox_inches="tight");plt.close(fig)
    pairs=[("logistic","Quadratic logistic"),("svc","RBF SVM"),("extra_trees","Extra trees"),("family_robust","Coherent-family robust")]
    fig,axes=plt.subplots(1,4,figsize=(12.4,4.0),sharey=True,layout="constrained")
    colors=["#377eb8","#e07b24","#3b9655","#9b59b6"]
    for ax,(method,label) in zip(axes,pairs):
        for source,color in zip("ABCD",colors):
            raw=results[source][f"rawlog_{method}"]["macro_panels"]["family_min_iou"]
            cov=results[source][f"covariant_{method}"]["macro_panels"]["family_min_iou"]
            ax.plot([0,1],[raw,cov],"o-",color=color,label=f"Source {source}",alpha=.85)
        ax.set_xticks([0,1],["Raw logs","Covariant"]);ax.set_xlim(-.2,1.2);ax.set_title(label);ax.grid(axis="y",alpha=.2)
    axes[0].set_ylabel("Macro family-minimum IoU");axes[-1].legend(fontsize=9,loc="lower right")
    fig.suptitle("The force–speed prior helps consistently; the robust objective does not win consistently",fontsize=12)
    fig.savefig(out/"benchmark_covariance.png",dpi=180,bbox_inches="tight")
    fig.savefig(out/"benchmark_covariance.pdf",bbox_inches="tight");plt.close(fig)
    summary={"scope":"Retrospective fixed-spec sensitivity; A-source primary, B/C/D conditional on fixed A-derived label definition",
             "count_sources":4,"main_metric":"mean of panel-wise minimum IoU over 15 global threshold pairs; not truth coverage",
             "findings":"Declared force-speed covariance improves these matched raw-coordinate comparisons; coherent-family objective fails to beat the strongest declared simple baseline in every source sweep.",
             "rows":rows}
    (out/"benchmark_aggregate.json").write_text(json.dumps(summary,indent=2)+"\n")
    print(f"Saved aggregate tables and four figure files under {out.relative_to(bd.ROOT)}")


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base",type=Path,default=bd.ROOT/"reports/research_20261013")
    p.add_argument("--out",type=Path,default=bd.ROOT/"research/2026-10-13/figures")
    a=p.parse_args(argv);build(a.base,a.out)


if __name__=="__main__":main()
