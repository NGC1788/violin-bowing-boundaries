#!/usr/bin/env python3
"""Reconstruct the (beta, force) trial grid from a trial audit and map problem windows onto it.

Reads only the per-trial table written by ``trial_audit.py``; no signal file is opened.
Beta levels are separated at the natural break between within-level jitter and
between-level spacing, so a level whose values straddle a rounding boundary is not
split. Force ranks order the measured column-1 window means inside one beta level.
Neither reconstruction is a published design specification; the report states how
strongly the data support each one.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import sys
import uuid

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
AUDIT_REPORTS = PROJECT_ROOT / "reports/trial_audit"
DEFAULT_REPORTS = PROJECT_ROOT / "reports/grid_summary"
MIN_BREAK_RATIO = 5.0
NEEDED = ("condition", "trial", "error", "beta", "c1_window_mean", "c2_max",
          "window_inside_plateau", "c2_window_steady_fraction")


class GridError(ValueError):
    """The trial table could not be interpreted."""


def natural_break(values, min_ratio: float = MIN_BREAK_RATIO) -> tuple[float, float]:
    """Gap threshold between jitter and level spacing, and the gap ratio at that break.

    ``inf`` means a single level. ``0.0`` means no clear break was found, so each distinct
    value is treated as its own level; the ratio is then reported for the caller to flag.
    """
    unique = np.unique(np.asarray(values, dtype=float))
    if unique.size <= 1:
        return math.inf, math.nan
    gaps = np.sort(np.diff(unique))
    if gaps.size < 2:
        return 0.0, math.nan
    ratios = gaps[1:] / gaps[:-1]
    k = int(np.argmax(ratios))
    if ratios[k] < min_ratio:
        return 0.0, float(ratios[k])
    return float(math.sqrt(gaps[k] * gaps[k + 1])), float(ratios[k])


def assign_levels(values, threshold: float) -> np.ndarray:
    """Level index per value, ascending in value; a new level starts where the gap exceeds threshold."""
    values = np.asarray(values, dtype=float)
    order = np.argsort(values, kind="stable")
    labels = np.empty(values.size, dtype=int)
    level = 0
    for position, index in enumerate(order):
        if position and values[index] - values[order[position - 1]] > threshold:
            level += 1
        labels[index] = level
    return labels


def _top(counter: dict, n: int = 3) -> dict:
    return dict(sorted(counter.items(), key=lambda item: (-item[1], item[0]))[:n])


def analyse_condition(rows: list[dict]) -> tuple[dict, list[dict]]:
    beta = np.array([r["beta"] for r in rows], dtype=float)
    trial = np.array([r["trial"] for r in rows], dtype=int)
    force = np.array([r["c1_window_mean"] for r in rows], dtype=float)
    inside = np.array([r["window_inside_plateau"] for r in rows], dtype=bool)
    steady = np.array([r["c2_window_steady_fraction"] for r in rows], dtype=float)

    threshold, ratio = natural_break(beta)
    level = assign_levels(beta, threshold)
    n_levels = int(level.max()) + 1
    counts = np.bincount(level, minlength=n_levels)
    centers = np.array([float(np.median(beta[level == L])) for L in range(n_levels)])

    rank = np.zeros(trial.size, dtype=int)
    contiguous, concordance, directions = 0, [], {"descending": 0, "ascending": 0}
    first_trial = np.zeros(n_levels, dtype=int)
    for L in range(n_levels):
        members = np.flatnonzero(level == L)
        by_force = members[np.argsort(force[members], kind="stable")]
        rank[by_force] = np.arange(1, members.size + 1)
        by_trial = members[np.argsort(trial[members], kind="stable")]
        first_trial[L] = trial[by_trial[0]]
        numbers = trial[by_trial]
        contiguous += int(numbers[-1] - numbers[0] + 1 == numbers.size)
        steps = np.diff(force[by_trial])
        if steps.size:
            down, up = float(np.mean(steps < 0)), float(np.mean(steps > 0))
            concordance.append(max(down, up))
            directions["descending" if down >= up else "ascending"] += 1

    level_order = centers[np.argsort(first_trial, kind="stable")]
    level_steps = np.diff(level_order)
    beta_sweep = ("descending" if level_steps.size and np.all(level_steps < 0) else
                  "ascending" if level_steps.size and np.all(level_steps > 0) else
                  "single level" if not level_steps.size else "not monotone")

    spacing = "undetermined"
    if n_levels >= 3 and np.all(centers > 0):
        lin, log = np.diff(centers), np.diff(np.log(centers))
        cv = lambda d: float(np.std(d) / np.mean(d)) if np.mean(d) > 0 else math.inf
        spacing = "closer to logarithmic" if cv(log) < cv(lin) else "closer to linear"

    uniform = bool(np.all(counts == counts[0]))
    rank_profile = {}
    if uniform:
        size = int(counts[0])
        medians = [float(np.median(force[rank == r])) for r in range(1, size + 1)]
        spreads = [float(np.median(np.abs(force[rank == r] - m))) for r, m in zip(range(1, size + 1), medians)]
        picks = sorted({1, max(1, size // 4), max(1, size // 2), max(1, 3 * size // 4), size})
        rank_profile = {"ranks_shown": {str(r): round(medians[r - 1], 4) for r in picks},
                        "median_abs_deviation_across_beta": round(float(np.median(spreads)), 4),
                        "force_by_rank": [round(m, 6) for m in medians]}

    nonpositive = force <= 0
    off = ~inside
    center_key = lambda L: f"{centers[L]:.4f}"
    summary = {
        "trials": int(trial.size),
        "c2_peak_levels": sorted({round(r["c2_max"], 4) for r in rows}),
        "beta": {"levels": n_levels, "break_ratio": None if math.isnan(ratio) else round(ratio, 2),
                 "threshold": None if math.isinf(threshold) else threshold,
                 "clear_break": bool(math.isinf(threshold) or threshold > 0),
                 "trials_per_level": {"min": int(counts.min()), "max": int(counts.max())},
                 "uniform_trials_per_level": uniform,
                 "centers_min": round(float(centers.min()), 5), "centers_max": round(float(centers.max()), 5),
                 "spacing": spacing, "sweep_order_by_trial": beta_sweep,
                 "trial_numbers_contiguous_in_levels": contiguous},
        "force": {"order_follows_trial_number": {
                      "levels_descending": directions["descending"], "levels_ascending": directions["ascending"],
                      "median_concordance": round(float(np.median(concordance)), 3) if concordance else None,
                      "min_concordance": round(float(np.min(concordance)), 3) if concordance else None},
                  **rank_profile},
        "nonpositive_force_windows": {"count": int(nonpositive.sum()),
                                      "by_rank": _top({int(r): int(c) for r, c in zip(*np.unique(rank[nonpositive], return_counts=True))})},
        "off_plateau_windows": {"count": int(off.sum()),
                                "by_beta_level": _top({center_key(int(L)): int(c) for L, c in zip(*np.unique(level[off], return_counts=True))}),
                                "by_force_rank": _top({int(r): int(c) for r, c in zip(*np.unique(rank[off], return_counts=True))}),
                                "steady_fraction_min": round(float(np.nanmin(steady)), 4) if steady.size else None},
    }
    grid_rows = [{"condition": rows[i]["condition"], "trial": int(trial[i]), "beta": float(beta[i]),
                  "beta_level": int(level[i]), "beta_level_center": float(centers[level[i]]),
                  "force_rank": int(rank[i]), "c1_window_mean": float(force[i]),
                  "window_inside_plateau": bool(inside[i]),
                  "c2_window_steady_fraction": float(steady[i])} for i in range(trial.size)]
    return summary, grid_rows


def load_trials(path: Path) -> tuple[dict[str, list[dict]], dict[str, int]]:
    by_condition: dict[str, list[dict]] = {}
    excluded: dict[str, int] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        missing = [c for c in NEEDED if c not in (reader.fieldnames or [])]
        if missing:
            raise GridError(f"trial table lacks columns: {missing}")
        for raw in reader:
            if raw["error"]:
                excluded[raw["condition"]] = excluded.get(raw["condition"], 0) + 1
                continue
            try:
                row = {"condition": raw["condition"], "trial": int(raw["trial"]), "beta": float(raw["beta"]),
                       "c1_window_mean": float(raw["c1_window_mean"]), "c2_max": float(raw["c2_max"]),
                       "window_inside_plateau": {"True": True, "False": False}[raw["window_inside_plateau"]],
                       "c2_window_steady_fraction": float(raw["c2_window_steady_fraction"])}
            except (KeyError, ValueError) as error:
                raise GridError(f"unreadable row for trial {raw.get('trial')!r}: {error}") from error
            by_condition.setdefault(row["condition"], []).append(row)
    if not by_condition:
        raise GridError(f"no readable trials in {path}")
    return by_condition, excluded


def plot_grid(analyses: dict[str, list[dict]], destination: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = sorted(analyses)
    figure, axes = plt.subplots(1, len(names), figsize=(4.6 * len(names), 5.0), squeeze=False,
                                constrained_layout=True)
    image = None
    for axis, name in zip(axes[0], names):
        rows = analyses[name]
        levels = max(r["beta_level"] for r in rows) + 1
        ranks = max(r["force_rank"] for r in rows)
        matrix = np.full((levels, ranks), np.nan)
        centers = {}
        for r in rows:
            matrix[r["beta_level"], r["force_rank"] - 1] = r["c2_window_steady_fraction"]
            centers[r["beta_level"]] = r["beta_level_center"]
        image = axis.imshow(matrix, origin="lower", aspect="auto", cmap="viridis", vmin=0, vmax=1,
                            extent=(0.5, ranks + 0.5, -0.5, levels - 0.5), interpolation="nearest")
        bad = [(r["force_rank"], r["beta_level"]) for r in rows if r["c1_window_mean"] <= 0]
        if bad:
            axis.scatter(*zip(*bad), marker="x", color="red", s=14, linewidths=0.8)
        ticks = np.linspace(0, levels - 1, min(levels, 6)).round().astype(int)
        axis.set_yticks(ticks, [f"{centers[t]:.3f}" for t in ticks])
        axis.set_xlabel("force rank (low → high)")
        if axis is axes[0][0]:
            axis.set_ylabel("β level")
        off = sum(not r["window_inside_plateau"] for r in rows)
        axis.set_title(f"{name}\noff-plateau {off}, column-1 ≤ 0: {len(bad)}", fontsize=10)
    figure.colorbar(image, ax=axes[0].tolist(), shrink=0.85, label="fraction of window on velocity plateau")
    figure.suptitle("Reconstructed (β, force) grid — dark cells: window not at constant velocity; × column-1 mean ≤ 0",
                    fontsize=10)
    figure.savefig(destination, dpi=130, bbox_inches="tight")
    plt.close(figure)


def format_report(report: dict) -> list[str]:
    lines = [f"GRID SUMMARY from {report['source']}"]
    for name, s in report["conditions"].items():
        b, f = s["beta"], s["force"]
        o, n = s["off_plateau_windows"], s["nonpositive_force_windows"]
        lines.append(f"[{name}] {s['trials']} trials | c2 peak {s['c2_peak_levels']}"
                     + (f" | excluded failed {report['excluded'].get(name, 0)}" if report["excluded"].get(name) else ""))
        lines.append(f"  beta: {b['levels']} levels (break ratio {b['break_ratio']}"
                     + ("" if b["clear_break"] else ", NO CLEAR BREAK") + f"), trials per level "
                     f"{b['trials_per_level']['min']}..{b['trials_per_level']['max']}, centers "
                     f"{b['centers_min']}..{b['centers_max']}, spacing {b['spacing']}")
        lines.append(f"  beta sweep by trial number: {b['sweep_order_by_trial']}; trial numbers contiguous in "
                     f"{b['trial_numbers_contiguous_in_levels']}/{b['levels']} levels")
        c = f["order_follows_trial_number"]
        lines.append(f"  force vs trial order: descending in {c['levels_descending']}, ascending in "
                     f"{c['levels_ascending']} levels; concordance median {c['median_concordance']} min {c['min_concordance']}")
        if "ranks_shown" in f:
            lines.append(f"  column-1 median by force rank {f['ranks_shown']}; typical deviation across beta "
                         f"{f['median_abs_deviation_across_beta']}")
        lines.append(f"  column-1 <= 0: {n['count']} windows, by rank {n['by_rank']}")
        lines.append(f"  off-plateau: {o['count']} windows, top beta levels {o['by_beta_level']}, "
                     f"top force ranks {o['by_force_rank']}")
    if report.get("plot"):
        lines.append(f"Plot: {report['plot']}")
    return lines


def resolve_trials(explicit: Path | None, audit_reports: Path) -> Path:
    if explicit:
        return explicit
    try:
        latest = json.loads((audit_reports / "latest.json").read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise GridError(f"no trial audit found ({error}); run trial-audit first") from error
    if latest.get("status") != "PASS":
        raise GridError(f"latest trial audit status is {latest.get('status')!r}; resolve it first")
    return Path(latest["run_dir"]) / "trials.csv"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--trials", type=Path, help="trials.csv; default: latest PASS trial audit")
    parser.add_argument("--audit-reports", type=Path, default=AUDIT_REPORTS)
    parser.add_argument("--reports-dir", type=Path, default=DEFAULT_REPORTS)
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args(argv)
    try:
        source = resolve_trials(args.trials, args.audit_reports)
        by_condition, excluded = load_trials(source)
        summaries, grids = {}, {}
        for name in sorted(by_condition):
            summaries[name], grids[name] = analyse_condition(by_condition[name])
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        run_dir = args.reports_dir / f"{stamp}_{uuid.uuid4().hex[:8]}"
        run_dir.mkdir(parents=True, exist_ok=False)
        report = {"source": str(source), "created_utc": stamp, "excluded": excluded, "conditions": summaries,
                  "scope": "Grid reconstructed from measured values in the trial audit; not a published design."}
        if not args.no_plot:
            plot_grid(grids, run_dir / "grid.png")
            report["plot"] = str(run_dir / "grid.png")
        with (run_dir / "grid.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(next(iter(grids.values()))[0]))
            writer.writeheader()
            for name in sorted(grids):
                for row in sorted(grids[name], key=lambda r: r["trial"]):
                    writer.writerow({k: (repr(v) if isinstance(v, float) else v) for k, v in row.items()})
        (run_dir / "summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print("\n".join(format_report(report)))
        return 0
    except GridError as error:
        print("GRID SUMMARY: FAIL —", error, file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
