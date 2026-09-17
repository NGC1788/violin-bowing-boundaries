#!/usr/bin/env python3
"""Provisional bowing-regime labels from bridge force, and playable-region maps.

Each trial's classification window of column 3 (published: bridge force) is reduced to
two interpretable quantities: periodicity (autocorrelation at the dominant period) and
slips per period (abrupt force changes, grouped). Helmholtz motion produces a sawtooth
with one abrupt return per period. Thresholds are PROVISIONAL, stated in every report,
and must be checked against the saved example waveforms. These rule-based labels are
not published reference labels; they are derived only from column 3 and never from the
bow force, bow velocity or beta that later models use as inputs.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import csv
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import sys
import uuid

import numpy as np

import grid_summary
import trial_audit


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REPORTS = PROJECT_ROOT / "reports/regime_map"
RATE_HZ = trial_audit.PUBLISHED_RATE_HZ
F0_RANGE_HZ = (40.0, 160.0)
CLASSES = ("helmholtz", "multiple_slip", "subharmonic", "aperiodic", "no_oscillation", "ambiguous")
COLORS = {"helmholtz": "#2a9d8f", "multiple_slip": "#e9c46a", "subharmonic": "#f4a261",
          "aperiodic": "#e76f51", "no_oscillation": "#d9d9d9", "ambiguous": "#8d99ae"}


class RegimeError(ValueError):
    """Inputs for regime mapping are missing or inconsistent."""


# ------------------------------------------------------------------ features

def dominant_period(x: np.ndarray, rate: float = RATE_HZ, f0_range=F0_RANGE_HZ) -> tuple[float, float]:
    """(periodicity, f0_hz) from the biased autocorrelation peak within the f0 range.

    The biased estimate decays with lag, so of equally strong multiples the shortest
    period wins; a genuinely period-doubled signal still peaks at the longer lag.
    """
    x = np.asarray(x, dtype=float) - float(np.mean(x))
    n = x.size
    lo, hi = int(rate / f0_range[1]), int(math.ceil(rate / f0_range[0]))
    if n < 2 * hi or not np.any(x):
        return math.nan, math.nan
    spectrum = np.fft.fft(x, n=2 * n)
    acf = np.fft.ifft(spectrum * np.conj(spectrum)).real[:n]
    acf /= acf[0]
    k = lo + int(np.argmax(acf[lo:hi + 1]))
    y0, y1, y2 = acf[k - 1], acf[k], acf[k + 1]
    denominator = y0 - 2 * y1 + y2
    lag = k + (0.5 * (y0 - y2) / denominator if denominator else 0.0)
    return float(y1), float(rate / lag)


def slips_per_period(x: np.ndarray, period: float, frac: float, merge_frac: float = 0.1) -> float:
    """Abrupt changes per period: runs of |diff| above ``frac`` of the typical per-period maximum.

    Runs closer than ``merge_frac`` of a period count once, so a smoothed return or
    ringing right after it is one slip.
    """
    if not (period > 2) or not math.isfinite(period):
        return math.nan
    d = np.abs(np.diff(np.asarray(x, dtype=float)))
    m = int(round(period))
    segments = d.size // m
    if segments < 2:
        return math.nan
    typical = float(np.median(d[:segments * m].reshape(segments, m).max(axis=1)))
    if typical <= 0:
        return math.nan
    mask = (d > frac * typical).astype(np.int8)
    edges = np.flatnonzero(np.diff(np.concatenate(([0], mask, [0]))))
    starts, stops = edges[0::2], edges[1::2]
    if starts.size == 0:
        return 0.0
    events, last_stop = 1, stops[0]
    for start, stop in zip(starts[1:], stops[1:]):
        if start - last_stop > merge_frac * period:
            events += 1
        last_stop = stop
    return events / (d.size / period)


def classify(amplitude: float, periodicity: float, f0: float, slips: float, reference_f0: float,
             thresholds: dict) -> str:
    if not (amplitude >= thresholds["min_amplitude"]):
        return "no_oscillation"
    if not math.isfinite(periodicity) or periodicity < thresholds["max_aperiodic"]:
        return "aperiodic"
    if periodicity < thresholds["min_periodicity"]:
        return "ambiguous"
    if math.isfinite(reference_f0) and math.isfinite(f0) and f0 < thresholds["subharmonic_ratio"] * reference_f0:
        return "subharmonic"
    if math.isfinite(slips) and abs(slips - 1.0) <= thresholds["slip_tolerance"]:
        return "helmholtz"
    if math.isfinite(slips) and slips > 1.0 + thresholds["slip_tolerance"]:
        return "multiple_slip"
    return "ambiguous"


def trial_features(task: tuple) -> dict:
    condition, directory, number, start, end, slip_frac = task
    row = {"condition": condition, "trial": number, "error": ""}
    try:
        base = Path(directory)
        signal = trial_audit.parse_signal(trial_audit._read_bounded(base / f"whole_{number}.csv",
                                                                    trial_audit.MAX_WHOLE_BYTES))
        window = signal[start:end + 1, 2]
        periodicity, f0 = dominant_period(window)
        row.update({"amplitude_std": float(np.std(window)), "periodicity": periodicity, "f0_hz": f0,
                    "slips_per_period": slips_per_period(window, RATE_HZ / f0, slip_frac) if math.isfinite(f0) else math.nan})
    except (OSError, trial_audit.TrialAuditError, IndexError) as error:
        row["error"] = str(error) or error.__class__.__name__
    return row


# ------------------------------------------------------------------ boundaries

def fit_boundaries(level_rank_class: list[tuple[int, int, str]], beta_centers: dict[int, float],
                   force_by_rank: list[float], max_hole: int = 2) -> dict:
    """Lower/upper Helmholtz boundary per beta level, then log-log slopes over uncensored levels.

    Helmholtz ranks separated by at most ``max_hole`` other ranks are bridged, so isolated
    misclassifications inside the region do not split it; the widest bridged run is used.
    Each boundary is the geometric mean of the forces either side of that run. A run
    touching the first or last force rank is censored (the boundary lies outside the
    sampled range) and excluded from the fit.
    """
    n_ranks = len(force_by_rank)
    by_level: dict[int, list[int]] = {}
    for level, rank, label in level_rank_class:
        if label == "helmholtz":
            by_level.setdefault(level, []).append(rank)
    lower, upper = [], []
    censored_lower = censored_upper = bridged = 0
    for level in sorted(beta_centers):
        ranks = sorted(set(by_level.get(level, [])))
        if not ranks:
            continue
        runs = np.split(np.array(ranks), np.flatnonzero(np.diff(ranks) > max_hole + 1) + 1)
        run = max(runs, key=lambda r: (r[-1] - r[0], len(r)))
        first, last = int(run[0]), int(run[-1])
        bridged += (last - first + 1) - len(run)
        beta = beta_centers[level]
        if first <= 1 + max_hole:  # an edge hole of <= max_hole ranks is indistinguishable from censoring
            censored_lower += 1
        else:
            pair = (force_by_rank[first - 2], force_by_rank[first - 1])
            if min(pair) > 0:
                lower.append((beta, math.sqrt(pair[0] * pair[1])))
        if last >= n_ranks - max_hole:
            censored_upper += 1
        else:
            pair = (force_by_rank[last - 1], force_by_rank[last])
            if min(pair) > 0:
                upper.append((beta, math.sqrt(pair[0] * pair[1])))

    def slope(points):
        if len(points) < 3:
            return {"n": len(points), "slope": None, "slope_se": None, "intercept": None, "beta_span": None}
        x, y = np.log([p[0] for p in points]), np.log([p[1] for p in points])
        s, c = np.polyfit(x, y, 1)
        se = None
        if len(points) >= 4:
            residual = y - (s * x + c)
            se = float(math.sqrt(np.sum(residual**2) / (len(points) - 2) / np.sum((x - x.mean())**2)))
        return {"n": len(points), "slope": round(float(s), 3), "slope_se": None if se is None else round(se, 3),
                "intercept": round(float(c), 4), "beta_span": round(float(math.exp(x.max() - x.min())), 2)}

    return {"lower": {**slope(lower), "theory_slope": -2, "censored_levels": censored_lower,
                      "points": [[round(b, 5), round(f, 5)] for b, f in lower]},
            "upper": {**slope(upper), "theory_slope": -1, "censored_levels": censored_upper,
                      "points": [[round(b, 5), round(f, 5)] for b, f in upper]},
            "levels_with_helmholtz": len(by_level), "levels_total": len(beta_centers),
            "max_hole": max_hole, "non_helmholtz_cells_bridged": bridged}


# ------------------------------------------------------------------ plotting

def _edges(centers: np.ndarray) -> np.ndarray:
    centers = np.asarray(centers, dtype=float)
    inner = np.sqrt(centers[1:] * centers[:-1])
    first = centers[0] ** 2 / inner[0]
    last = centers[-1] ** 2 / inner[-1]
    return np.concatenate(([first], inner, [last]))


def plot_maps(conditions: dict, destination: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    from matplotlib.patches import Patch

    names = sorted(conditions)
    figure, axes = plt.subplots(1, len(names), figsize=(4.8 * len(names), 6.0), squeeze=False,
                                constrained_layout=True, sharey=True)
    cmap = ListedColormap([COLORS[c] for c in CLASSES])
    for axis, name in zip(axes[0], names):
        data = conditions[name]
        centers = np.array([data["beta_centers"][L] for L in sorted(data["beta_centers"])])
        forces = np.array(data["force_by_rank"], dtype=float)
        positive = forces.copy()
        floor = positive[positive > 0].min() / 1.5
        positive[positive <= 0] = floor
        matrix = np.full((forces.size, centers.size), np.nan)
        for level, rank, label in data["cells"]:
            matrix[rank - 1, level] = CLASSES.index(label)
        axis.pcolormesh(_edges(centers), _edges(positive), np.ma.masked_invalid(matrix), cmap=cmap,
                        vmin=-0.5, vmax=len(CLASSES) - 0.5, shading="flat")
        off = data["off_plateau_points"]
        if off:
            axis.scatter(*zip(*off), s=6, color="black", marker=".", label="window off velocity plateau")
        low = data["nonpositive_points"]
        if low:
            axis.scatter(*zip(*low), s=14, color="red", marker="x", linewidths=0.7, label="column-1 mean ≤ 0")
        for key, style in (("lower", "--"), ("upper", ":")):
            fit = data["boundaries"][key]
            if fit["slope"] is not None:  # draw only across the beta range actually used in the fit
                used = np.array([p[0] for p in fit["points"]])
                xs = np.geomspace(used.min(), used.max(), 20)
                se = "" if fit["slope_se"] is None else f"±{fit['slope_se']}"
                axis.plot(xs, np.exp(fit["intercept"]) * xs ** fit["slope"], style, color="black", linewidth=1.2,
                          label=f"{key} boundary slope {fit['slope']}{se} (Schelleng {fit['theory_slope']}), n={fit['n']}")
        axis.set_xscale("log"); axis.set_yscale("log")
        edges = _edges(positive)
        axis.set_ylim(edges[0] / 1.3, edges[-1] * 1.3)
        axis.set_xlabel("β (relative bow-bridge distance)")
        if axis is axes[0][0]:
            axis.set_ylabel("column 1, median by force rank (published: bow force, N)")
        speed = ", ".join(f"{v:g}" for v in data["c2_peak"])
        axis.set_title(f"bow speed {speed} m/s (measured column 2)\n{name}", fontsize=10)
        axis.legend(fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.14), frameon=False)
    handles = [Patch(color=COLORS[c], label=c) for c in CLASSES]
    figure.legend(handles=handles, loc="outside upper center", ncols=len(CLASSES), fontsize=8)
    figure.savefig(destination, dpi=140)
    plt.close(figure)


def plot_examples(examples: list[dict], destination: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = [c for c in CLASSES if any(e["label"] == c for e in examples)]
    if not labels:
        return
    per = max(sum(e["label"] == c for e in examples) for c in labels)
    figure, axes = plt.subplots(len(labels), per, figsize=(3.4 * per, 1.9 * len(labels)), squeeze=False,
                                constrained_layout=True)
    for row, label in enumerate(labels):
        chosen = [e for e in examples if e["label"] == label]
        for col in range(per):
            axis = axes[row][col]
            if col >= len(chosen):
                axis.axis("off")
                continue
            e = chosen[col]
            t = np.arange(e["signal"].size) / RATE_HZ * 1000
            axis.plot(t, e["signal"], color=COLORS[label], linewidth=0.8)
            axis.set_title(f"{label} | {e['condition'][-4:]} #{e['trial']} β{e['beta']:.3f} r{e['rank']}\n"
                           f"per {e['periodicity']:.2f} slips {e['slips']:.2f} f0 {e['f0']:.1f}", fontsize=7)
            axis.tick_params(labelsize=6)
            if row == len(labels) - 1:
                axis.set_xlabel("ms from window start", fontsize=7)
    figure.savefig(destination, dpi=130)
    plt.close(figure)


# ------------------------------------------------------------------ orchestration

def run(audit_reports: Path, reports_dir: Path, workers: int, thresholds: dict, limit: int | None,
        plot: bool, progress=print) -> tuple[int, Path]:
    trials_path = grid_summary.resolve_trials(None, audit_reports)
    audit_summary = json.loads((trials_path.parent / "summary.json").read_text(encoding="utf-8"))
    root = Path(audit_summary["root"])
    directories = {d.name: d for d in trial_audit.discover(root)}
    by_condition, _ = grid_summary.load_trials(trials_path)
    windows = {}
    with trials_path.open(newline="", encoding="utf-8") as handle:
        for raw in csv.DictReader(handle):
            if not raw["error"]:
                windows[(raw["condition"], int(raw["trial"]))] = (int(raw["window_start"]), int(raw["window_end"]))

    tasks, grids = [], {}
    for name in sorted(by_condition):
        if name not in directories:
            raise RegimeError(f"condition folder not found under {root}: {name}")
        summary, rows = grid_summary.analyse_condition(by_condition[name])
        if not summary["beta"]["uniform_trials_per_level"] or "force_by_rank" not in summary["force"]:
            raise RegimeError(f"{name}: grid is not uniform; boundary mapping needs a complete grid")
        grids[name] = {"summary": summary, "rows": {r["trial"]: r for r in rows}}
        chosen = sorted(r["trial"] for r in rows)[:limit] if limit else sorted(r["trial"] for r in rows)
        tasks += [(name, str(directories[name]), n, *windows[(name, n)], thresholds["slip_frac"]) for n in chosen]

    progress(f"Classifying {len(tasks)} trial windows ({workers} workers). Thresholds are provisional.")
    features = []
    executor = ProcessPoolExecutor(max_workers=workers) if workers > 1 else None
    try:
        iterator = executor.map(trial_features, tasks, chunksize=8) if executor else map(trial_features, tasks)
        for i, row in enumerate(iterator, 1):
            features.append(row)
            if i % 500 == 0 or i == len(tasks):
                progress(f"  classified {i}/{len(tasks)}")
    finally:
        if executor:
            executor.shutdown()

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = reports_dir / f"{stamp}_{uuid.uuid4().hex[:8]}"
    run_dir.mkdir(parents=True, exist_ok=False)
    report = {"created_utc": stamp, "source_trials": str(trials_path), "thresholds_provisional": thresholds,
              "limit_per_condition": limit, "f0_search_hz": list(F0_RANGE_HZ),
              "label_scope": ("Rule-based, provisional, derived from column 3 only. Not published reference labels; "
                              "verify with examples.png before interpreting boundaries."),
              "conditions": {}}
    plot_data, examples, table = {}, [], []
    for name in sorted(grids):
        rows = [f for f in features if f["condition"] == name]
        good = [f for f in rows if not f["error"]]
        strong = [f["f0_hz"] for f in good if f["amplitude_std"] >= thresholds["min_amplitude"]
                  and math.isfinite(f["periodicity"]) and f["periodicity"] >= thresholds["min_periodicity"]]
        reference_f0 = float(np.median(strong)) if strong else math.nan
        grid_rows, gsum = grids[name]["rows"], grids[name]["summary"]
        cells, counts = [], {c: 0 for c in CLASSES}
        off_total = {c: 0 for c in CLASSES}
        low_total = {c: 0 for c in CLASSES}
        force_by_rank = gsum["force"]["force_by_rank"]
        for f in good:
            label = classify(f["amplitude_std"], f["periodicity"], f["f0_hz"], f["slips_per_period"], reference_f0, thresholds)
            g = grid_rows[f["trial"]]
            f.update(label=label, beta=g["beta"], beta_level=g["beta_level"], force_rank=g["force_rank"],
                     window_inside_plateau=g["window_inside_plateau"], c1_window_mean=g["c1_window_mean"])
            counts[label] += 1
            cells.append((g["beta_level"], g["force_rank"], label))
            if not g["window_inside_plateau"]:
                off_total[label] += 1
            if g["c1_window_mean"] <= 0:
                low_total[label] += 1
            table.append(f)
        centers = {}
        for g in grid_rows.values():
            centers[g["beta_level"]] = g["beta_level_center"]
        boundaries = fit_boundaries(cells, centers, force_by_rank, thresholds.get("max_hole", 2))
        f0s = [f["f0_hz"] for f in good if math.isfinite(f["f0_hz"])]
        report["conditions"][name] = {
            "trials_classified": len(good), "errors": len(rows) - len(good),
            "c2_peak": gsum["c2_peak_levels"], "reference_f0_hz": None if math.isnan(reference_f0) else round(reference_f0, 2),
            "f0_hz": trial_audit._describe(f0s, 2), "classes": counts,
            "classes_among_off_plateau": {k: v for k, v in off_total.items() if v},
            "classes_among_nonpositive_force": {k: v for k, v in low_total.items() if v},
            "boundaries": boundaries}
        plot_data[name] = {"beta_centers": centers, "force_by_rank": force_by_rank, "cells": cells,
                           "c2_peak": gsum["c2_peak_levels"], "boundaries": boundaries,
                           "off_plateau_points": [(centers[g["beta_level"]], force_by_rank[g["force_rank"] - 1])
                                                  for g in grid_rows.values() if not g["window_inside_plateau"]
                                                  and force_by_rank[g["force_rank"] - 1] > 0],
                           "nonpositive_points": [(centers[g["beta_level"]], max(force_by_rank[g["force_rank"] - 1], 1e-3))
                                                  for g in grid_rows.values() if g["c1_window_mean"] <= 0]}
        for label in CLASSES:
            members = sorted((f for f in good if f["label"] == label), key=lambda f: f["trial"])
            for f in [members[i] for i in np.linspace(0, len(members) - 1, min(3, len(members))).astype(int)] if members else []:
                examples.append({"condition": name, "trial": f["trial"], "label": label, "beta": f["beta"],
                                 "rank": f["force_rank"], "periodicity": f["periodicity"], "slips": f["slips_per_period"],
                                 "f0": f["f0_hz"]})

    fields = ["condition", "trial", "label", "beta", "beta_level", "force_rank", "c1_window_mean",
              "window_inside_plateau", "amplitude_std", "periodicity", "f0_hz", "slips_per_period"]
    with (run_dir / "regimes.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for f in sorted(table, key=lambda f: (f["condition"], f["trial"])):
            writer.writerow({k: (repr(float(v)) if isinstance(v, float) else v) for k, v in f.items() if k in fields})
    if plot:
        for e in examples:
            directory = directories[e["condition"]]
            start, _ = windows[(e["condition"], e["trial"])]
            signal = trial_audit.parse_signal(trial_audit._read_bounded(directory / f"whole_{e['trial']}.csv",
                                                                        trial_audit.MAX_WHOLE_BYTES))
            span = int(4 * RATE_HZ / e["f0"]) if math.isfinite(e["f0"]) and e["f0"] > 0 else 2000
            e["signal"] = signal[start:start + span, 2]
        plot_maps(plot_data, run_dir / "regime_map.png")
        plot_examples(examples, run_dir / "examples.png")
        report["plots"] = [str(run_dir / "regime_map.png"), str(run_dir / "examples.png")]
    (run_dir / "summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    for line in format_report(report):
        progress(line)
    return 0, run_dir


def format_report(report: dict) -> list[str]:
    t = report["thresholds_provisional"]
    lines = [f"REGIME MAP (provisional labels; thresholds {t})"]
    for name, c in report["conditions"].items():
        lines.append(f"[{name}] c2 peak {c['c2_peak']} | {c['trials_classified']} classified, errors {c['errors']} | "
                     f"reference f0 {c['reference_f0_hz']} Hz (f0 p25..p75 {c['f0_hz'].get('p25')}..{c['f0_hz'].get('p75')})")
        lines.append("  classes: " + " | ".join(f"{k} {v}" for k, v in c["classes"].items()))
        lo, up = c["boundaries"]["lower"], c["boundaries"]["upper"]
        lines.append(f"  lower boundary slope {lo['slope']}±{lo['slope_se']} (theory -2, n={lo['n']}, beta span x{lo['beta_span']}, "
                     f"censored {lo['censored_levels']}) | upper slope {up['slope']}±{up['slope_se']} (theory -1, n={up['n']}, "
                     f"beta span x{up['beta_span']}, censored {up['censored_levels']}) | "
                     f"Helmholtz in {c['boundaries']['levels_with_helmholtz']}/{c['boundaries']['levels_total']} beta levels, "
                     f"{c['boundaries']['non_helmholtz_cells_bridged']} interior cells bridged")
        lines.append(f"  labels of off-plateau windows {c['classes_among_off_plateau']} | "
                     f"of column-1<=0 windows {c['classes_among_nonpositive_force']}")
    for plot in report.get("plots", []):
        lines.append(f"Plot: {plot}")
    return lines


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--audit-reports", type=Path, default=grid_summary.AUDIT_REPORTS)
    parser.add_argument("--reports-dir", type=Path, default=DEFAULT_REPORTS)
    parser.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    parser.add_argument("--limit", type=int, default=None, help="classify only the first N trials per folder")
    parser.add_argument("--no-plot", action="store_true")
    parser.add_argument("--slip-frac", type=float, default=0.25)
    parser.add_argument("--slip-tolerance", type=float, default=0.25)
    parser.add_argument("--min-amplitude", type=float, default=0.02)
    parser.add_argument("--min-periodicity", type=float, default=0.8)
    parser.add_argument("--max-aperiodic", type=float, default=0.5)
    parser.add_argument("--subharmonic-ratio", type=float, default=0.75)
    parser.add_argument("--max-hole", type=int, default=2, help="non-Helmholtz ranks bridged inside a region")
    args = parser.parse_args(argv)
    thresholds = {"slip_frac": args.slip_frac, "slip_tolerance": args.slip_tolerance,
                  "min_amplitude": args.min_amplitude, "min_periodicity": args.min_periodicity,
                  "max_aperiodic": args.max_aperiodic, "subharmonic_ratio": args.subharmonic_ratio,
                  "max_hole": args.max_hole}
    if args.workers <= 0 or (args.limit is not None and args.limit <= 0):
        parser.error("--workers and --limit must be positive")
    try:
        code, _ = run(args.audit_reports, args.reports_dir, args.workers, thresholds, args.limit, not args.no_plot)
        return code
    except (RegimeError, grid_summary.GridError, trial_audit.TrialAuditError, OSError, KeyError) as error:
        print("REGIME MAP: FAIL —", error, file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
