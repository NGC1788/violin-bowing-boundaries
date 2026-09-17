#!/usr/bin/env python3
"""Provisional bowing-regime labels from bridge force, and playable-region maps.

Each trial's classification window of column 3 (published: bridge force) is folded at its
dominant period into one mean cycle. Helmholtz motion makes that cycle a sawtooth with a
single abrupt return (flyback) per period; double and multiple slipping add flybacks of
comparable size. The dataset authors classify with the detrended-staircase method of
Woodhouse and Galluzzo, which compares each step with the theoretical Helmholtz flyback
2 Z v_b / beta. Here flybacks are counted on the averaged cycle instead, so a label uses
waveform shape only; the flyback law is then tested on the Helmholtz-labelled trials as an
independent check. Thresholds are PROVISIONAL, stated in every report, and must be checked
against the saved example waveforms. A label never uses its own trial's bow force, bow
velocity or beta; column 1 only selects no-contact windows for one global noise floor.
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
SENSITIVITY_FRACS = (0.3, 0.4, 0.5, 0.6, 0.7)
DEFAULT_THRESHOLDS = {"flyback_frac": 0.5, "hysteresis_sigma": 3.0, "noise_floor_factor": 3.0, "min_amplitude": 0.02,
                      "min_periodicity": 0.8, "max_aperiodic": 0.5, "subharmonic_ratio": 0.75, "max_hole": 2}
# Comparison only: van Walstijn et al. (Acta Acustica 2026) Table 1 for a cello G2 string on the same
# monochord, Z = sqrt(T * rho * pi * r^2) with T = 161 N, rho = 11503 kg/m^3, r = 0.487 mm. The string in
# this archive may be a different type, so this value is never used for labelling.
REFERENCE_IMPEDANCE = 1.175
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


def fold_cycle(x: np.ndarray, period: float) -> tuple[np.ndarray, float] | None:
    """Mean of all whole periods resampled onto round(period) phases, and the standard error of that mean.

    Periods are cut at fractional positions, so a non-integer period does not drift across the window.
    """
    x = np.asarray(x, dtype=float)
    if not (math.isfinite(period) and period > 8):
        return None
    count = int(x.size // period)
    if count < 2:
        return None
    m = int(round(period))
    positions = (np.arange(count)[:, None] + np.arange(m)[None, :] / m) * period
    cycles = np.interp(positions.ravel(), np.arange(x.size), x).reshape(count, m)
    mean = cycles.mean(axis=0)
    return mean, float(math.sqrt(float(np.mean((cycles - mean) ** 2)) / count))


def turning_points(y: np.ndarray, hysteresis: float) -> np.ndarray:
    """Indices of alternating extrema, starting at y[0] (a maximum) and ending at the last sample.

    A reversal becomes a new leg only once it exceeds ``hysteresis``.
    """
    pivots, extreme, falling = [0], 0, True
    for i in range(1, y.size):
        if falling:
            if y[i] <= y[extreme]:
                extreme = i
            elif y[i] - y[extreme] > hysteresis:
                pivots.append(extreme)
                extreme, falling = i, False
        else:
            if y[i] >= y[extreme]:
                extreme = i
            elif y[extreme] - y[i] > hysteresis:
                pivots.append(extreme)
                extreme, falling = i, True
    if pivots[-1] != y.size - 1:
        pivots.append(y.size - 1)
    return np.array(pivots)


FLYBACK_KEYS = ("flybacks_per_period", "second_flyback_ratio", "flyback_height", "flyback_ms", "flyback_sign",
                "fold_noise")


def flyback_structure(x: np.ndarray, period: float, frac: float = 0.5, hysteresis_sigma: float = 3.0) -> dict:
    """Flybacks in the mean cycle of ``x`` folded at ``period`` samples.

    The cycle is split into legs between turning points; reversals under ``hysteresis_sigma``
    standard errors of the mean cycle are noise. The flyback direction is that of the leg
    that is both tallest and fastest (largest height^2 / duration). ``flybacks_per_period``
    counts legs in that direction at least ``frac`` of the tallest; ``second_flyback_ratio``
    is the second-tallest over the tallest, so a trial has one flyback exactly when that ratio
    is below ``frac``.
    """
    result = {"flybacks_per_period": math.nan, "second_flyback_ratio": math.nan, "flyback_height": math.nan,
              "flyback_ms": math.nan, "flyback_sign": 0, "fold_noise": math.nan, "cycle": None, "flyback_phases": []}
    folded = fold_cycle(x, period)
    if folded is None:
        return result
    mean, noise = folded
    m = mean.size
    start = int(np.argmax(mean))
    loop = np.append(np.roll(mean, -start), mean[start])
    pivots = turning_points(loop, hysteresis_sigma * noise)
    legs, durations = np.diff(loop[pivots]), np.diff(pivots)
    result.update(fold_noise=noise, cycle=mean)
    if legs.size < 2 or not np.any(legs):
        result["flybacks_per_period"] = 0
        return result
    sign = 1 if legs[int(np.argmax(legs**2 / durations))] > 0 else -1
    along = sign * legs
    chosen = np.flatnonzero(along > 0)
    heights = along[chosen]
    top = float(heights.max())
    ordered = np.sort(heights)[::-1]
    counted = chosen[heights >= frac * top]
    tallest = chosen[int(np.argmax(heights))]
    result.update(flybacks_per_period=int(counted.size),
                  second_flyback_ratio=float(ordered[1] / top) if ordered.size > 1 else 0.0,
                  flyback_height=top, flyback_ms=float(durations[tallest] * period / m / RATE_HZ * 1000),
                  flyback_sign=sign, flyback_phases=[float((start + pivots[i]) % m / m) for i in counted])
    return result


def classify(amplitude: float, periodicity: float, f0: float, flybacks: float, reference_f0: float,
             thresholds: dict) -> str:
    # A clearly periodic window is string motion however small: Helmholtz amplitude scales with v_b / beta,
    # so a fixed floor alone would erase weak but regular oscillation at large beta and low speed.
    if not (amplitude >= thresholds["min_amplitude"]) and not (periodicity >= thresholds["min_periodicity"]):
        return "no_oscillation"
    if not math.isfinite(periodicity) or periodicity < thresholds["max_aperiodic"]:
        return "aperiodic"
    if periodicity < thresholds["min_periodicity"]:
        return "ambiguous"
    if math.isfinite(reference_f0) and math.isfinite(f0) and f0 < thresholds["subharmonic_ratio"] * reference_f0:
        return "subharmonic"
    if flybacks == 1:
        return "helmholtz"
    if flybacks >= 2:
        return "multiple_slip"
    return "ambiguous"


def load_window(directory: Path, number: int, start: int, end: int, cached: bool) -> np.ndarray:
    """Rows start..end (inclusive, all columns) from the CSV, or the same rows from the audit's window cache."""
    if cached:
        window = np.load(trial_audit.cache_path(Path(directory), number), allow_pickle=False)
        if window.ndim != 2 or window.shape[0] != end - start + 1:
            raise trial_audit.TrialAuditError(f"cached window {number} has shape {window.shape}; "
                                              f"expected {end - start + 1} rows")
        return window
    signal = trial_audit.parse_signal(trial_audit._read_bounded(Path(directory) / f"whole_{number}.csv",
                                                                trial_audit.MAX_WHOLE_BYTES))
    return signal[start:end + 1]


def trial_features(task: tuple) -> dict:
    condition, directory, number, start, end, frac, hysteresis_sigma, cached = task
    row = {"condition": condition, "trial": number, "error": ""}
    try:
        window = load_window(Path(directory), number, start, end, cached)[:, 2]
        periodicity, f0 = dominant_period(window)
        shape = flyback_structure(window, RATE_HZ / f0 if math.isfinite(f0) else math.nan, frac, hysteresis_sigma)
        row.update({"amplitude_std": float(np.std(window)), "periodicity": periodicity, "f0_hz": f0,
                    **{key: shape[key] for key in FLYBACK_KEYS}})
    except (OSError, ValueError, trial_audit.TrialAuditError, IndexError) as error:
        row["error"] = str(error) or error.__class__.__name__
    return row


# ------------------------------------------------------------------ boundaries

def fit_boundaries(level_rank_class: list[tuple[int, int, str]], beta_centers: dict[int, float],
                   force_by_rank: list[float], max_hole: int = 2, excluded=frozenset()) -> dict:
    """Lower/upper Helmholtz boundary per beta level, then log-log slopes over uncensored levels.

    Helmholtz ranks separated by at most ``max_hole`` other ranks are bridged, so isolated
    misclassifications inside the region do not split it; the widest bridged run is used.
    Each boundary is the geometric mean of the forces either side of that run. A run
    touching the first or last force rank is censored (the boundary lies outside the
    sampled range) and excluded from the fit. ``excluded`` (level, rank) cells, such as windows off the
    velocity plateau, never count as Helmholtz, and a boundary next to one is censored as undetermined.
    """
    n_ranks = len(force_by_rank)
    by_level: dict[int, list[int]] = {}
    for level, rank, label in level_rank_class:
        if label == "helmholtz" and (level, rank) not in excluded:
            by_level.setdefault(level, []).append(rank)
    lower, upper = [], []
    censored_lower = censored_upper = bridged = excluded_lower = excluded_upper = 0
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
        elif (level, first - 1) in excluded:
            censored_lower += 1
            excluded_lower += 1
        else:
            pair = (force_by_rank[first - 2], force_by_rank[first - 1])
            if min(pair) > 0:
                lower.append((beta, math.sqrt(pair[0] * pair[1])))
        if last >= n_ranks - max_hole:
            censored_upper += 1
        elif (level, last + 1) in excluded:
            censored_upper += 1
            excluded_upper += 1
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
            "max_hole": max_hole, "non_helmholtz_cells_bridged": bridged,
            "censored_next_to_excluded": {"lower": excluded_lower, "upper": excluded_upper}}


def noise_floor(amplitudes, column1_means, factor: float, fallback: float, min_controls: int = 10) -> dict:
    """One global oscillation floor: ``factor`` x the median window std of no-contact windows (column-1 mean <= 0).

    Falls back to ``fallback`` when there are too few such windows; ``fallback`` is also the lower bound.
    """
    controls = np.array([a for a, c in zip(amplitudes, column1_means) if c <= 0 and math.isfinite(a)], dtype=float)
    info = {"controls": int(controls.size), "factor": factor, "fallback": fallback}
    if controls.size < min_controls:
        return {**info, "min_amplitude": fallback, "source": f"fallback (fewer than {min_controls} column-1<=0 windows)"}
    median = float(np.median(controls))
    return {**info, "control_median": round(median, 6), "control_p95": round(float(np.percentile(controls, 95)), 6),
            "min_amplitude": max(fallback, factor * median), "source": "column-1<=0 windows"}


def frac_sensitivity(trials: list[dict], beta_centers: dict[int, float], force_by_rank: list[float],
                     fracs=SENSITIVITY_FRACS, max_hole: int = 2, excluded=frozenset()) -> list[dict]:
    """Boundary slopes if the flyback fraction were each of ``fracs``, with every other gate unchanged."""
    out = []
    for frac in fracs:
        cells = [(f["beta_level"], f["force_rank"],
                  "helmholtz" if f["label"] in ("helmholtz", "multiple_slip") and f["second_flyback_ratio"] < frac
                  else "other") for f in trials]
        fit = fit_boundaries(cells, beta_centers, force_by_rank, max_hole, excluded)
        out.append({"flyback_frac": frac, "helmholtz": sum(c[2] == "helmholtz" for c in cells),
                    **{f"{side}_{key}": fit[side][key] for side in ("lower", "upper") for key in ("slope", "slope_se", "n")}})
    return out


def log_linear_fit(y, regressors: dict) -> dict:
    """Least squares log y = c + sum_k e_k log x_k with standard errors; a regressor spanning < 1.5x is dropped."""
    y = np.log(np.asarray(y, dtype=float))
    names = [k for k, x in regressors.items() if np.ptp(np.log(np.asarray(x, dtype=float))) >= math.log(1.5)]
    design = np.column_stack([np.ones(y.size)] + [np.log(np.asarray(regressors[k], dtype=float)) for k in names])
    coef, *_ = np.linalg.lstsq(design, y, rcond=None)
    residual = y - design @ coef
    dof = y.size - design.shape[1]
    covariance = residual @ residual / dof * np.linalg.inv(design.T @ design) if dof > 0 else None
    result = {"n": int(y.size), "intercept": round(float(coef[0]), 4),
              "rms_log_residual": round(float(np.sqrt(np.mean(residual**2))), 4)}
    for i, name in enumerate(names, 1):
        result[f"exponent_{name}"] = round(float(coef[i]), 3)
        result[f"exponent_{name}_se"] = None if covariance is None else round(float(math.sqrt(covariance[i, i])), 3)
    return result


def flyback_law(points) -> dict:
    """Fit log H = c + a log v_b + b log beta; ideal Helmholtz motion gives a = 1, b = -1 and H = 2 Z v_b / beta.

    ``points`` are (flyback height, bow speed, beta). The speed exponent is fitted only when speeds span
    at least 1.5x; ``impedance_kg_s`` is the distribution of H beta / (2 v_b) with both exponents fixed.
    """
    rows = [(h, v, b) for h, v, b in points if all(math.isfinite(q) and q > 0 for q in (h, v, b))]
    if len(rows) < 5:
        return {"n": len(rows)}
    height, speed, beta = (np.array(c, dtype=float) for c in zip(*rows))
    result = {**log_linear_fit(height, {"speed": speed, "beta": beta}),
              "theory": "flyback = 2 Z v_b / beta (speed exponent +1, beta exponent -1)"}
    z = height * beta / (2 * speed)
    result["impedance_kg_s"] = {q: round(float(np.percentile(z, pct)), 4) for q, pct in (("p25", 25), ("median", 50), ("p75", 75))}
    result["reference_impedance_kg_s"] = REFERENCE_IMPEDANCE
    return result


def schelleng_law(points) -> dict:
    """Pooled boundary fit log F = c + a log v_b + b log beta over (beta, force, speed) points from all folders.

    Schelleng: minimum force a = 1, b = -2; maximum force a = 1, b = -1.
    """
    rows = [(b, f, v) for b, f, v in points if all(math.isfinite(q) and q > 0 for q in (b, f, v))]
    if len(rows) < 5:
        return {"n": len(rows)}
    beta, force, speed = (np.array(c, dtype=float) for c in zip(*rows))
    return log_linear_fit(force, {"speed": speed, "beta": beta})


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
            cycle = e.get("cycle")
            if cycle is not None and label not in ("no_oscillation", "aperiodic"):  # mean cycle and counted flybacks
                m, period_ms = cycle.size, e["period"] / RATE_HZ * 1000
                repeats = int(math.ceil(t[-1] / period_ms)) if t.size else 0
                phase_t = (np.arange(repeats)[:, None] + np.arange(m)[None, :] / m) * period_ms
                axis.plot(phase_t.ravel(), np.tile(cycle, repeats), color="black", linewidth=0.5, alpha=0.7)
                for phase in e["phases"]:
                    marks = (np.arange(repeats) + phase) * period_ms
                    axis.scatter(marks, np.full(repeats, cycle[int(phase * m) % m]), s=10, color="red", zorder=3,
                                 marker="v" if e["sign"] < 0 else "^")
                axis.set_xlim(0, t[-1])
            axis.set_title(f"{label} | {e['condition'][-4:]} #{e['trial']} β{e['beta']:.3f} r{e['rank']}\n"
                           f"per {e['periodicity']:.2f} flybacks {e['flybacks']} r2 {e['r2']:.2f} f0 {e['f0']:.1f}",
                           fontsize=7)
            axis.tick_params(labelsize=6)
            if row == len(labels) - 1:
                axis.set_xlabel("ms from window start", fontsize=7)
    figure.savefig(destination, dpi=130)
    plt.close(figure)


def plot_features(table: list[dict], applied: dict, destination: Path) -> None:
    """Distributions behind each provisional threshold, per condition."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = sorted({f["condition"] for f in table})
    if not names:
        return
    figure, axes = plt.subplots(len(names), 3, figsize=(13, 2.8 * len(names)), squeeze=False, constrained_layout=True)
    for row, name in enumerate(names):
        rows = [f for f in table if f["condition"] == name]
        amplitude = np.array([f["amplitude_std"] for f in rows], dtype=float)
        axis = axes[row][0]
        axis.hist(np.log10(amplitude[amplitude > 0]), bins=60, color="#8d99ae")
        axis.axvline(math.log10(applied["min_amplitude"]), color="black", linestyle="--", linewidth=1)
        axis.set_title(f"{name}: log10 window std of column 3 (dashed: floor)", fontsize=8)
        periodic = [f["periodicity"] for f in rows if f["amplitude_std"] >= applied["min_amplitude"]
                    and math.isfinite(f["periodicity"])]
        axis = axes[row][1]
        axis.hist(periodic, bins=50, range=(0, 1), color="#8d99ae")
        for value in (applied["max_aperiodic"], applied["min_periodicity"]):
            axis.axvline(value, color="black", linestyle="--", linewidth=1)
        axis.set_title("periodicity above the floor (dashed: aperiodic / periodic gates)", fontsize=8)
        ratios = [f["second_flyback_ratio"] for f in rows if f["label"] in ("helmholtz", "multiple_slip")
                  and math.isfinite(f["second_flyback_ratio"])]
        axis = axes[row][2]
        axis.hist(ratios, bins=40, range=(0, 1), color="#2a9d8f")
        axis.axvline(applied["flyback_frac"], color="black", linestyle="--", linewidth=1)
        axis.set_title("second / largest flyback in periodic trials (dashed: flyback frac)", fontsize=8)
        for axis in axes[row]:
            axis.tick_params(labelsize=7)
    figure.savefig(destination, dpi=130)
    plt.close(figure)


def plot_flyback_law(table: list[dict], law: dict, destination: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(6.4, 5.2), constrained_layout=True)
    markers = "os^Dv"
    plotted = []
    for i, name in enumerate(sorted({f["condition"] for f in table})):
        rows = [f for f in table if f["condition"] == name and f["label"] == "helmholtz"
                and f["flyback_height"] > 0 and f["c2_window_mean"] > 0]
        if rows:
            x = [2 * f["c2_window_mean"] / f["beta"] for f in rows]
            plotted += x
            axis.scatter(x, [f["flyback_height"] for f in rows], s=9, marker=markers[i % len(markers)], alpha=0.6,
                         label=f"{name} (n={len(rows)})")
    z = law.get("impedance_kg_s", {}).get("median")
    if z and plotted:  # from the data range: linear autoscale limits can be negative before the log scale is set
        xs = np.geomspace(min(plotted), max(plotted), 20)
        axis.plot(xs, z * xs, color="black", linewidth=1, label=f"median Z = {z} kg/s")
        axis.plot(xs, REFERENCE_IMPEDANCE * xs, color="grey", linestyle="--", linewidth=1,
                  label=f"reference Z = {REFERENCE_IMPEDANCE} kg/s (other paper)")
    axis.set_xscale("log"); axis.set_yscale("log")
    axis.set_xlabel("2 v_b / β  (column 2 window mean / beta file, m/s)")
    axis.set_ylabel("largest flyback of mean cycle (column 3)")
    exponents = ", ".join(f"{k.split('_')[1]} {law[k]}±{law.get(k + '_se')}" for k in ("exponent_speed", "exponent_beta") if k in law)
    axis.set_title(f"Helmholtz-labelled trials: flyback law check\nfitted exponents {exponents or 'n/a'} (theory +1, -1)", fontsize=9)
    axis.legend(fontsize=7, frameon=False)
    figure.savefig(destination, dpi=140)
    plt.close(figure)


# ------------------------------------------------------------------ orchestration

def run(audit_reports: Path, reports_dir: Path, workers: int, thresholds: dict, limit: int | None,
        plot: bool, progress=print) -> tuple[int, Path]:
    trials_path = grid_summary.resolve_trials(None, audit_reports)
    audit_summary = json.loads((trials_path.parent / "summary.json").read_text(encoding="utf-8"))
    by_condition, _ = grid_summary.load_trials(trials_path)
    cached = bool(audit_summary.get("window_cache"))
    root = Path(audit_summary["window_cache"] if cached else audit_summary["root"])
    directories = ({name: root / name for name in by_condition if (root / name).is_dir()} if cached
                   else {d.name: d for d in trial_audit.discover(root)})
    windows, speeds = {}, {}
    with trials_path.open(newline="", encoding="utf-8") as handle:
        for raw in csv.DictReader(handle):
            if not raw["error"]:
                key = (raw["condition"], int(raw["trial"]))
                windows[key] = (int(raw["window_start"]), int(raw["window_end"]))
                speeds[key] = float(raw["c2_window_mean"])

    tasks, grids = [], {}
    for name in sorted(by_condition):
        if name not in directories:
            raise RegimeError(f"condition folder not found under {root}: {name}")
        summary, rows = grid_summary.analyse_condition(by_condition[name])
        if not summary["beta"]["uniform_trials_per_level"] or "force_by_rank" not in summary["force"]:
            raise RegimeError(f"{name}: grid is not uniform; boundary mapping needs a complete grid")
        grids[name] = {"summary": summary, "rows": {r["trial"]: r for r in rows}}
        chosen = sorted(r["trial"] for r in rows)[:limit] if limit else sorted(r["trial"] for r in rows)
        tasks += [(name, str(directories[name]), n, *windows[(name, n)], thresholds["flyback_frac"],
                   thresholds["hysteresis_sigma"], cached) for n in chosen]

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

    good_all = [f for f in features if not f["error"]]
    floor = noise_floor([f["amplitude_std"] for f in good_all],
                        [grids[f["condition"]]["rows"][f["trial"]]["c1_window_mean"] for f in good_all],
                        thresholds["noise_floor_factor"], thresholds["min_amplitude"])
    applied = {**thresholds, "min_amplitude": floor["min_amplitude"]}

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = reports_dir / f"{stamp}_{uuid.uuid4().hex[:8]}"
    run_dir.mkdir(parents=True, exist_ok=False)
    report = {"created_utc": stamp, "source_trials": str(trials_path), "window_source": "cache" if cached else "csv",
              "thresholds_provisional": thresholds,
              "noise_floor": floor, "thresholds_applied": applied,
              "limit_per_condition": limit, "f0_search_hz": list(F0_RANGE_HZ),
              "label_scope": ("Rule-based, provisional, from the shape of column 3 (plus one global noise floor from "
                              "column-1<=0 windows). Not published reference labels; verify with examples.png "
                              "before interpreting boundaries."),
              "conditions": {}}
    plot_data, examples, table = {}, [], []
    boundary_points = {"lower": [], "upper": []}
    for name in sorted(grids):
        rows = [f for f in features if f["condition"] == name]
        good = [f for f in rows if not f["error"]]
        strong = [f["f0_hz"] for f in good if f["amplitude_std"] >= applied["min_amplitude"]
                  and math.isfinite(f["periodicity"]) and f["periodicity"] >= applied["min_periodicity"]]
        reference_f0 = float(np.median(strong)) if strong else math.nan
        grid_rows, gsum = grids[name]["rows"], grids[name]["summary"]
        cells, counts = [], {c: 0 for c in CLASSES}
        off_total = {c: 0 for c in CLASSES}
        low_total = {c: 0 for c in CLASSES}
        force_by_rank = gsum["force"]["force_by_rank"]
        for f in good:
            label = classify(f["amplitude_std"], f["periodicity"], f["f0_hz"], f["flybacks_per_period"], reference_f0, applied)
            g = grid_rows[f["trial"]]
            f.update(label=label, beta=g["beta"], beta_level=g["beta_level"], force_rank=g["force_rank"],
                     window_inside_plateau=g["window_inside_plateau"], c1_window_mean=g["c1_window_mean"],
                     c2_window_mean=speeds[(name, f["trial"])])
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
        off_plateau = frozenset((g["beta_level"], g["force_rank"]) for g in grid_rows.values() if not g["window_inside_plateau"])
        boundaries = fit_boundaries(cells, centers, force_by_rank, thresholds.get("max_hole", 2), off_plateau)
        sensitivity = frac_sensitivity(good, centers, force_by_rank, SENSITIVITY_FRACS, thresholds.get("max_hole", 2), off_plateau)
        speed = float(np.median([f["c2_window_mean"] for f in good])) if good else math.nan
        for side in ("lower", "upper"):
            boundary_points[side] += [(b_, f_, speed) for b_, f_ in boundaries[side]["points"]]
        helmholtz_z = [f["flyback_height"] * f["beta"] / (2 * f["c2_window_mean"]) for f in good
                       if f["label"] == "helmholtz" and f["c2_window_mean"] > 0 and f["flyback_height"] > 0]
        f0s = [f["f0_hz"] for f in good if math.isfinite(f["f0_hz"])]
        report["conditions"][name] = {
            "trials_classified": len(good), "errors": len(rows) - len(good),
            "c2_peak": gsum["c2_peak_levels"], "reference_f0_hz": None if math.isnan(reference_f0) else round(reference_f0, 2),
            "f0_hz": trial_audit._describe(f0s, 2), "classes": counts,
            "classes_among_off_plateau": {k: v for k, v in off_total.items() if v},
            "classes_among_nonpositive_force": {k: v for k, v in low_total.items() if v},
            "boundaries": boundaries, "flyback_frac_sensitivity": sensitivity,
            "helmholtz_impedance_kg_s": trial_audit._describe(helmholtz_z, 4) if helmholtz_z else {}}
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
                                 "rank": f["force_rank"], "periodicity": f["periodicity"],
                                 "flybacks": f["flybacks_per_period"], "r2": f["second_flyback_ratio"], "f0": f["f0_hz"]})

    fields = ["condition", "trial", "label", "beta", "beta_level", "force_rank", "c1_window_mean",
              "window_inside_plateau", "c2_window_mean", "amplitude_std", "periodicity", "f0_hz", *FLYBACK_KEYS]
    with (run_dir / "regimes.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for f in sorted(table, key=lambda f: (f["condition"], f["trial"])):
            writer.writerow({k: (repr(float(v)) if isinstance(v, (float, np.floating)) else v)
                             for k, v in f.items() if k in fields})
    report["schelleng_law"] = {side: {**schelleng_law(boundary_points[side]), "theory_speed": 1,
                                      "theory_beta": -2 if side == "lower" else -1} for side in ("lower", "upper")}
    report["flyback_law"] = flyback_law([(f["flyback_height"], f["c2_window_mean"], f["beta"])
                                         for f in table if f["label"] == "helmholtz"])
    if plot:
        for e in examples:
            start, end = windows[(e["condition"], e["trial"])]
            window = load_window(directories[e["condition"]], e["trial"], start, end, cached)[:, 2]
            span = int(4 * RATE_HZ / e["f0"]) if math.isfinite(e["f0"]) and e["f0"] > 0 else 2000
            e["signal"] = window[:span]
            e["period"] = RATE_HZ / e["f0"] if math.isfinite(e["f0"]) and e["f0"] > 0 else math.nan
            shape = flyback_structure(window, e["period"], applied["flyback_frac"], applied["hysteresis_sigma"])
            e.update(cycle=shape["cycle"], phases=shape["flyback_phases"], sign=shape["flyback_sign"])
        plot_maps(plot_data, run_dir / "regime_map.png")
        plot_examples(examples, run_dir / "examples.png")
        plot_features(table, applied, run_dir / "features.png")
        plot_flyback_law(table, report["flyback_law"], run_dir / "flyback_law.png")
        report["plots"] = [str(run_dir / name) for name in ("regime_map.png", "examples.png", "features.png", "flyback_law.png")]
    (run_dir / "summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    for line in format_report(report):
        progress(line)
    return 0, run_dir


def format_report(report: dict) -> list[str]:
    t = report["thresholds_provisional"]
    lines = [f"REGIME MAP (provisional labels; thresholds {t})"]
    floor = report["noise_floor"]
    lines.append(f"noise floor: min_amplitude {floor['min_amplitude']:.4g} from {floor['source']} "
                 f"(controls {floor['controls']}, median {floor.get('control_median')}, p95 {floor.get('control_p95')}, "
                 f"factor {floor['factor']})")
    for name, c in report["conditions"].items():
        lines.append(f"[{name}] c2 peak {c['c2_peak']} | {c['trials_classified']} classified, errors {c['errors']} | "
                     f"reference f0 {c['reference_f0_hz']} Hz (f0 p25..p75 {c['f0_hz'].get('p25')}..{c['f0_hz'].get('p75')})")
        lines.append("  classes: " + " | ".join(f"{k} {v}" for k, v in c["classes"].items()))
        lo, up = c["boundaries"]["lower"], c["boundaries"]["upper"]
        lines.append(f"  lower boundary slope {lo['slope']}±{lo['slope_se']} (theory -2, n={lo['n']}, beta span x{lo['beta_span']}, "
                     f"censored {lo['censored_levels']}) | upper slope {up['slope']}±{up['slope_se']} (theory -1, n={up['n']}, "
                     f"beta span x{up['beta_span']}, censored {up['censored_levels']}) | "
                     f"Helmholtz in {c['boundaries']['levels_with_helmholtz']}/{c['boundaries']['levels_total']} beta levels, "
                     f"{c['boundaries']['non_helmholtz_cells_bridged']} interior cells bridged, censored next to off-plateau "
                     f"{c['boundaries']['censored_next_to_excluded']}")
        lines.append(f"  labels of off-plateau windows {c['classes_among_off_plateau']} | "
                     f"of column-1<=0 windows {c['classes_among_nonpositive_force']}")
        lines.append("  flyback-frac sensitivity: " + " | ".join(
            f"{s['flyback_frac']}: H {s['helmholtz']}, lower {s['lower_slope']}±{s['lower_slope_se']} (n {s['lower_n']}), "
            f"upper {s['upper_slope']}±{s['upper_slope_se']} (n {s['upper_n']})" for s in c["flyback_frac_sensitivity"]))
        z = c["helmholtz_impedance_kg_s"]
        lines.append(f"  Helmholtz H*beta/(2 v_b): median {z.get('median')} (p25 {z.get('p25')}, p75 {z.get('p75')}) kg/s")
    for side, fit in report.get("schelleng_law", {}).items():
        lines.append(f"SCHELLENG LAW {side} boundary (uncensored levels pooled over folders): n {fit.get('n')} | speed exponent "
                     f"{fit.get('exponent_speed')}±{fit.get('exponent_speed_se')} (theory +1) | beta exponent "
                     f"{fit.get('exponent_beta')}±{fit.get('exponent_beta_se')} (theory {fit['theory_beta']}) | "
                     f"rms log residual {fit.get('rms_log_residual')}")
    law = report.get("flyback_law", {})
    lines.append(f"FLYBACK LAW (Helmholtz-labelled, all folders): n {law.get('n')} | speed exponent "
                 f"{law.get('exponent_speed')}±{law.get('exponent_speed_se')} (theory +1) | beta exponent "
                 f"{law.get('exponent_beta')}±{law.get('exponent_beta_se')} (theory -1) | rms log residual "
                 f"{law.get('rms_log_residual')} | Z {law.get('impedance_kg_s')} kg/s (reference {REFERENCE_IMPEDANCE}, other paper)")
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
    parser.add_argument("--flyback-frac", type=float, default=DEFAULT_THRESHOLDS["flyback_frac"],
                        help="a flyback counts if at least this fraction of the largest in the mean cycle")
    parser.add_argument("--hysteresis-sigma", type=float, default=DEFAULT_THRESHOLDS["hysteresis_sigma"],
                        help="reversals under this many standard errors of the mean cycle are noise")
    parser.add_argument("--noise-floor-factor", type=float, default=DEFAULT_THRESHOLDS["noise_floor_factor"],
                        help="oscillation floor = factor x median std of column-1<=0 windows")
    parser.add_argument("--min-amplitude", type=float, default=DEFAULT_THRESHOLDS["min_amplitude"], help="floor fallback and lower bound")
    parser.add_argument("--min-periodicity", type=float, default=DEFAULT_THRESHOLDS["min_periodicity"])
    parser.add_argument("--max-aperiodic", type=float, default=DEFAULT_THRESHOLDS["max_aperiodic"])
    parser.add_argument("--subharmonic-ratio", type=float, default=DEFAULT_THRESHOLDS["subharmonic_ratio"])
    parser.add_argument("--max-hole", type=int, default=DEFAULT_THRESHOLDS["max_hole"], help="non-Helmholtz ranks bridged inside a region")
    args = parser.parse_args(argv)
    thresholds = {"flyback_frac": args.flyback_frac, "hysteresis_sigma": args.hysteresis_sigma,
                  "noise_floor_factor": args.noise_floor_factor, "min_amplitude": args.min_amplitude, "min_periodicity": args.min_periodicity,
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
