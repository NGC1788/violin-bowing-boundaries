#!/usr/bin/env python3
"""Audit every extracted trial of one archive: files, windows, and channel ranges.

Each ``whole_N.csv`` is read once. The result is a per-trial table and a
per-condition summary. Column meanings follow the published description
(bow force, bow velocity, bridge force, nut force) and are NOT verified here;
the report only records observations that are consistent or inconsistent with
that description. This is a structural data audit, not a regime label, a
physical calibration, or a research result.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import csv
from datetime import datetime, timezone
import io
import json
import math
import os
from pathlib import Path
import re
import sys
import uuid

import numpy as np

try:  # Locked on the server; optional for minimal local test environments.
    import pyarrow as pa
    import pyarrow.csv as pa_csv
except ImportError:  # pragma: no cover - exercised only without pyarrow
    pa = None
    pa_csv = None


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = PROJECT_ROOT / "data/interim/typeA_sample1"
DEFAULT_REPORTS = PROJECT_ROOT / "reports/trial_audit"
N_COLUMNS = 4
PUBLISHED_COLUMNS = ("bow force (N)", "bow velocity (m/s)", "bridge force (N)", "nut force (N)")
PUBLISHED_RATE_HZ = 50_000
MAX_WHOLE_BYTES = 256 * 1024**2
MAX_SMALL_BYTES = 4096
STEP_FLOOR = 1e-9
MAX_LISTED_FAILURES = 20
TRIAL_FILE = re.compile(r"(whole|beta|timestamp)_(\d+)\.csv")


class TrialAuditError(ValueError):
    """A trial could not be read or interpreted reliably."""


# ---------------------------------------------------------------- parsing

def _read_bounded(path: Path, limit: int) -> bytes:
    if path.is_symlink():
        raise TrialAuditError(f"symlink not accepted: {path.name}")
    size = path.stat().st_size
    if size > limit:
        raise TrialAuditError(f"{path.name} exceeds {limit} bytes")
    return path.read_bytes()


def parse_signal(data: bytes, parser: str = "auto") -> np.ndarray:
    """Headerless numeric CSV (LF or CRLF) -> float64 matrix with exactly N_COLUMNS columns."""
    if not data.strip():
        raise TrialAuditError("empty signal file")
    use_arrow = parser == "pyarrow" or (parser == "auto" and pa_csv is not None)
    try:
        if use_arrow:
            if pa_csv is None:
                raise TrialAuditError("pyarrow parser requested but not installed")
            table = pa_csv.read_csv(
                pa.py_buffer(data),
                read_options=pa_csv.ReadOptions(autogenerate_column_names=True, use_threads=False),
                convert_options=pa_csv.ConvertOptions(
                    column_types={f"f{i}": pa.float64() for i in range(N_COLUMNS)}),
            )
            if table.num_columns != N_COLUMNS:
                raise TrialAuditError(f"expected {N_COLUMNS} columns, found {table.num_columns}")
            matrix = np.column_stack([table.column(i).to_numpy() for i in range(N_COLUMNS)])
        else:
            matrix = np.loadtxt(io.BytesIO(data.replace(b"\r\n", b"\n")), delimiter=",",
                                dtype=np.float64, ndmin=2)
    except TrialAuditError:
        raise
    except Exception as error:  # parser-specific exception types
        raise TrialAuditError(f"signal parse error: {error}") from error
    if matrix.ndim != 2 or matrix.shape[1] != N_COLUMNS:
        raise TrialAuditError(f"expected {N_COLUMNS} columns, found shape {matrix.shape}")
    return np.ascontiguousarray(matrix, dtype=np.float64)


def _single_row_tokens(data: bytes, what: str) -> list[str]:
    lines = [line for line in data.decode("utf-8-sig").splitlines() if line.strip()]
    if len(lines) != 1:
        raise TrialAuditError(f"{what}: expected one row, found {len(lines)}")
    return [token.strip() for token in lines[0].split(",")]


def parse_beta(data: bytes) -> float:
    tokens = _single_row_tokens(data, "beta")
    if len(tokens) != 1:
        raise TrialAuditError(f"beta: expected one value, found {len(tokens)}")
    try:
        value = float(tokens[0])
    except ValueError as error:
        raise TrialAuditError(f"beta: not a number: {tokens[0]!r}") from error
    if not math.isfinite(value):
        raise TrialAuditError("beta: non-finite")
    return value


def parse_window(data: bytes) -> tuple[int, int]:
    tokens = _single_row_tokens(data, "timestamp")
    if len(tokens) != 2 or not all(re.fullmatch(r"[0-9]+", t) for t in tokens):
        raise TrialAuditError(f"timestamp: expected two non-negative integers, found {tokens!r}")
    start, end = int(tokens[0]), int(tokens[1])
    if end <= start:
        raise TrialAuditError(f"timestamp: end {end} not after start {start}")
    return start, end


# ---------------------------------------------------------------- statistics

def longest_true_run(mask: np.ndarray) -> tuple[int, int]:
    """Inclusive (start, end) of the longest contiguous True run, or (-1, -1)."""
    if mask.size == 0 or not mask.any():
        return -1, -1
    padded = np.concatenate(([False], mask.astype(bool), [False]))
    edges = np.flatnonzero(padded[1:] != padded[:-1])
    starts, stops = edges[0::2], edges[1::2]
    k = int(np.argmax(stops - starts))
    return int(starts[k]), int(stops[k] - 1)


def min_positive_step(values: np.ndarray) -> float:
    """Smallest gap between distinct finite values above STEP_FLOOR. A quantization hint only."""
    unique = np.unique(values[np.isfinite(values)])
    if unique.size < 2:
        return math.nan
    gaps = np.diff(unique)
    gaps = gaps[gaps > STEP_FLOOR]
    return float(gaps.min()) if gaps.size else math.nan


def analyse_trial(signal: np.ndarray, start: int, end: int, plateau_tol: float) -> dict:
    rows = int(signal.shape[0])
    non_finite = int((~np.isfinite(signal)).sum())
    if non_finite:
        raise TrialAuditError(f"{non_finite} non-finite values")
    if not (0 <= start < end <= rows - 1):
        raise TrialAuditError(f"window [{start}, {end}] outside rows 0..{rows - 1}")
    window = signal[start:end + 1]
    result: dict = {"rows": rows, "window_start": start, "window_end": end, "window_width": end - start}
    for j in range(N_COLUMNS):
        column, part, name = signal[:, j], window[:, j], f"c{j + 1}"
        result[f"{name}_min"] = float(column.min())
        result[f"{name}_max"] = float(column.max())
        result[f"{name}_window_mean"] = float(part.mean())
        result[f"{name}_window_std"] = float(part.std())
        result[f"{name}_window_min"] = float(part.min())
        result[f"{name}_window_max"] = float(part.max())
    velocity = signal[:, 1]
    peak = float(velocity.max())
    if peak > 0:
        steady = velocity >= (1.0 - plateau_tol) * peak
        run_start, run_end = longest_true_run(steady)
        result["c2_plateau_start"] = run_start
        result["c2_plateau_end"] = run_end
        result["c2_window_steady_fraction"] = float(steady[start:end + 1].mean())
        result["window_inside_plateau"] = bool(run_start <= start and end <= run_end)
    else:
        result["c2_plateau_start"] = result["c2_plateau_end"] = -1
        result["c2_window_steady_fraction"] = math.nan
        result["window_inside_plateau"] = False
    for j in (2, 3):
        result[f"c{j + 1}_window_min_step"] = min_positive_step(window[:, j])
        result[f"c{j + 1}_window_unique"] = int(np.unique(window[:, j]).size)
    return result


def audit_one(task: tuple) -> dict:
    """Audit one trial. Never raises; errors are returned in the row."""
    condition, directory, number, plateau_tol, parser = task
    row: dict = {"condition": condition, "trial": number, "error": ""}
    try:
        base = Path(directory)
        signal = parse_signal(_read_bounded(base / f"whole_{number}.csv", MAX_WHOLE_BYTES), parser)
        row["beta"] = parse_beta(_read_bounded(base / f"beta_{number}.csv", MAX_SMALL_BYTES))
        start, end = parse_window(_read_bounded(base / f"timestamp_{number}.csv", MAX_SMALL_BYTES))
        row.update(analyse_trial(signal, start, end, plateau_tol))
    except (OSError, TrialAuditError) as error:
        row["error"] = str(error) or error.__class__.__name__
    return row


# ---------------------------------------------------------------- discovery and summary

def discover(root: Path) -> dict[Path, dict[str, set[int]]]:
    """Map each directory containing trial files to the indices of whole/beta/timestamp files."""
    found: dict[Path, dict[str, set[int]]] = {}
    for path in root.rglob("*.csv"):
        match = TRIAL_FILE.fullmatch(path.name)
        if not match:
            continue
        kinds = found.setdefault(path.parent, {"whole": set(), "beta": set(), "timestamp": set()})
        kinds[match.group(1)].add(int(match.group(2)))
    return {directory: found[directory] for directory in sorted(found) if found[directory]["whole"]}


def _describe(values: list[float], digits: int = 6) -> dict:
    finite = np.asarray([v for v in values if isinstance(v, (int, float)) and math.isfinite(v)], dtype=float)
    if finite.size == 0:
        return {"n": 0}
    quantiles = np.quantile(finite, [0.0, 0.25, 0.5, 0.75, 1.0])
    return {"n": int(finite.size), **{key: round(float(q), digits)
            for key, q in zip(("min", "p25", "median", "p75", "max"), quantiles)}}


def summarise_condition(condition: str, directory: Path, kinds: dict[str, set[int]],
                        rows: list[dict], root: Path) -> dict:
    good = [r for r in rows if not r["error"]]
    failed = [r for r in rows if r["error"]]
    orphans = sorted((kinds["beta"] | kinds["timestamp"]) - kinds["whole"])
    width_counts: dict[str, int] = {}
    for r in good:
        width_counts[str(r["window_width"])] = width_counts.get(str(r["window_width"]), 0) + 1
    betas = sorted({round(r["beta"], 3) for r in good})
    speed_peaks = [r["c2_max"] for r in good]
    return {
        "directory": str(directory.relative_to(root)) if directory.is_relative_to(root) else str(directory),
        "trials_found": len(kinds["whole"]),
        "trials_audited": len(rows),
        "failed": len(failed),
        "failures": [{"trial": r["trial"], "error": r["error"]} for r in failed[:MAX_LISTED_FAILURES]],
        "orphan_companion_indices": orphans[:MAX_LISTED_FAILURES],
        "orphan_companion_count": len(orphans),
        "rows": _describe([r["rows"] for r in good], 0),
        "duration_s_if_published_rate": _describe([(r["rows"] - 1) / PUBLISHED_RATE_HZ for r in good], 4),
        "beta": {**_describe([r["beta"] for r in good], 4), "distinct_at_1e-3": len(betas),
                 "distinct_values_at_1e-3": betas},
        "window": {"width_counts": dict(sorted(width_counts.items())),
                   "start": _describe([r["window_start"] for r in good], 0),
                   "inside_c2_plateau": sum(bool(r["window_inside_plateau"]) for r in good),
                   "c2_steady_fraction": _describe([r["c2_window_steady_fraction"] for r in good], 4)},
        "c2_peak": _describe(speed_peaks, 4),
        "c2_peak_levels_at_1e-3": sorted({round(v, 3) for v in speed_peaks}),
        "c2_window_mean": _describe([r["c2_window_mean"] for r in good], 4),
        "c1_window_mean": {**_describe([r["c1_window_mean"] for r in good], 4),
                           "distinct_at_0.05": len({round(r["c1_window_mean"] / 0.05) for r in good})},
        "c1_window_std": _describe([r["c1_window_std"] for r in good], 4),
        "c3_window_std": _describe([r["c3_window_std"] for r in good], 4),
        "c4_window_min_step": _describe([r["c4_window_min_step"] for r in good], 7),
        "c4_window_unique": _describe([r["c4_window_unique"] for r in good], 0),
    }


def run_audit(root: Path, reports_dir: Path, limit: int | None, workers: int,
              plateau_tol: float, parser: str = "auto", progress=print) -> tuple[int, Path]:
    if not root.is_dir():
        raise TrialAuditError(f"extraction root not found: {root}")
    conditions = discover(root)
    if not conditions:
        raise TrialAuditError(f"no whole_N.csv files under {root}")
    tasks, plan = [], []
    for directory, kinds in conditions.items():
        numbers = sorted(kinds["whole"])[:limit] if limit else sorted(kinds["whole"])
        plan.append((directory, kinds, numbers))
        tasks += [(directory.name, str(directory), n, plateau_tol, parser) for n in numbers]
    progress(f"Auditing {len(tasks)} trials in {len(conditions)} condition folders "
             f"({'pyarrow' if parser != 'numpy' and pa_csv is not None else 'numpy'} parser, {workers} workers).")
    results: list[dict] = []
    if workers <= 1:
        iterator = map(audit_one, tasks)
        executor = None
    else:
        executor = ProcessPoolExecutor(max_workers=workers)
        iterator = executor.map(audit_one, tasks, chunksize=8)
    try:
        for i, row in enumerate(iterator, 1):
            results.append(row)
            if i % 500 == 0 or i == len(tasks):
                progress(f"  audited {i}/{len(tasks)}")
    finally:
        if executor is not None:
            executor.shutdown()

    by_condition: dict[str, list[dict]] = {}
    for row in results:
        by_condition.setdefault(row["condition"], []).append(row)
    summaries, warnings = [], []
    for directory, kinds, _numbers in plan:
        summary = summarise_condition(directory.name, directory, kinds, by_condition.get(directory.name, []), root)
        summaries.append(summary)
        good = summary["trials_audited"] - summary["failed"]
        outside = good - summary["window"]["inside_c2_plateau"]
        if outside:
            warnings.append(f"{directory.name}: {outside}/{good} windows are not fully inside the c2 plateau "
                            f"(tolerance {plateau_tol:.0%}); do not assume constant velocity in the window")
        if len(summary["c2_peak_levels_at_1e-3"]) > 1:
            warnings.append(f"{directory.name}: c2 peak differs across trials {summary['c2_peak_levels_at_1e-3']}")

    failed = sum(s["failed"] for s in summaries)
    orphans = sum(s["orphan_companion_count"] for s in summaries)
    status = "PASS" if failed == 0 and orphans == 0 else "FAIL"
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = reports_dir / f"{stamp}_{uuid.uuid4().hex[:8]}"
    run_dir.mkdir(parents=True, exist_ok=False)
    fieldnames: list[str] = []
    for row in results:
        fieldnames += [key for key in row if key not in fieldnames]
    with (run_dir / "trials.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, restval="")
        writer.writeheader()
        for row in sorted(results, key=lambda r: (r["condition"], r["trial"])):
            writer.writerow({k: (repr(v) if isinstance(v, float) else v) for k, v in row.items()})
    report = {
        "status": status,
        "created_utc": stamp,
        "root": str(root),
        "limit_per_condition": limit,
        "plateau_tolerance": plateau_tol,
        "parser": "pyarrow" if parser != "numpy" and pa_csv is not None else "numpy",
        "published_column_order_unverified": list(PUBLISHED_COLUMNS),
        "published_rate_hz_unverified": PUBLISHED_RATE_HZ,
        "scope": ("Structure, companion files, window bounds and channel ranges only. "
                  "No regime labels, units, calibration or research performance are established."),
        "trials_audited": len(results),
        "trials_failed": failed,
        "conditions": summaries,
        "warnings": warnings,
    }
    (run_dir / "summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    (reports_dir / "latest.json").write_text(json.dumps(
        {"status": status, "run_dir": str(run_dir), "created_utc": stamp}, indent=2), encoding="utf-8")
    progress(f"TRIAL AUDIT: {status}  ({len(results) - failed}/{len(results)} trials readable, "
             f"{orphans} orphan companion files)")
    progress(f"Summary: {run_dir / 'summary.json'}")
    return (0 if status == "PASS" else 1), run_dir


def show_latest(reports_dir: Path) -> int:
    """Compact text view of the latest run, sized for copying out of a remote desktop."""
    try:
        latest = json.loads((reports_dir / "latest.json").read_text(encoding="utf-8"))
        report = json.loads((Path(latest["run_dir"]) / "summary.json").read_text(encoding="utf-8"))
    except (OSError, ValueError, KeyError) as error:
        print("Could not read the latest trial audit:", error, file=sys.stderr)
        return 1
    print(f"TRIAL AUDIT: {report['status']}  {report['trials_audited'] - report['trials_failed']}/"
          f"{report['trials_audited']} trials, parser {report['parser']}, {report['created_utc']}"
          + (f", limit {report['limit_per_condition']}/folder" if report["limit_per_condition"] else ""))
    for c in report["conditions"]:
        b, w, r = c["beta"], c["window"], c["rows"]
        print(f"[{Path(c['directory']).name}] trials {c['trials_audited']}/{c['trials_found']} | "
              f"failed {c['failed']} | orphans {c['orphan_companion_count']}")
        if c["failed"]:
            print("  first failures:", c["failures"][:3])
        if r.get("n"):
            d = c["duration_s_if_published_rate"]
            print(f"  rows {r['min']:.0f}..{r['max']:.0f} (={d['min']}..{d['max']} s at published rate)")
            print(f"  beta {b['min']}..{b['max']}, distinct(1e-3) {b['distinct_at_1e-3']}")
            print(f"  window widths {w['width_counts']} | start {w['start']['min']:.0f}..{w['start']['max']:.0f}"
                  f" | inside c2 plateau {w['inside_c2_plateau']} | steady fraction min {w['c2_steady_fraction']['min']}")
            print(f"  c2 peak levels {c['c2_peak_levels_at_1e-3']} | window mean {c['c2_window_mean']['min']}"
                  f"..{c['c2_window_mean']['max']}")
            m = c["c1_window_mean"]
            print(f"  c1 window mean {m['min']}..{m['max']} (median {m['median']}), distinct(0.05) {m['distinct_at_0.05']}"
                  f" | c1 window std median {c['c1_window_std']['median']}")
            print(f"  c3 window std median {c['c3_window_std']['median']} | c4 min step median "
                  f"{c['c4_window_min_step']['median']}, unique median {c['c4_window_unique']['median']:.0f}")
    for warning in report["warnings"]:
        print("WARNING:", warning)
    return 0 if report["status"] == "PASS" else 1


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT, help="Extracted archive directory")
    parser.add_argument("--reports-dir", type=Path, default=DEFAULT_REPORTS)
    parser.add_argument("--limit", type=int, default=None, help="Audit only the first N trials per folder")
    parser.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    parser.add_argument("--plateau-tol", type=float, default=0.01,
                        help="Relative tolerance below the trial's peak c2 that still counts as plateau")
    parser.add_argument("--parser", choices=("auto", "pyarrow", "numpy"), default="auto")
    parser.add_argument("--show", action="store_true", help="Print the latest saved audit; reads no trial data")
    args = parser.parse_args(argv)
    if args.show:
        return show_latest(args.reports_dir)
    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be positive")
    if args.workers <= 0:
        parser.error("--workers must be positive")
    if not (0 < args.plateau_tol < 1):
        parser.error("--plateau-tol must be between 0 and 1")
    try:
        code, _ = run_audit(args.root.resolve(), args.reports_dir, args.limit, args.workers,
                            args.plateau_tol, args.parser)
        return code
    except TrialAuditError as error:
        print("TRIAL AUDIT: FAIL —", error, file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\nInterrupted. Trial data were only read; nothing was modified.", file=sys.stderr)
        return 130


if __name__ == "__main__":
    sys.exit(main())
