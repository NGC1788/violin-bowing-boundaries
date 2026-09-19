"""Collect the small tables and example waveforms a presentation needs.

The regime-map runs keep their per-trial table (``regimes.csv``) and report
(``summary.json``) next to the plots. Figures for a talk need to be redrawn
from those numbers, not cropped out of the diagnostic plots, so this copies
them together with a few raw bridge-force windows into one directory and
prints its size. Nothing is deleted and no analysis is repeated.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
import tarfile
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REPORTS = PROJECT_ROOT / "reports"
DEFAULT_OUT = Path.home() / "deck_export"
PLOTS = ("regime_map.png", "examples.png", "features.png", "flyback_law.png")
# One window is 0.4 s at 50 kHz; a few periods are enough to show the shape.
EXAMPLE_PERIODS = 6
EXAMPLES_PER_LABEL = 2


def progress(message: str) -> None:
    print(message, flush=True)


def newest_runs(reports: Path) -> dict[str, Path]:
    """The most recent regime-map run directory for each grid."""
    runs: dict[str, Path] = {}
    for table in sorted(reports.rglob("regimes.csv")):
        run = table.parent
        # .../<grid>/regime_map/<stamp>_<uuid>/regimes.csv -> grid, else the run's own name.
        parts = run.parts
        grid = parts[-3] if len(parts) >= 3 and parts[-2] == "regime_map" else run.name
        if grid not in runs or run.name > runs[grid].name:
            runs[grid] = run
    return runs


def read_table(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def choose_examples(rows: list[dict]) -> list[dict]:
    """A couple of clear trials per regime label, spread over the label's trials."""
    by_label: dict[str, list[dict]] = {}
    for row in rows:
        label = row.get("label") or ""
        if label and label != "ambiguous":
            by_label.setdefault(label, []).append(row)
    chosen: list[dict] = []
    for label, members in sorted(by_label.items()):
        members.sort(key=lambda r: int(r["trial"]))
        picks = np.linspace(0, len(members) - 1, min(EXAMPLES_PER_LABEL, len(members)))
        chosen += [members[int(index)] for index in picks]
    return chosen


def window_lookup(trials_path: Path) -> dict[tuple[str, int], tuple[int, int]]:
    lookup: dict[tuple[str, int], tuple[int, int]] = {}
    for raw in read_table(trials_path):
        if not raw.get("error"):
            lookup[(raw["condition"], int(raw["trial"]))] = (int(raw["window_start"]), int(raw["window_end"]))
    return lookup


def export_examples(summary: dict, rows: list[dict], destination: Path) -> int:
    """Save the bridge-force trace of a few example trials. Returns how many were saved."""
    trials_path = Path(summary.get("source_trials", ""))
    if not trials_path.is_file():
        progress(f"  examples skipped: trials table not found ({trials_path})")
        return 0
    audit = json.loads((trials_path.parent / "summary.json").read_text(encoding="utf-8"))
    cached = bool(audit.get("window_cache"))
    root = Path(audit["window_cache"] if cached else audit["root"])
    windows = window_lookup(trials_path)

    arrays: dict[str, np.ndarray] = {}
    meta: list[dict] = []
    for row in choose_examples(rows):
        condition, trial = row["condition"], int(row["trial"])
        key = (condition, trial)
        if key not in windows:
            continue
        start, end = windows[key]
        source = root / condition / f"window_{trial}.npy"
        if not source.is_file():
            progress(f"  examples skipped: window cache missing ({source})")
            return len(arrays)
        window = np.load(source, allow_pickle=False)
        if window.ndim != 2 or window.shape[0] != end - start + 1:
            progress(f"  examples skipped: window {trial} has shape {window.shape}")
            continue
        f0 = float(row.get("f0_hz") or "nan")
        rate = float(summary.get("rate_hz") or 50000.0)
        span = int(EXAMPLE_PERIODS * rate / f0) if np.isfinite(f0) and f0 > 0 else 2000
        name = f"{condition}__{trial}"
        arrays[name] = np.asarray(window[:span, 2], dtype=np.float32)
        meta.append({"name": name, "condition": condition, "trial": trial, "label": row["label"],
                     "beta": row.get("beta"), "force_rank": row.get("force_rank"),
                     "c1_window_mean": row.get("c1_window_mean"), "c2_window_mean": row.get("c2_window_mean"),
                     "f0_hz": row.get("f0_hz"), "periodicity": row.get("periodicity"),
                     "flybacks_per_period": row.get("flybacks_per_period"), "rate_hz": rate})
    if arrays:
        np.savez_compressed(destination / "examples.npz", **arrays)
        (destination / "examples.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return len(arrays)


def export_run(grid: str, run: Path, out: Path) -> dict:
    destination = out / grid
    destination.mkdir(parents=True, exist_ok=True)
    progress(f"- {grid}  <- {run}")
    shutil.copy2(run / "regimes.csv", destination / "regimes.csv")
    summary_path = run / "summary.json"
    summary: dict = {}
    if summary_path.is_file():
        shutil.copy2(summary_path, destination / "summary.json")
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    for plot in PLOTS:
        if (run / plot).is_file():
            shutil.copy2(run / plot, destination / plot)
    rows = read_table(run / "regimes.csv")
    saved = export_examples(summary, rows, destination) if summary else 0
    progress(f"  trials {len(rows)}, example windows {saved}")
    return {"grid": grid, "run": str(run), "trials": len(rows), "examples": saved}


def export_comparisons(reports: Path, out: Path) -> int:
    """Simulation-versus-measurement reports: agreement, Helmholtz IoU, confusion, slopes."""
    destination = out / "comparison"
    found = 0
    for path in sorted(reports.rglob("comparison.json")):
        destination.mkdir(parents=True, exist_ok=True)
        # .../simulation/<diagram>/<run>/comparison.json
        name = f"{path.parent.parent.name}__{path.parent.name}.json"
        shutil.copy2(path, destination / name)
        found += 1
    progress(f"- comparison reports {found}")
    return found


def export_calibration(reports: Path, out: Path) -> int:
    """The calibration search trace, one evaluation per line."""
    source = reports / "calibration"
    if not source.is_dir():
        progress("- calibration trace not found")
        return 0
    destination = out / "calibration"
    destination.mkdir(parents=True, exist_ok=True)
    lines = 0
    for path in sorted(source.glob("*.jsonl")):
        shutil.copy2(path, destination / path.name)
        lines += sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip())
    progress(f"- calibration evaluations {lines}")
    return lines


def directory_size(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reports", type=Path, default=DEFAULT_REPORTS, help="reports directory to search")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT, help="directory to write (created if absent)")
    parser.add_argument("--archive", action="store_true", help="also write <out>.tar.gz")
    options = parser.parse_args(argv)

    if not options.reports.is_dir():
        progress(f"reports directory not found: {options.reports}")
        return 1
    runs = newest_runs(options.reports)
    if not runs:
        progress(f"no regimes.csv under {options.reports}")
        return 1

    options.out.mkdir(parents=True, exist_ok=True)
    manifest = [export_run(grid, run, options.out) for grid, run in sorted(runs.items())]
    comparisons = export_comparisons(options.reports, options.out)
    calibration = export_calibration(options.reports, options.out)
    (options.out / "manifest.json").write_text(
        json.dumps({"grids": manifest, "comparison_reports": comparisons, "calibration_evaluations": calibration},
                   ensure_ascii=False, indent=2), encoding="utf-8")

    if options.archive:
        archive = options.out.with_suffix(".tar.gz")
        with tarfile.open(archive, "w:gz") as tar:
            tar.add(options.out, arcname=options.out.name)
        progress(f"archive {archive} ({archive.stat().st_size / 1e6:.1f} MB)")

    progress(f"EXPORT DONE: {len(manifest)} grids, {directory_size(options.out) / 1e6:.1f} MB in {options.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
