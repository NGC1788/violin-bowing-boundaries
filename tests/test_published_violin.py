"""External benchmark contracts: missing labels, coordinates, blocks and audit."""
import csv
import importlib.util
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
HERE = ROOT / "research/2026-10-13/schoonderwaldt2008"
spec = importlib.util.spec_from_file_location("published_benchmark", HERE / "benchmark_published.py")
benchmark = importlib.util.module_from_spec(spec)
spec.loader.exec_module(benchmark)


def test_missing_and_unanalyzed_cells_excluded():
    all_rows, selected, X, y, beta, ids = benchmark.load_cells(HERE / "published_cells.csv")
    assert len(all_rows) == 6*11*24
    assert len(selected) == 1041
    assert len(ids) == len(set(ids))
    assert set(r["string"] for r in selected) == {"D4"}
    assert all(r["published_category"] not in ("unplotted_unknown", "helmholtz_not_systematically_analyzed") for r in selected)
    assert sum(r["strict_systematically_analyzed"] == "True" for r in all_rows) == 1171
    assert (y.sum(), (~y).sum()) == (664, 377)


def test_manual_marker_audit_agrees_with_extracted_cells():
    all_rows, *_ = benchmark.load_cells(HERE / "published_cells.csv")
    index = {(r["figure"], r["panel"], r["force_rank_0based"], r["beta_column_0based"]): r for r in all_rows}
    with (HERE / "manual_checks.csv").open(newline="") as handle:
        checks = list(csv.DictReader(handle))
    assert len(checks) == 24
    for check in checks:
        key = tuple(check[k] for k in ("figure", "panel", "force_rank_0based", "beta_column_0based"))
        assert index[key]["published_category"] == check["visual_category"]


def test_source_folds_are_disjoint_complete_contiguous_beta_blocks():
    _, _, X, y, beta, ids = benchmark.load_cells(HERE / "published_cells.csv")
    for speed in benchmark.SOURCE_SPEEDS:
        levels = beta[X[:,2] == speed]
        held = []
        for train, valid in benchmark.blocked_folds(levels):
            assert set(levels[train]).isdisjoint(levels[valid])
            columns = np.unique(levels[valid])
            assert np.all(np.diff(columns) == 1)
            held.extend(valid.tolist())
        assert sorted(held) == list(range(len(levels)))


def test_static_and_uniform_have_identical_source_coordinates():
    for speed in benchmark.SOURCE_SPEEDS:
        X = np.array([[.03, .05, speed], [.15, 3., speed]])
        np.testing.assert_array_equal(benchmark.coordinates(X, 0., speed), benchmark.coordinates(X, 1., speed))
        X[:,2] *= 2
        delta = benchmark.coordinates(X, 1., speed) - benchmark.coordinates(X, 0., speed)
        np.testing.assert_allclose(delta[:,0], 0.)
        np.testing.assert_allclose(delta[:,1], -np.log(2))


def test_unknown_gap_is_not_filled_or_given_a_transition_bracket():
    all_rows, *_ = benchmark.load_cells(HERE / "published_cells.csv")
    changed = [dict(row) for row in all_rows]
    for row in changed:
        if row["string"] == "D4" and float(row["bow_speed_nominal_m_s"]) == .1 and row["beta_column_0based"] == "0":
            rank = int(row["force_rank_0based"])
            row["binary_helmholtz"] = "1" if rank in (5, 7) else "0"
            row["strict_systematically_analyzed"] = "False" if rank == 6 else "True"
    record = next(r for r in benchmark.audit_grid(changed) if r["speed"] == .1 and r["beta_column"] == 0)
    assert record["positive_runs_without_gap_filling"] == [[5,5], [7,7]]
    assert record["unknown_force_ranks"] == [6]
    assert record["positive_adjacent_to_unknown"]
    assert len(record["lower_adjacent_transition_brackets_N"]) == 1
    assert len(record["upper_adjacent_transition_brackets_N"]) == 1
