"""Grid reconstruction against a planted (beta, force) grid with known problem cells."""

import contextlib
import csv
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest

try:
    import numpy as np
except ImportError:  # pragma: no cover
    np = None

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "grid_summary.py"
grid = None
if np is not None:
    SPEC = importlib.util.spec_from_file_location("grid_summary_under_test", SCRIPT)
    grid = importlib.util.module_from_spec(SPEC)
    sys.modules[SPEC.name] = grid
    SPEC.loader.exec_module(grid)
try:
    import matplotlib  # noqa: F401
    HAVE_MPL = True
except ImportError:  # pragma: no cover
    HAVE_MPL = False

BETAS_DESC = [0.2, 0.12, 0.0655, 0.02]      # 0.0655 +/- 1e-4 straddles a 3-decimal rounding boundary
FORCES_ASC = [0.1, 0.4, 1.0, 2.0, 4.0]
JITTER = [-1e-4, -5e-5, 0.0, 5e-5, 1e-4]
OFF_PLATEAU = {(0.2, 4), (0.2, 3), (0.12, 4)}  # (beta, force index ascending)
NONPOSITIVE = (0.02, 0)


def planted_rows(condition="2024-03-27_r_v1"):
    """Trial numbers sweep beta descending, and force descending within each beta level."""
    rows = []
    for block, beta in enumerate(BETAS_DESC):
        for fidx in range(len(FORCES_ASC)):
            position = len(FORCES_ASC) - 1 - fidx
            force = -0.02 if (beta, fidx) == NONPOSITIVE else FORCES_ASC[fidx]
            off = (beta, fidx) in OFF_PLATEAU
            rows.append({"condition": condition, "trial": block * 5 + position + 1, "error": "",
                         "beta": beta + JITTER[position], "c1_window_mean": force, "c2_max": 0.05,
                         "window_inside_plateau": not off, "c2_window_steady_fraction": 0.7 if off else 1.0})
    return rows


def write_table(path, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(grid.NEEDED))
        writer.writeheader()
        for row in rows:
            writer.writerow({k: (repr(v) if isinstance(v, float) else v) for k, v in row.items()})


@unittest.skipIf(np is None, "numpy not installed")
class BreakTests(unittest.TestCase):
    def test_natural_break(self):
        values = [b + j for b in BETAS_DESC for j in JITTER]
        threshold, ratio = grid.natural_break(values)
        self.assertGreater(threshold, 1e-4)
        self.assertLess(threshold, 0.0655 - 0.02 - 2e-4)
        self.assertGreater(ratio, 100)
        self.assertTrue(np.isinf(grid.natural_break([0.3, 0.3])[0]))
        self.assertEqual(grid.natural_break([0.1, 0.2])[0], 0.0)

    def test_level_straddling_rounding_boundary_is_not_split(self):
        values = [b + j for b in BETAS_DESC for j in JITTER]
        self.assertGreater(len({round(v, 3) for v in values}), len(BETAS_DESC))  # the fixture really straddles
        labels = grid.assign_levels(values, grid.natural_break(values)[0])
        self.assertEqual(len(set(labels)), len(BETAS_DESC))

    def test_evenly_spaced_values_have_no_clear_break(self):
        summary, _ = grid.analyse_condition([
            {"condition": "c", "trial": i + 1, "beta": 0.01 * (i + 1), "c1_window_mean": 1.0, "c2_max": 0.1,
             "window_inside_plateau": True, "c2_window_steady_fraction": 1.0} for i in range(12)])
        self.assertFalse(summary["beta"]["clear_break"])


@unittest.skipIf(np is None, "numpy not installed")
class ReconstructionTests(unittest.TestCase):
    def test_reconstructs_planted_grid_and_problem_cells(self):
        summary, rows = grid.analyse_condition(planted_rows())
        b, f = summary["beta"], summary["force"]
        self.assertEqual((b["levels"], b["uniform_trials_per_level"], b["clear_break"]), (4, True, True))
        self.assertEqual(b["trials_per_level"], {"min": 5, "max": 5})
        self.assertEqual((b["sweep_order_by_trial"], b["trial_numbers_contiguous_in_levels"]), ("descending", 4))
        order = f["order_follows_trial_number"]
        self.assertEqual((order["levels_descending"], order["levels_ascending"], order["median_concordance"]), (4, 0, 1.0))
        self.assertEqual(f["force_by_rank"], FORCES_ASC)
        self.assertEqual(summary["nonpositive_force_windows"], {"count": 1, "by_rank": {1: 1}})
        off = summary["off_plateau_windows"]
        self.assertEqual(off["count"], 3)
        self.assertEqual(off["by_beta_level"], {"0.2000": 2, "0.1200": 1})
        self.assertEqual(off["by_force_rank"], {5: 2, 4: 1})
        self.assertEqual(off["steady_fraction_min"], 0.7)
        heaviest = [r for r in rows if r["force_rank"] == 5 and r["beta_level"] == 3]
        self.assertEqual([r["trial"] for r in heaviest], [1])


@unittest.skipIf(np is None, "numpy not installed")
class CommandTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        base = Path(self.tmp.name)
        self.audit_reports, self.reports = base / "audit", base / "grid"
        run_dir = self.audit_reports / "run1"
        run_dir.mkdir(parents=True)
        rows = planted_rows() + planted_rows("2024-03-27_r_v3")
        failed = dict(rows[0], condition="2024-03-27_r_v3", trial=999, error="signal parse error", beta="",
                      c1_window_mean="", c2_max="", window_inside_plateau="", c2_window_steady_fraction="")
        write_table(run_dir / "trials.csv", rows + [failed])
        self.latest = self.audit_reports / "latest.json"
        self.latest.write_text(json.dumps({"status": "PASS", "run_dir": str(run_dir)}), encoding="utf-8")

    def tearDown(self):
        self.tmp.cleanup()

    def run_main(self, *extra):
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            code = grid.main(["--audit-reports", str(self.audit_reports), "--reports-dir", str(self.reports), *extra])
        return code, out.getvalue(), err.getvalue()

    def test_end_to_end_report_and_grid_table(self):
        code, text, _ = self.run_main("--no-plot")
        self.assertEqual(code, 0)
        self.assertIn("beta: 4 levels", text)
        self.assertIn("excluded failed 1", text)
        self.assertLess(len(text.splitlines()), 20)
        (run_dir,) = list(self.reports.iterdir())
        with (run_dir / "grid.csv").open(newline="", encoding="utf-8") as handle:
            table = list(csv.DictReader(handle))
        self.assertEqual(len(table), 40)
        for row in table:  # every numeric field must be plain text a later reader can parse
            for key in ("beta", "beta_level_center", "c1_window_mean", "c2_window_steady_fraction"):
                float(row[key])
            int(row["beta_level"]), int(row["force_rank"])
        self.assertEqual({row["window_inside_plateau"] for row in table}, {"True", "False"})

    @unittest.skipUnless(HAVE_MPL, "matplotlib not installed")
    def test_plot_is_written(self):
        code, text, _ = self.run_main()
        self.assertEqual(code, 0)
        (run_dir,) = list(self.reports.iterdir())
        self.assertGreater((run_dir / "grid.png").stat().st_size, 1000)
        self.assertIn("Plot:", text)

    def test_refuses_non_pass_audit(self):
        self.latest.write_text(json.dumps({"status": "FAIL", "run_dir": "x"}), encoding="utf-8")
        code, _, err = self.run_main("--no-plot")
        self.assertEqual(code, 1)
        self.assertIn("FAIL", err)

    def test_missing_columns_rejected(self):
        bad = Path(self.tmp.name) / "bad.csv"
        bad.write_text("condition,trial\nx,1\n", encoding="utf-8")
        with self.assertRaises(grid.GridError):
            grid.load_trials(bad)


@unittest.skipIf(np is None, "numpy not installed")
class PipelineTests(unittest.TestCase):
    """trial_audit.py output must be readable by grid_summary.py without any conversion."""

    def test_trial_audit_table_feeds_grid_summary(self):
        spec = importlib.util.spec_from_file_location("trial_audit_for_pipeline", SCRIPT.parent / "trial_audit.py")
        audit = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = audit
        spec.loader.exec_module(audit)
        tests_dir = Path(__file__).resolve().parent
        spec_t = importlib.util.spec_from_file_location("trial_audit_fixtures", tests_dir / "test_trial_audit.py")
        fixtures = importlib.util.module_from_spec(spec_t)
        spec_t.loader.exec_module(fixtures)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "extract" / "archive"
            number = 0
            for beta in (0.2, 0.0655, 0.02):
                for force in (0.5, 1.0, 2.0):
                    number += 1
                    fixtures.write_trial(root / "2024-03-27_r_v2", number, fixtures.make_signal(0.125, force=force),
                                         beta + (number % 3 - 1) * 1e-4, (300, 500))
            code, run_dir = audit.run_audit(root.parent, Path(tmp) / "audit", None, 1, 0.01, progress=fixtures.quiet)
            self.assertEqual(code, 0)
            by_condition, excluded = grid.load_trials(run_dir / "trials.csv")
            summary, _ = grid.analyse_condition(by_condition["2024-03-27_r_v2"])
            self.assertEqual((summary["beta"]["levels"], summary["beta"]["uniform_trials_per_level"], excluded), (3, True, {}))
            self.assertEqual(summary["force"]["force_by_rank"], [0.5, 1.0, 2.0])


if __name__ == "__main__":
    unittest.main()
