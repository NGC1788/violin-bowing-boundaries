"""Trial audit against synthetic trials with exactly known structure and values."""

import contextlib
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

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "trial_audit.py"
audit = None
if np is not None:
    SPEC = importlib.util.spec_from_file_location("trial_audit_under_test", SCRIPT)
    audit = importlib.util.module_from_spec(SPEC)
    sys.modules[SPEC.name] = audit
    SPEC.loader.exec_module(audit)

ROWS = 1000
PLATEAU = (200, 799)  # inclusive, by construction below


def make_signal(peak: float, force: float = 2.5, lsb: float = 0.125) -> "np.ndarray":
    """Ramps stay below 50% of peak, so the 1%-tolerance plateau is exactly rows 200..799."""
    i = np.arange(ROWS)
    velocity = np.where(i < 200, peak * i / 400, np.where(i < 800, peak, peak * 0.4 * (999 - i) / 199))
    bridge = 1.5 * np.sin(2 * np.pi * i / 50)
    nut = lsb * ((i % 7) - 3)
    return np.column_stack([np.full(ROWS, force), velocity, bridge, nut])


def write_trial(directory: Path, number: int, signal, beta, window, newline=b"\r\n"):
    directory.mkdir(parents=True, exist_ok=True)
    body = newline.join(",".join(repr(float(x)) for x in row).encode() for row in signal) + newline
    (directory / f"whole_{number}.csv").write_bytes(body)
    (directory / f"beta_{number}.csv").write_bytes(repr(beta).encode() + newline)
    (directory / f"timestamp_{number}.csv").write_bytes(f"{window[0]},{window[1]}".encode() + newline)


def quiet(*_args):
    return None


@unittest.skipIf(np is None, "numpy not installed")
class ParsingTests(unittest.TestCase):
    def test_crlf_and_lf_parse_identically(self):
        signal = make_signal(0.25)[:20]
        lf = b"\n".join(",".join(repr(float(x)) for x in r).encode() for r in signal) + b"\n"
        crlf = lf.replace(b"\n", b"\r\n")
        a, b = audit.parse_signal(lf, "numpy"), audit.parse_signal(crlf, "numpy")
        self.assertTrue(np.array_equal(a, b))
        self.assertTrue(np.array_equal(a, signal))

    def test_wrong_column_count_rejected(self):
        bad = b"1,2,3,4\r\n1,2,3\r\n"
        for parser in ("numpy",) + (("pyarrow",) if audit.pa_csv is not None else ()):
            with self.subTest(parser=parser), self.assertRaises(audit.TrialAuditError):
                audit.parse_signal(bad, parser)
        with self.assertRaises(audit.TrialAuditError):
            audit.parse_signal(b"1,2,3\r\n4,5,6\r\n", "numpy")
        with self.assertRaises(audit.TrialAuditError):
            audit.parse_signal(b"\r\n", "numpy")

    @unittest.skipIf(audit is None or audit.pa_csv is None, "pyarrow not installed")
    def test_parsers_agree_bit_for_bit(self):
        signal = make_signal(0.2, force=4.09334036339256)
        data = b"\r\n".join(",".join(repr(float(x)) for x in r).encode() for r in signal) + b"\r\n"
        self.assertTrue(np.array_equal(audit.parse_signal(data, "pyarrow"), audit.parse_signal(data, "numpy")))

    def test_beta_and_window_are_strict(self):
        self.assertEqual(audit.parse_beta(b"0.200108073997398\r\n"), 0.200108073997398)
        self.assertEqual(audit.parse_window(b"138202,158200\r\n"), (138202, 158200))
        for bad in (b"nan\r\n", b"0.1,0.2\r\n", b"0.1\r\n0.2\r\n", b"x\r\n"):
            with self.subTest(beta=bad), self.assertRaises(audit.TrialAuditError):
                audit.parse_beta(bad)
        for bad in (b"1.5,3\r\n", b"5,2\r\n", b"5,5\r\n", b"1,2,3\r\n", b"-1,4\r\n", b""):
            with self.subTest(window=bad), self.assertRaises(audit.TrialAuditError):
                audit.parse_window(bad)


@unittest.skipIf(np is None, "numpy not installed")
class StatisticTests(unittest.TestCase):
    def test_longest_true_run(self):
        run = audit.longest_true_run
        self.assertEqual(run(np.array([], dtype=bool)), (-1, -1))
        self.assertEqual(run(np.zeros(5, dtype=bool)), (-1, -1))
        self.assertEqual(run(np.ones(5, dtype=bool)), (0, 4))
        self.assertEqual(run(np.array([1, 0, 1, 1, 1, 0, 1, 1], dtype=bool)), (2, 4))

    def test_min_positive_step(self):
        self.assertEqual(audit.min_positive_step(np.array([0.375, -0.125, 0.25, 0.25])), 0.125)
        self.assertTrue(np.isnan(audit.min_positive_step(np.array([1.0, 1.0]))))

    def test_tiny_quantities_are_not_rounded_away(self):
        values = [3e-8, 5e-8, 4e-8]
        self.assertEqual(audit._describe(values, None)["min"], 3e-8)
        self.assertEqual(audit._describe(values, 7)["min"], 0.0)  # why full precision is needed
        gap = audit.min_positive_step(np.array([1.0, 1.0 + 3e-8, 1.0 + 7e-8]))
        self.assertAlmostEqual(gap / 3e-8, 1.0, places=6)  # 1.0 + 3e-8 is not exactly representable

    def test_exact_window_statistics(self):
        result = audit.analyse_trial(make_signal(0.25), 300, 500, 0.01)
        self.assertEqual((result["c2_plateau_start"], result["c2_plateau_end"]), PLATEAU)
        self.assertTrue(result["window_inside_plateau"])
        self.assertEqual(result["c2_window_steady_fraction"], 1.0)
        self.assertEqual(result["c2_window_mean"], 0.25)
        self.assertEqual(result["c1_window_mean"], 2.5)
        self.assertEqual(result["c1_window_std"], 0.0)
        self.assertEqual(result["c4_window_min_step"], 0.125)
        self.assertEqual(result["c4_window_unique"], 7)
        self.assertEqual(result["window_width"], 200)

    def test_window_on_ramp_is_measured_not_hidden(self):
        result = audit.analyse_trial(make_signal(0.25), 100, 299, 0.01)
        self.assertFalse(result["window_inside_plateau"])
        self.assertEqual(result["c2_window_steady_fraction"], 0.5)

    def test_invalid_trials_raise(self):
        signal = make_signal(0.25)
        with self.assertRaises(audit.TrialAuditError):
            audit.analyse_trial(signal, 900, ROWS, 0.01)  # last valid index is ROWS - 1
        signal[5, 0] = np.nan
        with self.assertRaises(audit.TrialAuditError):
            audit.analyse_trial(signal, 300, 500, 0.01)


@unittest.skipIf(np is None, "numpy not installed")
class AuditRunTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        base = Path(self.tmp.name)
        self.root = base / "extract" / "archive_root"
        self.reports = base / "reports"
        self.slow = self.root / "2024-03-27_r_v1"
        self.fast = self.root / "2024-03-27_r_v3"
        for number, beta in ((1, 0.2), (2, 0.02), (10, 0.11)):
            write_trial(self.slow, number, make_signal(0.125), beta, (300, 500))
            write_trial(self.fast, number, make_signal(0.25, force=1.0), beta, (300, 500))

    def tearDown(self):
        self.tmp.cleanup()

    def run_audit(self, **kwargs):
        options = {"limit": None, "workers": 1, "plateau_tol": 0.01, "parser": "auto", "progress": quiet}
        options.update(kwargs)
        return audit.run_audit(self.root.parent, self.reports, **options)

    def summary(self, run_dir):
        return json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))

    def test_pass_with_exact_summary_and_numeric_order(self):
        for parser in ("auto", "numpy"):
            with self.subTest(parser=parser):
                code, run_dir = self.run_audit(parser=parser)
                report = self.summary(run_dir)
                self.assertEqual((code, report["status"], report["trials_audited"]), (0, "PASS", 6))
                slow, fast = report["conditions"]
                self.assertEqual(Path(slow["directory"]).name, "2024-03-27_r_v1")
                self.assertEqual(slow["c2_peak_levels_at_1e-3"], [0.125])
                self.assertEqual(fast["c2_peak_levels_at_1e-3"], [0.25])
                self.assertEqual(slow["beta"]["distinct_values_at_1e-3"], [0.02, 0.11, 0.2])
                self.assertEqual(slow["beta"]["trials_per_level_at_1e-3"]["max"], 1)
                self.assertEqual(slow["window"]["width_counts"], {"200": 3})
                self.assertEqual(slow["window"]["inside_c2_plateau"], 3)
                self.assertEqual(fast["c1_window_mean"]["median"], 1.0)
                self.assertEqual(report["warnings"], [])
                trials = (run_dir / "trials.csv").read_text(encoding="utf-8").splitlines()
                self.assertEqual(len(trials), 7)
                order = [line.split(",")[:2] for line in trials[1:4]]
                self.assertEqual(order, [["2024-03-27_r_v1", "1"], ["2024-03-27_r_v1", "2"], ["2024-03-27_r_v1", "10"]])
                latest = json.loads((self.reports / "latest.json").read_text(encoding="utf-8"))
                self.assertEqual(Path(latest["run_dir"]), run_dir)

    def test_show_is_compact_and_reports_status(self):
        self.run_audit()
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            code = audit.show_latest(self.reports)
        text = buffer.getvalue()
        self.assertEqual(code, 0)
        self.assertIn("TRIAL AUDIT: PASS", text)
        self.assertLess(len(text.splitlines()), 20)

    def test_limit_takes_first_trials_in_numeric_order(self):
        _, run_dir = self.run_audit(limit=2)
        rows = (run_dir / "trials.csv").read_text(encoding="utf-8").splitlines()[1:]
        self.assertEqual(sorted({line.split(",")[1] for line in rows}), ["1", "2"])

    def test_trials_per_beta_level_counts_repeats(self):
        write_trial(self.slow, 11, make_signal(0.125), 0.20004, (300, 500))  # rounds to the 0.2 level
        _, run_dir = self.run_audit()
        per_level = self.summary(run_dir)["conditions"][0]["beta"]["trials_per_level_at_1e-3"]
        self.assertEqual((per_level["min"], per_level["max"]), (1, 2))

    def test_ramp_window_is_a_warning_not_a_failure(self):
        write_trial(self.fast, 2, make_signal(0.25, force=1.0), 0.02, (100, 299))
        code, run_dir = self.run_audit()
        report = self.summary(run_dir)
        self.assertEqual((code, report["status"]), (0, "PASS"))
        self.assertEqual(report["conditions"][1]["window"]["inside_c2_plateau"], 2)
        self.assertEqual(len(report["warnings"]), 1)

    def test_structural_problems_fail_and_are_listed(self):
        cases = {
            "missing timestamp": lambda: (self.slow / "timestamp_2.csv").unlink(),
            "orphan beta": lambda: (self.slow / "beta_99.csv").write_bytes(b"0.1\r\n"),
            "non-finite": lambda: (self.slow / "whole_1.csv").write_bytes(
                (self.slow / "whole_1.csv").read_bytes().replace(b"2.5,", b"nan,", 1)),
            "window past end": lambda: (self.slow / "timestamp_10.csv").write_bytes(b"900,1000\r\n"),
            "short row": lambda: (self.slow / "whole_2.csv").write_bytes(
                (self.slow / "whole_2.csv").read_bytes() + b"1.0,2.0,3.0\r\n"),
        }
        for label, damage in cases.items():
            with self.subTest(case=label):
                self.tearDown()
                self.setUp()
                damage()
                code, run_dir = self.run_audit()
                report = self.summary(run_dir)
                self.assertEqual((code, report["status"]), (1, "FAIL"))
                slow = report["conditions"][0]
                self.assertTrue(slow["failed"] or slow["orphan_companion_count"])

    def test_missing_root_and_empty_root(self):
        with self.assertRaises(audit.TrialAuditError):
            audit.run_audit(Path(self.tmp.name) / "nope", self.reports, None, 1, 0.01, progress=quiet)
        empty = Path(self.tmp.name) / "empty"
        empty.mkdir()
        with self.assertRaises(audit.TrialAuditError):
            audit.run_audit(empty, self.reports, None, 1, 0.01, progress=quiet)

    def test_show_without_reports_fails_cleanly(self):
        with contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(audit.show_latest(Path(self.tmp.name) / "no-reports"), 1)


if __name__ == "__main__":
    unittest.main()
