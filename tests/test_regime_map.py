"""Provisional regime labels and boundary fits on signals and grids with known answers."""

import csv
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest

try:
    import numpy as np
except ImportError:  # pragma: no cover
    np = None

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
regime = None
if np is not None:
    sys.path.insert(0, str(SCRIPTS))
    SPEC = importlib.util.spec_from_file_location("regime_map_under_test", SCRIPTS / "regime_map.py")
    regime = importlib.util.module_from_spec(SPEC)
    sys.modules[SPEC.name] = regime
    SPEC.loader.exec_module(regime)
try:
    import matplotlib  # noqa: F401
    HAVE_MPL = True
except ImportError:  # pragma: no cover
    HAVE_MPL = False

RATE = 50_000
THRESHOLDS = {"slip_frac": 0.25, "slip_tolerance": 0.25, "min_amplitude": 0.02, "min_periodicity": 0.8,
              "max_aperiodic": 0.5, "subharmonic_ratio": 0.75}


def sawtooth(f0, n=20_000, amp=0.8, second=0.0, smooth=1, noise=0.0, ringing=0.0, ring_tau=1e-3, seed=0):
    """Bridge-force-like sawtooth: one abrupt return per period; ``second`` adds a return at half period."""
    t = np.arange(n) / RATE
    x = amp * (np.mod(t * f0, 1.0) - 0.5)
    if second:
        x = x + second * (np.mod(t * 2 * f0, 1.0) - 0.5)
    if ringing:
        since = np.mod(t * f0, 1.0) / f0  # seconds since the last return
        x = x + ringing * np.exp(-since / ring_tau) * np.sin(2 * np.pi * 2000 * since)
    if smooth > 1:
        x = np.convolve(x, np.ones(smooth) / smooth, mode="same")
    if noise:
        x = x + np.random.default_rng(seed).normal(0, noise, n)
    return x


def label_of(x, reference_f0=98.0):
    periodicity, f0 = regime.dominant_period(x)
    slips = regime.slips_per_period(x, RATE / f0, THRESHOLDS["slip_frac"]) if np.isfinite(f0) else float("nan")
    return regime.classify(float(np.std(x)), periodicity, f0, slips, reference_f0, THRESHOLDS)


@unittest.skipIf(np is None, "numpy not installed")
class FeatureTests(unittest.TestCase):
    def test_period_and_periodicity_of_a_sawtooth(self):
        periodicity, f0 = regime.dominant_period(sawtooth(98.0))
        self.assertGreater(periodicity, 0.95)
        self.assertAlmostEqual(f0, 98.0, delta=0.5)

    def test_one_slip_per_period_survives_smoothing_noise_and_ringing(self):
        for name, signal in {"clean": sawtooth(98.0),
                             "smoothed+noise": sawtooth(98.0, smooth=7, noise=0.005),
                             "ringing": sawtooth(98.0, smooth=7, ringing=0.35)}.items():
            with self.subTest(signal=name):
                self.assertAlmostEqual(regime.slips_per_period(signal, RATE / 98.0, 0.25), 1.0, delta=0.1)

    def test_ringing_fixture_really_splits_into_separate_runs(self):
        # Without merging, the ringing after each return would count as several slips.
        unmerged = regime.slips_per_period(sawtooth(98.0, smooth=7, ringing=0.35), RATE / 98.0, 0.25, merge_frac=0.0)
        self.assertGreater(unmerged, 3.0)

    def test_two_slips_per_period(self):
        self.assertAlmostEqual(regime.slips_per_period(sawtooth(98.0, second=0.4), RATE / 98.0, 0.25), 2.0, delta=0.1)


@unittest.skipIf(np is None, "numpy not installed")
class ClassificationTests(unittest.TestCase):
    def test_known_regimes(self):
        rng = np.random.default_rng(3)
        cases = {
            "helmholtz": sawtooth(98.0, smooth=7, noise=0.005),
            "multiple_slip": sawtooth(98.0, second=0.4),
            "subharmonic": sawtooth(49.0),
            "aperiodic": rng.normal(0, 0.3, 20_000),
            "no_oscillation": rng.normal(0, 0.003, 20_000),
        }
        for expected, signal in cases.items():
            with self.subTest(expected=expected):
                self.assertEqual(label_of(signal), expected)

    def test_subharmonic_needs_a_reference(self):
        self.assertNotEqual(label_of(sawtooth(49.0), reference_f0=float("nan")), "subharmonic")


def schelleng_grid(a, b, betas, forces):
    """Helmholtz strictly between F_min = a/beta^2 and F_max = b/beta."""
    cells = []
    for level, beta in enumerate(betas):
        for rank, force in enumerate(forces, 1):
            cells.append((level, rank, "helmholtz" if a / beta**2 < force < b / beta else "aperiodic"))
    return cells


@unittest.skipIf(np is None, "numpy not installed")
class BoundaryTests(unittest.TestCase):
    def setUp(self):
        self.betas = np.geomspace(0.02, 0.2, 12)
        self.forces = list(np.geomspace(0.01, 10, 40))
        self.centers = dict(enumerate(map(float, self.betas)))

    def test_recovers_schelleng_slopes(self):
        fit = regime.fit_boundaries(schelleng_grid(0.0008, 0.12, self.betas, self.forces), self.centers, self.forces)
        self.assertEqual((fit["lower"]["censored_levels"], fit["upper"]["censored_levels"]), (0, 0))
        self.assertAlmostEqual(fit["lower"]["slope"], -2.0, delta=0.2)
        self.assertAlmostEqual(fit["upper"]["slope"], -1.0, delta=0.2)
        self.assertEqual(fit["levels_with_helmholtz"], 12)
        step = self.forces[1] / self.forces[0]
        for beta, force in fit["lower"]["points"]:  # each boundary lies within half a force step of the truth
            self.assertLess(abs(np.log(force / (0.0008 / beta**2))), np.log(step) * 0.5 + 1e-9)
        for beta, force in fit["upper"]["points"]:
            self.assertLess(abs(np.log(force / (0.12 / beta))), np.log(step) * 0.5 + 1e-9)

    def test_isolated_misclassifications_inside_the_region_do_not_move_boundaries(self):
        cells = schelleng_grid(0.0008, 0.12, self.betas, self.forces)
        holes = 0
        for i, (level, rank, label) in enumerate(cells):  # flip scattered interior cells, never the edges
            if label == "helmholtz" and i % 5 == 0:
                neighbours = {c[1] for c in cells if c[0] == level and c[2] == "helmholtz"}
                if rank - 1 in neighbours and rank + 1 in neighbours:
                    cells[i] = (level, rank, "ambiguous")
                    holes += 1
        self.assertGreater(holes, 10)  # the fixture really contains interior holes
        fit = regime.fit_boundaries(cells, self.centers, self.forces)
        self.assertAlmostEqual(fit["lower"]["slope"], -2.0, delta=0.2)
        self.assertAlmostEqual(fit["upper"]["slope"], -1.0, delta=0.2)
        self.assertEqual(fit["lower"]["n"], 12)

    def test_misclassified_edge_cell_does_not_uncensor_a_boundary(self):
        # True upper boundary lies above the sampled forces at every level; flip the top cell in half the levels.
        cells = schelleng_grid(0.0008, 1e6, self.betas, self.forces)
        top = len(self.forces)
        cells = [(L, k, "ambiguous" if k == top and L % 2 == 0 else lab) for L, k, lab in cells]
        fit = regime.fit_boundaries(cells, self.centers, self.forces)
        self.assertEqual((fit["upper"]["n"], fit["upper"]["censored_levels"]), (0, 12))

    def test_slope_standard_error_is_reported(self):
        fit = regime.fit_boundaries(schelleng_grid(0.0008, 0.12, self.betas, self.forces), self.centers, self.forces)
        self.assertIsNotNone(fit["lower"]["slope_se"])
        self.assertLess(fit["lower"]["slope_se"], 0.2)

    def test_boundary_outside_the_sampled_range_is_censored_not_fitted(self):
        fit = regime.fit_boundaries(schelleng_grid(0.00002, 0.12, self.betas, self.forces), self.centers, self.forces)
        self.assertGreater(fit["lower"]["censored_levels"], 0)
        self.assertEqual(fit["lower"]["n"] + fit["lower"]["censored_levels"], 12)


@unittest.skipIf(np is None, "numpy not installed")
class PipelineTests(unittest.TestCase):
    """trial_audit -> grid_summary -> regime_map on a tiny archive with planted regimes."""

    def write_trial(self, directory, number, beta, force, c3):
        rows = c3.size
        i = np.arange(rows)
        velocity = np.where(i < 1000, 0.1 * i / 2000, 0.1)
        signal = np.column_stack([np.full(rows, force), velocity, c3, np.zeros(rows)])
        directory.mkdir(parents=True, exist_ok=True)
        text = "\r\n".join(",".join(repr(float(v)) for v in row) for row in signal) + "\r\n"
        (directory / f"whole_{number}.csv").write_text(text, encoding="ascii")
        (directory / f"beta_{number}.csv").write_text(repr(beta) + "\r\n", encoding="ascii")
        (directory / f"timestamp_{number}.csv").write_text("1000,5999\r\n", encoding="ascii")

    def test_planted_labels_are_recovered(self):
        audit = sys.modules.get("trial_audit") or __import__("trial_audit")
        rng = np.random.default_rng(5)
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp) / "extract" / "archive" / "2024-03-27_r_v2"
            planted, number = {}, 0
            for beta in (0.2, 0.05):
                for force in (2.0, 1.0, 0.5):  # descending within a beta level, like the real sweep
                    number += 1
                    helm = force == 1.0
                    c3 = sawtooth(98.0, n=6000, smooth=5, noise=0.003, seed=number) if helm else rng.normal(0, 0.3, 6000)
                    self.write_trial(folder, number, beta + (number % 3 - 1) * 1e-5, force, c3)
                    planted[number] = "helmholtz" if helm else "aperiodic"
            audit_dir, reports = Path(tmp) / "audit", Path(tmp) / "regime"
            code, _ = audit.run_audit(folder.parent.parent, audit_dir, None, 1, 0.01, progress=lambda *_: None)
            self.assertEqual(code, 0)
            lines = []
            code, run_dir = regime.run(audit_dir, reports, 1, THRESHOLDS, None, HAVE_MPL, progress=lines.append)
            self.assertEqual(code, 0)
            with (run_dir / "regimes.csv").open(newline="", encoding="utf-8") as handle:
                got = {int(r["trial"]): r["label"] for r in csv.DictReader(handle)}
            self.assertEqual(got, planted)
            summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["conditions"]["2024-03-27_r_v2"]["classes"]["helmholtz"], 2)
            self.assertTrue(any(line.startswith("  classes:") for line in lines))
            if HAVE_MPL:
                self.assertGreater((run_dir / "regime_map.png").stat().st_size, 1000)
                self.assertGreater((run_dir / "examples.png").stat().st_size, 1000)


if __name__ == "__main__":
    unittest.main()
