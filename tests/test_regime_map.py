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
THRESHOLDS = {"flyback_frac": 0.5, "hysteresis_sigma": 3.0, "noise_floor_factor": 3.0, "min_amplitude": 0.02,
              "min_periodicity": 0.8, "max_aperiodic": 0.5, "subharmonic_ratio": 0.75}
PERIOD = RATE / 98.0


def cycle_signal(f0=98.0, n=20_000, drops=((0.0, 1.0),), smooth=1, noise=0.0, ripple=0.0, ripple_k=7, ringing=0.0,
                 ring_tau=1e-3, seed=0):
    """Bridge-force-like staircase: a linear ramp with abrupt returns of the given (phase, height) per period.

    One drop is Helmholtz; several are multiple slipping. A negative height makes the return rise.
    ``ripple`` adds a sinusoid locked at ``ripple_k`` times f0 and ``ringing`` a decaying 2 kHz tone after each period start.
    """
    t = np.arange(n) / RATE
    phase = f0 * t
    x = sum(h for _, h in drops) * phase - sum(h * np.floor(phase - p) for p, h in drops)
    if ripple:
        x = x + ripple * np.sin(2 * np.pi * ripple_k * phase)
    if ringing:
        since = np.mod(phase, 1.0) / f0
        x = x + ringing * np.exp(-since / ring_tau) * np.sin(2 * np.pi * 2000 * since)
    x = x - x.mean()
    if smooth > 1:
        x = np.convolve(x, np.ones(smooth) / smooth, mode="same")
    if noise:
        x = x + np.random.default_rng(seed).normal(0, noise, n)
    return x


def sawtooth(f0, n=20_000, amp=0.8, smooth=1, noise=0.0, seed=0):
    return cycle_signal(f0, n, ((0.0, amp),), smooth, noise, seed=seed)


def flybacks(x, **kw):
    return regime.flyback_structure(x, PERIOD, kw.pop("frac", 0.5), **kw)


def label_of(x, reference_f0=98.0):
    periodicity, f0 = regime.dominant_period(x)
    shape = regime.flyback_structure(x, RATE / f0 if np.isfinite(f0) else float("nan"), THRESHOLDS["flyback_frac"],
                                     THRESHOLDS["hysteresis_sigma"])
    return regime.classify(float(np.std(x)), periodicity, f0, shape["flybacks_per_period"], reference_f0, THRESHOLDS)


@unittest.skipIf(np is None, "numpy not installed")
class FeatureTests(unittest.TestCase):
    def test_period_and_periodicity_of_a_sawtooth(self):
        periodicity, f0 = regime.dominant_period(sawtooth(98.0))
        self.assertGreater(periodicity, 0.95)
        self.assertAlmostEqual(f0, 98.0, delta=0.5)

    def test_spread_return_under_sample_noise_is_one_flyback(self):
        # Server run 20260917T104233Z: clean sawtooths scored 0.03 slips/period because sample noise
        # crossed a per-sample |diff| threshold along the ramp and chained runs merged the whole window.
        for seed in range(5):
            with self.subTest(seed=seed):
                shape = flybacks(cycle_signal(drops=((0.0, 1.5),), smooth=21, noise=0.02, seed=seed))
                self.assertEqual(shape["flybacks_per_period"], 1)
                self.assertLess(shape["second_flyback_ratio"], 0.1)
                # Smoothing over 21 of 510 samples rounds both corners by ~10 samples of ramp (1.5 * 10/510 each).
                self.assertAlmostEqual(shape["flyback_height"], 1.5 * (1 - 20 / 510), delta=0.03)
                self.assertEqual(shape["flyback_sign"], -1)

    def test_hysteresis_keeps_a_noisy_return_whole(self):
        signals = [cycle_signal(drops=((0.0, 1.5),), smooth=21, noise=0.2, seed=seed) for seed in range(8)]
        self.assertEqual([flybacks(x)["flybacks_per_period"] for x in signals], [1] * 8)
        # Guard: without hysteresis the fixture really splits some returns.
        self.assertTrue(any(flybacks(x, hysteresis_sigma=0.0)["flybacks_per_period"] != 1 for x in signals))

    def test_ripple_and_moderate_ringing_stay_one_flyback(self):
        for name, x in {"ripple": cycle_signal(ripple=0.15, smooth=5, noise=0.01),
                        "ringing": cycle_signal(drops=((0.0, 0.8),), smooth=7, ringing=0.15)}.items():
            with self.subTest(signal=name):
                shape = flybacks(x)
                self.assertEqual(shape["flybacks_per_period"], 1)
                self.assertGreater(shape["second_flyback_ratio"], 0.15)  # the extra legs exist but are small

    def test_rising_return_is_found_with_its_sign(self):
        shape = flybacks(cycle_signal(drops=((0.0, -1.0),), smooth=5, noise=0.01))
        self.assertEqual((shape["flybacks_per_period"], shape["flyback_sign"]), (1, 1))
        self.assertAlmostEqual(shape["flyback_height"], 1.0, delta=0.05)

    def test_double_and_triple_slipping(self):
        double = flybacks(cycle_signal(drops=((0.0, 0.6), (0.5, 0.4)), smooth=5, noise=0.01))
        self.assertEqual(double["flybacks_per_period"], 2)
        self.assertAlmostEqual(double["second_flyback_ratio"], 0.4 / 0.6, delta=0.08)
        circular = lambda a, b: min(abs(a - b), 1 - abs(a - b))  # noqa: E731
        self.assertEqual(len(double["flyback_phases"]), 2)
        for target in (0.0, 0.5):  # a return starts at the peak just before its phase
            self.assertLess(min(circular(p, target) for p in double["flyback_phases"]), 0.02)
        triple = flybacks(cycle_signal(drops=((0.0, 0.4), (0.33, 0.35), (0.66, 0.3)), smooth=5, noise=0.01))
        self.assertEqual(triple["flybacks_per_period"], 3)

    def test_second_ratio_decides_the_count_at_any_frac(self):
        x = cycle_signal(drops=((0.0, 0.6), (0.5, 0.4)))
        ratio = flybacks(x)["second_flyback_ratio"]
        self.assertEqual(flybacks(x, frac=ratio - 0.01)["flybacks_per_period"], 2)
        self.assertEqual(flybacks(x, frac=ratio + 0.01)["flybacks_per_period"], 1)

    def test_fold_uses_fractional_periods(self):
        f0 = 97.9  # period 510.73 samples
        mean, _ = regime.fold_cycle(cycle_signal(f0=f0, drops=((0.0, 1.0),)), RATE / f0)
        # A return between samples is shared by at most two phase samples; integer-period cutting would
        # drift 0.27 samples per period and smear it over ~10.
        steps = np.abs(np.diff(np.append(mean, mean[0])))  # circular: the return may straddle the cycle edge
        self.assertGreater(float(np.sum(np.sort(steps)[-2:])), 0.9)

    def test_too_short_or_undefined_period(self):
        self.assertIsNone(regime.fold_cycle(np.zeros(100), float("nan")))
        self.assertTrue(np.isnan(flybacks(np.zeros(500))["flybacks_per_period"]))


@unittest.skipIf(np is None, "numpy not installed")
class ClassificationTests(unittest.TestCase):
    def test_known_regimes(self):
        rng = np.random.default_rng(3)
        cases = {
            "helmholtz": sawtooth(98.0, smooth=7, noise=0.005),
            "multiple_slip": cycle_signal(drops=((0.0, 0.5), (0.5, 0.4))),
            "subharmonic": sawtooth(49.0),
            "aperiodic": rng.normal(0, 0.3, 20_000),
            "no_oscillation": rng.normal(0, 0.003, 20_000),
        }
        for expected, signal in cases.items():
            with self.subTest(expected=expected):
                self.assertEqual(label_of(signal), expected)

    def test_subharmonic_needs_a_reference(self):
        self.assertNotEqual(label_of(sawtooth(49.0), reference_f0=float("nan")), "subharmonic")

    def test_noise_floor_from_no_contact_windows(self):
        amplitudes = [0.03, 0.04, 0.05] * 5 + [1.0, 2.0]
        column1 = [-0.01, 0.0, -0.02] * 5 + [1.0, 2.0]
        floor = regime.noise_floor(amplitudes, column1, 3.0, 0.02)
        self.assertEqual(floor["controls"], 15)
        self.assertAlmostEqual(floor["min_amplitude"], 0.12)
        few = regime.noise_floor(amplitudes[:3] + [1.0], column1[:3] + [1.0], 3.0, 0.02)
        self.assertEqual(few["min_amplitude"], 0.02)
        self.assertEqual(regime.noise_floor([0.001] * 20, [0.0] * 20, 3.0, 0.02)["min_amplitude"], 0.02)

    def test_flyback_law_recovers_exponents_and_impedance(self):
        rng = np.random.default_rng(1)
        points = [(2 * 0.9 * v / b * float(np.exp(rng.normal(0, 0.05))), v * (1 + rng.normal(0, 0.01)), b)
                  for v in (0.05, 0.1, 0.2) for b in np.geomspace(0.02, 0.2, 15)]
        law = regime.flyback_law(points)
        self.assertAlmostEqual(law["exponent_speed"], 1.0, delta=0.05)
        self.assertAlmostEqual(law["exponent_beta"], -1.0, delta=0.05)
        self.assertAlmostEqual(law["impedance_kg_s"]["median"], 0.9, delta=0.03)
        one_speed = regime.flyback_law([p for p in points if abs(p[1] - 0.1) < 0.01])
        self.assertNotIn("exponent_speed", one_speed)
        self.assertAlmostEqual(one_speed["exponent_beta"], -1.0, delta=0.08)
        self.assertEqual(regime.flyback_law(points[:3]), {"n": 3})


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

    def test_frac_sensitivity_matches_labels_and_is_monotone(self):
        rng = np.random.default_rng(2)
        trials = []
        for level, beta in enumerate(self.betas):
            for rank, force in enumerate(self.forces, 1):
                inside = 0.0008 / beta**2 < force < 0.12 / beta
                ratio = float(rng.uniform(0.0, 0.45) if inside else rng.uniform(0.55, 1.0))
                label = "helmholtz" if ratio < 0.5 else "multiple_slip"
                trials.append({"beta_level": level, "force_rank": rank, "label": label, "second_flyback_ratio": ratio})
        rows = regime.frac_sensitivity(trials, self.centers, self.forces, (0.3, 0.5, 0.7))
        by_frac = {r["flyback_frac"]: r for r in rows}
        self.assertEqual(by_frac[0.5]["helmholtz"], sum(t["label"] == "helmholtz" for t in trials))
        fit = regime.fit_boundaries([(t["beta_level"], t["force_rank"], t["label"]) for t in trials], self.centers, self.forces)
        self.assertEqual(by_frac[0.5]["lower_slope"], fit["lower"]["slope"])
        self.assertLess(by_frac[0.3]["helmholtz"], by_frac[0.5]["helmholtz"])
        self.assertLess(by_frac[0.5]["helmholtz"], by_frac[0.7]["helmholtz"])

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
            for beta in (0.2, 0.1, 0.05, 0.035, 0.02):
                for force in (2.0, 1.0, -0.01, -0.02):  # descending within a beta level, like the real sweep
                    number += 1
                    if force == 1.0:
                        c3, planted[number] = sawtooth(98.0, n=6000, smooth=5, noise=0.003, seed=number), "helmholtz"
                    elif force > 0:
                        c3, planted[number] = rng.normal(0, 0.3, 6000), "aperiodic"
                    else:  # no bow contact: above the 0.02 fallback, below 3x the no-contact median
                        c3, planted[number] = rng.normal(0, 0.05, 6000), "no_oscillation"
                    self.write_trial(folder, number, beta + (number % 3 - 1) * 1e-5, force, c3)
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
            self.assertEqual(summary["conditions"]["2024-03-27_r_v2"]["classes"]["helmholtz"], 5)
            condition = summary["conditions"]["2024-03-27_r_v2"]
            self.assertEqual([r["flyback_frac"] for r in condition["flyback_frac_sensitivity"]], list(regime.SENSITIVITY_FRACS))
            self.assertAlmostEqual(condition["helmholtz_impedance_kg_s"]["median"], 0.8 * 0.05 / (2 * 0.1), delta=0.01)
            self.assertEqual(summary["noise_floor"]["controls"], 10)
            self.assertAlmostEqual(summary["noise_floor"]["min_amplitude"], 0.15, delta=0.01)
            self.assertEqual(summary["flyback_law"]["n"], 5)
            self.assertNotIn("exponent_speed", summary["flyback_law"])  # one speed only
            self.assertTrue(any(line.startswith("  classes:") for line in lines))
            self.assertTrue(any(line.startswith("FLYBACK LAW") for line in lines))
            with (run_dir / "regimes.csv").open(newline="", encoding="utf-8") as handle:
                for row in csv.DictReader(handle):
                    for key in ("flybacks_per_period", "second_flyback_ratio", "flyback_height", "c2_window_mean"):
                        float(row[key])  # every numeric field parses (no numpy reprs)
            if HAVE_MPL:
                for name in ("regime_map.png", "examples.png", "features.png", "flyback_law.png"):
                    self.assertGreater((run_dir / name).stat().st_size, 1000)


if __name__ == "__main__":
    unittest.main()
