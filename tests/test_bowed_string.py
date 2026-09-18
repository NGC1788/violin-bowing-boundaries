"""Waveguide bowed-string simulator: friction solution, pitch, Schelleng regime order and the flyback law."""

import importlib.util
from pathlib import Path
import sys
import unittest

try:
    import numpy as np
except ImportError:  # pragma: no cover
    np = None

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if np is not None:
    sys.path.insert(0, str(SCRIPTS))
    import regime_map as rm  # noqa: E402
    SPEC = importlib.util.spec_from_file_location("bowed_string_under_test", SCRIPTS / "bowed_string.py")
    bs = importlib.util.module_from_spec(SPEC)
    sys.modules[SPEC.name] = bs
    SPEC.loader.exec_module(bs)

RATE = 50_000


def stroke(betas, speeds, forces, seconds=0.6, rate=RATE, string=None):
    string = string or bs.StringParams()
    points = int(seconds * 1000) + 2
    t = (np.arange(points) + 0.5) / 1000
    velocity = np.minimum(t[:, None] / 0.1, 1.0) * np.asarray(speeds, float)[None, :]
    force = np.broadcast_to(np.asarray(forces, float), (points, len(betas))).copy()
    steps = int(seconds * rate)
    every = rate // RATE
    return bs.simulate(bs.Backend(), string, bs.FrictionParams(), betas, velocity, force, 1000, rate, 0.0, steps,
                       every, record_from=steps - 20_000 * every)


def describe(window):
    window = window.astype(float)
    periodicity, f0 = rm.dominant_period(window)
    shape = rm.flyback_structure(window, RATE / f0 if np.isfinite(f0) else float("nan"))
    label = rm.classify(float(np.std(window)), periodicity, f0, shape["flybacks_per_period"], 98.0, rm.DEFAULT_THRESHOLDS)
    return label, f0, shape["flyback_height"]


@unittest.skipIf(np is None, "numpy not installed")
class FrictionTests(unittest.TestCase):
    def setUp(self):
        self.be, self.fr, self.y = bs.Backend(), bs.FrictionParams(), 1 / (2 * 0.946)

    def solve(self, w0, fn, sticking):
        f, stick, slip = bs.friction_step(self.be, np.array([w0]), self.y, np.array([fn]), np.array([sticking]), self.fr)
        return float(f[0]), bool(stick[0]), float(slip[0])

    def test_stick_moves_the_string_with_the_bow(self):
        f, stick, slip = self.solve(0.2, 1.0, True)  # needs 0.2/Y = 0.378 N <= mu_s F_N = 0.67 N
        self.assertTrue(stick)
        self.assertAlmostEqual(f, 0.2 / self.y)
        self.assertEqual(slip, 0.0)

    def test_slip_solves_the_friction_curve(self):
        f, stick, slip = self.solve(1.0, 1.0, True)
        self.assertFalse(stick)
        mu = self.fr.mu_d + (self.fr.mu_s - self.fr.mu_d) * self.fr.v0 / (self.fr.v0 + slip)
        self.assertAlmostEqual(slip, 1.0 - self.y * f, places=10)  # string velocity = v_h + Y f
        self.assertAlmostEqual(f, 1.0 * mu, places=10)
        f_neg, _, slip_neg = self.solve(-1.0, 1.0, True)
        self.assertAlmostEqual((f_neg, slip_neg), (-f, -slip))

    def test_hysteresis_and_no_contact(self):
        # Load line crossing the curve three times: sticking stays stuck, slipping stays slipping.
        fr = bs.FrictionParams(mu_s=1.0, mu_d=0.2, v0=0.01)
        f1, stick1, _ = bs.friction_step(self.be, np.array([0.4]), self.y, np.array([1.0]), np.array([True]), fr)
        f2, stick2, slip2 = bs.friction_step(self.be, np.array([0.4]), self.y, np.array([1.0]), np.array([False]), fr)
        self.assertTrue(bool(stick1[0]))
        self.assertFalse(bool(stick2[0]))
        self.assertGreater(float(slip2[0]), 0.0)
        f, stick, slip = self.solve(0.3, 0.0, True)
        self.assertEqual((f, stick), (0.0, False))
        self.assertAlmostEqual(slip, 0.3)


@unittest.skipIf(np is None, "numpy not installed")
class WaveguideTests(unittest.TestCase):
    def test_schelleng_order_pitch_and_flyback_law(self):
        # beta 0.1, v 0.1: Schelleng F_max = 2 Z v / (beta (mu_s - mu_d)) = 5.8 N.
        out = stroke([0.1] * 4, [0.1, 0.1, 0.1, 0.05], [0.25, 4.0, 10.0, 1.5])
        low, helm, high, slow = (describe(out[:, j]) for j in range(4))
        self.assertEqual(low[0], "multiple_slip")
        self.assertEqual(high[0], "aperiodic")
        self.assertEqual((helm[0], slow[0]), ("helmholtz", "helmholtz"))
        self.assertAlmostEqual(helm[1], 98.08, delta=0.2)  # reflection-filter delay compensated
        theory = 2 * 0.946 * 0.1 / 0.1
        self.assertAlmostEqual(helm[2] / theory, 1.0, delta=0.12)
        self.assertAlmostEqual(helm[2] / slow[2], 2.0, delta=0.3)  # flyback proportional to bow speed

    def test_small_beta_needs_a_finer_rate(self):
        with self.assertRaisesRegex(ValueError, "beta too small"):
            stroke([0.0005], [0.1], [1.0], seconds=0.05)


if __name__ == "__main__":
    unittest.main()


@unittest.skipIf(np is None, "numpy not installed")
class PerStrokeParameterTests(unittest.TestCase):
    """Arrays of string and friction values let several parameter sets share one batch."""

    def test_batched_parameters_match_separate_runs(self):
        settings = [dict(q1=2000.0, corner_hz=6000.0, mu_s=0.6725, mu_d=0.346, v0=0.06),
                    dict(q1=600.0, corner_hz=12000.0, mu_s=0.9, mu_d=0.25, v0=0.15)]
        betas, speeds, forces = [0.1, 0.05], [0.1, 0.1], [3.0, 2.0]
        separate = []
        for setting in settings:
            string = bs.StringParams(q1=setting["q1"], corner_hz=setting["corner_hz"])
            friction = bs.FrictionParams(setting["mu_s"], setting["mu_d"], setting["v0"])
            separate.append(stroke_with(betas, speeds, forces, string, friction))
        tiled = bs.StringParams(q1=np.array([s["q1"] for s in settings for _ in betas]),
                                corner_hz=np.array([s["corner_hz"] for s in settings for _ in betas]))
        friction = bs.FrictionParams(np.array([s["mu_s"] for s in settings for _ in betas]),
                                     np.array([s["mu_d"] for s in settings for _ in betas]),
                                     np.array([s["v0"] for s in settings for _ in betas]))
        together = stroke_with(betas * 2, speeds * 2, forces * 2, tiled, friction)
        for index, one in enumerate(separate):
            for column in range(len(betas)):
                np.testing.assert_allclose(together[:, index * len(betas) + column], one[:, column], atol=1e-9)

    def test_scalar_parameters_are_unchanged(self):
        plain = stroke_with([0.1], [0.1], [4.0], bs.StringParams(), bs.FrictionParams())
        arrayed = stroke_with([0.1], [0.1], [4.0], bs.StringParams(q1=np.array([2000.0])), bs.FrictionParams())
        np.testing.assert_allclose(plain, arrayed, atol=1e-12)


def stroke_with(betas, speeds, forces, string, friction, seconds=0.4, rate=RATE):
    points = int(seconds * 1000) + 2
    t = (np.arange(points) + 0.5) / 1000
    velocity = np.minimum(t[:, None] / 0.1, 1.0) * np.asarray(speeds, float)[None, :]
    force = np.broadcast_to(np.asarray(forces, float), (points, len(betas))).copy()
    steps = int(seconds * rate)
    return bs.simulate(bs.Backend(), string, friction, betas, velocity, force, 1000, rate, 0.0, steps, 1,
                       record_from=steps - 10_000)
