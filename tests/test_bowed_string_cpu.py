"""CPU acceleration must preserve passive reference dynamics and contact balance."""
import sys
from pathlib import Path
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import bowed_string as bs
try:
    from bowed_string_cpu import simulate_cpu, _thiran_step
except ImportError:
    simulate_cpu = None


@unittest.skipIf(simulate_cpu is None, "numba not installed")
class CompiledPhysicsTests(unittest.TestCase):
    def profiles(self):
        t = (np.arange(202) + 0.5) / 1000
        velocity = np.minimum(t[:, None] / 0.04, 1) * np.array([[0.1, -0.1, 0.05]])
        force = np.broadcast_to([4.0, 2.0, 0.0], velocity.shape).copy()
        return velocity, force

    def test_waveform_parity_and_contact_dissipation(self):
        velocity, force = self.profiles()
        string = bs.StringParams(q1=np.array([2000., 600., 900.]), corner_hz=np.array([6000., 12000., 9000.]))
        friction = bs.FrictionParams(mu_s=np.array([0.6725, 0.9, 0.7]), mu_d=0.25, v0=0.08)
        args = (string, friction, [0.1, 0.05, 0.8], velocity, force, 1000, 50_000, 0., 10_000, 2, 5000)
        expected = bs.simulate(bs.Backend(), *args)
        actual, diagnostics = simulate_cpu(*args, return_diagnostics=True)
        np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=2e-7)
        self.assertGreaterEqual(float(diagnostics["min_friction_dissipation_watts"].min()), -1e-14)
        self.assertLess(float(diagnostics["max_load_line_residual"].max()), 1e-12)
        np.testing.assert_array_equal(actual[:, 2], 0)
        self.assertTrue(np.all((diagnostics["slip_fraction"] >= 0) & (diagnostics["slip_fraction"] <= 1)))

    def test_thread_count_preserves_waveform(self):
        velocity, force = self.profiles()
        args = (bs.StringParams(), bs.FrictionParams(), [0.1, 0.2, 0.1], velocity, force,
                1000, 50_000, 0., 3000)
        np.testing.assert_array_equal(simulate_cpu(*args, threads=1), simulate_cpu(*args, threads=2))

    def test_nonpassive_history_refused_and_memory_guard(self):
        velocity, force = self.profiles()
        with self.assertRaisesRegex(ValueError, "nonpassive"):
            simulate_cpu(bs.StringParams(q1=3726.62, corner_hz=1896.41), bs.FrictionParams(),
                         [0.1] * 3, velocity, force, 1000, 50_000, 0., 100)
        with self.assertRaisesRegex(ValueError, "memory budget"):
            simulate_cpu(bs.StringParams(), bs.FrictionParams(), [0.1] * 3, velocity, force,
                         1000, 50_000, 0., 100_000_000)

    def test_internal_window_slip_trace_has_integral_indicators(self):
        velocity, force = self.profiles()
        _, diagnostics = simulate_cpu(bs.StringParams(), bs.FrictionParams(), [0.1] * 3,
                                      velocity, force, 1000, 50_000, 0., 10_000, record_from=5000,
                                      return_diagnostics=True)
        trace = diagnostics["slip_fraction_by_output"]
        self.assertTrue(np.all((trace == 0) | (trace == 1)))
        np.testing.assert_allclose(trace.mean(axis=0), diagnostics["slip_fraction"], atol=1e-7)

    def test_free_decay_matches_passive_loop_q_not_legacy_recurrence(self):
        # Excite briefly, remove contact, and independently fit fundamental decay.
        # The old recurrence gives roughly Q=1147 instead of the intended ~1985.
        t = (np.arange(4002) + 0.5) / 1000
        velocity = np.full((t.size, 1), 0.1)
        force = np.where(t[:, None] < 0.02, 4.0, 0.0)
        string = bs.StringParams()
        waveform = simulate_cpu(string, bs.FrictionParams(), [0.1], velocity, force,
                                1000, 50_000, 0.0, 200_000, output_every=5)[:, 0]
        amplitudes, centers = [], []
        for start in np.arange(0.3, 3.4, 0.2):
            lo = int(start * 10_000)
            tt = np.arange(lo, lo + 4000) / 10_000
            basis = np.column_stack([np.cos(2 * np.pi * string.f0 * tt),
                                     np.sin(2 * np.pi * string.f0 * tt), np.ones(tt.size)])
            weight = np.hanning(tt.size)
            fit = np.linalg.lstsq(weight[:, None] * basis, waveform[lo:lo + 4000] * weight, rcond=None)[0]
            amplitudes.append(np.hypot(fit[0], fit[1]))
            centers.append(start + 0.2)
        decay = np.polyfit(centers, np.log(amplitudes), 1)[0]
        fitted_q = -np.pi * string.f0 / decay
        expected_q = bs.linear_decay_diagnostics(string, [0.1], 50_000)["effective_q_at_nominal_f0"][0]
        self.assertAlmostEqual(fitted_q / expected_q, 1.0, delta=0.01)

    def test_thiran_impulse_has_unit_magnitude_and_requested_dc_delay(self):
        for fraction in (.2, .5, .8):
            coefficient = (1 - fraction) / (1 + fraction)
            previous_input, previous_output = 0., 0.
            response = []
            for n in range(4096):
                value = float(n == 0)
                current = _thiran_step(value, previous_input, previous_output, coefficient)
                response.append(current)
                previous_input, previous_output = value, current
            transform = np.fft.rfft(response)
            np.testing.assert_allclose(abs(transform), 1., atol=1e-12)
            measured_delay = -np.angle(transform[1]) / (2 * np.pi / 4096)
            self.assertAlmostEqual(measured_delay, fraction, delta=1e-6)

    def test_thiran_integer_delay_bypass_matches_linear_exactly(self):
        velocity, force = self.profiles()
        # pole=0, fs/f0=500, beta gives integer round trips on both sides.
        string = bs.StringParams(f0=100., corner_hz=1e9)
        args = (string, bs.FrictionParams(), [.1, .2, .8], velocity, force, 1000., 50_000, 0., 10_000)
        linear = simulate_cpu(*args)
        allpass = simulate_cpu(*args, feedback_delay="thiran1")
        np.testing.assert_array_equal(allpass, linear)


if __name__ == "__main__":
    unittest.main()
