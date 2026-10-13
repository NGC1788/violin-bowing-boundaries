"""Independent wave-algebra and counterexample tests for the research prototype."""
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import numpy as np


spec = spec_from_file_location("endpoint_audit", Path(__file__).resolve().parents[1] /
                              "research/2026-10-13/endpoint_observer_audit.py")
audit = module_from_spec(spec)
spec.loader.exec_module(audit)


def test_forward_reflection_reconstruction_equals_known_smoothing():
    rng = np.random.default_rng(7)
    outgoing = rng.normal(size=6000)
    z, half_delay, round_delay, gain, pole = .946, 21.37, 41.19, .99, .8
    truth = outgoing - audit.reflection(outgoing, gain, pole, round_delay)
    force = 2 * z * audit.delay(outgoing, half_delay)
    observed = audit.estimate_contact(force, z, half_delay, round_delay, gain, pole)
    np.testing.assert_allclose(observed[300:-300], audit.smoothing(truth, half_delay)[300:-300], atol=2e-12)


def test_noise_bound_and_shared_scale_blindness():
    rng = np.random.default_rng(10)
    n, g, p, eps, z = 4000, .99, .8, .03, .946
    noise = rng.uniform(-eps, eps, n)
    h, d = 18.23, 35.97
    transformed = audit.estimate_contact(noise, z, h, d, g, p)
    assert max(abs(transformed[100:-100])) <= (1 + g) * eps / (2 * z)
    # Independent identical sensors still cannot reveal their shared calibration scale.
    force = rng.normal(size=n)
    correct = audit.estimate_contact(force, z, h, d, g, p)
    biased = audit.estimate_contact(force, 1.1 * z, h, d, g, p)
    np.testing.assert_allclose(biased, correct / 1.1, atol=1e-14)


def test_both_fractional_endpoint_observations_can_hide_same_mode():
    alternation = (-1.) ** np.arange(1000)
    for half_delay in (17.5, 82.5):
        np.testing.assert_array_equal(audit.delay(alternation, half_delay)[100:-100], 0.)
    assert max(abs(alternation)) == 1
    assert audit.ideal_toy()["integer_delay_identity_max_error"] < 1e-14


def test_endpoint_clone_matches_frozen_production_and_contact_states():
    string = audit.bs.StringParams(f0=np.array([98.08, 440.]), impedance=1., q1=200., corner_hz=6000.)
    (signals, stick), cfg, velocity, force = audit.simulate_endpoints(string, [.05, .1], [.1, .2], [1., 4.],
                                                                    rate=50_000, duration=.15)
    reference, diagnostics = audit.simulate_cpu(string, audit.bs.FrictionParams(), [.05, .1], velocity, force,
                                                1000., 50_000, 0., 7500, return_diagnostics=True)
    np.testing.assert_array_equal(signals[:, :, 0].astype(np.float32), reference)
    np.testing.assert_array_equal(1 - stick, diagnostics["slip_fraction_by_output"])
    np.testing.assert_allclose(signals[6000:, :, 2][stick[6000:].astype(bool)],
                               np.broadcast_to([.1, .2], stick[6000:].shape)[stick[6000:].astype(bool)], atol=1e-13)
