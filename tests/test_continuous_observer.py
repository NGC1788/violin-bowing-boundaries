import numpy as np
import pytest
import os
from pathlib import Path
import subprocess
import sys

from continuous_observer import ContinuousObserver


def brute(observer, p, weights):
    scores = []
    for f in observer.f_states:
        for q in observer.q_states:
            y = observer.labels(f, q)
            denominator = np.sum(weights * (p + (1-p)*y))
            scores.append(np.sum(weights*p*y)/denominator if denominator else 1.)
    return min(scores), max(scores)


@pytest.mark.parametrize("soft", [False, True])
def test_matches_exhaustive_all_states(soft):
    rng = np.random.default_rng(20261013)
    for _ in range(50):
        n = 23
        r = rng.choice([.2,.3,.35,.41,.5,.61,.7,.8], n)
        q = rng.choice([.7,.75,.78,.8,.82,.85,.9], n)
        observer = ContinuousObserver(r, q, rng.random(n) > .1)
        p = rng.random(n) if soft else (rng.random(n) > .5).astype(float)
        w = rng.random(n)
        lo, hi = brute(observer, p, w)
        assert observer.extreme(p, sample_weight=w)["iou"] == pytest.approx(lo, abs=1e-12)
        assert observer.extreme(p, maximum=True, sample_weight=w)["iou"] == pytest.approx(hi, abs=1e-12)


def test_strict_fraction_and_inclusive_periodicity_endpoints():
    o = ContinuousObserver([.3,.7,.299], [.85,.85,.75])
    np.testing.assert_array_equal(o.labels(.3,.75), [False,False,True])
    np.testing.assert_array_equal(o.labels(.7,.85), [True,False,False])
    assert o.extreme([1,1,1])["iou"] == 0


def test_coarse_grid_can_miss_interior_worst_case():
    # At .35 one false-positive prediction disappears? The admissible target gains
    # a missed positive before it gains a correctly predicted positive at .36.
    o = ContinuousObserver([.1,.34,.36], [.9,.9,.9])
    p = np.array([1.,0.,1.])
    exact = o.extreme(p)["iou"]
    coarse = min(np.sum(p*o.labels(f,.8))/np.sum(np.maximum(p,o.labels(f,.8)))
                 for f in (.3,.4,.5,.6,.7))
    assert exact == pytest.approx(1/3)
    assert coarse == pytest.approx(1/2)


def test_empty_union_and_zero_weights():
    o = ContinuousObserver([.4,.6], [.8,.82])
    assert o.extreme([0.,0.])["iou"] == 0
    assert o.extreme([0.,0.], maximum=True)["iou"] == 1
    assert o.extreme([0.,1.], sample_weight=[1.,0.])["iou"] == 0
    with pytest.raises(ValueError):
        o.extreme([.5,np.nan])


def test_independent_breakpoint_enumeration_including_degenerate_bounds():
    """Do not use the oracle's own state construction as the reference enumerator."""
    rng = np.random.default_rng(986325)
    for _ in range(200):
        n = int(rng.integers(1, 22))
        low, high = sorted(rng.choice(np.arange(-3, 4) / 4, 2, replace=True))
        qlow, qhigh = sorted(rng.choice(np.arange(-3, 4) / 4, 2, replace=True))
        ratio = rng.choice(np.arange(-4, 5) / 4, n)
        periodicity = rng.choice(np.arange(-4, 5) / 4, n)
        eligible = rng.random(n) > .2
        observer = ContinuousObserver(ratio, periodicity, eligible,
                                      fractions=(low, high), cutoffs=(qlow, qhigh))
        p = rng.choice([0., .001, .2, .7, 1.], n)
        weights = rng.choice([0., 1e-7, .1, 1., 10.], n)
        if not weights.any():
            weights[0] = 1.
        f_breaks = np.unique(np.r_[low, high, ratio[(ratio >= low) & (ratio <= high)]])
        q_breaks = np.unique(np.r_[qlow, qhigh,
                                   periodicity[(periodicity >= qlow) & (periodicity <= qhigh)]])
        fractions = np.r_[f_breaks, (f_breaks[:-1] + f_breaks[1:]) / 2]
        cutoffs = np.r_[q_breaks, (q_breaks[:-1] + q_breaks[1:]) / 2]
        scores = []
        for fraction in fractions:
            for cutoff in cutoffs:
                label = eligible & (ratio < fraction) & (periodicity >= cutoff)
                denominator = np.sum(weights * (p + (1-p) * label))
                scores.append(np.sum(weights * p * label) / denominator if denominator else 1.)
        for maximum, expected in [(False, min(scores)), (True, max(scores))]:
            result = observer.extreme(p, maximum=maximum, sample_weight=weights)
            assert result["iou"] == pytest.approx(expected, abs=1e-12)


def test_cached_numba_oracle_survives_a_second_python_process(tmp_path):
    """Recursive cached Numba calls previously passed once, then segfaulted on reuse."""
    root = Path(__file__).resolve().parents[1]
    environment = os.environ.copy()
    environment.pop("NUMBA_DISABLE_JIT", None)
    environment.update({"NUMBA_CACHE_DIR": str(tmp_path / "numba_cache"),
                        "PYTHONPATH": str(root / "scripts"),
                        "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
                        "VECLIB_MAXIMUM_THREADS": "1", "NUMBA_NUM_THREADS": "2"})
    program = """
from continuous_observer import ContinuousObserver
o = ContinuousObserver([.1, .34, .36], [.9, .9, .9])
assert abs(o.extreme([1., 0., 1.])["iou"] - 1/3) < 1e-12
assert abs(o.extreme([1., 0., 1.], maximum=True)["iou"] - 2/3) < 1e-12
print("cached oracle passed")
"""
    for process_number in (1, 2):
        result = subprocess.run([sys.executable, "-c", program], cwd=root, env=environment,
                                text=True, capture_output=True, timeout=30)
        assert result.returncode == 0, (process_number, result.returncode, result.stderr)
        assert "cached oracle passed" in result.stdout


def test_continuous_rule_matches_all_fixed_family_members_on_released_inputs():
    from benchmark_data import load_dataset
    from set_envelope import label_family
    dataset = load_dataset(include_ambiguous=True)
    observer = ContinuousObserver.from_features(dataset["output_features"],
                                                dataset["reference_f0_hz"])
    family = label_family(dataset["output_features"], dataset["reference_f0_hz"])
    for index, (fraction, cutoff) in enumerate((f, q) for f in (.3, .4, .5, .6, .7)
                                             for q in (.75, .8, .85)):
        np.testing.assert_array_equal(observer.labels(fraction, cutoff), family[:, index])


def test_large_finite_weights_do_not_overflow_normalization():
    observer = ContinuousObserver([.1, .1], [.9, .9])
    for maximum in (False, True):
        ordinary = observer.extreme([1., 0.], maximum=maximum, sample_weight=[1., 1.])
        huge = observer.extreme([1., 0.], maximum=maximum, sample_weight=[1e308, 1e308])
        assert ordinary["iou"] == pytest.approx(.5)
        assert huge["iou"] == pytest.approx(ordinary["iou"])
        assert np.isfinite(huge["residual"])
