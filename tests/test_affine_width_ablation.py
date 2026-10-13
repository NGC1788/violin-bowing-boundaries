import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import affine_width_ablation as ablation
from test_affine_transport import example


def test_width_ablation_fixes_only_k_and_uses_same_objective():
    X, y, _, speeds = example()
    source = speeds != .2
    model = ablation.fit_translation(X[source], y[source], .01, .03)
    theta = np.asarray(model["theta"])
    assert theta[8] == 0.
    assert model["unique_starts"] == 1
    assert theta[5] <= -1e-6
    norm = ablation.af.normalization(X[source])
    assert model["normalization"] == norm
    actual, _ = ablation.af.objective(theta, ablation.af.raw_coordinates(X[source]), y[source], norm, .01, .03)
    assert actual == model["objective"]
    b, u, t = ablation.af.raw_coordinates(X)
    np.testing.assert_allclose(ablation.af.transport(b, u, t, *theta[6:]), u - (theta[6] + theta[7] * b) * t)
    np.testing.assert_array_equal(ablation.af.predict_frozen(model, X),
                                   ablation.af.predict_frozen(json.loads(json.dumps(model)), X))


def test_width_ablation_is_invariant_to_poisoned_target_labels(tmp_path):
    X, y, levels, speeds = example()
    source = speeds != .2
    ids = np.array([f"row{i}" for i in range(len(y))])
    def select(labels, name):
        return ablation.select_source(X[source], labels[source], labels[source, None], levels[source], speeds[source],
                                       ids[source], tmp_path / name, lambdas=(.01,), warp_lambdas=(.01,))
    original = select(y, "original.jsonl")
    poisoned = y.copy()
    poisoned[~source] = ~poisoned[~source]
    assert original == select(poisoned, "poisoned.jsonl")
