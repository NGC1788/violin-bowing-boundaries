import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import twospeed_known_prior as prior
from test_affine_transport import example


def test_ensemble_is_exact_mean_of_frozen_source_models():
    X, y, _, speeds = example()
    source = speeds != .2
    models = prior.fit_pair(X[source], y[source], speeds[source], .01)
    for exponents in prior.PRIORS.values():
        expected = (prior.thaw(models[0]).predict_proba(X, exponents)[:, 1] +
                    prior.thaw(models[1]).predict_proba(X, exponents)[:, 1]) / 2
        actual = prior.predict_pair(json.loads(json.dumps(models)), X, exponents)
        np.testing.assert_array_equal(actual, expected)


def test_poisoned_target_labels_do_not_affect_ensemble_prior_selection(tmp_path):
    X, y, levels, speeds = example()
    source = speeds != .2
    ids = np.array([f"row{i}" for i in range(len(y))])
    def select(labels, path):
        return prior.select_source(X[source], labels[source], labels[source, None], levels[source], speeds[source],
                                   ids[source], path, regularizations=(.01, .1))
    result = select(y, tmp_path / "first.jsonl")
    poisoned = y.copy()
    poisoned[~source] = ~poisoned[~source]
    assert result == select(poisoned, tmp_path / "poisoned.jsonl")
