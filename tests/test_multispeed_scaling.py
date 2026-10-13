"""Identifiability, fold isolation and frozen target evaluation for speed scaling."""
import json
from pathlib import Path
import sys

import numpy as np
import pytest
from threadpoolctl import threadpool_limits

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import multispeed_scaling as ms


def fixture(alpha=1.5, levels=15, ranks=31):
    rows = []
    for specimen in ("A", "B"):
        for speed in (.05, .1, .2):
            for level, b in enumerate(np.linspace(-1.2, 1.2, levels), 1):
                for rank, f in enumerate(np.linspace(-3., 3., ranks), 1):
                    q = f - alpha * np.log(speed / .1)
                    y = (q - .2 * b) ** 2 + .15 * b ** 2 < 1.
                    rows.append((specimen, speed, level, rank, .1 * np.exp(b), np.exp(f), y))
    X = np.array([[r[4], r[5], r[1]] for r in rows])
    y = np.asarray([r[6] for r in rows], int)
    return {"X": X, "y_fixed": y, "string": np.asarray([r[0] for r in rows]),
            "speed": np.asarray([r[1] for r in rows]), "beta_level": np.asarray([r[2] for r in rows]),
            "ids": np.asarray([f"{r[0]}/{r[1]}/{r[2]}/{r[3]}" for r in rows]),
            "group": np.asarray([f"{r[0]}/{r[1]:g}" for r in rows]), "reference_f0_hz": 100.,
            "output_features": {"periodicity": np.full(len(rows), .95), "f0_hz": np.full(len(rows), 100.),
                                "second_flyback_ratio": np.where(y, .1, .9),
                                "flybacks_per_period": np.ones(len(rows))},
            "manifest": {"scope": "planted synthetic fixture"}}


def test_beta_columns_stay_together_across_both_speeds():
    data = fixture(levels=10, ranks=3)
    train, _ = ms.build_masks(data, "A", (.05, .2))
    levels, speeds = data["beta_level"][train], data["speed"][train]
    folds = ms.two_speed_folds(levels, speeds)
    hits = np.zeros(len(levels), int)
    for source, valid in folds:
        assert not np.intersect1d(levels[source], levels[valid]).size
        assert np.unique(speeds[source]).size == np.unique(speeds[valid]).size == 2
        hits[valid] += 1
    np.testing.assert_array_equal(hits, 1)
    with pytest.raises(ValueError, match="exactly two"):
        ms.two_speed_folds(levels, np.full_like(speeds, .1))


def test_two_speed_source_cv_recovers_planted_exponent(tmp_path):
    data = fixture()
    train, _ = ms.build_masks(data, "A", (.05, .2))
    family = ms.label_family(data["output_features"], data["reference_f0_hz"])
    with threadpool_limits(limits=1):
        candidates, selected = ms.select_source(data["X"][train], data["y_fixed"][train], family[train],
                                               data["beta_level"][train], data["speed"][train], data["ids"][train],
                                               tmp_path / "fits.jsonl", exponents=(0., 1., 1.5, 2.), cs=(10.,))
    assert candidates[selected["learned_alpha"]]["alpha"] == 1.5
    assert candidates[selected["learned_alpha"]]["metrics"]["nominal_iou"] > .95


def test_frozen_json_prediction_matches_estimator(tmp_path):
    data = fixture(levels=10, ranks=15)
    train, _ = ms.build_masks(data, "A", (.05, .2))
    with threadpool_limits(limits=1):
        for alpha in (0., 1.5, None):
            estimator = ms.fit_estimator(data["X"][train], data["y_fixed"][train], alpha=alpha, C=10.)
            frozen = json.loads(json.dumps(ms.freeze_estimator(estimator, alpha, 10.)))
            expected = estimator.predict_proba(ms.coordinates(data["X"], alpha))[:, 1]
            np.testing.assert_allclose(ms.predict_frozen(frozen, data["X"]), expected, atol=1e-13)


def test_target_label_poison_cannot_change_frozen_models_and_freeze_precedes_evaluation(tmp_path, monkeypatch):
    data = fixture(levels=10, ranks=15)
    original_evaluate = ms.evaluate_panels
    active = {"directory": None}

    def guarded_evaluation(*args, **kwargs):
        directory = active["directory"]
        assert (directory / "all_models_frozen.json").is_file()
        return original_evaluate(*args, **kwargs)

    monkeypatch.setattr(ms, "evaluate_panels", guarded_evaluation)
    frozen = []
    with threadpool_limits(limits=1):
        for trial in (0, 1):
            directory = tmp_path / str(trial)
            active["directory"] = directory
            if trial:
                train, _ = ms.build_masks(data, "A", (.05, .2))
                data["y_fixed"][~train] = 1 - data["y_fixed"][~train]
                data["output_features"]["second_flyback_ratio"] = np.where(data["y_fixed"], .1, .9)
            ms.run(directory, sources=("A",), pairs=((.05, .2),), dataset=data,
                   exponents=(0., 1., 1.5), cs=(1.,))
            frozen.append(json.loads((directory / "A__0.05_0.2/frozen_models.json").read_text()))
    # CV diagnostics, IDs, selected candidates and every fitted coefficient must agree.
    assert frozen[0] == frozen[1]
    with pytest.raises(FileExistsError):
        ms.run(tmp_path / "0", dataset=data)
