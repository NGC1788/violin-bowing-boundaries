"""Only byte-identical source-only reconstruction may publish a frozen checkpoint."""
import copy
import json

import joblib
import numpy as np
import pytest

import restore_frozen_models as restore


@pytest.fixture
def restoration(tmp_path, monkeypatch):
    root = tmp_path / "repo"
    root.mkdir()
    monkeypatch.setattr(restore.bd, "ROOT", root)
    data = {"train_mask": np.array([True, False]), "ids": np.array(["source", "unrelated"]),
            "X": np.array([[.1, .4, .1], [.2, 99., .2]]), "y_fixed": np.array([1, 0]),
            "output_features": {}, "reference_f0_hz": 98.}
    monkeypatch.setattr(restore.bd, "load_dataset", lambda include_ambiguous: data)
    monkeypatch.setattr(restore, "label_family", lambda features, f0: np.ones((2, 15), bool))
    monkeypatch.setattr(restore.bs, "specifications", lambda: [{"name": "model"}])
    calls = []
    expected_model = {"fitted_from_source": True, "coefficient": 3.}

    def fit(spec, parameters, X, nominal, family):
        np.testing.assert_array_equal(X, data["X"][:1])
        np.testing.assert_array_equal(nominal, [True])
        assert family.shape == (1, 15)
        calls.append("fit")
        return copy.deepcopy(expected_model), {}

    monkeypatch.setattr(restore.bs, "fit_model", fit)
    reference = tmp_path / "expected.joblib"
    joblib.dump(expected_model, reference, compress=3)
    frozen = {"implementation_sha256": {}, "training_ids_sha256": restore.bd.id_hash(data["ids"][:1]),
              "all_declared_models": {"model": {"path": "reports/model.joblib", "params": {},
                                                "sha256": restore.el.digest(reference)}}}
    frozen["freeze_sha256"] = restore.el.canonical_digest(frozen, "freeze_sha256")
    manifest = root / "freeze.json"
    manifest.write_text(json.dumps(frozen))
    original_check = restore.el.check_freeze
    monkeypatch.setattr(restore.el, "check_freeze", lambda path: original_check(path, root=root))
    return root, manifest, frozen, calls, data


def test_missing_models_restore_exactly_and_existing_models_skip_refit(restoration):
    root, manifest, _, calls, _ = restoration
    before = manifest.read_bytes()
    result = restore.restore(manifest)
    assert result == {"status": "PASS", "verified_models": 1,
                      "frozen_manifest_unchanged": True, "no_target_access": True}
    assert calls == ["fit"]
    assert (root / "reports/model.joblib").exists()
    restore.restore(manifest)
    assert calls == ["fit"]
    restore.restore(manifest, verify_rebuild=True)
    assert calls == ["fit", "fit"]
    assert manifest.read_bytes() == before


def test_rebuild_mismatch_never_publishes_checkpoint(restoration, monkeypatch):
    root, manifest, _, _, _ = restoration
    monkeypatch.setattr(restore.bs, "fit_model", lambda *args: ({"wrong_model": True}, {}))
    with pytest.raises(ValueError, match="rebuild differs"):
        restore.restore(manifest)
    assert not (root / "reports/model.joblib").exists()


def test_existing_mismatch_and_manifest_tamper_fail_before_refit(restoration):
    root, manifest, frozen, calls, _ = restoration
    model = root / "reports/model.joblib"
    model.parent.mkdir()
    model.write_bytes(b"must preserve")
    with pytest.raises(ValueError, match="existing checkpoint differs"):
        restore.restore(manifest)
    assert model.read_bytes() == b"must preserve"
    assert calls == []
    frozen["training_ids_sha256"] = "tampered"
    manifest.write_text(json.dumps(frozen))
    with pytest.raises(ValueError, match="fingerprint mismatch"):
        restore.restore(manifest)
    assert calls == []


def test_changed_membership_or_escape_path_cannot_restore(restoration):
    root, manifest, frozen, calls, data = restoration
    data["train_mask"][:] = [False, True]
    with pytest.raises(ValueError, match="training membership changed"):
        restore.restore(manifest)
    data["train_mask"][:] = [True, False]
    frozen["all_declared_models"]["model"]["path"] = "../outside.joblib"
    frozen["freeze_sha256"] = restore.el.canonical_digest(frozen, "freeze_sha256")
    manifest.write_text(json.dumps(frozen))
    with pytest.raises(ValueError, match="escapes repository"):
        restore.restore(manifest)
    assert calls == []
