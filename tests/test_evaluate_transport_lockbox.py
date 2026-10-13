"""Portable additive freeze, source isolation and fail-closed target provenance."""
from copy import deepcopy
import json
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import evaluate_transport_lockbox as transport


def write(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload))
    return path


@pytest.fixture
def data():
    beta, force, speed = np.meshgrid([.03, .05, .1, .15, .2], [.2, 2.], [.05, .1, .2], indexing="ij")
    X = np.column_stack((beta.ravel(), force.ravel(), speed.ravel()))
    y = X[:, 1] < 1
    return {"X": X, "ids": np.array([f"A/{i}" for i in range(len(X))]), "string": np.repeat("A", len(X)),
            "speed": X[:, 2], "y_fixed": y.astype(int), "reference_f0_hz": 98.,
            "output_features": {"periodicity": np.full(len(X), .95), "f0_hz": np.full(len(X), 98.),
                                "second_flyback_ratio": np.where(y, .1, .9),
                                "flybacks_per_period": np.ones(len(X)), "amplitude_std": np.ones(len(X))},
            "manifest": {"sources": [{"role": "measured", "string": "A", "sha256": "input-hash"}]}}


@pytest.fixture
def specifications(data):
    sources = {}
    for pair in transport.PAIRS:
        affine = {"theta": [1., 0., 0., 0., 0., -1., 1., 0., .2],
                  "normalization": {"mean": [0., 0.], "scale": [1., 1.]}}
        zero = deepcopy(affine)
        zero["theta"][8] = 0.
        simple = {"alpha": 1., "C": 1., "polynomial_powers": [[1, 0], [0, 1]],
                  "polynomial_feature_mean": [0., 0.], "polynomial_feature_scale": [1., 1.],
                  "logistic_coefficient": [0., -1.], "logistic_intercept": 0.}
        raw = {**deepcopy(simple), "alpha": None, "polynomial_powers": np.eye(3, dtype=int).tolist(),
               "polynomial_feature_mean": [0.] * 3, "polynomial_feature_scale": [1.] * 3,
               "logistic_coefficient": [0., -1., 0.]}
        sources[transport.pair_key(pair)] = {"training_speeds": list(pair),
            "source": transport.source_arrays(data, pair)[0],
            "models": {"affine_flow": affine, "translation_k0": zero, "learned_alpha": simple, "raw_3d": raw},
            "selected_simple_baseline": "learned_alpha",
            "source_cv_family_min_iou": {"affine_flow": .6, "translation_k0": .5, "learned_alpha": .7, "raw_3d": .4}}
    frozen = {"freeze_sha256": "main-models", "lockbox": {"record": "17782542", "file": "TypeA_sample2.7z", "md5": "expected-md5"}}
    evaluation = {"spec_sha256": "main-evaluation", "feature_pipeline_sha256": {"regime_map.py": "feature-hash"}}
    preprocessing = {"preprocessing_spec_sha256": "main-preprocessing", "implementation_sha256": {"collection.py": "collection-hash"}}
    spec = {"created_utc": "2026-10-13T09:00:00+00:00", "main_model_sha256": "main-models",
        "main_evaluation_spec_sha256": "main-evaluation", "main_preprocessing_spec_sha256": "main-preprocessing",
        "lockbox": frozen["lockbox"], "sources": sources, "model_count": 12,
        "source_dataset_sha256": transport.dataset_hashes(data), "implementation_sha256": {"code": "test-hash"},
        "primary_source": "A__0.05_0.1", "primary_endpoint": "macro panel minimum-family IoU",
        "statistical_scope": "one sample2, weak observations"}
    return spec, frozen, evaluation, preprocessing


def test_source_fingerprint_ignores_targets_but_detects_training_inputs(data):
    pair = (.05, .1)
    before, ids = transport.source_arrays(data, pair)
    poisoned = deepcopy(data)
    target = poisoned["speed"] == .2
    poisoned["X"][target, 1] *= 10
    poisoned["y_fixed"][target] = 1 - poisoned["y_fixed"][target]
    poisoned["output_features"]["second_flyback_ratio"][target] = 1 - poisoned["output_features"]["second_flyback_ratio"][target]
    assert transport.source_arrays(poisoned, pair)[0] == before
    poisoned["X"][~target, 1] *= 2
    assert transport.source_arrays(poisoned, pair)[0] != before
    assert len(ids) == 20


def test_immutable_spec_rejects_code_source_membership_and_model_tampering(tmp_path, monkeypatch, data, specifications):
    spec, *main = specifications
    monkeypatch.setattr(transport, "main_specs", lambda: tuple(main))
    monkeypatch.setattr(transport, "implementation_hashes", lambda: {"code": "test-hash"})
    monkeypatch.setattr(transport.bd, "load_dataset", lambda **kwargs: data)
    path = tmp_path / "spec.json"
    def save(value):
        value["spec_sha256"] = transport.el.canonical_digest(value, "spec_sha256")
        write(path, value)
    save(spec)
    assert transport.check_spec(path)[0]["model_count"] == 12
    changed = deepcopy(spec)
    changed["sources"]["A__0.05_0.1"]["models"]["affine_flow"]["theta"][8] += .1
    write(path, changed)  # no recomputed digest
    with pytest.raises(ValueError, match="fingerprint"):
        transport.check_spec(path)
    changed = deepcopy(spec)
    changed["sources"]["A__0.05_0.1"]["source"]["ids_sha256"] = "wrong"
    save(changed)
    with pytest.raises(ValueError, match="membership/provenance"):
        transport.check_spec(path)
    save(spec)
    monkeypatch.setattr(transport, "implementation_hashes", lambda: {"code": "changed"})
    with pytest.raises(ValueError, match="implementation changed"):
        transport.check_spec(path)


def test_freeze_requires_no_decoding_and_cannot_overwrite(tmp_path, monkeypatch, data, specifications):
    spec, *main = specifications
    monkeypatch.setattr(transport, "ROOT", tmp_path)
    state_path = tmp_path / "reports/lockbox_A2_pipeline_state.json"
    write(state_path, {"phase": "waiting_for_existing_download"})
    monkeypatch.setattr(transport, "main_specs", lambda: tuple(main))
    monkeypatch.setattr(transport.bd, "load_dataset", lambda **kwargs: data)
    monkeypatch.setattr(transport, "collect_sources", lambda *args: spec["sources"])
    monkeypatch.setattr(transport, "implementation_hashes", lambda: {"code": "test-hash"})
    path = tmp_path / "spec.json"
    frozen = transport.freeze_spec(path, state_path=state_path)
    assert frozen["original_fourteen_model_protocol_unchanged"]
    with pytest.raises(FileExistsError):
        transport.freeze_spec(path, state_path=state_path)
    decoded = tmp_path / "data/extracted/TypeA_sample2/trial.csv"
    decoded.parent.mkdir(parents=True)
    decoded.write_text("decoded waveform")
    with pytest.raises(ValueError, match="decoded/evaluated"):
        transport.freeze_spec(tmp_path / "second.json", state_path=state_path)
    assert not (tmp_path / "second.json").exists()
    decoded.unlink()
    write(state_path, {"phase": "extracting_and_auditing"})
    with pytest.raises(ValueError, match="no longer waiting"):
        transport.freeze_spec(tmp_path / "second.json", state_path=state_path)


def test_development_file_and_selection_checks_fail_closed(tmp_path):
    payload = {"source_candidates": [{"lambda": .1, "warp_lambda": .1, "metrics": {"family_min_iou": .4}},
                                     {"lambda": .01, "warp_lambda": .01, "metrics": {"family_min_iou": .8}}],
               "selected_index": 1, "model": {"lambda": .01, "warp_lambda": .01}}
    assert transport.validate_selection(payload) == .8
    payload["selected_index"] = 0
    with pytest.raises(ValueError, match="source-CV"):
        transport.validate_selection(payload)
    model_path = write(tmp_path / "model.json", {"theta": [1.]})
    write(tmp_path / "protocol.json", {})
    write(tmp_path / "all_models_frozen.json", {"key": {"path": "model.json", "sha256": transport.el.digest(model_path)}})
    assert transport.frozen_run_file(tmp_path, "key")[0] == {"theta": [1.]}
    model_path.write_text("changed")
    with pytest.raises(ValueError, match="checksum/path"):
        transport.frozen_run_file(tmp_path, "key")


@pytest.fixture
def completed(tmp_path, monkeypatch, specifications, data):
    spec, frozen, evaluation, preprocessing = specifications
    receipt = {"status": "verified", "whole_archive_verified": True, "archive_filename": frozen["lockbox"]["file"],
        "record": frozen["lockbox"]["record"], "actual_md5": frozen["lockbox"]["md5"], "bytes": 123,
        "verified_utc": "2026-10-13T09:30:00+00:00"}
    directory = tmp_path / "preprocess/diagram"
    directory.mkdir(parents=True)
    csv_path = directory / "regimes.csv"
    csv_path.write_text("fixture waveform features")
    summary_path = write(directory / "summary.json", {})
    target = {"X": data["X"], "family": transport.label_family(data["output_features"], 98.),
              "nominal": data["y_fixed"] == 1, "ids": np.array([f"A2/{i}" for i in range(30)]),
              "groups": np.array([f"A2/{speed:g}" for speed in data["speed"]]),
              "beta_level": np.repeat(np.arange(5), 6), "input_csv_sha256": transport.el.digest(csv_path),
              "counts": {"raw": 30, "eligible": 30}}
    monkeypatch.setattr(transport.el, "load_target", lambda *args, **kwargs: target)
    configuration = {"archive": {key: receipt[key] for key in ("record", "archive_filename", "actual_md5", "bytes")},
        "frozen_model_sha256": frozen["freeze_sha256"], "evaluation_spec_sha256": evaluation["spec_sha256"],
        "preprocessing_spec_sha256": preprocessing["preprocessing_spec_sha256"],
        "feature_pipeline_sha256": evaluation["feature_pipeline_sha256"], "collection_sha256": "collection-hash", "workers": 2}
    binding_path = write(tmp_path / "preprocess/lockbox_provenance.json", {"status": "complete", "configuration": configuration,
        "output_sha256": {"diagram/regimes.csv": transport.el.digest(csv_path), "diagram/summary.json": transport.el.digest(summary_path)}})
    main_path = write(tmp_path / "main/manifest.json", {"status": "PASS", "frozen_model_fingerprint": frozen["freeze_sha256"],
        "evaluation_spec_sha256": evaluation["spec_sha256"], "archive_verification": receipt,
        "input_csv_sha256": target["input_csv_sha256"], "feature_summary_sha256": transport.el.digest(summary_path),
        "ids_sha256": transport.bd.id_hash(target["ids"])})
    results_path = write(tmp_path / "main/results.json", {"original_fourteen_models": "unchanged fixture"})
    state_path = write(tmp_path / "state.json", {"phase": "complete", "no_training_or_adaptation": True,
        "results": str(results_path), "preprocessing_provenance": str(binding_path)})
    return state_path, target, main_path, csv_path


def test_complete_main_provenance_and_timing_required(completed, specifications):
    state, target, main_path, csv_path = completed
    assert transport.completed_target(state, *specifications)[0] is target
    manifest = json.loads(main_path.read_text())
    manifest["archive_verification"]["verified_utc"] = "2026-10-13T08:30:00+00:00"
    write(main_path, manifest)
    with pytest.raises(ValueError, match="before the additive"):
        transport.completed_target(state, *specifications)
    manifest["archive_verification"]["verified_utc"] = "2026-10-13T09:30:00+00:00"
    write(main_path, manifest)
    csv_path.write_text("tampered")
    with pytest.raises(ValueError, match="output hash/path"):
        transport.completed_target(state, *specifications)


def test_all_twelve_predictions_without_fitting_and_preserve_main(tmp_path, monkeypatch, completed, specifications):
    state, target, main_path, _ = completed
    before = main_path.read_bytes()
    spec = specifications[0]
    spec["spec_sha256"] = "fixture-freeze"
    monkeypatch.setattr(transport, "check_spec", lambda path: specifications)
    def forbidden(*args, **kwargs):
        raise AssertionError("evaluation attempted source fitting or selection")
    monkeypatch.setattr(transport.af, "fit_source", forbidden)
    monkeypatch.setattr(transport.ms, "fit_estimator", forbidden)
    monkeypatch.setattr(transport.af, "select_source", forbidden)
    out = tmp_path / "supplementary"
    result = transport.evaluate(state, tmp_path / "unused-spec.json", out)
    assert result["model_count"] == 12 and result["no_fit_or_adaptation"]
    assert result["primary_source"] == "A__0.05_0.1"
    assert result["primary_result"]["selected_simple_baseline"] == "learned_alpha"
    predictions = np.load(out / "predictions.npz")
    for pair, source in spec["sources"].items():
        for method, model in source["models"].items():
            expected = (transport.af.predict_frozen(model, target["X"]) if method in ("affine_flow", "translation_k0")
                        else transport.ms.predict_frozen(model, target["X"]))
            np.testing.assert_array_equal(predictions[f"{pair}__{method}"], expected)
    assert main_path.read_bytes() == before
    with pytest.raises(FileExistsError):
        transport.evaluate(state, tmp_path / "unused-spec.json", out)


def test_incomplete_main_never_evaluates_or_creates_outputs(tmp_path, monkeypatch, specifications):
    monkeypatch.setattr(transport, "check_spec", lambda path: specifications)
    state = write(tmp_path / "state.json", {"phase": "extracting_and_auditing"})
    out = tmp_path / "result"
    with pytest.raises(ValueError, match="complete successfully first"):
        transport.evaluate(state, tmp_path / "spec.json", out)
    assert not out.exists()
    with pytest.raises(TimeoutError):
        transport.wait_for_main(state, 0)
    write(state, {"phase": "failed", "error": "archive corrupt"})
    with pytest.raises(RuntimeError, match="archive corrupt"):
        transport.wait_for_main(state, 100)


def test_public_embedded_models_validate_without_reports_or_pickle_checkpoints(tmp_path):
    """A fresh checkout must not rely on ignored development/model artifacts."""
    root = Path(__file__).resolve().parents[1]
    shutil.copytree(root / "scripts", tmp_path / "scripts", ignore=shutil.ignore_patterns("__pycache__"))
    shutil.copytree(root / "research/inputs", tmp_path / "research/inputs")
    destination = tmp_path / "research/2026-10-13"
    destination.mkdir()
    for name in ("transport_lockbox_spec.json", "lockbox_frozen_manifest_portable.json",
                 "lockbox_evaluation_spec.json", "lockbox_preprocessing_spec.json"):
        shutil.copyfile(root / "research/2026-10-13" / name, destination / name)
    command = ("import sys;sys.path.insert(0,'scripts');import evaluate_transport_lockbox as t;"
               "s,*_=t.check_spec();assert s['model_count']==12;assert len(s['sources'])==3;"
               "assert not t.ROOT.joinpath('reports').exists()")
    result = subprocess.run([sys.executable, "-c", command], cwd=tmp_path, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr
