"""The continuation wrapper must not decode/evaluate unverified or stale inputs."""
import copy
import hashlib
import json
import os
from pathlib import Path

import pytest

import continue_lockbox as runner


@pytest.fixture
def pipeline(tmp_path, monkeypatch):
    root = tmp_path / "repo"
    root.mkdir()
    monkeypatch.setattr(runner.bd, "ROOT", root)
    lockbox = {"record": "17782542", "file": "2024-04-19_TypeA_sample2.7z",
               "md5": hashlib.md5(b"verified archive fixture").hexdigest()}
    frozen = {"lockbox": lockbox, "freeze_sha256": "frozen-models"}
    spec = {"spec_sha256": "frozen-observation", "feature_pipeline_sha256": {"regime_map.py": "pinned"}}
    monkeypatch.setattr(runner.el, "check_freeze", lambda path: copy.deepcopy(frozen))
    monkeypatch.setattr(runner.el, "check_spec", lambda path, value: copy.deepcopy(spec))
    monkeypatch.setattr(runner.collection, "seven_zip", lambda: "test-7zz")
    monkeypatch.setattr(runner, "check_preprocessing_spec",
                        lambda path, frozen, spec: {"preprocessing_spec_sha256": "pinned-wrapper"})
    archive = root / "data/raw/zenodo" / lockbox["record"] / lockbox["file"]
    archive.parent.mkdir(parents=True)
    calls = []

    def process(layout, part, name, path, workers, progress):
        assert runner.el.verify_archive(path, lockbox)["whole_archive_verified"] is True
        calls.append("process")
        report = layout.reports / "2024-04-19_TypeA_sample2" / "run"
        report.mkdir(parents=True, exist_ok=True)
        (report / "regimes.csv").write_text("fixture features")
        (report / "summary.json").write_text("{}")
        return {"status": "complete", "diagrams": {"A2": {"regime_run": str(report)}}}

    def evaluate(csv, summary, freeze, specification, out, verification):
        proof = json.loads(verification.read_text())
        runner.el.validate_receipt(proof, lockbox)
        assert calls == ["process"]
        calls.append("evaluate")
        out.mkdir(parents=True, exist_ok=True)
        (out / "results.json").write_text("{}")

    monkeypatch.setattr(runner.collection, "process_archive", process)
    monkeypatch.setattr(runner.el, "evaluate", evaluate)
    return root, archive, frozen, spec, calls


@pytest.mark.parametrize("relative", [True, False])
def test_custom_output_and_verified_processing_order(pipeline, tmp_path, monkeypatch, relative):
    root, archive, frozen, spec, calls = pipeline
    archive.write_bytes(b"verified archive fixture")
    monkeypatch.chdir(root)
    out = Path("reports/custom_A2") if relative else tmp_path / "outside-repository" / "custom_A2"
    runner.main(["--out", str(out)])
    assert calls == ["process", "evaluate"]
    absolute = out.resolve()
    state = json.loads(absolute.with_name(absolute.name + "_pipeline_state.json").read_text())
    assert state["phase"] == "complete"
    binding = absolute.with_name(absolute.name + "_preprocess") / "lockbox_provenance.json"
    provenance = json.loads(binding.read_text())
    assert provenance["configuration"]["archive"]["actual_md5"] == frozen["lockbox"]["md5"]
    assert provenance["configuration"]["evaluation_spec_sha256"] == spec["spec_sha256"]
    assert len(provenance["output_sha256"]) == 2


def test_incomplete_or_corrupt_archive_never_reaches_preprocessing(pipeline):
    root, archive, _, _, calls = pipeline
    out = root / "reports/test"
    archive.with_name(archive.name + ".part").write_bytes(b"partial")
    with pytest.raises(TimeoutError):
        runner.main(["--out", str(out)])
    assert calls == []
    archive.write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="corrupt archive"):
        runner.main(["--out", str(out)])
    assert calls == []


def test_atomic_download_publication_race_is_not_reported_as_stopped(pipeline, monkeypatch):
    root, archive, _, _, calls = pipeline
    lock = archive.with_name(archive.name + ".download.lock")
    lock.write_text("pid=99999999\n")

    def publish_during_lock_check(path):
        archive.write_bytes(b"verified archive fixture")
        path.unlink()
        return True

    monkeypatch.setattr(runner.zenodo_catalog, "stale_lock", publish_during_lock_check)
    runner.main(["--out", str(root / "reports/race"), "--wait-seconds", "1"])
    assert calls == ["process", "evaluate"]


def test_existing_evaluation_preserves_its_complete_pipeline_state(pipeline):
    root, _, _, _, calls = pipeline
    out = root / "reports/finished"
    out.mkdir(parents=True)
    (out / "results.json").write_text("previous result")
    state = out.with_name(out.name + "_pipeline_state.json")
    state.write_text('{"phase":"complete"}')
    with pytest.raises(FileExistsError):
        runner.main(["--out", str(out)])
    assert state.read_text() == '{"phase":"complete"}'
    assert calls == []


def test_preprocessing_rejects_unbound_changed_or_tampered_cache(pipeline):
    root, archive, frozen, spec, _ = pipeline
    archive.write_bytes(b"verified archive fixture")
    receipt = runner.el.verify_archive(archive, frozen["lockbox"])
    out = root / "reports/provenance"
    directory = out.with_name(out.name + "_preprocess")
    directory.mkdir(parents=True)
    (directory / "unbound.json").write_text("{}")
    with pytest.raises(ValueError, match="no bound provenance"):
        runner.prepare_preprocessing(out, frozen, spec, receipt, "pinned-wrapper")
    (directory / "unbound.json").unlink()
    layout, path, binding = runner.prepare_preprocessing(out, frozen, spec, receipt, "pinned-wrapper")
    changed = copy.deepcopy(spec)
    changed["spec_sha256"] = "another evaluation"
    with pytest.raises(ValueError, match="provenance changed"):
        runner.prepare_preprocessing(out, frozen, changed, receipt, "pinned-wrapper")
    output = directory / "regimes.csv"
    output.write_text("original")
    binding["output_sha256"] = {"regimes.csv": runner.el.digest(output)}
    runner.collection.write_json(path, binding)
    runner.prepare_preprocessing(out, frozen, spec, receipt, "pinned-wrapper")
    output.write_text("tampered")
    with pytest.raises(ValueError, match="output changed"):
        runner.prepare_preprocessing(out, frozen, spec, receipt, "pinned-wrapper")


def test_live_processing_lock_blocks_parallel_extraction_and_is_preserved(tmp_path):
    lock = tmp_path / "sample.preprocess.lock"
    with runner.processing_lock(lock):
        assert lock.read_text() == f"pid={os.getpid()}\n"
        with pytest.raises(RuntimeError, match="already running"):
            with runner.processing_lock(lock):
                pytest.fail("second process must not enter")
        assert lock.exists()
    assert not lock.exists()


def test_preprocessing_freeze_checks_implementation_and_cannot_be_overwritten(tmp_path, monkeypatch):
    frozen = {"freeze_sha256": "models", "lockbox": {"record": "17782542", "file": "A2.7z", "md5": "a" * 32}}
    spec = {"spec_sha256": "observer"}
    path = tmp_path / "preprocessing.json"
    monkeypatch.setattr(runner, "preprocessing_code_hashes", lambda: {"pipeline.py": "original"})
    result = runner.freeze_preprocessing_spec(path, frozen, spec)
    assert runner.check_preprocessing_spec(path, frozen, spec) == result
    before = path.read_bytes()
    with pytest.raises(FileExistsError):
        runner.freeze_preprocessing_spec(path, frozen, spec)
    assert path.read_bytes() == before
    monkeypatch.setattr(runner, "preprocessing_code_hashes", lambda: {"pipeline.py": "changed"})
    with pytest.raises(ValueError, match="code changed"):
        runner.check_preprocessing_spec(path, frozen, spec)
    result["workers"] = 8
    path.write_text(json.dumps(result))
    with pytest.raises(ValueError, match="fingerprint mismatch"):
        runner.check_preprocessing_spec(path, frozen, spec)
