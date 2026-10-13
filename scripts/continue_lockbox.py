#!/usr/bin/env python3
"""Finish the pinned sample2 evaluation after an already-running download.

No training, downloading, parameter selection, or archive deletion occurs here.
The existing collection pipeline removes only its disposable extracted CSVs after
validation/caching. Progress and failure are saved for an unattended current run.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import time

from threadpoolctl import threadpool_limits

import benchmark_data as bd
import collection
import evaluate_lockbox as el
import zenodo_catalog

DEFAULT_PREPROCESS_SPEC = bd.ROOT / "research/2026-10-13/lockbox_preprocessing_spec.json"


def preprocessing_code_hashes():
    return {Path(module.__file__).name: el.digest(Path(module.__file__))
            for module in (collection, zenodo_catalog)} | {Path(__file__).name: el.digest(Path(__file__))}


def freeze_preprocessing_spec(path, frozen, evaluation_spec):
    if path.exists():
        raise FileExistsError("preprocessing specification already exists; preserve the lock")
    specification = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "frozen_model_sha256": frozen["freeze_sha256"],
        "evaluation_spec_sha256": evaluation_spec["spec_sha256"],
        "lockbox": {key: frozen["lockbox"][key] for key in ("record", "file", "md5")},
        "implementation_sha256": preprocessing_code_hashes(),
        "workers": 2,
        "requires_whole_archive_md5_before_decoding": True,
        "requires_bound_per_output_preprocessing_cache": True,
        "does_not_change_models_or_observation_family": True,
    }
    specification["preprocessing_spec_sha256"] = el.canonical_digest(specification, "preprocessing_spec_sha256")
    # Exclusive creation also prevents two freeze commands from replacing one another.
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as handle:
        json.dump(specification, handle, indent=2)
        handle.write("\n")
    return specification


def check_preprocessing_spec(path, frozen, evaluation_spec):
    specification = json.loads(path.read_text())
    if specification.get("preprocessing_spec_sha256") != el.canonical_digest(specification, "preprocessing_spec_sha256"):
        raise ValueError("preprocessing specification fingerprint mismatch")
    if (specification["frozen_model_sha256"] != frozen["freeze_sha256"]
            or specification["evaluation_spec_sha256"] != evaluation_spec["spec_sha256"]):
        raise ValueError("preprocessing specification refers to another evaluation freeze")
    if specification["lockbox"] != {key: frozen["lockbox"][key] for key in ("record", "file", "md5")}:
        raise ValueError("preprocessing specification refers to another archive")
    if specification["implementation_sha256"] != preprocessing_code_hashes():
        raise ValueError("preprocessing code changed after specification freeze")
    return specification


def display_path(path):
    """Use portable repository paths when possible, including for relative --out."""
    path = Path(path).resolve()
    return str(path.relative_to(bd.ROOT.resolve())) if path.is_relative_to(bd.ROOT.resolve()) else str(path)


@contextmanager
def processing_lock(path):
    """Do not let two wrappers extract/delete the same archive's staging folders."""
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        handle = path.open("x", encoding="utf-8")
    except FileExistsError:
        if not zenodo_catalog.stale_lock(path):
            raise RuntimeError("sample2 preprocessing is already running; preserve its lock")
        path.unlink()
        handle = path.open("x", encoding="utf-8")
    try:
        with handle:
            handle.write(f"pid={os.getpid()}\n")
        yield
    finally:
        path.unlink(missing_ok=True)


def prepare_preprocessing(out, frozen, spec, receipt, preprocessing_sha256):
    """Only resume caches bound to the verified archive and current frozen pipeline."""
    layout = collection.Layout(bd.ROOT / "data", out.with_name(out.name + "_preprocess"), {})
    path = layout.reports / "lockbox_provenance.json"
    configuration = {
        "archive": {key: receipt[key] for key in ("record", "archive_filename", "actual_md5", "bytes")},
        "frozen_model_sha256": frozen["freeze_sha256"],
        "evaluation_spec_sha256": spec["spec_sha256"],
        "preprocessing_spec_sha256": preprocessing_sha256,
        "feature_pipeline_sha256": spec["feature_pipeline_sha256"],
        "collection_sha256": el.digest(Path(collection.__file__)),
        "workers": 2,
    }
    if path.exists():
        binding = json.loads(path.read_text())
        if binding.get("configuration") != configuration:
            raise ValueError("preprocessing provenance changed; preserve the cache and choose a new --out")
        for name, expected in binding.get("output_sha256", {}).items():
            target = (layout.reports / name).resolve()
            if not target.is_relative_to(layout.reports.resolve()) or el.digest(target) != expected:
                raise ValueError("cached preprocessing output changed; refusing unverified reuse")
    else:
        if layout.reports.exists() and any(layout.reports.iterdir()):
            raise ValueError("nonempty preprocessing cache has no bound provenance; choose a new --out")
        binding = {"configuration": configuration, "status": "in_progress"}
        collection.write_json(path, binding)
    return layout, path, binding


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wait-seconds", type=float, default=0)
    parser.add_argument("--out", type=Path, default=bd.ROOT / "reports/research_20261013/lockbox_A2")
    parser.add_argument("--preprocessing-spec", type=Path, default=DEFAULT_PREPROCESS_SPEC)
    parser.add_argument("--freeze-preprocessing-spec", action="store_true",
                        help="pin wrapper/extraction code without opening target waveforms")
    args = parser.parse_args(argv)
    if not 0 <= args.wait_seconds <= 86400:
        parser.error("wait must be between 0 and 86400 seconds")
    if args.freeze_preprocessing_spec:
        frozen = el.check_freeze(el.DEFAULT_FREEZE)
        spec = el.check_spec(el.DEFAULT_SPEC, frozen)
        print(json.dumps(freeze_preprocessing_spec(args.preprocessing_spec, frozen, spec), indent=2))
        return
    args.out = args.out.resolve()
    # A repeated command must not turn an already-complete state's status to failed.
    if args.out.exists() and any(args.out.iterdir()):
        raise FileExistsError("preserve the existing lockbox result; choose an empty output directory")
    state_path = args.out.with_name(args.out.name + "_pipeline_state.json")

    def state(phase, **extra):
        payload = {"phase": phase, "updated_utc": datetime.now(timezone.utc).isoformat(), **extra}
        collection.write_json(state_path, payload)
        print(json.dumps(payload, ensure_ascii=False), flush=True)

    try:
        frozen = el.check_freeze(el.DEFAULT_FREEZE)
        spec = el.check_spec(el.DEFAULT_SPEC, frozen)
        preprocessing = check_preprocessing_spec(args.preprocessing_spec, frozen, spec)
        collection.seven_zip()  # Fail early, before waiting for a large download.
        lockbox = frozen["lockbox"]
        archive = bd.ROOT / "data/raw/zenodo" / str(lockbox["record"]) / lockbox["file"]
        download_lock = archive.with_name(archive.name + ".download.lock")
        state("waiting_for_existing_download", archive=display_path(archive),
              wait_seconds=args.wait_seconds, frozen_model_sha256=frozen["freeze_sha256"],
              preprocessing_spec_sha256=preprocessing["preprocessing_spec_sha256"])
        deadline = time.monotonic() + args.wait_seconds
        while not archive.is_file():
            if time.monotonic() >= deadline:
                raise TimeoutError("complete archive is absent; resume its download and rerun")
            if not download_lock.is_file() or zenodo_catalog.stale_lock(download_lock):
                # The downloader publishes atomically and removes its lock: both
                # can happen between the loop condition and the lock inspection.
                if archive.is_file():
                    break
                raise RuntimeError("download stopped before completion; its partial file remains resumable")
            time.sleep(min(10, max(0, deadline - time.monotonic())))
        state("verifying_complete_archive")
        receipt = el.verify_archive(archive, lockbox)
        receipt_path = args.out.with_name(args.out.name + "_archive_verification.json")
        collection.write_json(receipt_path, receipt)
        state("extracting_and_auditing", archive_verification=display_path(receipt_path))
        with processing_lock(archive.with_name(archive.name + ".preprocess.lock")), threadpool_limits(limits=1):
            layout, binding_path, binding = prepare_preprocessing(
                args.out, frozen, spec, receipt, preprocessing["preprocessing_spec_sha256"])
            result = collection.process_archive(layout, 2, lockbox["file"], archive, workers=2,
                                                progress=lambda x: print(x, flush=True))
            if result["status"] != "complete" or len(result["diagrams"]) != 1:
                raise RuntimeError("sample2 preprocessing did not complete exactly one diagram")
            diagram = next(iter(result["diagrams"].values()))
            directory = Path(diagram["regime_run"]).resolve()
            if not directory.is_relative_to(layout.reports.resolve()):
                raise ValueError("preprocessing returned a report outside its bound output directory")
            binding.update(status="complete", output_sha256={
                str((directory / name).relative_to(layout.reports)): el.digest(directory / name)
                for name in ("regimes.csv", "summary.json")})
            collection.write_json(binding_path, binding)
            state("evaluating_frozen_models", regime_directory=display_path(directory),
                  preprocessing_provenance=display_path(binding_path))
            el.evaluate(directory / "regimes.csv", directory / "summary.json", el.DEFAULT_FREEZE,
                        el.DEFAULT_SPEC, args.out, verification=receipt_path)
        state("complete", results=display_path(args.out / "results.json"),
              preprocessing_provenance=display_path(binding_path), no_training_or_adaptation=True)
    except Exception as error:
        state("failed", error=f"{type(error).__name__}: {error}")
        raise


if __name__ == "__main__":
    main()
