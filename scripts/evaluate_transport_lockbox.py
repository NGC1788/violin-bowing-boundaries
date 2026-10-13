#!/usr/bin/env python3
"""Separate additive sample2 protocol: frozen transport models, no fitting.

This amendment is made after development results, before local A2 decoding. It
does not replace the original fourteen-model protocol or claim preregistration.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from threadpoolctl import threadpool_limits

import affine_transport as af
import benchmark_data as bd
import benchmark_sets as bs
import continue_lockbox as continuation
import evaluate_lockbox as el
import multispeed_scaling as ms
from set_envelope import label_family

ROOT = bd.ROOT
DEFAULT_SPEC = ROOT / "research/2026-10-13/transport_lockbox_spec.json"
DEFAULT_STATE = ROOT / "reports/research_20261013/lockbox_A2_pipeline_state.json"
DEFAULT_FLOW = ROOT / "reports/affine_transport/2026-10-13-v2"
DEFAULT_K0 = ROOT / "reports/affine_width_ablation/2026-10-13"
METHODS = ("affine_flow", "translation_k0", "learned_alpha", "raw_3d")
PAIRS = ((.05, .1), (.05, .2), (.1, .2))


def pair_key(pair):
    return f"A__{pair[0]:g}_{pair[1]:g}"


def resolve_path(path):
    path = Path(path)
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, allow_nan=False)
        handle.write("\n")


def main_specs():
    # The addendum needs main protocol definitions, not its fourteen pickle files.
    # Thus embedded transport coefficients remain usable on a fresh clone.
    frozen = json.loads(el.DEFAULT_FREEZE.read_text())
    if frozen.get("freeze_sha256") != el.canonical_digest(frozen, "freeze_sha256"):
        raise ValueError("main model manifest fingerprint mismatch")
    for name, digest in frozen["implementation_sha256"].items():
        if el.digest(ROOT / "scripts" / name) != digest:
            raise ValueError(f"main frozen implementation changed: {name}")
    evaluation = el.check_spec(el.DEFAULT_SPEC, frozen)
    preprocessing = continuation.check_preprocessing_spec(continuation.DEFAULT_PREPROCESS_SPEC, frozen, evaluation)
    return frozen, evaluation, preprocessing


def implementation_hashes():
    return {name: el.digest(ROOT / "scripts" / name) for name in
            (Path(__file__).name, "affine_transport.py", "multispeed_scaling.py", "affine_width_ablation.py")}


def source_arrays(data, pair):
    train = (data["string"] == "A") & np.isin(data["speed"], pair)
    if set(np.unique(data["speed"][train])) != set(pair):
        raise ValueError("source must contain exactly the specified two A speeds")
    family = label_family(data["output_features"], data["reference_f0_hz"])[train]
    nominal = data["y_fixed"][train] == 1
    if not np.array_equal(family[:, 7], nominal):
        raise ValueError("source nominal rule differs from the fixed family")
    h = hashlib.sha256()
    for array in (data["X"][train].astype("<f8"), nominal.astype("u1"), family.astype("u1")):
        h.update(json.dumps(array.shape).encode())
        h.update(np.ascontiguousarray(array).tobytes())
    return {"n": int(train.sum()), "ids_sha256": bd.id_hash(data["ids"][train]),
            "predictors_and_labels_sha256": h.hexdigest()}, data["ids"][train]


def dataset_hashes(data):
    return {f"{row['role']}/{row['string']}": row["sha256"] for row in data["manifest"]["sources"]}


def frozen_run_file(run, key):
    run = Path(run).resolve()
    manifest_path = run / "all_models_frozen.json"
    manifest = json.loads(manifest_path.read_text())
    entry = manifest[key]
    path = (run / entry["path"]).resolve()
    if not path.is_relative_to(run) or el.digest(path) != entry["sha256"]:
        raise ValueError("development frozen model checksum/path mismatch")
    return json.loads(path.read_text()), {"model_file_sha256": entry["sha256"],
        "run_manifest_sha256": el.digest(manifest_path), "run_protocol_sha256": el.digest(run / "protocol.json")}


def validate_selection(selection):
    candidates, index = selection["source_candidates"], selection["selected_index"]
    winner = max(range(len(candidates)), key=lambda i: candidates[i]["metrics"]["family_min_iou"])
    model = selection["model"]
    if index != winner or any(model[k] != candidates[index][k] for k in ("lambda", "warp_lambda")):
        raise ValueError("source-CV selection or final regularization mismatch")
    return candidates[index]["metrics"]["family_min_iou"]


def collect_sources(flow_run, k0_run, data):
    full_protocol = json.loads((flow_run / "protocol.json").read_text())
    zero_protocol = json.loads((k0_run / "protocol.json").read_text())
    old_inputs = json.loads((flow_run / "input_manifests.json").read_text())["cello"]
    if old_inputs != data["manifest"]:
        raise ValueError("development source inputs/observation definitions changed")
    if full_protocol["code_sha256"] != el.digest(Path(af.__file__)):
        raise ValueError("full-flow fitting implementation changed")
    for name, expected in zero_protocol["code_sha256"].items():
        if el.digest(ROOT / "scripts" / name) != expected:
            raise ValueError("translation fitting implementation changed")
    sources = {}
    for pair in PAIRS:
        key = "cello__" + pair_key(pair)
        full, full_provenance = frozen_run_file(flow_run, key)
        zero, zero_provenance = frozen_run_file(k0_run, key)
        fingerprint, ids = source_arrays(data, pair)
        for payload in (full, zero):
            if (payload["source"] != "A" or payload["domain"] != "cello"
                    or tuple(payload["training_speeds"]) != pair
                    or payload["training_ids_sha256"] != fingerprint["ids_sha256"]
                    or payload["training_ids"] != ids.tolist()):
                raise ValueError("source provenance or training IDs mismatch")
        if (zero["full_flow_frozen_file_sha256"] != full_provenance["model_file_sha256"]
                or zero["full_flow_model"] != full["flow"]["model"]):
            raise ValueError("matched ablation refers to another full-flow model")
        scores = {"affine_flow": validate_selection(full["flow"]),
                  "translation_k0": validate_selection(zero["translation"])}
        if zero["translation"]["model"]["theta"][8] != 0.:
            raise ValueError("matched translation model must have k exactly zero")
        models = {"affine_flow": full["flow"]["model"], "translation_k0": zero["translation"]["model"],
                  **full["baseline_models"]}
        candidates = full["baseline_source_selection"]
        for method in ("learned_alpha", "raw_3d"):
            eligible = [i for i, c in enumerate(candidates) if (c["alpha"] is None) == (method == "raw_3d")]
            best = max(eligible, key=lambda i: candidates[i]["metrics"]["family_min_iou"])
            selected = candidates[best]
            if any(models[method][k] != selected[k] for k in ("alpha", "C")):
                raise ValueError("simple baseline is not its source-CV winner")
            scores[method] = selected["metrics"]["family_min_iou"]
        # Deterministic declared tie-break: learned_alpha before raw_3d.
        baseline = max(("learned_alpha", "raw_3d"), key=scores.__getitem__)
        sources[pair_key(pair)] = {"training_speeds": pair, "source": fingerprint,
            "models": models, "source_cv_family_min_iou": scores, "selected_simple_baseline": baseline,
            "source_selection": {"flow": full["flow"]["source_candidates"],
                                 "k0": zero["translation"]["source_candidates"], "simple": candidates},
            "development_provenance": {"flow": full_provenance, "k0": zero_provenance,
                "input_manifest_sha256": el.digest(flow_run / "input_manifests.json"),
                "source_dataset_manifest_sha256": data["manifest"]["manifest_sha256"]}}
    return sources


def unseen_audit(state_path, lockbox):
    """Local non-exposure evidence, not proof about people or other computers."""
    state_path = Path(state_path).resolve()
    state = json.loads(state_path.read_text())
    if state.get("phase") != "waiting_for_existing_download":
        raise ValueError("cannot freeze: main pipeline is no longer waiting for the archive")
    archive = ROOT / "data/raw/zenodo" / str(lockbox["record"]) / lockbox["file"]
    if archive.exists() or archive.with_name(archive.name + ".preprocess.lock").exists():
        raise ValueError("cannot freeze after complete archive or preprocessing appears")
    allowed = {archive.with_name(archive.name + suffix).resolve() for suffix in
               (".part", ".part.chunks.json", ".download.lock")}
    stem = archive.stem.lower()
    findings = []
    for base in (ROOT / "data", ROOT / "reports"):
        for path in base.rglob("*"):
            if not path.is_file():
                continue
            relative = str(path.relative_to(ROOT)).lower()
            if stem in relative or "lockbox_a2" in relative:
                if path.resolve() not in allowed | {state_path}:
                    raise ValueError("potential A2 decoded/evaluated artifact exists: " + str(path.relative_to(ROOT)))
                findings.append(str(path.relative_to(ROOT)))
    if archive.exists() or json.loads(state_path.read_text()).get("phase") != "waiting_for_existing_download":
        raise ValueError("archive/pipeline state changed during the non-exposure check")
    return {"checked_utc": datetime.now(timezone.utc).isoformat(), "state_sha256": el.digest(state_path),
            "phase": state["phase"], "allowed_local_artifacts": sorted(findings),
            "scope": "Local filesystem and running pipeline state only; cannot certify human or external exposure history."}


def freeze_spec(path=DEFAULT_SPEC, *, flow_run=DEFAULT_FLOW, k0_run=DEFAULT_K0, state_path=DEFAULT_STATE):
    path = Path(path)
    if path.exists():
        raise FileExistsError("preserve the existing additive specification")
    frozen, evaluation, preprocessing = main_specs()
    data = bd.load_dataset(include_ambiguous=True)
    sources = collect_sources(Path(flow_run), Path(k0_run), data)
    evidence = unseen_audit(state_path, frozen["lockbox"])
    spec = {"created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "ADDITIVE PROSPECTIVE HOLDOUT CHECK, specified after development and before local A2 decoding; not preregistration",
        "original_fourteen_model_protocol_unchanged": True,
        "main_model_sha256": frozen["freeze_sha256"], "main_evaluation_spec_sha256": evaluation["spec_sha256"],
        "main_preprocessing_spec_sha256": preprocessing["preprocessing_spec_sha256"],
        "implementation_sha256": implementation_hashes(), "lockbox": frozen["lockbox"],
        "source_dataset_sha256": dataset_hashes(data),
        "non_exposure_evidence": evidence, "sources": sources, "model_count": 12,
        "primary_source": pair_key(PAIRS[0]), "primary_endpoint": "unweighted mean over three A2 speed panels of each panel's minimum IoU over 15 fixed coherent rules",
        "primary_comparisons": ["affine_flow minus translation_k0", "affine_flow minus source-CV-selected simple baseline"],
        "secondary_endpoints": "all other source pairs, all individual panels, nominal/Brier and globally shared-rule IoU; report all models regardless of outcome",
        "statistical_scope": "one new Type-A specimen/session, three dependent panels; no independent-replicate p-value, physical ground-truth or intact-violin claim",
        "observer": "exact existing A-referenced family and fixed preprocessing; target outputs enter labels only",
        "selection": "no target fit/adaptation/selection; source CV only; source-pair (.05,.1) fixed primary after development before decoding"}
    spec["spec_sha256"] = el.canonical_digest(spec, "spec_sha256")
    write_json(path, spec)
    return spec


def check_spec(path=DEFAULT_SPEC):
    spec = json.loads(Path(path).read_text())
    if spec.get("spec_sha256") != el.canonical_digest(spec, "spec_sha256"):
        raise ValueError("transport specification fingerprint mismatch")
    if spec["implementation_sha256"] != implementation_hashes():
        raise ValueError("transport implementation changed after freeze")
    frozen, evaluation, preprocessing = main_specs()
    if (spec["main_model_sha256"] != frozen["freeze_sha256"]
            or spec["main_evaluation_spec_sha256"] != evaluation["spec_sha256"]
            or spec["main_preprocessing_spec_sha256"] != preprocessing["preprocessing_spec_sha256"]):
        raise ValueError("transport amendment refers to another main protocol")
    if spec["lockbox"] != frozen["lockbox"] or spec["model_count"] != 12:
        raise ValueError("transport target/model count changed")
    data = bd.load_dataset(include_ambiguous=True)
    if spec["source_dataset_sha256"] != dataset_hashes(data):
        raise ValueError("source dataset digest changed")
    if set(spec["sources"]) != {pair_key(pair) for pair in PAIRS}:
        raise ValueError("exactly the three declared source pairs are required")
    for pair in PAIRS:
        source = spec["sources"][pair_key(pair)]
        fingerprint, _ = source_arrays(data, pair)
        if source["source"] != fingerprint or set(source["models"]) != set(METHODS):
            raise ValueError("embedded source model membership/provenance mismatch")
        if source["models"]["translation_k0"]["theta"][8] != 0.:
            raise ValueError("translation model no longer has k=0")
        expected = max(("learned_alpha", "raw_3d"), key=source["source_cv_family_min_iou"].__getitem__)
        if source["selected_simple_baseline"] != expected:
            raise ValueError("source-selected baseline changed")
    return spec, frozen, evaluation, preprocessing


def completed_target(state_path, spec, frozen, evaluation, preprocessing):
    state = json.loads(Path(state_path).read_text())
    if state.get("phase") != "complete" or not state.get("no_training_or_adaptation"):
        raise ValueError("main lockbox pipeline must complete successfully first")
    results_path = resolve_path(state["results"])
    main_manifest_path = results_path.with_name("manifest.json")
    main = json.loads(main_manifest_path.read_text())
    if (main.get("status") != "PASS" or main.get("frozen_model_fingerprint") != frozen["freeze_sha256"]
            or main.get("evaluation_spec_sha256") != evaluation["spec_sha256"]):
        raise ValueError("main evaluation manifest does not match the fixed protocol")
    receipt = el.validate_receipt(main["archive_verification"], frozen["lockbox"])
    if datetime.fromisoformat(receipt["verified_utc"]) <= datetime.fromisoformat(spec["created_utc"]):
        raise ValueError("archive was verified before the additive protocol freeze")
    binding_path = resolve_path(state["preprocessing_provenance"])
    binding = json.loads(binding_path.read_text())
    config = binding["configuration"]
    expected = {"archive": {key: receipt[key] for key in ("record", "archive_filename", "actual_md5", "bytes")},
        "frozen_model_sha256": frozen["freeze_sha256"], "evaluation_spec_sha256": evaluation["spec_sha256"],
        "preprocessing_spec_sha256": preprocessing["preprocessing_spec_sha256"],
        "feature_pipeline_sha256": evaluation["feature_pipeline_sha256"],
        "collection_sha256": preprocessing["implementation_sha256"]["collection.py"], "workers": 2}
    if binding.get("status") != "complete" or config != expected:
        raise ValueError("preprocessing provenance mismatch/incomplete")
    paths = {}
    for relative, expected_hash in binding["output_sha256"].items():
        path = (binding_path.parent / relative).resolve()
        if not path.is_relative_to(binding_path.parent.resolve()) or el.digest(path) != expected_hash:
            raise ValueError("preprocessing output hash/path mismatch")
        if path.name in paths:
            raise ValueError("duplicate preprocessing output")
        paths[path.name] = path
    if set(paths) != {"regimes.csv", "summary.json"} or paths["regimes.csv"].parent != paths["summary.json"].parent:
        raise ValueError("one complete regime table and summary are required")
    target = el.load_target(paths["regimes.csv"], json.loads(paths["summary.json"].read_text()),
                            frozen, evaluation, frozen["lockbox"]["file"].removesuffix(".7z"))
    if (target["input_csv_sha256"] != main["input_csv_sha256"]
            or el.digest(paths["summary.json"]) != main["feature_summary_sha256"]
            or bd.id_hash(target["ids"]) != main["ids_sha256"]):
        raise ValueError("supplementary inputs differ from the completed main evaluation")
    return target, {"main_manifest_sha256": el.digest(main_manifest_path),
        "main_results_sha256": el.digest(results_path), "preprocessing_provenance_sha256": el.digest(binding_path),
        "archive_verification": receipt, "input_csv_sha256": target["input_csv_sha256"],
        "feature_summary_sha256": el.digest(paths["summary.json"])}


def evaluate(state_path, spec_path, out):
    out = Path(out)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError("preserve the first supplementary evaluation")
    spec, frozen, evaluation, preprocessing = check_spec(spec_path)
    target, provenance = completed_target(state_path, spec, frozen, evaluation, preprocessing)
    predictions, results = {}, {}
    for key, source in spec["sources"].items():
        scores = {}
        for name, model in source["models"].items():
            probability = (af.predict_frozen(model, target["X"]) if name in ("affine_flow", "translation_k0")
                           else ms.predict_frozen(model, target["X"]))
            predictions[f"{key}__{name}"] = probability
            scores[name] = bs.evaluate_panels(probability, target["nominal"], target["family"], target["groups"],
                                              np.ones(len(probability), bool))
        endpoint = {name: row["macro_panels"]["family_min_iou"] for name, row in scores.items()}
        results[key] = {"training_speeds": source["training_speeds"], "methods": scores,
            "primary_endpoint_by_model": endpoint, "selected_simple_baseline": source["selected_simple_baseline"],
            "affine_minus_k0": endpoint["affine_flow"] - endpoint["translation_k0"],
            "affine_minus_selected_simple": endpoint["affine_flow"] - endpoint[source["selected_simple_baseline"]]}
    manifest = {"evaluated_utc": datetime.now(timezone.utc).isoformat(), "status": "PASS",
        "additive_spec_sha256": spec["spec_sha256"], "spec_created_utc": spec["created_utc"],
        "main_protocol_unchanged": True, "no_fit_or_adaptation": True, "model_count": len(predictions),
        "primary_source": spec["primary_source"], "primary_endpoint": spec["primary_endpoint"],
        "primary_result": results[spec["primary_source"]], "counts": target["counts"],
        "ids_sha256": bd.id_hash(target["ids"]), "scope": spec["statistical_scope"], **provenance}
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / "predictions.npz", ids=target["ids"], family=target["family"], nominal=target["nominal"],
                        groups=target["groups"], beta_level=target["beta_level"], **predictions)
    write_json(out / "results.json", results)
    write_json(out / "manifest.json", manifest)
    return manifest


def wait_for_main(path, seconds):
    if not 0 <= seconds <= 86400:
        raise ValueError("wait must be between zero and 86400 seconds")
    deadline = time.monotonic() + seconds
    while True:
        state = json.loads(Path(path).read_text())
        if state.get("phase") == "complete":
            return
        if state.get("phase") == "failed":
            raise RuntimeError("main pipeline failed: " + state.get("error", "unknown failure"))
        if time.monotonic() >= deadline:
            raise TimeoutError("main pipeline not complete; supplementary evaluation has not run")
        time.sleep(min(10, max(0, deadline - time.monotonic())))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, default=DEFAULT_SPEC)
    parser.add_argument("--freeze-spec", action="store_true")
    parser.add_argument("--flow-run", type=Path, default=DEFAULT_FLOW)
    parser.add_argument("--k0-run", type=Path, default=DEFAULT_K0)
    parser.add_argument("--wait-state", type=Path, default=DEFAULT_STATE)
    parser.add_argument("--wait-seconds", type=float, default=0)
    parser.add_argument("--out", type=Path, default=ROOT / "reports/research_20261013/transport_lockbox_A2")
    args = parser.parse_args(argv)
    if args.freeze_spec:
        spec = freeze_spec(args.spec, flow_run=args.flow_run, k0_run=args.k0_run, state_path=args.wait_state)
        print(json.dumps({"spec_sha256": spec["spec_sha256"], "created_utc": spec["created_utc"], "models": 12}))
        return
    check_spec(args.spec)  # fail immediately if frozen inputs changed, before waiting
    wait_for_main(args.wait_state, args.wait_seconds)
    with threadpool_limits(limits=1):
        manifest = evaluate(args.wait_state, args.spec, args.out)
    print(json.dumps({k: manifest[k] for k in ("status", "additive_spec_sha256", "primary_source", "model_count")}))


if __name__ == "__main__":
    main()
