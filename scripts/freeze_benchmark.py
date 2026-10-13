#!/usr/bin/env python3
"""Freeze A-source models and evaluation hashes before decoding the sample2 lockbox."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
from threadpoolctl import threadpool_limits

import benchmark_data as bd
import benchmark_sets as bs
import set_envelope


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def freeze(benchmark: Path, out: Path, public_manifest: Path):
    if (out.exists() and any(out.iterdir())) or public_manifest.exists():
        raise FileExistsError("freeze outputs already exist; never silently replace a lockbox freeze")
    benchmark_manifest = json.loads((benchmark/"manifest.json").read_text())
    # Verify the exact code which generated the source CV before re-fitting.
    for name, digest in benchmark_manifest["script_sha256"].items():
        if sha(Path(__file__).with_name(name)) != digest:
            raise ValueError(f"benchmark implementation changed before freeze: {name}")
    if benchmark_manifest["source"] != "A/0.1":
        raise ValueError("the prospective protocol uses source A/0.1")
    selection = json.loads((benchmark/"selection.json").read_text())
    data = bd.load_dataset(include_ambiguous=True)
    mask = data["train_mask"]
    family = set_envelope.label_family(data["output_features"], data["reference_f0_hz"])
    nominal = data["y_fixed"] == 1
    specs = bs.specifications()
    standards = [s for s in specs if s["kind"] != "set"]
    # This decision uses SOURCE cross-validation only, not retrospective transfer scores.
    baseline = max(standards, key=lambda s: selection[s["name"]]["cv"][selection[s["name"]]["primary_index"]]["metrics"]["family_min_iou"])["name"]
    out.mkdir(parents=True, exist_ok=True)
    old_predictions = np.load(benchmark/"predictions.npz", allow_pickle=False)
    np.testing.assert_array_equal(old_predictions["ids"], data["ids"])
    models = {}
    for spec in specs:
        picked = selection[spec["name"]]
        item = picked["cv"][picked["primary_index"]]
        model, diagnostic = bs.fit_model(spec, item["params"], data["X"][mask], nominal[mask], family[mask])
        predicted = model.predict_proba(data["X"])[:, 1]
        error = float(np.max(np.abs(predicted-old_predictions[spec["name"]])))
        if error > 1e-10:
            raise ValueError(f"frozen re-fit does not reproduce saved predictions: {spec['name']} error={error}")
        destination = out / f"{spec['name']}.joblib"
        joblib.dump(model, destination, compress=3)
        # Immediate trusted-local checkpoint round trip; never unpickle external artifacts.
        reloaded = joblib.load(destination)
        np.testing.assert_allclose(reloaded.predict_proba(data["X"][:20]), model.predict_proba(data["X"][:20]), rtol=0, atol=1e-14)
        models[spec["name"]] = {"params": item["params"], "source_cv_family_min_iou": item["metrics"]["family_min_iou"],
                                "path": destination.relative_to(bd.ROOT).as_posix(), "sha256": sha(destination),
                                "reproduction_max_abs_error": error, "fit_diagnostics": diagnostic}
    manifest = {
        "frozen_utc": datetime.now(timezone.utc).isoformat(),
        "status": "frozen_before_sample2_waveform_decode",
        "scope": "Source A/0.1 model states; retrospective model development, prospective sample2 evaluation only after external non-exposure check.",
        "selected_simple_baseline": baseline,
        "selection_rule": "highest pooled source-only blocked-CV coherent-family minimum IoU among declared standard models; no target ranking used",
        "primary_comparisons": [baseline, "covariant_family_robust", "covariant_box_robust", "covariant_nominal_iou", "covariant_nominal_bce", "covariant_consensus_bce"],
        "all_declared_models": models,
        "observation_family": {"flyback_fraction": [.3,.4,.5,.6,.7], "periodicity": [.75,.8,.85], "subharmonic_ratio": .75,
                               "reference_f0_hz": data["reference_f0_hz"], "thresholds": data["manifest"]["classifier_reference"]["thresholds"]},
        "eligibility": "positive mean measured bow force and entire analysis window within 1%-tolerance velocity plateau; retain ambiguous cells in family; no output-dependent row exclusions",
        "evaluation": "Predict only beta/F/v. Report nominal and coherent-family min/mean/max IoU per speed and macro; same-rule global IoU and rectangular envelope bounds. Never tune the family or models after decoding.",
        "lockbox": {"record": "17782542", "file": "2024-04-19_TypeA_sample2.7z", "md5": "ec520fecf6c5011ef8491b335edc1cee",
                    "caveat": "Type A repeat specimen/setup only; withdraw non-exposure claim if earlier unrecorded usage is discovered."},
        "training_ids_sha256": bd.id_hash(data["ids"][mask]), "data_manifest_sha256": data["manifest"]["manifest_sha256"],
        "source_cv_selection_sha256": sha(benchmark/"selection.json"),
        "benchmark_manifest_sha256": sha(benchmark/"manifest.json"),
        "implementation_sha256": {p.name: sha(p) for p in (Path(bs.__file__), Path(bd.__file__), Path(set_envelope.__file__), Path(__file__))},
        "local_artifact_notice": "Joblib files are trusted-local Python checkpoints. Source paths are project-relative; no raw waveforms or private machine paths are included.",
    }
    manifest["freeze_sha256"] = hashlib.sha256(json.dumps(manifest,sort_keys=True,separators=(",",":"),allow_nan=False).encode()).hexdigest()
    payload = json.dumps(manifest, indent=2, allow_nan=False)+"\n"
    (out/"frozen_manifest.json").write_text(payload)
    public_manifest.parent.mkdir(parents=True, exist_ok=True)
    public_manifest.write_text(payload)
    print(json.dumps({"status":"PASS","baseline":baseline,"models":len(models),"freeze_sha256":manifest["freeze_sha256"]},indent=2))
    return manifest


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark",type=Path,default=bd.ROOT/"reports/research_20261013/benchmark_A")
    parser.add_argument("--out",type=Path,default=bd.ROOT/"reports/research_20261013/frozen_A")
    parser.add_argument("--manifest",type=Path,default=bd.ROOT/"research/2026-10-13/lockbox_frozen_manifest.json")
    args=parser.parse_args(argv)
    with threadpool_limits(limits=1): freeze(args.benchmark,args.out,args.manifest)


if __name__=="__main__": main()
