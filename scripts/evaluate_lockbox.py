#!/usr/bin/env python3
"""Evaluate frozen A models on the verified sample2 archive, without fitting anything.

The saved target `label`, target reference pitch and target noise-floor estimates
are ignored. The fixed A pitch and fifteen global rules are applied to the saved
signal features. Waveform period/flyback extraction remains the frozen pipeline.
"""
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
import regime_map
import set_envelope

DEFAULT_FREEZE = bd.ROOT / "research/2026-10-13/lockbox_frozen_manifest_portable.json"
DEFAULT_SPEC = bd.ROOT / "research/2026-10-13/lockbox_evaluation_spec.json"


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def canonical_digest(obj, field):
    body = {key:value for key,value in obj.items() if key != field}
    return hashlib.sha256(json.dumps(body, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def check_freeze(path: Path, root=bd.ROOT):
    frozen = json.loads(path.read_text())
    if frozen.get("freeze_sha256") != canonical_digest(frozen, "freeze_sha256"):
        raise ValueError("frozen manifest fingerprint mismatch")
    for name, expected in frozen["implementation_sha256"].items():
        if digest(root/"scripts"/name) != expected:
            raise ValueError(f"frozen implementation checksum mismatch: {name}")
    for name, item in frozen["all_declared_models"].items():
        modelpath = (root/item["path"]).resolve()
        if not modelpath.is_relative_to(root.resolve()):
            raise ValueError("model path escapes project root")
        if digest(modelpath) != item["sha256"]:
            raise ValueError(f"frozen model checksum mismatch: {name}")
    return frozen


def freeze_spec(freeze_path: Path, output: Path):
    if output.exists():
        raise FileExistsError("evaluation specification already exists; preserve the lock")
    frozen = check_freeze(freeze_path)
    feature_paths = [bd.ROOT/"scripts"/name for name in ("regime_map.py", "trial_audit.py", "grid_summary.py")]
    spec = {"created_utc":datetime.now(timezone.utc).isoformat(), "frozen_model_fingerprint":frozen["freeze_sha256"],
            "evaluator_sha256":digest(Path(__file__)),
            "feature_pipeline_sha256":{p.name:digest(p) for p in feature_paths},
            "feature_settings":{"flyback_frac":.5,"hysteresis_sigma":3.,"f0_search_hz":[40.,160.]},
            "sample_rate_hz":50000,"expected_raw_rows":6000,"expected_speed_panels":3,
            "target_outputs_used_only_for_labels":True,
            "predictor_inputs":["beta","force_N","velocity_m_s"],
            "original_target_label_ignored":True,
            "notes":"The fold-cycle noise estimate is per waveform and enters the fixed hysteresis procedure; it is not the diagram-level no-contact floor. Target diagram f0/noise fitting is ignored in the final observation family."}
    spec["spec_sha256"] = canonical_digest(spec, "spec_sha256")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(spec,indent=2)+"\n")
    return spec


def check_spec(path, frozen, root=bd.ROOT):
    spec=json.loads(Path(path).read_text())
    if spec.get("spec_sha256") != canonical_digest(spec,"spec_sha256"):
        raise ValueError("evaluation specification fingerprint mismatch")
    if spec["frozen_model_fingerprint"] != frozen["freeze_sha256"]:
        raise ValueError("evaluation specification refers to another model freeze")
    if spec["evaluator_sha256"] != digest(Path(__file__)):
        raise ValueError("evaluator changed after specification freeze")
    for name, expected in spec["feature_pipeline_sha256"].items():
        if digest(root/"scripts"/name) != expected:
            raise ValueError(f"feature pipeline changed: {name}")
    return spec


def validate_receipt(receipt, lockbox):
    """Metadata listing an expected checksum is not evidence of completed download."""
    if not receipt.get("whole_archive_verified") or receipt.get("status") != "verified":
        raise ValueError("whole archive verification receipt is required")
    if receipt.get("actual_md5") != lockbox["md5"] or receipt.get("archive_filename") != lockbox["file"]:
        raise ValueError("verification receipt is for another or corrupt archive")
    if str(receipt.get("record")) != str(lockbox["record"]):
        raise ValueError("verification receipt record mismatch")
    if not isinstance(receipt.get("bytes"), int) or receipt["bytes"] <= 0:
        raise ValueError("verification receipt has no positive byte count")
    return receipt


def verify_archive(archive: Path, lockbox):
    if archive.name != lockbox["file"] or not archive.is_file():
        raise ValueError("the pinned complete sample2 archive is required")
    before=archive.stat()
    md5=hashlib.md5()
    with archive.open("rb") as handle:
        for chunk in iter(lambda:handle.read(8*1024*1024),b""):
            md5.update(chunk)
    after=archive.stat()
    if (before.st_size,before.st_mtime_ns)!=(after.st_size,after.st_mtime_ns):
        raise ValueError("archive changed during checksum verification")
    receipt={"status":"verified","whole_archive_verified":True,"archive_filename":archive.name,
             "record":lockbox["record"],"actual_md5":md5.hexdigest(),"bytes":after.st_size,
             "verified_utc":datetime.now(timezone.utc).isoformat()}
    return validate_receipt(receipt,lockbox)


def load_target(csv_path: Path, summary: dict, frozen: dict, spec: dict, diagram: str, *, expected_rows=None):
    expected = frozen["lockbox"]["file"].removesuffix(".7z")
    if not (diagram == expected or diagram.startswith(expected+"__")) or "sample1" in diagram:
        raise ValueError("only the pinned sample2 diagram is eligible")
    source = str(summary.get("source_trials", ""))
    if expected not in source or "sample1" in source:
        raise ValueError("feature summary does not establish sample2 provenance")
    settings=spec["feature_settings"]
    provisional=summary.get("thresholds_provisional",{})
    for key in ("flyback_frac","hysteresis_sigma"):
        if provisional.get(key) != settings[key]:
            raise ValueError(f"signal feature setting changed: {key}")
    if summary.get("f0_search_hz") != settings["f0_search_hz"]:
        raise ValueError("signal pitch-search setting changed")
    raw=bd.read_source_bytes(csv_path)
    csv_hash=hashlib.sha256(raw).hexdigest()
    if csv_hash in bd.EXPECTED_SHA256.values():
        raise ValueError("existing sample1 table cannot be used as the lockbox")
    rows=bd.read_regimes(csv_path,content=raw)
    required=spec["expected_raw_rows"] if expected_rows is None else expected_rows
    if len(rows)!=required:
        raise ValueError("incomplete or unexpected lockbox trial count")
    conditions=sorted({r["condition"] for r in rows})
    if len(conditions)!=3 or {c.split("_")[-1] for c in conditions}!={"v1","v2","v3"}:
        raise ValueError("lockbox requires exactly the three verified speed conditions")
    if any(sum(r["condition"]==c for r in rows)!=required//3 for c in conditions):
        raise ValueError("unequal or incomplete lockbox panels")
    selected=[r for r in rows if r["window_inside_plateau"] and r["c1_window_mean"]>0]
    if not selected or len({r["condition"] for r in selected})!=3:
        raise ValueError("a lockbox speed panel has no eligible rows")
    ids=np.array([f"{diagram}/{r['condition']}/{r['trial']:04d}" for r in selected])
    if len(set(ids))!=len(ids):
        raise ValueError("duplicate lockbox trial IDs")
    X=np.array([[r["beta"],r["c1_window_mean"],r["c2_window_mean"]] for r in selected])
    features={key:np.array([r[key] for r in selected]) for key in ("periodicity","f0_hz","second_flyback_ratio","flybacks_per_period","amplitude_std")}
    family=set_envelope.label_family(features,frozen["observation_family"]["reference_f0_hz"],
                                     frozen["observation_family"]["flyback_fraction"],frozen["observation_family"]["periodicity"])
    # Reconstruct the central rule, not any target-fitted original labels.
    nominal=family[:,7]
    return {"X":X,"family":family,"nominal":nominal,"ids":ids,
            "groups":np.array([f"A2/{r['nominal_speed']:g}" for r in selected]),
            "beta_level":np.array([r["beta_level"] for r in selected]),
            "input_csv_sha256":csv_hash,"counts":{"raw":len(rows),"eligible":len(selected),
                "off_plateau":sum(not r["window_inside_plateau"] for r in rows),
                "nonpositive_force":sum(r["c1_window_mean"]<=0 for r in rows)},
            "legacy_label_disagreements":int(np.count_nonzero(nominal != np.array([r["label"]=="helmholtz" for r in selected])))}


def evaluate(csv_path, summary_path, freeze_path, spec_path, out, *, archive=None, verification=None):
    if out.exists() and any(out.iterdir()):
        raise FileExistsError("evaluation output exists; preserve the first lockbox evaluation")
    frozen=check_freeze(freeze_path)
    spec=check_spec(spec_path,frozen)
    receipt=(verify_archive(archive,frozen["lockbox"]) if archive is not None else
             validate_receipt(json.loads(verification.read_text()),frozen["lockbox"]))
    summary=json.loads(summary_path.read_text())
    diagram=frozen["lockbox"]["file"].removesuffix(".7z")
    target=load_target(csv_path,summary,frozen,spec,diagram)
    old=bd.load_dataset(include_ambiguous=True)
    if set(target["ids"]) & set(old["ids"]):
        raise ValueError("lockbox and development IDs overlap")
    out.mkdir(parents=True,exist_ok=True)
    predictions,results={},{}
    for name,item in frozen["all_declared_models"].items():
        model=joblib.load(bd.ROOT/item["path"])
        probability=model.predict_proba(target["X"])[:,1]
        predictions[name]=probability
        results[name]=bs.evaluate_panels(probability,target["nominal"],target["family"],target["groups"],np.ones(len(probability),bool))
    np.savez_compressed(out/"predictions.npz",ids=target["ids"],family=target["family"],nominal=target["nominal"],
                        groups=target["groups"],beta_level=target["beta_level"],**predictions)
    manifest={"evaluated_utc":datetime.now(timezone.utc).isoformat(),"status":"PASS",
              "frozen_model_fingerprint":frozen["freeze_sha256"],"evaluation_spec_sha256":spec["spec_sha256"],
              "archive_verification":receipt,"input_csv_sha256":target["input_csv_sha256"],
              "feature_summary_sha256":digest(summary_path),"ids_sha256":bd.id_hash(target["ids"]),
              "counts":target["counts"],"selected_simple_baseline":frozen["selected_simple_baseline"],
              "primary_comparisons":frozen["primary_comparisons"],
              "legacy_label_disagreements":target["legacy_label_disagreements"],
              "no_fit_or_adaptation":True,
              "scope":"Previously unseen public Type A sample2 only if the non-exposure history holds; outcomes are weak observation-rule labels, not physical ground truth."}
    (out/"manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
    (out/"results.json").write_text(json.dumps(results,indent=2)+"\n")
    print(json.dumps(manifest,indent=2))
    return results


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--freeze",type=Path,default=DEFAULT_FREEZE)
    p.add_argument("--spec",type=Path,default=DEFAULT_SPEC)
    p.add_argument("--freeze-spec",action="store_true",help="record evaluator/feature hashes without reading new data")
    p.add_argument("--csv",type=Path)
    p.add_argument("--summary",type=Path)
    p.add_argument("--out",type=Path)
    proof=p.add_mutually_exclusive_group();proof.add_argument("--archive",type=Path);proof.add_argument("--verification",type=Path)
    a=p.parse_args(argv)
    if a.freeze_spec:
        print(json.dumps(freeze_spec(a.freeze,a.spec),indent=2));return
    if not all((a.csv,a.summary,a.out)) or not (a.archive or a.verification):
        p.error("evaluation requires --csv --summary --out and either --archive or --verification")
    with threadpool_limits(limits=1):
        evaluate(a.csv,a.summary,a.freeze,a.spec,a.out,archive=a.archive,verification=a.verification)


if __name__=="__main__":main()
