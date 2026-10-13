#!/usr/bin/env python3
"""Rebuild missing trusted-local checkpoints from pinned source inputs/parameters.

Never replaces the public freeze or silently changes model hashes. A byte mismatch
fails closed: use the recorded environment rather than treating a refit as the
original checkpoint. No target sample or source-CV search is needed.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import tempfile

import joblib
from threadpoolctl import threadpool_limits

import benchmark_data as bd
import benchmark_sets as bs
import evaluate_lockbox as el
from set_envelope import label_family


def restore(manifest=el.DEFAULT_FREEZE, *, verify_rebuild=False):
    frozen = json.loads(Path(manifest).read_text())
    if frozen["freeze_sha256"] != el.canonical_digest(frozen, "freeze_sha256"):
        raise ValueError("frozen manifest fingerprint mismatch")
    for name, expected in frozen["implementation_sha256"].items():
        if el.digest(bd.ROOT / "scripts" / name) != expected:
            raise ValueError(f"frozen implementation changed: {name}")
    data = bd.load_dataset(include_ambiguous=True)
    train = data["train_mask"]
    if bd.id_hash(data["ids"][train]) != frozen["training_ids_sha256"]:
        raise ValueError("training membership changed")
    family = label_family(data["output_features"], data["reference_f0_hz"])
    nominal = data["y_fixed"] == 1
    specs = {item["name"]: item for item in bs.specifications()}
    verified = []
    with threadpool_limits(limits=1):
        for name, item in frozen["all_declared_models"].items():
            destination = (bd.ROOT / item["path"]).resolve()
            if not destination.is_relative_to(bd.ROOT.resolve()):
                raise ValueError("checkpoint destination escapes repository")
            if destination.exists() and el.digest(destination) != item["sha256"]:
                raise ValueError(f"existing checkpoint differs: {name}; preserving it")
            if destination.exists() and not verify_rebuild:
                verified.append(name)
                continue
            destination.parent.mkdir(parents=True, exist_ok=True)
            # The same filesystem makes publication of a verified missing model atomic.
            with tempfile.TemporaryDirectory(prefix="restore-", dir=destination.parent) as tmp:
                temporary = Path(tmp) / destination.name
                model, _ = bs.fit_model(specs[name], item["params"], data["X"][train],
                                        nominal[train], family[train])
                joblib.dump(model, temporary, compress=3)
                if el.digest(temporary) != item["sha256"]:
                    raise ValueError(f"rebuild differs from frozen bytes: {name}; no model was replaced")
                if not destination.exists():
                    temporary.replace(destination)
            verified.append(name)
            print(f"Verified exact reconstruction: {name}", flush=True)
    el.check_freeze(Path(manifest))
    return {"status": "PASS", "verified_models": len(verified),
            "frozen_manifest_unchanged": True, "no_target_access": True}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", type=Path, default=el.DEFAULT_FREEZE)
    p.add_argument("--verify-rebuild", action="store_true")
    args = p.parse_args(argv)
    print(json.dumps(restore(args.manifest, verify_rebuild=args.verify_rebuild), indent=2))


if __name__ == "__main__":
    main()
