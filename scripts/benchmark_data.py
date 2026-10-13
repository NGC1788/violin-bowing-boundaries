#!/usr/bin/env python3
"""Frozen retrospective tabular benchmark from backed-up, provisional regime labels.

Only beta, measured bow force and bow velocity enter X. Bridge-force features are
used for label diagnostics, never as predictors. Existing physics outputs used
target-derived impedance/noise/pitch, so they are historical diagnostics, not a
fair zero-shot comparator. All four specimens have already been inspected.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import io
import json
import math
from pathlib import Path

import numpy as np

import regime_map

ROOT = Path(__file__).resolve().parents[1]
FEATURE_NAMES = ("beta", "force_N", "velocity_m_s")
# Explicit runs: never select by latest.json, glob order or duplicated A simulation.
SOURCES = (
    ("A", "2024-03-25_TypeA_sample1", "20260917T135154Z_3982fcc7", "20260919T060704Z_47169eda", "20260919T062942Z_f02993f1"),
    ("B", "2024-04-10_TypeB_sample1__2024-04-10", "20260917T154209Z_aa60a05c", "20260919T070011Z_8ccdc396", "20260919T072355Z_5df18cf0"),
    ("C", "2024-04-12_TypeC_sample1__2024-04-15", "20260917T163822Z_54d11e71", "20260919T072408Z_8306725d", "20260919T074703Z_c841637f"),
    ("D", "2024-04-15_TypeD_sample1__2024-04-17", "20260917T172527Z_6dbcf479", "20260919T074717Z_f3212a15", "20260919T081004Z_2ab933f2"),
)
LABELS = frozenset(regime_map.CLASSES)
NOMINAL_SPEEDS = {"v1": 0.05, "v2": 0.1, "v3": 0.2}
TRAIN_STRING = "A"
TRAIN_SPEED = 0.1
MAX_SOURCE_BYTES = 8 * 1024 * 1024
EXPECTED_SHA256 = {
    ("A", "measured"): "53afd784fdd8bbb010c69f0b4779f3a18b4eb3ddace9514f02d51748ef64da61",
    ("A", "simulated"): "e814094b64a41be776b2a889e26d5eeef8f69596c62d81d93685926d4fb04dc7",
    ("B", "measured"): "0e002bde53b9e12951a4fcfef95c1ee43e7c4f433ea23e996f4a3badd537416f",
    ("B", "simulated"): "cce26b42dab8c82b481c013f930d29e91ca0824de616c4a5cc4163609208b6f7",
    ("C", "measured"): "386c10e4faae2a400bd6bc4153e8d0042d2390e52c783acd355bc93d2d22225e",
    ("C", "simulated"): "fcc15fb8ea5742f478d50c08e8e0d3bd8b7587b5690dbb2f0fa7e10fff39a41c",
    ("D", "measured"): "398eb11e129935edb57c0e15cd6f4a7f40fde5ecc1b37f390de341d9924ec4b7",
    ("D", "simulated"): "43912c440a96e9a50a13139c9d6297cdc7433cbf6d47755a02f2e83b9c838c57",
}


def id_hash(ids) -> str:
    """Hash stable membership, independent of row order."""
    return hashlib.sha256(("\n".join(sorted(str(x) for x in ids)) + "\n").encode()).hexdigest()


def read_source_bytes(path: Path) -> bytes:
    """Read a bounded CSV or its lossless release copy; checksum is of decoded bytes."""
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rb") as handle:
        payload = handle.read(MAX_SOURCE_BYTES + 1)
    if len(payload) > MAX_SOURCE_BYTES:
        raise ValueError("regime table exceeds fixed byte budget")
    return payload


def read_regimes(path: Path, *, content: bytes | None = None) -> list[dict]:
    """Strict parsing avoids NaN inputs, duplicate joins and truthy 'False' strings."""
    rows = []
    seen = set()
    cells = set()
    with io.StringIO((read_source_bytes(path) if content is None else content).decode("utf-8"), newline="") as handle:
        for raw in csv.DictReader(handle):
            row = dict(raw)
            for name in ("trial", "beta_level", "force_rank"):
                row[name] = int(raw[name])
            for name in ("beta", "c1_window_mean", "c2_window_mean", "amplitude_std", "periodicity", "f0_hz", "flybacks_per_period", "second_flyback_ratio"):
                row[name] = float(raw[name])
            if not all(math.isfinite(row[k]) for k in ("beta", "c1_window_mean", "c2_window_mean")):
                raise ValueError(f"non-finite physical predictor in {path.name}")
            if row["beta"] <= 0 or row["c2_window_mean"] <= 0:
                raise ValueError("beta and velocity must be positive")
            if raw["window_inside_plateau"] not in ("True", "False"):
                raise ValueError("invalid window_inside_plateau boolean")
            row["window_inside_plateau"] = raw["window_inside_plateau"] == "True"
            if row["label"] not in LABELS:
                raise ValueError(f"unknown provisional label: {row['label']}")
            key = (row["condition"], row["trial"])
            cell = (row["condition"], row["beta_level"], row["force_rank"])
            if key in seen or cell in cells:
                raise ValueError("duplicate trial or grid cell")
            seen.add(key)
            cells.add(cell)
            speed_key = row["condition"].split("_")[-1]
            if speed_key not in NOMINAL_SPEEDS:
                raise ValueError(f"unknown verified condition: {row['condition']}")
            row["nominal_speed"] = NOMINAL_SPEEDS[speed_key]
            # Declared mapping is checked against observed plateau velocities below.
            if row["window_inside_plateau"] and not math.isclose(row["c2_window_mean"], row["nominal_speed"], rel_tol=0.011):
                raise ValueError("condition speed differs from verified nominal speed")
            rows.append(row)
    return sorted(rows, key=lambda r: (r["condition"], r["trial"]))


def fixed_reference(rows: list[dict]) -> dict:
    """Estimate diagnostic label nuisance settings solely from the A/0.1 training panel.

    Include its no-contact controls before filtering F > 0; never use target panels.
    This cannot reverse historical threshold development on all specimens.
    """
    train = [r for r in rows if r["string"] == TRAIN_STRING and r["nominal_speed"] == TRAIN_SPEED]
    if not train:
        raise ValueError("training reference panel A/0.1 is absent")
    floor = regime_map.noise_floor([r["amplitude_std"] for r in train],
                                  [r["c1_window_mean"] for r in train], 3.0, 0.02)
    threshold = float(floor["min_amplitude"])
    strong = [r["f0_hz"] for r in train if r["amplitude_std"] >= threshold
              and math.isfinite(r["periodicity"]) and r["periodicity"] >= 0.8
              and math.isfinite(r["f0_hz"])]
    if not strong:
        raise ValueError("no periodic training rows for reference pitch")
    return {"training_panel": "A/0.1", "training_ids_sha256": id_hash(r["id"] for r in train),
            "noise_floor": floor, "f0_hz": float(np.median(strong)),
            "thresholds": {**regime_map.DEFAULT_THRESHOLDS, "min_amplitude": threshold}}


def relabel(row: dict, reference: dict) -> str:
    return regime_map.classify(row["amplitude_std"], row["periodicity"], row["f0_hz"],
                               row["flybacks_per_period"], reference["f0_hz"], reference["thresholds"])


def assemble_dataset(measured: list[dict], simulations: dict[str, dict], *, include_ambiguous=False) -> dict:
    """Pure assembly entry point, useful for independent checks with small fixtures."""
    ids = [r["id"] for r in measured]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate global trial IDs")
    if set(ids) != set(simulations):
        raise ValueError("measured and simulated trial IDs do not match exactly")
    reference = fixed_reference(measured)
    eligible = [r for r in measured if r["window_inside_plateau"] and r["c1_window_mean"] > 0]
    selected = [r for r in eligible if include_ambiguous or r["label"] != "ambiguous"]
    selected = sorted(selected, key=lambda r: r["id"])
    if not selected:
        raise ValueError("no eligible rows")
    original = np.array([r["label"] for r in selected])
    fixed = np.array([relabel(r, reference) for r in selected])
    legacy_sim = np.array([simulations[r["id"]]["label"] for r in selected])
    fixed_sim = np.array([relabel(simulations[r["id"]], reference) for r in selected])
    dataset = {
        "feature_names": FEATURE_NAMES,
        "X": np.array([[r["beta"], r["c1_window_mean"], r["c2_window_mean"]] for r in selected]),
        "y": np.where(original == "ambiguous", -1, (original == "helmholtz").astype(int)),
        "y_fixed": np.where(fixed == "ambiguous", -1, (fixed == "helmholtz").astype(int)),
        "ids": np.array([r["id"] for r in selected]),
        "string": np.array([r["string"] for r in selected]),
        "speed": np.array([r["nominal_speed"] for r in selected]),
        "group": np.array([f"{r['string']}/{r['nominal_speed']:g}" for r in selected]),
        "beta_level": np.array([r["beta_level"] for r in selected], dtype=int),
        "force_rank": np.array([r["force_rank"] for r in selected], dtype=int),
        "labels": original,
        "fixed_labels": fixed,
        "legacy_sim_labels": legacy_sim,
        "reclassified_sim_labels": fixed_sim,
        "legacy_sim_prediction": (legacy_sim == "helmholtz").astype(int),
        "reclassified_sim_prediction": (fixed_sim == "helmholtz").astype(int),
        "output_features": {key: np.array([r[key] for r in selected], dtype=float)
                            for key in ("periodicity", "f0_hz", "flybacks_per_period", "second_flyback_ratio", "amplitude_std")},
        "reference_f0_hz": reference["f0_hz"],
    }
    dataset["train_mask"] = (dataset["string"] == TRAIN_STRING) & (dataset["speed"] == TRAIN_SPEED)
    manifest = {
        "scope": "Retrospective development benchmark; all four specimens were previously inspected.",
        "target": "Rule-generated provisional Helmholtz label, not author-verified ground truth.",
        "feature_names": list(FEATURE_NAMES),
        "predictor_exclusions": "No measured bridge-force features, trial index, force rank, labels, target impedance, target pitch or target noise in X.",
        "counts": {"raw": len(measured), "off_plateau": sum(not r["window_inside_plateau"] for r in measured),
                   "nonpositive_force": sum(r["c1_window_mean"] <= 0 for r in measured),
                   "eligible": len(eligible), "ambiguous_eligible": sum(r["label"] == "ambiguous" for r in eligible),
                   "selected": len(selected), "train": int(dataset["train_mask"].sum())},
        "include_ambiguous": bool(include_ambiguous),
        "classifier_reference": reference,
        "simulation_scope": "Historical target-adapted diagnostic. Corrected classifier settings do not undo target-derived simulation inputs.",
        "n_label_changes_with_fixed_reference": int(np.count_nonzero(original != fixed)),
        "n_sim_label_changes_with_fixed_reference": int(np.count_nonzero(legacy_sim != fixed_sim)),
        "n_binary_sim_changes_with_fixed_reference": int(np.count_nonzero((legacy_sim == "helmholtz") != (fixed_sim == "helmholtz"))),
        "ids_sha256": id_hash(dataset["ids"]),
        "split_ids_sha256": {"train_A_0.1": id_hash(dataset["ids"][dataset["train_mask"]]),
                             "evaluation_retrospective": id_hash(dataset["ids"][~dataset["train_mask"]])},
        "groups": {g: int(np.count_nonzero(dataset["group"] == g)) for g in sorted(set(dataset["group"]))},
        "independence": "Four sample1 string specimens, three speed panels each. Rows/grid cells are not independent replications of instruments. For spatial resampling use complete beta columns; for transfer report each specimen/speed panel.",
    }
    dataset["manifest"] = manifest
    return dataset


def load_dataset(root: Path | str | None = None, *, include_ambiguous=False) -> dict:
    root = Path(root) if root is not None else ROOT
    measured, simulations, files = [], {}, []
    for string, diagram, measured_run, simulation_run, simulated_regime in SOURCES:
        paths = {
            "measured": root / "reports/collection" / diagram / "regime_map" / measured_run / "regimes.csv",
            "simulated": root / "reports/simulation" / diagram / simulation_run / "regime_map" / simulated_regime / "regimes.csv",
        }
        for role, path in paths.items():
            release = root / "research/inputs" / f"{string}_{role}_regimes.csv.gz"
            physical_path = path if path.exists() else release
            payload = read_source_bytes(physical_path)
            digest = hashlib.sha256(payload).hexdigest()
            if digest != EXPECTED_SHA256[(string, role)]:
                raise ValueError(f"frozen source checksum mismatch: {path.relative_to(root)}")
            rows = read_regimes(physical_path, content=payload)
            if len(rows) != 6000:
                raise ValueError(f"expected 6000 {role} rows for {string}, found {len(rows)}")
            files.append({"role": role, "string": string, "path": path.relative_to(root).as_posix(),
                          "sha256": digest, "rows": len(rows),
                          "release_copy": release.relative_to(root).as_posix()})
            for row in rows:
                row.update(string=string, id=f"{diagram}/{row['condition']}/{row['trial']:04d}")
                if role == "measured":
                    measured.append(row)
                else:
                    simulations[row["id"]] = row
    dataset = assemble_dataset(measured, simulations, include_ambiguous=include_ambiguous)
    dataset["manifest"]["sources"] = files
    # This fingerprints immutable inputs and configuration, not a claim of preregistration.
    payload = json.dumps(dataset["manifest"], sort_keys=True, separators=(",", ":"), allow_nan=False)
    dataset["manifest"]["manifest_sha256"] = hashlib.sha256(payload.encode()).hexdigest()
    return dataset


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--include-ambiguous", action="store_true")
    args = parser.parse_args(argv)
    data = load_dataset(args.root, include_ambiguous=args.include_ambiguous)
    output = json.dumps(data["manifest"], ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(output, encoding="utf-8")
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
