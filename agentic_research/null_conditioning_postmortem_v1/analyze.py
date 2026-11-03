"""Decompose sealed source predictions using the original participant weights."""
import json
from pathlib import Path

import numpy as np

from ..donor_postmortem_v1.analyze import load, losses, sha

HERE = Path(__file__).resolve().parent
PARENT = HERE.parent / "null_conditioning_learning_v1"


def analyze():
    result_path = PARENT / "RESULTS.json"
    result = json.loads(result_path.read_text())
    review = json.loads((PARENT / "EXECUTION_REVIEW.json").read_text())
    seal_path = PARENT / "OUTPUT_SEAL.json"
    freeze_path = PARENT / "FREEZE.json"
    freeze = json.loads(freeze_path.read_text())
    if (
        review["verdict"] != "INDEPENDENT_REPRODUCTION_PASS_STOP"
        or review["blockers"]
        or review["results_sha256"] != sha(result_path)
        or review["freeze_sha256"] != sha(freeze_path)
        or review["output_seal_sha256"] != sha(seal_path)
    ):
        raise ValueError("Independent review does not bind the saved results")
    jobs = json.loads(seal_path.read_text())["jobs"]
    if len(jobs) != 12 or sum(map(len, jobs.values())) != 60:
        raise ValueError("Incomplete output seal")
    for job, artifacts in jobs.items():
        for name, digest in artifacts.items():
            path = PARENT / "local" / job / name
            if path.is_symlink() or sha(path) != digest:
                raise ValueError("Output seal mismatch")

    saved = load(PARENT / "local/all_predictions.npz", result["all_predictions_sha256"])
    meta = load(
        HERE.parent / "path_source_gate_v1/local/source_metadata.npz",
        freeze["inputs"]["source_metadata.npz"],
    )
    guard_path = HERE.parent / "path_source_gate_v1/local/future_labels.npz"
    guard_hash = freeze["inputs"]["future_labels.npz"]
    if guard_path.is_symlink() or sha(guard_path) != guard_hash:
        raise ValueError("Guard input seal mismatch")
    with np.load(guard_path, allow_pickle=False) as archive:
        guard = archive["guard"]
    if sha(guard_path) != guard_hash or guard.shape != (1442,) or guard.dtype != bool or (~guard).sum() != 15:
        raise ValueError("Guard input mismatch")
    truth = saved["truth"]
    people, fold = meta["people"], meta["fold"]
    _, index, counts = np.unique(people, return_inverse=True, return_counts=True)
    if truth.shape != (1442, 9) or len(counts) != 34 or not np.isin(truth, [0, 1]).all():
        raise ValueError("Unexpected source population or event matrix")
    weights = 1 / (len(counts) * counts[index])
    methods = ["null_conditioned", "random_matched", "markov"]
    scores = {method: losses(truth, saved[method]) for method in methods}
    for method, score in scores.items():
        if not np.isclose(weights @ score.mean(1), result["aggregate"][method]["log_loss"], atol=1e-12, rtol=0):
            raise ValueError("Saved macro score mismatch")

    comparisons = {}
    for baseline in methods[1:]:
        delta = scores["null_conditioned"] - scores[baseline]
        contribution = weights[:, None] * delta / truth.shape[1]
        total = float(contribution.sum())
        cells = {
            str(k): {
                str(label): (contribution * (fold[:, None] == k) * (truth == label)).sum(0).tolist()
                for label in (0, 1)
            }
            for k in range(3)
        }
        per_person = np.bincount(index, weights=delta.mean(1)) / counts
        label_gaps = {str(label): float(contribution[truth == label].sum()) for label in (0, 1)}
        fold_gaps = {str(k): float(contribution[fold == k].sum()) for k in range(3)}
        event_gaps = contribution.sum(0)
        guard_gaps = {
            "invalid": float(contribution[~guard].sum()),
            "valid_negative_events": float(contribution[guard[:, None] & (truth == 0)].sum()),
            "valid_positive_events": float(contribution[guard[:, None] & (truth == 1)].sum()),
        }
        reconstructions = [
            sum(sum(sum(events) for events in labels.values()) for labels in cells.values()),
            per_person.mean(), sum(label_gaps.values()), sum(fold_gaps.values()), event_gaps.sum(), sum(guard_gaps.values()),
        ]
        error = max(abs(float(value) - total) for value in reconstructions)
        if error > 1e-12:
            raise ValueError("Additive decomposition failed")
        comparisons[baseline] = {
            "gap_null_minus_baseline": total,
            "by_fold": fold_gaps,
            "by_event_label": label_gaps,
            "by_guard_and_event_label": guard_gaps,
            "by_event_column": event_gaps.tolist(),
            "by_fold_label_event": cells,
            "participants_higher_loss": int((per_person > 1e-12).sum()),
            "participants_lower_loss": int((per_person < -1e-12).sum()),
            "maximum_reconstruction_error": error,
        }
    return {
        "scope": "Exploratory analysis of sealed source predictions",
        "source_result_sha256": review["results_sha256"],
        "participants": len(counts),
        "rows": len(truth),
        "comparisons": comparisons,
    }


if __name__ == "__main__":
    print(json.dumps(analyze(), indent=2, allow_nan=False))
