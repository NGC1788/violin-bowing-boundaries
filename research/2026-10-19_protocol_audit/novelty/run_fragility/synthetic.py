"""Fixed, bounded synthetic behavior check; no empirical data are used."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np

from core import first_run_costs, witness_first_run
from global_profile import optimize_intercept, optimize_incompatibility


ROOT = Path(__file__).resolve().parent
SEED = 20261021
K = 2
EPSILON = 0.2
MODES = ("both", "delete", "add")
SLOPES = (-2.0, -1.2)


def scan_start(labels):
    """Direct operational evaluator, independent of DP internals."""
    for s in range(len(labels) - K + 1):
        if all(labels[s:s + K]):
            return s
    return None


def finite_start(s, forces):
    return s is not None and 1 <= s <= len(forces) - K and 0 < forces[s - 1] < forces[s]


def interval(s, forces, beta, slope):
    return (
        math.log(forces[s - 1]) - slope * math.log(beta) - EPSILON,
        math.log(forces[s]) - slope * math.log(beta) + EPSILON,
    )


def observed_compatibility(labels, forces, beta, slope, selected):
    bounds = []
    for j in selected:
        s = scan_start(labels[j])
        assert finite_start(s, forces)
        bounds.append(interval(s, forces, beta[j], slope))
    lo = max(x[0] for x in bounds)
    hi = min(x[1] for x in bounds)
    return lo <= hi, [lo, hi]


def build_options(labels, forces, beta, slope, mode, selected, support_only=False):
    columns = []
    for j in selected:
        table = first_run_costs(labels[j], K, mode)
        choices = []
        for s in range(1, len(table["costs"])):
            if not finite_start(s, forces):
                continue
            lo, hi = interval(s, forces, beta[j], slope)
            choices.append({"lo": lo, "hi": hi, "cost": 0 if support_only else float(table["costs"][s]), "start": s})
        columns.append(choices)
    return columns


def verified_result(result, options, labels, forces, beta, slope, mode, selected, target):
    """Independently apply the optimizer's concrete edit certificates."""
    if not result["feasible"]:
        return {"feasible": False, "cost": None, "reason": result["reason"]}
    edited = labels.copy()
    flips = []
    requested = []
    for local_j, choice in enumerate(result["choices"]):
        j = selected[local_j]
        s = options[local_j][choice]["start"]
        witness = witness_first_run(labels[j], s, K, mode)
        assert witness["feasible"]
        edited[j] = witness["labels"]
        assert scan_start(edited[j]) == s
        requested.append(int(s))
        for i in np.flatnonzero(edited[j] != labels[j]):
            source, dest = int(labels[j, i]), int(edited[j, i])
            if mode == "delete":
                assert source == 1 and dest == 0
            if mode == "add":
                assert source == 0 and dest == 1
            flips.append({"column": int(j), "index": int(i), "from": source, "to": dest})
    actual = int(np.count_nonzero(edited != labels))
    assert actual == result["cost"] == len(flips)
    verdict, intersection = observed_compatibility(edited, forces, beta, slope, selected)
    assert verdict is target
    return {
        "feasible": True, "cost": actual, "first_starts": requested,
        "edited_columns": len({x["column"] for x in flips}),
        "additions": sum(x["to"] == 1 for x in flips),
        "deletions": sum(x["to"] == 0 for x in flips),
        "flips": flips, "intersection": intersection,
        "verified_verdict": "compatible" if verdict else "excluded",
    }


def law_analysis(labels, forces, beta, slope, mode, selected):
    original, intersection = observed_compatibility(labels, forces, beta, slope, selected)
    options = build_options(labels, forces, beta, slope, mode, selected)
    restore = verified_result(
        optimize_intercept(options), options, labels, forces, beta, slope, mode, selected, True,
    )
    exclude = verified_result(
        optimize_incompatibility(options), options, labels, forces, beta, slope, mode, selected, False,
    )
    support_options = build_options(labels, forces, beta, slope, "both", selected, True)
    support = optimize_intercept(support_options)["feasible"]
    if not restore["feasible"]:
        restore["status"] = "grid_support_infeasible" if not support else "direction_constraints_infeasible"
    else:
        restore["status"] = "attainable"
    if not exclude["feasible"]:
        exclude["status"] = "no_admissible_exclusion"
    else:
        exclude["status"] = "attainable"
    unchanged = restore if original else exclude
    assert unchanged["feasible"] and unchanged["cost"] == 0
    reversal = exclude if original else restore
    assert not reversal["feasible"] or reversal["cost"] > 0
    return {
        "slope": slope, "mode": mode, "original_compatible": original,
        "original_intersection": intersection, "grid_support_feasible": support,
        "restoration": restore, "exclusion": exclude,
        "decision_reversal_direction": "exclude" if original else "restore",
        "decision_reversal_cost": reversal["cost"],
        "decision_reversal_status": reversal["status"],
    }


def clean_boundary_restoration(labels, clean, mode):
    total = 0
    for j, source in enumerate(labels):
        w = witness_first_run(source, scan_start(clean[j]), K, mode)
        if not w["feasible"]:
            return None
        assert scan_start(w["labels"]) == scan_start(clean[j])
        total += w["cost"]
    return total


def perturbations(clean, rng):
    yield "clean", 0, clean.copy()
    for repeat in range(4):
        noisy = clean.copy()
        indices = rng.choice(clean.size, 3, replace=False)
        noisy.flat[indices] = 1 - noisy.flat[indices]
        yield "random_sparse", repeat, noisy
    for repeat in range(4):
        noisy = clean.copy()
        for j in rng.choice(len(clean), 2, replace=False):
            s = scan_start(clean[j])
            assert clean[j, s] == 1
            noisy[j, s] = 0
        yield "boundary_adjacent", repeat, noisy
    for repeat in range(4):
        noisy = clean.copy()
        j = int(rng.integers(len(clean)))
        s = scan_start(clean[j])
        assert np.all(clean[j, s:s + 4] == 1)
        noisy[j, s:s + 4] = 0
        yield "boundary_burst", repeat, noisy


def summarize(cases):
    out = {}
    for kind in ("clean", "random_sparse", "boundary_adjacent", "boundary_burst"):
        group = [c for c in cases if c["perturbation"] == kind]
        correct = [c["laws"][str(c["generating_slope"])]["both"] for c in group]
        alternative = [c["laws"][str(next(p for p in SLOPES if p != c["generating_slope"]))]["both"] for c in group]
        def values(seq):
            return {"min": min(seq), "max": max(seq), "mean": float(np.mean(seq))}
        out[kind] = {
            "cases": len(group), "injected_label_changes": values([c["injected_label_changes"] for c in group]),
            "clean_boundary_restoration": values([c["clean_boundary_restoration"]["both"] for c in group]),
            "generating_law_currently_compatible": sum(r["original_compatible"] for r in correct),
            "generating_law_restoration": values([r["restoration"]["cost"] for r in correct]),
            "generating_law_reversal": values([r["decision_reversal_cost"] for r in correct]),
            "alternative_law_currently_compatible": sum(r["original_compatible"] for r in alternative),
            "alternative_law_restoration": values([r["restoration"]["cost"] for r in alternative]),
            "alternative_law_reversal": values([r["decision_reversal_cost"] for r in alternative]),
        }
    return out


def main():
    rng = np.random.default_rng(SEED)
    beta = np.geomspace(0.025, 0.2, 6)
    forces = np.geomspace(0.01, 20.0, 48)
    cases = []
    clean_conditions = []
    witnesses = 0
    for slope in SLOPES:
        lower = 0.4 * (beta / 0.1) ** slope
        clean = ((forces[None, :] >= lower[:, None]) & (forces[None, :] <= 4 * lower[:, None])).astype(np.uint8)
        clean_conditions.append({"generating_slope": slope, "true_lower_force": lower.tolist(), "labels": clean.tolist(), "first_starts": [scan_start(x) for x in clean]})
        assert all(finite_start(scan_start(x), forces) for x in clean)
        for kind, repeat, noisy in perturbations(clean, rng):
            starts = [scan_start(x) for x in noisy]
            selected = [j for j, s in enumerate(starts) if finite_start(s, forces)]
            # Same fixed eligibility rule as the root plan; no replacement
            # of a noise realization if it makes an endpoint censored.
            assert len(selected) >= 2
            changed = int(np.count_nonzero(clean != noisy))
            restorations = {mode: clean_boundary_restoration(noisy, clean, mode) for mode in MODES}
            assert restorations["both"] <= changed
            laws = {}
            for target in SLOPES:
                laws[str(target)] = {}
                for mode in MODES:
                    r = law_analysis(noisy, forces, beta, target, mode, selected)
                    laws[str(target)][mode] = r
                    witnesses += r["restoration"]["feasible"] + r["exclusion"]["feasible"]
            truth = laws[str(slope)]["both"]
            assert truth["restoration"]["cost"] <= restorations["both"]
            if kind == "clean":
                assert truth["original_compatible"]
                assert not laws[str(next(p for p in SLOPES if p != slope))]["both"]["original_compatible"]
            cases.append({
                "id": f"p{slope:g}_{kind}_{repeat}", "generating_slope": slope,
                "perturbation": kind, "repeat": repeat, "labels": noisy.tolist(),
                "first_starts": starts, "selected_columns": selected,
                "injected_label_changes": changed,
                "injected_flips": [{"column": int(j), "index": int(i), "from": int(clean[j, i]), "to": int(noisy[j, i])} for j, i in np.argwhere(clean != noisy)],
                "clean_boundary_restoration": restorations, "laws": laws,
            })
    narrow_forces = np.geomspace(0.4, 0.6, 12)
    narrow_labels = np.tile([0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0], (6, 1)).astype(np.uint8)
    narrow = {str(p): law_analysis(narrow_labels, narrow_forces, beta, p, "both", list(range(6))) for p in SLOPES}
    assert all(v["restoration"]["status"] == "grid_support_infeasible" for v in narrow.values())
    assert all(v["decision_reversal_cost"] is None for v in narrow.values())
    fixed_plan = (ROOT / "SYNTHETIC.md").read_text().split("## Results\n", 1)[0]
    result = {
        "seed": SEED, "k": K, "epsilon": EPSILON,
        "beta": beta.tolist(), "forces": forces.tolist(), "clean_conditions": clean_conditions,
        "cases": cases, "summary": summarize(cases),
        "checks": {"cases": len(cases), "law_mode_comparisons": len(cases) * len(SLOPES) * len(MODES), "verified_global_witnesses": witnesses, "all_clean_generating_laws_compatible": True, "all_original_boundary_restoration_costs_bounded_by_injected_changes": True},
        "support_example": {"forces": narrow_forces.tolist(), "labels": narrow_labels.tolist(), "laws": narrow},
        "source_sha256": {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in ("core.py", "global_profile.py", "synthetic.py")},
        "fixed_plan_sha256": hashlib.sha256(fixed_plan.encode()).hexdigest(),
        "scope": "Operational label and interval compatibility validation; no empirical or causal truth claim.",
    }
    (ROOT / "synthetic_results.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"checks": result["checks"], "summary": result["summary"], "support": {p: x["restoration"]["status"] for p, x in narrow.items()}}, indent=2))


if __name__ == "__main__":
    main()
