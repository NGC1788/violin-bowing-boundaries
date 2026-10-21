"""Deterministic join of all prior minimum obstructions to exhaustive MLE refits.

No optimizer, feature processing, labels, or adaptive classifier search. The
family extension is explicitly post hoc and all ranks are descriptive.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

TIE_ATOL = 1e-9
METRICS = ("abs_delta_beta", "toward_target", "T_reduction")


def read(path):
    return json.loads(path.read_text())


def rank(values, target):
    values = np.asarray(values)
    tied = np.abs(values - target) <= TIE_ATOL
    below = (values < target) & ~tied
    above = (values > target) & ~tied
    assert np.all(tied | below | above)
    return {"below": int(below.sum()), "tied": int(tied.sum()),
            "above": int(above.sum()), "n": len(values),
            "midrank": float((below.sum() + .5 * tied.sum()) / len(values))}


def describe(values):
    values = np.asarray(values, dtype=float)
    assert values.size and np.isfinite(values).all()
    return {"minimum": float(np.min(values)), "median": float(np.median(values)),
            "maximum": float(np.max(values))}


def best_member(members, metric):
    # Fixed tie tolerance; IDs supply the deterministic requested tie-break.
    best = max(m[metric] for m in members)
    tied = [m for m in members if abs(m[metric] - best) <= TIE_ATOL]
    return min(tied, key=lambda m: tuple(m["ids"]))


def aggregate(cases):
    result = {"n_cases": len(cases), "n_certificates": sum(c["family_size"] for c in cases)}
    if not cases:
        return result
    result.update({
        "family_size": describe([c["family_size"] for c in cases]),
        "cases_with_some_away_from_target": sum(c["counts"]["away_from_target"] > 0 for c in cases),
        "cases_with_all_away_from_target": sum(c["counts"]["away_from_target"] == c["family_size"] for c in cases),
        "cases_with_both_toward_directions": sum(c["counts"]["away_from_target"] > 0 and c["counts"]["toward_target"] > 0 for c in cases),
        "cases_with_some_T_increase": sum(c["counts"]["T_increase"] > 0 for c in cases),
        "cases_with_all_T_increase": sum(c["counts"]["T_increase"] == c["family_size"] for c in cases),
        "cases_with_both_T_directions": sum(c["counts"]["T_increase"] > 0 and c["counts"]["T_decrease"] > 0 for c in cases),
        "member_direction_counts": {k: sum(c["counts"][k] for c in cases) for k in
                                    ("away_from_target", "toward_target", "T_increase", "T_decrease")},
        "cases_best_T_reduction_positive": sum(c["best_for_T"]["T_reduction"] > TIE_ATOL for c in cases),
        "cases_best_T_matches_unrestricted_control_max": sum(c["best_T_gap_to_all_control_max"] <= TIE_ATOL for c in cases),
        "cases_lex_choice_matches_best_T": sum(abs(c["best_for_T"]["T_reduction"] - c["previous_choice"]["T_reduction"]) <= TIE_ATOL for c in cases),
        "best_T_improvement_over_previous": describe([c["best_for_T"]["T_reduction"] - c["previous_choice"]["T_reduction"] for c in cases]),
        "best_T_gap_to_all_control_max": describe([c["best_T_gap_to_all_control_max"] for c in cases]),
        "best_T_selected_metrics": {m: describe([c["best_for_T"][m] for c in cases]) for m in METRICS},
        "best_T_rank_among_all_controls": describe([c["best_for_T"]["ranks"]["T_reduction"]["midrank"] for c in cases]),
        "best_T_rank_at_least_95_percentile": sum(c["best_for_T"]["ranks"]["T_reduction"]["midrank"] >= .95 for c in cases),
        "previous_choice_T_rank_among_all_controls": describe([c["previous_choice"]["ranks"]["T_reduction"]["midrank"] for c in cases]),
    })
    return result


def main(production, out):
    rows = read(production / "rows.json")
    baseline = read(production / "baselines.json")["sessions"]
    selected = read(production / "selected.json")
    assert selected["status"] == "PASS"
    selected_map = {(c["group"], c["p"]): c for c in selected["cases"]}
    prior = read(production / "prior_obstructions.json")
    jobs = {}
    provenance = {}
    for path in (production / "rows.json", production / "baselines.json",
                 production / "selected.json", production / "prior_obstructions.json"):
        provenance[str(path.relative_to(production))] = hashlib.sha256(path.read_bytes()).hexdigest()
    for path in sorted((production / "controls").glob("*.json")):
        job = read(path)
        assert not job["failures"]
        jobs[(job["group"], job["k"])] = (job, path.with_suffix(".npz"))

    cases, total_pairs, total_triples = [], 0, 0
    for old in prior["groups"]:
        if old["compatible_cells"]:
            continue
        group, p, k = old["group"], old["p"], old["minimum_obstruction_cardinality"]
        assert k in (2, 3)
        family = old["all_certifying_pairs" if k == 2 else "all_certifying_triples"]
        assert family and len({tuple(m["ids"]) for m in family}) == len(family)
        total_pairs += len(family) if k == 2 else 0
        total_triples += len(family) if k == 3 else 0
        job, npz_path = jobs[(group, k)]
        session = job["session"]
        rr = [r for r in rows if r["session"] == session]
        row_map = {r["id"]: i for i, r in enumerate(rr)}
        z = np.load(npz_path)
        assert z["free_success"].all() and z[str(p) + "_success"].all()
        assert not z["free_scale_hit"].any() and not z[str(p) + "_scale_hit"].any()
        combo_map = {tuple(map(int, v)): i for i, v in enumerate(z["combinations"])}
        assert len(combo_map) == job["n_controls"]
        base = baseline[session]
        full_beta = float(base["free"]["params"][2])
        full_T = float(2 * (base["fixed"][str(p)]["nll"] - base["free"]["nll"]))
        after_beta = z["free_params"][:, 2]
        after_T = 2 * (z[str(p) + "_nll"] - z["free_nll"])
        all_metrics = {
            "abs_delta_beta": np.abs(after_beta - full_beta),
            "toward_target": abs(full_beta - p) - np.abs(after_beta - p),
            "T_reduction": full_T - after_T,
        }
        members = []
        for m in family:
            ids = sorted(m["ids"])
            indices = tuple(sorted(row_map[s] for s in ids))
            assert len(indices) == k
            assert all(rr[i]["group"] == group and rr[i]["eligible"] for i in indices)
            index = combo_map[indices]
            member = {"ids": ids, "row_indices": list(indices), "control_index": index,
                      "beta": [rr[i]["beta"] if "beta" in rr[i] else float(np.exp(rr[i]["lb"])) for i in indices],
                      "numerical_only_certificate": m["invalid_cells"] == 0,
                      "delta_beta": float(after_beta[index] - full_beta),
                      "beta_after": float(after_beta[index]), "T_after": float(after_T[index]),
                      "sigma_after": float(np.exp(z["free_params"][index, 3])),
                      **{name: float(values[index]) for name, values in all_metrics.items()}}
            member["ranks"] = {name: rank(values, member[name]) for name, values in all_metrics.items()}
            members.append(member)
        members.sort(key=lambda m: tuple(m["ids"]))
        previous_ids = sorted(selected_map[(group, p)]["ids"])
        previous = [m for m in members if m["ids"] == previous_ids]
        assert len(previous) == 1
        best = best_member(members, "T_reduction")
        old_sel = selected_map[(group, p)]
        assert abs(previous[0]["delta_beta"] - old_sel["delta_beta"]) < 2e-4
        assert abs(previous[0]["T_reduction"] - old_sel["T_reduction"]) < 1e-6
        stats = {name: describe([m[name] for m in members]) |
                 {"rank_among_all_controls": describe([m["ranks"][name]["midrank"] for m in members]),
                  "all_controls": describe(values)} for name, values in all_metrics.items()}
        case = {"group": group, "session": session, "p": p, "k": k,
                "arbitrary_geometry_feasible": old["arbitrary_geometry_feasible"],
                "family_size": len(members), "control_population_size": len(after_beta),
                "control_job_number": job["number"], "beta_before": full_beta, "T_before": full_T,
                "counts": {
                    "away_from_target": sum(m["toward_target"] < -TIE_ATOL for m in members),
                    "toward_target": sum(m["toward_target"] > TIE_ATOL for m in members),
                    "T_increase": sum(m["T_reduction"] < -TIE_ATOL for m in members),
                    "T_decrease": sum(m["T_reduction"] > TIE_ATOL for m in members),
                    "numerical_only": sum(m["numerical_only_certificate"] for m in members),
                },
                "metrics": stats, "previous_choice": previous[0], "best_for_T": best,
                "best_T_gap_to_all_control_max": float(np.max(all_metrics["T_reduction"]) - best["T_reduction"]),
                "members": members}
        assert case["best_T_gap_to_all_control_max"] >= -TIE_ATOL
        cases.append(case)
        provenance[str(npz_path.relative_to(production))] = hashlib.sha256(npz_path.read_bytes()).hexdigest()
    assert len(cases) == 77 and total_pairs == 7718 and total_triples == 24
    cases.sort(key=lambda c: (c["group"], c["p"]))
    result = {"status": "PASS", "post_hoc_extension": True, "no_new_fits": True,
              "tie_tolerance": TIE_ATOL,
              "rank_definition": "(strictly below + half tied)/all controls; selected member included; not a p-value",
              "selection_definition": "largest T reduction among all previous minimum-cardinality geometric certificates; ties within 1e-9 by lexicographic sorted IDs; selected after outcomes",
              "counts": {"cases": len(cases), "minimum_pairs": total_pairs, "minimum_triples": total_triples},
              "summary": {"all": aggregate(cases),
                          "classifier_specific_52": aggregate([c for c in cases if c["arbitrary_geometry_feasible"]]),
                          "geometry_infeasible_25": aggregate([c for c in cases if not c["arbitrary_geometry_feasible"]]),
                          "triples": aggregate([c for c in cases if c["k"] == 3]),
                          "by_slope": {str(p): aggregate([c for c in cases if c["p"] == p]) for p in (-2., -1.2)}},
              "cases": cases, "source_sha256": provenance}
    out.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"status": result["status"], "counts": result["counts"],
                      "summary": result["summary"]}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--production", type=Path, default=Path(__file__).resolve().parents[1] / "production")
    parser.add_argument("--out", type=Path, default=Path(__file__).with_name("family.json"))
    args = parser.parse_args()
    main(args.production, args.out)
