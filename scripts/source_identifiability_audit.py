#!/usr/bin/env python3
"""Retrospective source boundary geometry and identification audit; no model fit.

Keep disconnected observations, acquisition censoring and every minimum-Hamming
single-interval approximation distinct. No labels are overwritten. A2 is never read.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

import affine_transport as af
import benchmark_data as bd
from set_envelope import label_family


def component_brackets(log_force, first, last):
    f = np.asarray(log_force, float)
    return {"first": int(first), "last": int(last),
            "lower": [float(f[first - 1]) if first else -np.inf, float(f[first])],
            "upper": [float(f[last]), float(f[last + 1]) if last + 1 < len(f) else np.inf],
            "width": [float(f[last] - f[first]),
                      float(f[last + 1] - f[first - 1]) if first and last + 1 < len(f) else np.inf]}


def envelope(hypotheses):
    if not hypotheses:
        return None
    return {"lower": [min(h["lower"][0] for h in hypotheses), max(h["lower"][1] for h in hypotheses)],
            "upper": [min(h["upper"][0] for h in hypotheses), max(h["upper"][1] for h in hypotheses)],
            "width": [min(h["width"][0] for h in hypotheses), max(h["width"][1] for h in hypotheses)],
            "center": [min((h["lower"][0] + h["upper"][0]) / 2 for h in hypotheses),
                       max((h["lower"][1] + h["upper"][1]) / 2 for h in hypotheses)]}


def boundary_geometry(force, labels):
    force, raw_y = np.asarray(force, float), np.asarray(labels)
    if force.ndim != 1 or force.shape != raw_y.shape or len(force) < 1 or np.any(force <= 0) or not np.isfinite(force).all():
        raise ValueError("positive finite force vector and matching binary labels required")
    if not np.isin(raw_y, [0, 1]).all():
        raise ValueError("labels must be binary")
    y = raw_y.astype(bool)
    order = np.argsort(force, kind="stable")
    force, y = force[order], y[order]
    f = np.log(force)
    changes = np.diff(np.r_[False, y, False].astype(int))
    starts, ends = np.flatnonzero(changes == 1), np.flatnonzero(changes == -1) - 1
    components = [component_brackets(f, first, last) for first, last in zip(starts, ends)]
    # Every best contiguous positive subset; empty prediction is also allowed.
    gains = np.r_[0, np.cumsum(np.where(y, 1, -1))]
    best_gain, ranges = 0, []
    # A deterministic force interval cannot split repeated measurements at the
    # exact same force. Conflicting ties can make empty/nonempty bands equally
    # optimal, so retain those zero-gain alternatives as well.
    allowed_first = np.r_[True, np.diff(force) > 0]
    allowed_last = np.r_[np.diff(force) > 0, True]
    for first in np.flatnonzero(allowed_first):
        for last in np.flatnonzero(allowed_last & (np.arange(len(y)) >= first)):
            gain = int(gains[last + 1] - gains[first])
            if gain > best_gain:
                best_gain, ranges = gain, [(first, last)]
            elif gain == best_gain:
                ranges.append((first, last))
    hypotheses = [component_brackets(f, first, last) for first, last in ranges]
    conflicting_ties = any(np.unique(y[force == value]).size > 1 for value in np.unique(force))
    return {"n": len(y), "positive": int(y.sum()), "force_range": [float(force[0]), float(force[-1])],
            "components": components, "component_count": len(components),
            "left_acquisition_edge_positive": bool(y[0]), "right_acquisition_edge_positive": bool(y[-1]),
            "force_ties_with_conflicting_labels": bool(conflicting_ties),
            "exact_single_band": envelope(components) if len(components) == 1 and not conflicting_ties else None,
            "minimum_disagreement_count": int(y.sum()) - best_gain,
            "minimum_disagreement_fraction": (int(y.sum()) - best_gain) / len(y),
            "optimal_positive_intervals": hypotheses, "optimal_positive_interval_count": len(hypotheses),
            # A tied optimal empty prediction leaves the location of a possible
            # off-grid band unidentified. Keep nonempty alternatives available
            # for inspection, but do not turn that conditional envelope into
            # unconditional boundary constraints.
            "minimum_disagreement_nonempty_band_envelope": envelope(hypotheses),
            "minimum_disagreement_band_envelope": envelope(hypotheses) if best_gain > 0 else None,
            "empty_optimal": best_gain == 0,
            "scope": "No observed positive or an empty best prediction does not rule out an unobserved off-grid band."}


def finite_both(band):
    return band is not None and np.isfinite(band["lower"] + band["upper"]).all()


def difference_interval(first, second, delta_t):
    if not np.isfinite(delta_t) or delta_t <= 0:
        raise ValueError("strictly ordered finite source log-speeds required")
    return [(second[0] - first[1]) / delta_t, (second[1] - first[0]) / delta_t]


def width_generator_interval(first, second, delta_t):
    # Width ratio is exp(k delta_t); these are conservative scalar interval bounds.
    if not np.isfinite(delta_t) or delta_t <= 0:
        raise ValueError("strictly ordered finite source log-speeds required")
    for interval in (first, second):
        if (len(interval) != 2 or np.isnan(interval).any() or interval[0] < 0
                or interval[0] > interval[1] or interval[1] <= 0):
            raise ValueError("ordered nonnegative width bounds with positive possible width required")
    low = np.log(second[0] / first[1]) / delta_t if second[0] > 0 and np.isfinite(first[1]) else -np.inf
    high = np.log(second[1] / first[0]) / delta_t if first[0] > 0 and np.isfinite(second[1]) else np.inf
    return [float(low), float(high)]


def extrapolation_weights(source_speeds, target_speed):
    if (len(source_speeds) != 2 or not np.isfinite([*source_speeds, target_speed]).all()
            or not 0 < source_speeds[0] < source_speeds[1] or target_speed <= 0):
        raise ValueError("two strictly ordered positive source speeds and a positive target speed required")
    first, second = source_speeds
    r = np.log(target_speed / first) / np.log(second / first)
    return {"first": float(1 - r), "second": float(r), "absolute_weight_sum": float(abs(1 - r) + abs(r)),
            "curvature_remainder_factor": float(.5 * abs(np.log(target_speed / first) * np.log(target_speed / second)))}


def geometry_rows(data, domain):
    family = data["family"]
    rows = []
    for source in np.unique(data["string"]):
        if domain == "published" and source != "D4":
            continue  # E strict cells are boundary-focused, not a full-grid target.
        for speed in np.unique(data["speed"][data["string"] == source]):
            panel = (data["string"] == source) & (data["speed"] == speed)
            for level in np.unique(data["beta_level"][panel]):
                keep = panel & (data["beta_level"] == level)
                for rule in range(family.shape[1]):
                    rows.append({"domain": domain, "source": str(source), "speed": float(speed), "beta_level": int(level),
                                 "beta_median": float(np.median(data["X"][keep, 0])), "rule": rule,
                                 **boundary_geometry(data["X"][keep, 1], family[keep, rule])})
    return rows


def panel_summary(rows):
    return {"columns": len(rows), "samples": sum(r["n"] for r in rows),
            "no_observed_positive": sum(r["positive"] == 0 for r in rows),
            "multiple_observed_components": sum(r["component_count"] > 1 for r in rows),
            "left_censored_columns": sum(r["left_acquisition_edge_positive"] for r in rows),
            "right_censored_columns": sum(r["right_acquisition_edge_positive"] for r in rows),
            "exact_fully_bracketed_columns": sum(finite_both(r["exact_single_band"]) for r in rows),
            "minimum_disagreement_fully_bracketed_columns": sum(finite_both(r["minimum_disagreement_band_envelope"]) for r in rows),
            "minimum_disagreement_count": sum(r["minimum_disagreement_count"] for r in rows),
            "minimum_disagreement_fraction": sum(r["minimum_disagreement_count"] for r in rows) / sum(r["n"] for r in rows),
            "columns_with_nonunique_best_interval": sum(r["optimal_positive_interval_count"] > 1 for r in rows)}


def pair_summary(first, second, pair, target_speeds, field):
    a, b = {r["beta_level"]: r for r in first}, {r["beta_level"]: r for r in second}
    shared = sorted(set(a) & set(b))
    dt = np.log(pair[1] / pair[0])
    details = []
    for level in shared:
        band1, band2 = a[level][field], b[level][field]
        if band1 is None or band2 is None:
            continue
        exponent_lower = difference_interval(band1["lower"], band2["lower"], dt)
        exponent_upper = difference_interval(band1["upper"], band2["upper"], dt)
        k = width_generator_interval(band1["width"], band2["width"], dt)
        details.append({"beta_level": level, "fully_bracketed_both_speeds": finite_both(band1) and finite_both(band2),
                        "lower_exponent": exponent_lower, "upper_exponent": exponent_upper, "k_interval": k,
                        "k_sign_identified": k[0] > 0 or k[1] < 0,
                        "known_asymmetric_exponents_compatible": exponent_lower[0] <= 0 <= exponent_lower[1]
                            and exponent_upper[0] <= 1 <= exponent_upper[1],
                        "finite_target_boundary_interval_widths": {str(v): {
                            side: abs(extrapolation_weights(pair, v)["first"]) * (band1[side][1] - band1[side][0]) +
                                  abs(extrapolation_weights(pair, v)["second"]) * (band2[side][1] - band2[side][0])
                            for side in ("lower", "upper")} for v in target_speeds}})
    k_lo = max((d["k_interval"][0] for d in details), default=-np.inf)
    k_hi = min((d["k_interval"][1] for d in details), default=np.inf)
    common_slopes = {}
    for side in ("lower", "upper"):
        low = max((d[side + "_exponent"][0] for d in details), default=-np.inf)
        high = min((d[side + "_exponent"][1] for d in details), default=np.inf)
        common_slopes[side] = {"interval": [low, high], "nonempty": low <= high,
                              "contains_known_exponent": low <= (0 if side == "lower" else 1) <= high,
                              "minimum_uniform_log_boundary_expansion_for_any_common_slope": max(0., low - high) * dt / 4,
                              "expansion_scope": "Outer relaxation only: each endpoint bracket may expand by delta at both source speeds; no calibrated noise bound."}
    target_uncertainty = {}
    for speed in target_speeds:
        target_uncertainty[str(speed)] = {}
        for side in ("lower", "upper"):
            values = np.asarray([d["finite_target_boundary_interval_widths"][str(speed)][side] for d in details])
            finite = values[np.isfinite(values)]
            target_uncertainty[str(speed)][side] = {"finite_columns": len(finite),
                "median_log_force_width_if_finite": float(np.median(finite)) if len(finite) else None,
                "unbounded_or_unidentified_columns": len(shared) - len(finite)}
    return {"common_beta_columns": len(shared), "both_have_band": len(details),
            "both_fully_bracketed": sum(d["fully_bracketed_both_speeds"] for d in details),
            "k_sign_identified_columns": sum(d["k_sign_identified"] for d in details),
            "known_asymmetric_compatible_columns": sum(d["known_asymmetric_exponents_compatible"] for d in details),
            "shared_k_intersection": [k_lo, k_hi], "shared_k_intersection_nonempty": k_lo <= k_hi,
            "minimum_uniform_log_width_expansion_for_common_k": max(0., k_lo - k_hi) * dt / 4,
            "shared_constant_boundary_slopes": common_slopes,
            "shared_k_scope": "Necessary outer interval relaxation, not sufficient joint feasibility. Conditional on available exact or minimum-disagreement bands; omitted/no-band columns are NOT evidence of compatibility.",
            "target_boundary_uncertainty": target_uncertainty}, details


def specimen_differences(rows, nominal_rule=7):
    selected = [r for r in rows if r["domain"] == "cello" and r["rule"] == nominal_rule]
    index = {(r["source"], r["speed"], r["beta_level"]): r for r in selected}
    results = []
    for first, second in itertools.combinations(("A", "B", "C", "D"), 2):
        for speed in (.05, .1, .2):
            eligible = lower = upper = 0
            for level in range(40):
                a = index[first, speed, level]["minimum_disagreement_band_envelope"]
                b = index[second, speed, level]["minimum_disagreement_band_envelope"]
                if a is None or b is None:
                    continue
                eligible += 1
                lower += max(a["lower"][0], b["lower"][0]) > min(a["lower"][1], b["lower"][1])
                upper += max(a["upper"][0], b["upper"][0]) > min(a["upper"][1], b["upper"][1])
            results.append({"first": first, "second": second, "speed": speed, "eligible_beta_columns": eligible,
                            "disjoint_lower_brackets": int(lower), "disjoint_upper_brackets": int(upper)})
    return results


def sanitize(x):
    if isinstance(x, dict):
        return {k: sanitize(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [sanitize(v) for v in x]
    if isinstance(x, (np.integer, np.bool_)):
        return x.item()
    if isinstance(x, (float, np.floating)):
        return float(x) if np.isfinite(x) else None
    return x


def write(path, content):
    with Path(path).open("x") as stream:
        json.dump(sanitize(content), stream, indent=2, allow_nan=False)
        stream.write("\n")


def main(output, flow_run):
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("preserve previous audits")
    output.mkdir(parents=True, exist_ok=True)
    protocol = {"created_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "RETROSPECTIVE source identifiability diagnosis; not a fitted method, no A2 access",
        "geometry": "sorted measured positive forces within nominal beta/speed columns; beta/velocity measurement variations are not modeled",
        "exact_brackets": "all observed positive components, never fill holes; exterior endpoint is unbounded when positive at acquisition edge",
        "approximation": "all minimum-Hamming contiguous positive intervals; descriptive contamination required for a band, not relabeled physical truth",
        "empty_ties": "If the empty prediction is also optimal, nonempty alternatives remain recorded but no unconditional boundary envelope is asserted.",
        "family": "all15 coherent cello observer rules separately; one published author label rule; E excluded due selective interior analysis",
        "diagnostic_before_outcome_join": ["fraction with both boundaries finitely bracketed at both source speeds", "maximum absolute interpolation-weight sum across declared held-out source speeds"],
        "curvature": "Two source speeds do not bound an unrestricted second derivative; linear interpolation uncertainty remains conditional.",
        "null_values": "unbounded endpoints or unavailable summaries; lower bracket left endpoint null means minus infinity, upper bracket right endpoint null means plus infinity",
        "code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    write(output / "protocol.json", protocol)
    cello = bd.load_dataset(include_ambiguous=True)
    cello["family"] = label_family(cello["output_features"], cello["reference_f0_hz"])
    published = af.load_published()
    datasets = {"cello": cello, "published": published}
    write(output / "source_manifests.json", {domain: data["manifest"] for domain, data in datasets.items()})
    rows = sum([geometry_rows(data, domain) for domain, data in datasets.items()], [])
    write(output / "all_column_geometry.json", rows)
    panels, pairs, details = [], [], []
    for domain, data in datasets.items():
        sources = ("A", "B", "C", "D") if domain == "cello" else ("D4",)
        speeds = (.05, .1, .2) if domain == "cello" else (.05, .1, .15, .2)
        for source in sources:
            for rule in range(data["family"].shape[1]):
                subset = [r for r in rows if r["domain"] == domain and r["source"] == source and r["rule"] == rule]
                for speed in speeds:
                    panel = [r for r in subset if r["speed"] == speed]
                    panels.append({"domain": domain, "source": source, "rule": rule, "speed": speed, **panel_summary(panel)})
                for pair in itertools.combinations(speeds, 2):
                    target_speeds = [speed for speed in speeds if speed not in pair]
                    p = {"domain": domain, "source": source, "rule": rule, "training_speeds": pair,
                         "heldout_speeds": target_speeds,
                         "extrapolation": {str(v): extrapolation_weights(pair, v) for v in target_speeds}}
                    for name, field in (("exact", "exact_single_band"), ("minimum_disagreement", "minimum_disagreement_band_envelope")):
                        summary, individual = pair_summary([r for r in subset if r["speed"] == pair[0]],
                            [r for r in subset if r["speed"] == pair[1]], pair, target_speeds, field)
                        p[name] = summary
                        details.append({"domain": domain, "source": source, "rule": rule, "pair": pair, "geometry": name, "columns": individual})
                    pairs.append(p)
    write(output / "all_panel_summaries.json", panels)
    write(output / "all_pair_constraints.json", details)
    # Only now join already-exposed historical outcome scores for a descriptive diagnostic.
    old = json.loads((Path(flow_run) / "results.json").read_text())
    nominal_pairs = [p for p in pairs if p["rule"] == (7 if p["domain"] == "cello" else 0)]
    for p in nominal_pairs:
        pair = p["training_speeds"]
        key = f"{p['domain']}__{p['source']}__{pair[0]:g}_{pair[1]:g}"
        p["historical_same_source_heldout_speed_scores"] = {m: v["source_heldout_speed"]["macro_panels"]["family_min_iou"]
                                                           for m, v in old[key]["methods"].items()}
    correlations = {}
    for domain in datasets:
        selected = [p for p in nominal_pairs if p["domain"] == domain]
        target_error = [1 - p["historical_same_source_heldout_speed_scores"]["affine_flow"] for p in selected]
        bounded = [p["minimum_disagreement"]["both_fully_bracketed"] / p["minimum_disagreement"]["common_beta_columns"] for p in selected]
        leverage = [max(e["absolute_weight_sum"] for e in p["extrapolation"].values()) for p in selected]
        correlations[domain] = {"tasks": len(selected), "scope": "descriptive dependent tasks; no p-value or validated failure predictor",
            "source_bounded_fraction_vs_error_spearman": float(spearmanr(bounded, target_error).statistic),
            "source_speed_leverage_vs_error_spearman": float(spearmanr(leverage, target_error).statistic)}
    summary = {"protocol": protocol, "nominal_panels": [p for p in panels if p["rule"] == (7 if p["domain"] == "cello" else 0)],
               "nominal_pairs": nominal_pairs, "all_rule_pair_summaries": pairs,
               "specimen_minimum_disagreement_bracket_differences": specimen_differences(rows),
               "retrospective_diagnostic_association": correlations,
               "row_count": len(rows), "no_label_changes_or_model_fitting": True}
    write(output / "summary.json", summary)
    print(json.dumps(sanitize({"columns": len(rows), "panels": summary["nominal_panels"], "association": correlations}), indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--flow-run", type=Path, default=af.ROOT / "reports/affine_transport/2026-10-13-v2")
    args = parser.parse_args()
    main(args.output, args.flow_run)
