#!/usr/bin/env python3
"""Minimum extra label disagreement imposed by one shared log-band width law.

This is a retrospective model-invalidation diagnostic, not a predictor or new
physical law. For TWO source speeds, independently positioned positive bands
must obey W_2 = exp(k * log(v_2/v_1)) W_1 at every nominal beta column. Centers
remain free, so this is a necessary relaxation of affine flow evaluated on
exactly constant controls within each nominal column. The cello predictor uses
slightly varying measured controls, so this is not its certified loss bound.
All finite force intervals, ties, off-grid bands and interval alternatives are
retained through exact finite cost envelopes. Floating arithmetic is checked by
constructive witnesses; it is not an interval-arithmetic proof certificate.
The command never reads A2 or any held-out outcomes.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gzip
import hashlib
import heapq
import itertools
import json
from pathlib import Path
import time

import numpy as np

import affine_transport as af
import benchmark_data as bd
import set_envelope as se


def _point_inside(lower, upper):
    """Choose a finite, strictly interior point, never a sweep endpoint."""
    if not lower < upper:
        raise ValueError("nonempty open interval required")
    if np.isneginf(lower) and np.isposinf(upper):
        return 0.
    if np.isneginf(lower):
        result = upper - max(1., abs(upper) * .1)
    elif np.isposinf(upper):
        result = lower + max(1., abs(lower) * .1)
    else:
        result = lower + (upper - lower) / 2
    if not lower < result < upper or not np.isfinite(result):
        raise ArithmeticError("no representable interior witness; numerical ambiguity")
    return float(result)


def enumerate_bands(force, labels):
    """All different on-grid interval predictions and an arbitrary-width off-grid band.

    A nonempty observed interval [i,j] permits width
    [log(F_j)-log(F_i), log(F_next)-log(F_previous)), intersected with W>0.
    The upper endpoint is strict, because adjacent negative cells must remain
    outside the closed physical band. Repeated identical forces cannot split.
    Any band with cost >= the off-grid cost is harmlessly dominated by the
    off-grid band, which supports EVERY strictly positive width.
    """
    force, raw = np.asarray(force, float), np.asarray(labels)
    if (force.ndim != 1 or force.size == 0 or raw.shape != force.shape
            or not np.isfinite(force).all() or np.any(force <= 0)
            or not np.isin(raw, [0, 1]).all()):
        raise ValueError("positive finite forces and same-shaped binary labels required")
    order = np.argsort(force, kind="stable")
    force, y = force[order], raw[order].astype(int)
    f = np.log(force)
    # Distinct physical force values must remain distinct in the working log
    # representation; otherwise silently treating them as equal would be wrong.
    if np.any((np.diff(force) > 0) & (np.diff(f) <= 0)):
        raise ArithmeticError("distinct forces collapse in floating log coordinates")
    n, positives = len(y), int(y.sum())
    gain = np.r_[0, np.cumsum(2 * y - 1)]
    bands = [{"first": None, "last": None, "cost": positives,
              "width_lower": 0., "width_upper": np.inf}]
    for first in range(n):
        if first and force[first] == force[first - 1]:
            continue
        for last in range(first, n):
            if last + 1 < n and force[last] == force[last + 1]:
                continue
            cost = int(positives - gain[last + 1] + gain[first])
            if cost >= positives:
                continue
            lower = float(f[last] - f[first])
            upper = float(f[last + 1] - f[first - 1]) if first and last + 1 < n else np.inf
            if not lower < upper:
                raise ArithmeticError("empty floating width bracket")
            bands.append({"first": first, "last": last, "cost": cost,
                          "width_lower": lower, "width_upper": upper})
    return {"log_force": f, "labels": y, "order": order,
            "bands": bands, "independent_cost": min(b["cost"] for b in bands)}


def lower_envelope(intervals):
    """Minimum cost on every OPEN cell between interval endpoints.

    intervals contain lower, upper, cost and arbitrary witness metadata. Width
    intervals include their lower endpoint whereas k intervals do not. This
    routine deliberately represents open cells in either case. Endpoint-only
    improvements cannot occur here: width intervals are lower-closed/upper-open,
    and k intervals are open. A minimum over all positive widths / real k is
    therefore realized on some represented open cell. Fixed k must instead use
    strict membership in the original k intervals.
    """
    if not intervals:
        raise ValueError("at least one interval required")
    starts = {}
    points = set()
    for index, interval in enumerate(intervals):
        lo, hi = interval["lower"], interval["upper"]
        if not lo < hi or np.isnan(lo) or np.isnan(hi):
            raise ValueError("ordered non-NaN endpoints required")
        starts.setdefault(lo, []).append((interval["cost"], index))
        points.update((lo, hi))
    points = sorted(points)
    heap, result = [], []
    for lo, hi in zip(points, points[1:]):
        for item in starts.get(lo, ()):
            heapq.heappush(heap, item)
        while heap and intervals[heap[0][1]]["upper"] <= lo:
            heapq.heappop(heap)
        if not heap:
            continue
        cost, index = heap[0]
        if result and result[-1]["index"] == index and result[-1]["upper"] == lo:
            result[-1]["upper"] = hi
        else:
            result.append({"lower": lo, "upper": hi, "cost": cost, "index": index})
    return result


def width_envelope(column):
    intervals = [{"lower": b["width_lower"], "upper": b["width_upper"], "cost": b["cost"]}
                 for b in column["bands"]]
    return lower_envelope(intervals)


def pair_intervals(first, second, delta_t):
    """All attainable pair costs after exact single-column width compression."""
    if not np.isfinite(delta_t) or delta_t <= 0:
        raise ValueError("positive finite source log-speed separation required")
    a, b = width_envelope(first), width_envelope(second)
    intervals = []
    for left, right in itertools.product(a, b):
        # Width-envelope cells may share a cost but differ in realizing band.
        # Their lower boundaries are available in the original bands. For the
        # global infimum, open-cell compression preserves every attainable cost.
        lo = (np.log(right["lower"]) - np.log(left["upper"])) / delta_t \
            if right["lower"] > 0 and np.isfinite(left["upper"]) else -np.inf
        hi = (np.log(right["upper"]) - np.log(left["lower"])) / delta_t \
            if left["lower"] > 0 and np.isfinite(right["upper"]) else np.inf
        if not lo < hi:
            raise ArithmeticError("collapsed pair interval")
        intervals.append({"lower": float(lo), "upper": float(hi), "cost": left["cost"] + right["cost"],
                          "first_band": left["index"], "second_band": right["index"],
                          "first_width_cell": [left["lower"], left["upper"]],
                          "second_width_cell": [right["lower"], right["upper"]]})
    return intervals, {"first_bands": len(first["bands"]), "second_bands": len(second["bands"]),
                       "first_width_cells": len(a), "second_width_cells": len(b)}


def _band_witness(column, band_index, width):
    band, f = column["bands"][band_index], column["log_force"]
    if not (width > 0 and width >= band["width_lower"] and width < band["width_upper"]):
        raise ArithmeticError("width outside realizing band's exact bracket")
    if band["first"] is None:
        lower = float(f[-1] + 1.)
    else:
        i, j = band["first"], band["last"]
        lo = max(f[i - 1] if i else -np.inf, f[j] - width)
        hi = min(f[i], f[j + 1] - width if j + 1 < len(f) else np.inf)
        # The lower width endpoint is closed. At exactly the span between two
        # positive observations, the only realizing endpoints may be those two
        # observations themselves. Their negative neighbors remain excluded.
        if lo == hi and lo <= f[i] and lo + width >= f[j] \
                and (not i or lo > f[i - 1]) \
                and (j + 1 == len(f) or lo + width < f[j + 1]):
            lower = float(lo)
        else:
            lower = _point_inside(lo, hi)
    upper = lower + width
    pred = (f >= lower) & (f <= upper)
    disagreements = np.flatnonzero(pred != column["labels"])
    expected = np.zeros(len(f), bool)
    if band["first"] is not None:
        expected[band["first"]:band["last"] + 1] = True
    if not np.array_equal(pred, expected) or len(disagreements) != band["cost"]:
        raise ArithmeticError("constructive endpoints fail observed-label verification")
    if not np.isclose(upper - lower, width, rtol=2e-11, atol=2e-14):
        raise ArithmeticError("endpoint subtraction lost width precision")
    return {"lower_log_force": lower, "upper_log_force": float(upper), "width": float(width),
            "first_positive_sorted_index": band["first"], "last_positive_sorted_index": band["last"],
            "cost": band["cost"], "disagreement_original_indices": column["order"][disagreements].tolist()}


def pair_witness(first, second, interval, k, delta_t):
    if not interval["lower"] < k < interval["upper"]:
        raise ArithmeticError("k is not strictly inside its realizing pair interval")
    rho = float(np.exp(k * delta_t))
    if not np.isfinite(rho) or rho <= 0:
        raise ArithmeticError("width ratio cannot be represented finitely and positively")
    a, b = interval["first_width_cell"], interval["second_width_cell"]
    low, high = max(a[0], b[0] / rho), min(a[1], b[1] / rho)
    width1 = _point_inside(low, high)
    width2 = rho * width1
    first_result = _band_witness(first, interval["first_band"], width1)
    second_result = _band_witness(second, interval["second_band"], width2)
    if first_result["cost"] + second_result["cost"] != interval["cost"]:
        raise AssertionError("pair witness cost mismatch")
    return {"first": first_result, "second": second_result, "cost": interval["cost"]}


def solve_shared_width(columns, delta_t):
    """Globally minimum integer disagreement over shared k and free band centers."""
    if not columns:
        raise ValueError("at least one paired column required")
    envelopes, pair_lists, diagnostics = [], [], []
    baseline = 0
    for first, second in columns:
        pairs, diagnostic = pair_intervals(first, second, delta_t)
        envelope = lower_envelope(pairs)
        if envelope[0]["lower"] != -np.inf or envelope[-1]["upper"] != np.inf:
            raise AssertionError("off-grid bands must make every k feasible")
        envelopes.append(envelope)
        pair_lists.append(pairs)
        diagnostics.append({**diagnostic, "pair_intervals": len(pairs), "k_cost_cells": len(envelope)})
        baseline += first["independent_cost"] + second["independent_cost"]
    # Every individual cost envelope covers R away from its finite breakpoints.
    # At a breakpoint, open candidate intervals can only REMOVE alternatives;
    # hence a point can never improve on both adjacent open-cell costs.
    events = {}
    cost = sum(e[0]["cost"] for e in envelopes)
    for e in envelopes:
        for previous, current in zip(e, e[1:]):
            if previous["upper"] != current["lower"]:
                raise AssertionError("incomplete k cost envelope")
            events[current["lower"]] = events.get(current["lower"], 0) + current["cost"] - previous["cost"]
    points = [-np.inf, *sorted(events), np.inf]
    global_cells = []
    for lo, hi in zip(points, points[1:]):
        cost += events.get(lo, 0)
        # Keep breakpoints out of merged cells: endpoint spikes are possible.
        # The public certificate must not declare every merged k attainable.
        global_cells.append({"lower": lo, "upper": hi, "cost": int(cost)})
    best = min(c["cost"] for c in global_cells)
    optimum = next(c for c in global_cells if c["cost"] == best)
    k = _point_inside(optimum["lower"], optimum["upper"])
    witnesses, fixed_zero_cost = [], 0
    for (first, second), pairs in zip(columns, pair_lists):
        feasible = [p for p in pairs if p["lower"] < k < p["upper"]]
        chosen = min(feasible, key=lambda p: p["cost"])
        witnesses.append(pair_witness(first, second, chosen, k, delta_t))
        # Use strict membership, not the cost of a neighboring sweep cell.
        fixed_zero_cost += min(p["cost"] for p in pairs if p["lower"] < 0. < p["upper"])
    if sum(w["cost"] for w in witnesses) != best or best < baseline:
        raise AssertionError("global constructive objective mismatch")
    return {"independent_band_cost": int(baseline), "shared_width_cost": int(best),
            "extra_disagreements": int(best - baseline), "constant_width_cost": int(fixed_zero_cost),
            "constant_width_extra_disagreements": int(fixed_zero_cost - baseline),
            "witness_k": k, "witness_ratio": float(np.exp(k * delta_t)),
            "witnesses": witnesses, "cost_cells": global_cells, "column_diagnostics": diagnostics,
            "verified_constructive_witness": True}


def _portable(value):
    if isinstance(value, dict):
        return {k: _portable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_portable(v) for v in value]
    if isinstance(value, (np.integer, np.bool_)):
        return value.item()
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    return value


def _write(path, value):
    with Path(path).open("x") as stream:
        json.dump(_portable(value), stream, indent=2, allow_nan=False)
        stream.write("\n")


def run(output, *, all_observers=False):
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("preserve historical certificate outputs")
    output.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    protocol = {"created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "Retrospective source-only diagnostic; no A2, no target outcomes, no predictor fit",
        "law": "W2=exp(k*log(v2/v1))*W1; one k shared across nominal beta columns; centers completely free",
        "scope": "Necessary width-only relaxation of affine flow on an exactly nominal control grid; G>0 rules out zero-extra-repair nominal realization, G=0 does not validate full flow or physical truth. Not a certified loss bound for the cello predictor with varying measured controls.",
        "observer": ("All fifteen prespecified coherent cello observer rules, each separately; published D author classifications. No best-observer selection."
                     if all_observers else "Original nominal cello rule y_fixed; published D author classifications. No best-observer selection."),
        "loss": "Unweighted row disagreement counts; paired sources only; no deletion or alteration of labels",
        "empty_observed_prediction": "Nonempty positive-width band outside sampled force range; arbitrary width permitted because centers are free",
        "endpoint_convention": "Closed positive band; nonempty observed interval width lower inclusive (except W>0), upper strict. Pair k bounds strict.",
        "numerics": "Finite exhaustive cost-envelope optimization in floating log coordinates; every chosen witness reclassifies all source rows. Not interval arithmetic.",
        "null_values": "A null lower endpoint is negative infinity; null upper endpoint is positive infinity",
        "control_limit": "Measured beta/speed variations within nominal columns are not modeled",
        "code_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in
                        (Path(__file__), Path(af.__file__), Path(bd.__file__), Path(se.__file__))}}
    _write(output / "protocol.json", protocol)
    datasets = {"cello": bd.load_dataset(include_ambiguous=True), "published": af.load_published()}
    family = se.label_family(datasets["cello"]["output_features"], datasets["cello"]["reference_f0_hz"])
    if not np.array_equal(family[:, 7], datasets["cello"]["y_fixed"] == 1):
        raise AssertionError("nominal observer no longer agrees with retained nominal labels")
    _write(output / "input_manifests.json", {name: data["manifest"] for name, data in datasets.items()})
    rules = range(15) if all_observers else (7,)
    tasks = [("cello", s, pair, rule) for s, pair, rule in itertools.product(("A", "B", "C", "D"),
                 itertools.combinations((.05, .1, .2), 2), rules)]
    tasks += [("published", "D4", pair, 0) for pair in itertools.combinations((.05, .1, .15, .2), 2)]
    results = []
    cache = {}
    for domain, source, speeds, rule in tasks:
        data = datasets[domain]
        labels = family[:, rule] if domain == "cello" else data["y_fixed"] == 1
        paired, metadata, ids = [], [], []
        levels = sorted(np.unique(data["beta_level"][data["string"] == source]).tolist())
        for level in levels:
            pair = []
            for speed in speeds:
                mask = (data["string"] == source) & (data["beta_level"] == level) & (data["speed"] == speed)
                if not np.any(mask):
                    raise ValueError("every nominal beta column must occur at both source speeds")
                key = (domain, source, speed, level, rule)
                if key not in cache:
                    cache[key] = enumerate_bands(data["X"][mask, 1], labels[mask])
                pair.append(cache[key])
                ids.extend(data["ids"][mask].tolist())
            paired.append(tuple(pair))
            metadata.append({"beta_level": level, "first_n": len(pair[0]["labels"]), "second_n": len(pair[1]["labels"])})
        result = solve_shared_width(paired, float(np.log(speeds[1] / speeds[0])))
        total = sum(m["first_n"] + m["second_n"] for m in metadata)
        result.update({"domain": domain, "source": source, "source_speeds": list(speeds), "source_rows": total,
            "rule_index": rule,
            "rule_parameters": ({"fraction": (.3, .4, .5, .6, .7)[rule // 3], "periodicity": (.75, .8, .85)[rule % 3]}
                                if domain == "cello" else {"author_classification": True}),
            "source_ids_sha256": bd.id_hash(np.asarray(ids)), "columns": metadata,
            "extra_disagreement_fraction": result["extra_disagreements"] / total,
            "independent_band_fraction": result["independent_band_cost"] / total})
        task = f"{domain}__{source}__{speeds[0]:g}_{speeds[1]:g}"
        if all_observers:
            task += f"__rule{rule}"
        with gzip.open(output / f"{task}.json.gz", "xt") as stream:
            json.dump(_portable(result), stream, separators=(",", ":"), allow_nan=False)
        compact = {key: value for key, value in result.items() if key not in ("cost_cells", "witnesses", "column_diagnostics", "columns")}
        compact.update({"task": task, "k_cost_cells": len(result["cost_cells"]),
                        "pair_intervals": sum(d["pair_intervals"] for d in result["column_diagnostics"])})
        results.append(compact)
        print(task, "band", result["independent_band_cost"], "shared", result["shared_width_cost"],
              "extra", result["extra_disagreements"], "k", result["witness_k"], flush=True)
    summary = {"protocol": protocol, "all_tasks": results, "seconds": time.monotonic() - started}
    _write(output / "summary.json", summary)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--all-observers", action="store_true", help="Report all fifteen coherent cello rules separately")
    arguments = parser.parse_args()
    run(arguments.output, all_observers=arguments.all_observers)
