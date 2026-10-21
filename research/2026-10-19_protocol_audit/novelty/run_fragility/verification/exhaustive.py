"""Independent exhaustive oracle for first-K-run edit distances.

The oracle enumerates all target bit strings and computes Hamming distance.
It deliberately does not use run automata, shortest paths, or core helpers.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent


def load_file(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def bit_strings(n):
    """Lexicographically ordered arrays, including the unique empty string."""
    vals = np.arange(2**n, dtype=np.uint32)
    shifts = np.arange(n - 1, -1, -1, dtype=np.uint32)
    return ((vals[:, None] >> shifts[None, :]) & 1).astype(np.int8)


def first_run(bits, k):
    """Literal slices only; None denotes no run."""
    for start in range(max(0, len(bits) - k + 1)):
        if all(int(value) == 1 for value in bits[start : start + k]):
            return start
    return None


def oracle_all(n, k, mode):
    """Return all sources, target run starts, and all-source minimal costs."""
    strings = bit_strings(n)
    starts = list(range(max(0, n - k + 1))) + [None]
    target_starts = [first_run(target, k) for target in strings]
    dist = (strings[:, None, :] != strings[None, :, :]).sum(axis=2).astype(float)
    if mode == "delete":
        dist[(strings[None, :, :] > strings[:, None, :]).any(axis=2)] = np.inf
    elif mode == "add":
        dist[(strings[None, :, :] < strings[:, None, :]).any(axis=2)] = np.inf
    elif mode != "both":
        raise ValueError(mode)
    result = np.full((len(strings), len(starts)), np.inf)
    for col, wanted in enumerate(starts):
        eligible = np.array([value == wanted for value in target_starts])
        if eligible.any():
            result[:, col] = np.min(dist[:, eligible], axis=1)
    return strings, starts, result


def check_witness(source, target, wanted, k, mode, expected):
    target = np.asarray(target)
    assert target.shape == source.shape, (source, target)
    assert np.isin(target, [0, 1]).all(), target
    if mode == "delete":
        assert np.all(target <= source), (source, target)
    if mode == "add":
        assert np.all(target >= source), (source, target)
    assert first_run(target, k) == wanted, (source, target, wanted)
    assert np.count_nonzero(target != source) == expected, (source, target, expected)


def adapt_costs(result, starts):
    """Adapt the public core API without sharing any oracle implementation."""
    if "costs" not in result:
        raise AssertionError(f"Unknown core API: {list(result)}")
    # Core API is finalized by the implementation agent; indices are checked
    # against explicit starts here rather than inferred from its helper code.
    costs = np.asarray(result["costs"], dtype=float)
    assert len(costs) == len(starts) - 1
    return np.r_[costs, float(result["no_run_cost"])]


def adapt_witness(result):
    for key in ("labels", "target", "edited_labels", "witness"):
        if key in result:
            return result[key]
    raise AssertionError(f"Unknown witness API: {list(result)}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-n", type=int, default=9)
    args = parser.parse_args()
    core = load_file("independently_verified_run_core", HERE.parent / "core.py")
    started = time.time()
    counts = dict(source_configurations=0, endpoint_costs=0, finite_witnesses=0,
                  impossible_witnesses=0,
                  enumerated_source_target_mode_pairs=0)
    rows = []
    for n in range(args.max_n + 1):
        n_count = dict(source_configurations=0, endpoint_costs=0, finite_witnesses=0,
                       impossible_witnesses=0)
        for k in (1, 2, 3, 4):
            for mode in ("both", "delete", "add"):
                sources, starts, wanted_costs = oracle_all(n, k, mode)
                counts["enumerated_source_target_mode_pairs"] += len(sources) ** 2
                for source, wanted in zip(sources, wanted_costs):
                    assert core.first_run_start(source, k=k) == first_run(source, k)
                    actual = adapt_costs(core.first_run_costs(source.tolist(), k=k, mode=mode), starts)
                    assert actual.shape == wanted.shape, (n, k, mode, actual, wanted)
                    np.testing.assert_array_equal(actual, wanted,
                        err_msg=f"n={n}, k={k}, mode={mode}, source={source.tolist()}")
                    n_count["source_configurations"] += 1
                    n_count["endpoint_costs"] += len(starts)
                    for start, cost in zip(starts, wanted):
                        witness = core.witness_first_run(source.tolist(), start, k=k, mode=mode)
                        if np.isfinite(cost):
                            check_witness(source, adapt_witness(witness), start, k, mode, cost)
                            if "cost" in witness:
                                assert witness["cost"] == cost
                            assert witness["feasible"]
                            np.testing.assert_array_equal(witness["flipped_indices"],
                                np.flatnonzero(np.asarray(adapt_witness(witness)) != source))
                            n_count["finite_witnesses"] += 1
                        else:
                            assert not witness["feasible"]
                            assert witness["cost"] == np.inf
                            n_count["impossible_witnesses"] += 1
        counts.update({name: counts[name] + value for name, value in n_count.items()})
        rows.append(dict(n=n, **n_count))
        print(f"n={n}: {n_count}", flush=True)
    report = dict(status="passed", max_n=args.max_n, k_values=[1, 2, 3, 4],
                  modes=["both", "delete", "add"], counts=counts,
                  rows=rows, elapsed_seconds=time.time() - started,
                  oracle="Full binary target enumeration with pairwise Hamming distance")
    (HERE / "exhaustive_results.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
