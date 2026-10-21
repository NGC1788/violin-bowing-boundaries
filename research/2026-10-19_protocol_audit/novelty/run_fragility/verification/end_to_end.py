"""Tiny end-to-end label enumeration for the declared finite-band verdict.

Enumerates target label strings directly, maps their literal first run to a
force bracket, then enumerates all columns' target combinations. This tests
that DP compression plus global optimization retains the true label minimum.
It does not test a likelihood-ratio, confidence interval, or physical truth.
"""
from pathlib import Path
import json
import time
import numpy as np

from exhaustive import load_file, bit_strings, first_run, check_witness
from global_oracle import exhaustive_combinations

HERE = Path(__file__).resolve().parent


def interval(forces, beta, start, exponent, epsilon):
    return (float(np.log(forces[start - 1]) - exponent * np.log(beta) - epsilon),
            float(np.log(forces[start]) - exponent * np.log(beta) + epsilon))


def main():
    core = load_file("end_to_end_core", HERE.parent / "core.py")
    glob = load_file("end_to_end_global", HERE.parent / "global_profile.py")
    rng = np.random.default_rng(20261021)
    started = time.time()
    count = combinations = 0
    for n in (3, 4, 5):
        strings = bit_strings(n)
        for k in (1, 2, 3):
            for fixture in range(10):
                sources = rng.integers(0, 2, size=(3, n))
                forces = [np.exp(np.cumsum(rng.uniform(.1, 1, n))) for _ in range(3)]
                betas = [.05, .1, .2]
                exponent = float(rng.choice([-2., -1.2, 0., 1.]))
                epsilon = float(rng.choice([0., .1, .2, .4]))
                for mode in ("both", "delete", "add"):
                    literal = []
                    compressed = []
                    for source, force, beta in zip(sources, forces, betas):
                        literals = []
                        for target in strings:
                            run = first_run(target, k)
                            if run is None or run == 0:
                                continue  # finite-positive-bracket-only estimand
                            if mode == "delete" and np.any(target > source):
                                continue
                            if mode == "add" and np.any(target < source):
                                continue
                            low, high = interval(force, beta, run, exponent, epsilon)
                            literals.append(dict(lo=low, hi=high,
                                                 cost=int(np.count_nonzero(source != target)),
                                                 state=run))
                        literal.append(literals)
                        costs = core.first_run_costs(source, k=k, mode=mode)["costs"]
                        options = []
                        for run in range(1, len(costs)):
                            low, high = interval(force, beta, run, exponent, epsilon)
                            options.append(dict(lo=low, hi=high, cost=costs[run], state=run))
                        compressed.append(options)
                    for compatible in (True, False):
                        oracle = exhaustive_combinations(literal, compatible)
                        algorithm = glob.optimize_intercept if compatible else glob.optimize_incompatibility
                        actual = algorithm(compressed)
                        assert actual["cost"] == oracle["cost"], (n, k, fixture, mode, compatible, actual, oracle)
                        assert actual["feasible"] == bool(np.isfinite(oracle["cost"]))
                        if actual["feasible"]:
                            total_flips = 0
                            lows, highs = [], []
                            for source, force, beta, options, choice in zip(sources, forces, betas, compressed, actual["choices"]):
                                option = options[choice]
                                run = option["state"]
                                witness = core.witness_first_run(source, run, k=k, mode=mode)
                                check_witness(source, witness["labels"], run, k, mode, option["cost"])
                                total_flips += witness["cost"]
                                lo, hi = interval(force, beta, first_run(witness["labels"], k), exponent, epsilon)
                                lows.append(lo)
                                highs.append(hi)
                            assert total_flips == actual["cost"]
                            assert (max(lows) <= min(highs)) == compatible
                        count += 1
                        combinations += oracle["combinations"]
    report = dict(status="passed", cases=count, target_label_combinations=combinations,
                  seed=20261021, elapsed_seconds=time.time() - started,
                  interpretation="Exact minimum label flips for finite first-run interval compatibility or incompatibility; not MLE rejection")
    (HERE / "end_to_end_results.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
