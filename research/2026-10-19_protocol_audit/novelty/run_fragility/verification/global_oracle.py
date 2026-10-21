"""Check the global intercept algorithm by full candidate combinations.

No endpoint sweep is used in this oracle: every choice of one interval per
column is tested directly for a nonempty closed intersection.
"""
from __future__ import annotations

import importlib.util
import itertools
import json
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent


def exhaustive_combinations(columns, compatible=True):
    best = np.inf
    choices = None
    combinations = 0
    for selection in itertools.product(*(range(len(column)) for column in columns)):
        combinations += 1
        selected = [column[index] for column, index in zip(columns, selection)]
        low = max((option["lo"] for option in selected), default=-np.inf)
        high = min((option["hi"] for option in selected), default=np.inf)
        cost = sum(option["cost"] for option in selected)
        if ((low <= high) == compatible) and cost < best:
            best = cost
            choices = selection
    return dict(cost=best, choices=choices, combinations=combinations)


def make_fixtures(seed=20261021, count=1500):
    fixtures = [
        [],
        [[]],
        [[dict(lo=0., hi=0., cost=2., state=0)]],
        [[dict(lo=-np.inf, hi=0., cost=1., state=0)],
         [dict(lo=0., hi=np.inf, cost=3., state=0)]],
        [[dict(lo=-np.inf, hi=-1., cost=0., state=0)],
         [dict(lo=0., hi=np.inf, cost=0., state=0)]],
        [[dict(lo=-np.inf, hi=np.inf, cost=4., state=0)],
         [dict(lo=-np.inf, hi=np.inf, cost=1., state=0)]],
        [[dict(lo=0., hi=1., cost=np.inf, state=0)],
         [dict(lo=0., hi=1., cost=0., state=0)]],
        [[dict(lo=0., hi=1., cost=1., state=0),
          dict(lo=1., hi=2., cost=0., state=1)],
         [dict(lo=1., hi=1., cost=0., state=0)]],
    ]
    rng = np.random.default_rng(seed)
    endpoints = np.arange(-5, 6) / 2
    for _ in range(count):
        columns = []
        for _ in range(int(rng.integers(1, 6))):
            options = []
            for state in range(int(rng.integers(1, 6))):
                left, right = np.sort(rng.choice(endpoints, size=2))
                if rng.random() < .08:
                    left = -np.inf
                if rng.random() < .08:
                    right = np.inf
                cost = float(rng.integers(0, 6))
                if rng.random() < .08:
                    cost = np.inf
                options.append(dict(lo=float(left), hi=float(right), cost=cost, state=state))
            columns.append(options)
        fixtures.append(columns)
    return fixtures


def main():
    spec = importlib.util.spec_from_file_location("verified_global_profile", HERE.parent / "global_profile.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    start = time.time()
    count = 0
    feasible = 0
    combinations = 0
    for compatible in (True, False):
        algorithm = module.optimize_intercept if compatible else module.optimize_incompatibility
        for number, columns in enumerate(make_fixtures()):
            expected = exhaustive_combinations(columns, compatible=compatible)
            actual = algorithm(columns)
            combinations += expected["combinations"]
            assert actual["cost"] == expected["cost"], (compatible, number, columns, actual, expected)
            assert actual["feasible"] == bool(np.isfinite(expected["cost"])), (compatible, number, actual, expected)
            if actual["feasible"]:
                feasible += 1
                selected = [column[index] for column, index in zip(columns, actual["choices"])]
                assert len(actual["choices"]) == len(columns), (number, actual)
                assert sum(option["cost"] for option in selected) == actual["cost"], (number, actual)
                low = max((option["lo"] for option in selected), default=-np.inf)
                high = min((option["hi"] for option in selected), default=np.inf)
                assert (low <= high) == compatible, (compatible, number, actual, selected)
                if compatible:
                    point = actual["intercept"]
                    assert np.isfinite(point), (number, actual)
                    assert all(option["lo"] <= point <= option["hi"] for option in selected), (number, actual, selected)
            count += 1
    report = dict(status="passed", fixtures=count, feasible=feasible,
                  enumerated_combinations=combinations, seed=20261021,
                  elapsed_seconds=time.time() - start,
                  oracle="Full Cartesian choice enumeration, closed interval intersection and nonintersection")
    (HERE / "global_results.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
