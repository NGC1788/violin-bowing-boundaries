"""Public input contract, boundary cases, and deterministic witness checks."""
from pathlib import Path
import json
import numpy as np
from exhaustive import load_file

HERE = Path(__file__).resolve().parent


def main():
    core = load_file("run_core_validation", HERE.parent / "core.py")
    rejected = 0
    cases = [
        lambda: core.first_run_costs([0, 2]),
        lambda: core.first_run_costs([0, -1]),
        lambda: core.first_run_costs([0, .5]),
        lambda: core.first_run_costs([0, np.nan]),
        lambda: core.first_run_costs([0, np.inf]),
        lambda: core.first_run_costs([[0, 1]]),
        lambda: core.first_run_costs(1),
        lambda: core.first_run_costs(["0", "1"]),
        lambda: core.first_run_costs(np.array([0, 1], dtype=object)),
        lambda: core.first_run_costs([0j, 1j]),
        lambda: core.first_run_costs([0, 1], k=0),
        lambda: core.first_run_costs([0, 1], k=-1),
        lambda: core.first_run_costs([0, 1], k=2.5),
        lambda: core.first_run_costs([0, 1], k=True),
        lambda: core.first_run_costs([0, 1], mode="unknown"),
        lambda: core.witness_first_run([0, 1], -1),
        lambda: core.witness_first_run([0, 1], 1),
        lambda: core.witness_first_run([0, 1], .5),
        lambda: core.witness_first_run([0, 1], True),
        lambda: core.witness_first_run([], 0),
    ]
    for call in cases:
        try:
            call()
        except (ValueError, TypeError):
            rejected += 1
        else:
            raise AssertionError(f"Invalid input case {rejected} was accepted")
    # Float and bool binary vectors should be accepted as the same labels.
    for labels in ([0., 1., 1.], [False, True, True], np.array([0, 1, 1], dtype=np.uint8)):
        result = core.first_run_costs(labels, k=np.int64(2))
        assert result["costs"][1] == 0
    tie_source = [1, 1, 1, 1, 1]
    repeated = [core.witness_first_run(tie_source, None, k=2, mode="delete") for _ in range(5)]
    for item in repeated[1:]:
        np.testing.assert_array_equal(item["labels"], repeated[0]["labels"])
    supplied = np.array([0, 0, 0, 1, 1], dtype=np.uint8)
    original = supplied.copy()
    core.first_run_costs(supplied)
    core.witness_first_run(supplied, 0)
    np.testing.assert_array_equal(supplied, original)
    report = dict(status="passed", invalid_cases_rejected=rejected,
                  valid_dtype_cases=3, deterministic_repetitions=5,
                  input_mutation_check="unchanged",
                  note="Labels are ordered inputs: force sorting and force ties belong to the caller, not this core.")
    (HERE / "validation_results.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
