"""Cross-check cached endpoint replacement against recomputing all columns."""
import unittest

import numpy as np

from agentic_research.engine_v2.harness.data import rule_bank
from agentic_research.law_decision_pilot_v1.pilot import certificates, intervals, possible_starts
from agentic_research.law_counterfactual_selection_v1.experiment import (
    choose, outer, response_patterns, shortlist,
)


class CounterfactualTest(unittest.TestCase):
    def test_shortlist_scores_equal_full_recomputation(self):
        rng = np.random.default_rng(902)
        patterns = response_patterns(rule_bank("nine"))
        self.assertEqual(len(np.unique(patterns, axis=0)), 10)
        probabilities = np.arange(1, 11) / 55
        forces = np.exp(np.arange(8) * .12)
        lower, upper = intervals([{"force": forces, "beta": beta} for beta in (1, 1.05, 1.1)])
        checked = 0
        for _ in range(60):
            partial = patterns[rng.integers(len(patterns), size=(3, 8))].transpose(0, 2, 1)
            missing = rng.random((3, 8)) < .6
            partial = np.where(missing[:, None], -1, partial)
            possible = possible_starts(partial)
            if certificates(possible, lower, upper)[0] is not None:
                continue
            left, right = outer(possible, lower, upper)
            live = left.max(0) <= right.min(0)
            cols, ranks, priorities = shortlist(partial, possible, lower, upper, left, right, live)
            scores = []
            for col, rank, priority in zip(cols, ranks, priorities):
                count, width = [], []
                for pattern in patterns:
                    updated = partial.copy()
                    updated[col, :, rank] = pattern
                    a, b = outer(possible_starts(updated), lower, upper)
                    lo, hi = a.max(0), b.min(0)
                    remain = (lo <= hi) & live
                    count.append(remain.sum())
                    width.append(np.where(remain, np.maximum(hi - lo, 0), 0).sum())
                scores.append((round(np.dot(count, probabilities), 12),
                               round(np.dot(width, probabilities), 12), -priority, rank, col))
            winner = min(scores)
            self.assertEqual(choose(partial, possible, lower, upper, patterns, probabilities),
                             (winner[-1], winner[-2]))
            checked += 1
        self.assertGreater(checked, 20)


if __name__ == "__main__":
    unittest.main()
