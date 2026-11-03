import itertools
import unittest

import numpy as np

from agentic_research.law_decision_pilot_v1.pilot import certificates, possible_starts


def completions(partial):
    missing = np.flatnonzero(partial == -1)
    for bits in itertools.product((0, 1), repeat=len(missing)):
        result = partial.copy()
        result[missing] = bits
        yield result


def first(bits):
    return next((i for i, pair in enumerate(zip(bits, bits[1:])) if tuple(pair) == (1, 1)),
                len(bits) - 1)


class PilotTests(unittest.TestCase):
    def test_all_partial_sequences_up_to_length_six(self):
        for size in range(2, 7):
            for values in itertools.product((-1, 0, 1), repeat=size):
                partial = np.array(values)
                expected = {first(bits) for bits in completions(partial)}
                self.assertEqual(set(np.flatnonzero(possible_starts(partial))), expected)

    def test_every_partial_pair_of_short_columns(self):
        lower = np.array([[[np.nan, 0, 1, np.nan]], [[np.nan, 0.5, 3, np.nan]]])
        upper = lower + 1
        for values in itertools.product((-1, 0, 1), repeat=8):
            partial = np.array(values).reshape(2, 1, 4)
            prediction, resolved, _ = certificates(possible_starts(partial), lower, upper)
            if prediction is None:
                self.assertFalse(resolved)
                continue
            for bits in completions(partial.ravel()):
                starts = [first(col) for col in bits.reshape(2, 4)]
                lo = lower[np.arange(2), 0, starts]
                hi = upper[np.arange(2), 0, starts]
                self.assertEqual(prediction, bool(lo.max() <= hi.min()))


if __name__ == "__main__":
    unittest.main()
