import unittest

import numpy as np

from agentic_research.law_continuous_v1.certificate import Observations, U, V, decide, representatives
from agentic_research.law_decision_pilot_v1.pilot import intervals


def exact(obs, lower, upper):
    guarded = obs.guard
    for u in representatives(obs.periodicity[guarded], U):
        for v in representatives(obs.ratio[guarded], V):
            lo, hi = [], []
            for c, bits in enumerate(obs.labels(u, v)):
                i = next((i for i in range(len(bits) - 1) if bits[i] == bits[i+1] == 1), None)
                if i is None or not np.isfinite(lower[c, 0, i]):
                    break
                lo.append(lower[c, 0, i])
                hi.append(upper[c, 0, i])
            else:
                if max(lo) <= min(hi):
                    return True
    return False


class CertificateTest(unittest.TestCase):
    def test_continuous_endpoint_and_partial_soundness(self):
        rng = np.random.default_rng(32)
        lo, hi = intervals([{"force": np.exp(np.arange(6)*.3), "beta": b} for b in (1., 1.08)])
        outcomes = set()
        for _ in range(100):
            obs = Observations.empty((2, 6))
            per = rng.choice([.69, .7, .8, .9], size=(2, 6))
            ratio = rng.choice([.29, .3, .5, .7], size=(2, 6))
            order = rng.permutation(12)
            early = []
            for k, position in enumerate(order):
                col, rank = divmod(position, 6)
                obs.reveal(col, rank, False, True, per[col, rank], ratio[col, rank])
                if k in (0, 4, 8):
                    early.append(decide(obs, lo, hi)[0])
            target = exact(obs, lo, hi)
            self.assertEqual(decide(obs, lo, hi)[0], target)
            self.assertTrue(all(x is None or x == target for x in early))
            outcomes.add(target)
        self.assertEqual(outcomes, {True, False})


if __name__ == "__main__":
    unittest.main()
