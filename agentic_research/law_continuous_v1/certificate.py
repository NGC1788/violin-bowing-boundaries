"""Sound partial-observation certificate for a continuous threshold rectangle."""
from dataclasses import dataclass

import numpy as np

from agentic_research.law_decision_pilot_v1.pilot import certificates, possible_starts

U = (.7, .9)
V = (.3, .7)


@dataclass(frozen=True)
class Observations:
    known: np.ndarray
    fixed: np.ndarray
    guard: np.ndarray
    periodicity: np.ndarray
    ratio: np.ndarray

    @classmethod
    def empty(cls, shape):
        return cls(np.zeros(shape, bool), np.zeros(shape, bool), np.zeros(shape, bool),
                   np.zeros(shape), np.zeros(shape))

    def reveal(self, column, rank, fixed, guard, periodicity, ratio):
        self.known[column, rank] = True
        self.fixed[column, rank] = fixed
        self.guard[column, rank] = guard
        self.periodicity[column, rank] = periodicity
        self.ratio[column, rank] = ratio

    def labels(self, u, v):
        result = self.fixed | (self.guard & (self.periodicity >= u) & (self.ratio < v))
        return np.where(self.known, result, -1).astype(np.int8)


def representatives(scores, domain):
    low, high = domain
    return np.r_[np.unique(scores[(scores >= low) & (scores < high)]), high]


def decide(observations, lower, upper, cap=None):
    """Return decision and rectangle count. None denotes unresolved."""
    guarded = observations.known & observations.guard
    us = representatives(observations.periodicity[guarded], U)
    vs = representatives(observations.ratio[guarded], V)
    rectangles = [(0, len(us) - 1, 0, len(vs) - 1)]
    visited = 0
    unresolved = False

    # Try the existing baseline threshold first for a positive witness.
    anchor = observations.labels(.8, .5)
    decision, _, _ = certificates(possible_starts(anchor[:, None]), lower, upper)
    if decision is True:
        return True, visited

    while rectangles:
        if cap is not None and visited >= cap:
            return None, visited
        u0, u1, v0, v1 = rectangles.pop()
        visited += 1
        minimum = observations.labels(us[u1], vs[v0])
        maximum = observations.labels(us[u0], vs[v1])
        partial = np.where(minimum == maximum, minimum, -1)
        decision, _, _ = certificates(possible_starts(partial[:, None]), lower, upper)
        if decision is True:
            return True, visited
        if decision is False:
            continue
        if u0 == u1 and v0 == v1:
            unresolved = True
        elif u1 - u0 >= v1 - v0:
            mid = (u0 + u1) // 2
            rectangles.extend(((mid + 1, u1, v0, v1), (u0, mid, v0, v1)))
        else:
            mid = (v0 + v1) // 2
            rectangles.extend(((u0, u1, mid + 1, v1), (u0, u1, v0, mid)))
    return (None if unresolved else False), visited
