"""Seed policy: the published counterfactual interval lookahead (uniform responses)."""
from agentic_research.evolve_v4.helpers import UNIFORM10, counterfactual_choose, response_patterns


def choose(view):
    patterns = response_patterns(view.rules)
    return counterfactual_choose(view.partial, view.possible, view.lower, view.upper,
                                 patterns, UNIFORM10)
