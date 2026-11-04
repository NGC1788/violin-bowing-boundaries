"""Building blocks a candidate policy may import (same names as on the Mac)."""
import numpy as np

from agentic_research.evolve_v4.core import (  # noqa: F401
    certificates, counterfactual_choose, outer, possible_starts, response_patterns, shortlist,
)

UNIFORM10 = np.full(10, .1)
