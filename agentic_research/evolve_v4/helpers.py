"""Building blocks a candidate policy may import (read-only library)."""
import numpy as np

from agentic_research.law_decision_pilot_v1.pilot import certificates, possible_starts  # noqa: F401
from agentic_research.law_counterfactual_selection_v1.experiment import (  # noqa: F401
    choose as counterfactual_choose, outer, response_patterns, shortlist,
)

UNIFORM10 = np.full(10, .1)
