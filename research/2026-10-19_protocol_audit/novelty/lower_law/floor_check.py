"""POST HOC follow-up (decided after seeing lower_law results): does the session's force floor (label-free) track the
lower-boundary beta exponent? floor = median over beta columns of the lowest measured t = 0.2 s window force (all speeds);
also the share of strokes reading below 0.05 N (offset/zero region). Claude, 2026-10-20."""
import json, sys
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr
H = Path.home(); sys.path.insert(0, str(H / "violin-rne/novelty/session_variance"))
import session_variance as SV  # noqa: E402
import score_bt2s2 as BT  # noqa: E402  (path added by lower_law import below)
