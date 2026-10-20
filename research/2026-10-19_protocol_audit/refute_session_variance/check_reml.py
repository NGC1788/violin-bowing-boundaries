"""Own REML vs session_variance.reml() on the claim's own levels/SEs (post hoc)."""
import json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent)); import stats as ST
sys.path.insert(0, str(Path.home() / "violin-rne/novelty/session_variance")); import session_variance as SV
R = json.loads((Path.home() / "violin-rne/novelty/session_variance/results.json").read_text())
for w in ("upper", "lower"):
    lev = R[w]["levels"]; s = list(lev); y = [lev[k]["level"] for k in s]; se = [lev[k]["se"] for k in s]; g = [SV.SGROUP[k] for k in s]
    a = SV.reml(y, se, g); b = ST.reml(y, se, g); m = ST.mom(y, se, g)
    print(w, "claim reml():", {k: (np.round(v, 4) if not isinstance(v, (list, int)) else v) for k, v in a.items()})
    print(w, "own   reml  :", {k: (np.round(v, 4) if not isinstance(v, list) else v) for k, v in b.items()}, "MoM", m)
