"""claim_audit: rerun session_variability (read-only) + joint-MLE speed exponent per B_T2 repeat for comparability."""
import sys
from collections import defaultdict
sys.path.insert(0, "/Users/jeongjihoon/violin-rne/claude_extra")
import numpy as np
import verify_astra_boundaries as V
import session_variability as S

S.STEMS = ["2024-10-23_variability_setup"]
S.main()
rows, beta, speed = V.load("2024-10-23_variability_setup")
col = V.columns(beta)
for rep in ("rep_1", "rep_2"):
    r2 = {k: v for k, v in rows.items() if f"/{rep}/" in k[0] + "/"}
    b2 = {k: beta[k] for k in r2}
    s2 = {f: v for f, v in speed.items() if f"/{rep}/" in f + "/"}
    for t in V.TIMES:
        D = V.intervals(r2, b2, col, s2, t)
        x = V.fit(D)
        print(f"joint MLE {rep} t={t}: speed exp {x[1]:+.3f} beta exp {x[2]:+.3f} sigma {np.exp(x[3]):.2f} cols {len(D)} left {np.sum(~np.isfinite(D[:,2]))} right {np.sum(~np.isfinite(D[:,3]))}")
