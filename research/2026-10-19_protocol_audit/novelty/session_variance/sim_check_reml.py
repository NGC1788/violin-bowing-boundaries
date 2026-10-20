"""Simulation check of reml() (Claude, 2026-10-20): same group layout and SEs as the upper analysis."""
import json, numpy as np
import session_variance as S
r = json.load(open("results.json"))["upper"]
sess = list(S.SESSIONS); g = [S.SGROUP[s] for s in sess]; se = np.array([r["levels"][s]["se"] for s in sess])
groups = sorted(set(g)); rng = np.random.default_rng(5)
for tau, om in ((0.06, 0.07), (0.0, 0.07), (0.10, 0.0)):
    est = []
    for _ in range(200):
        a = {k: rng.normal(0, tau) for k in groups}
        y = np.array([a[gi] + rng.normal(0, om) + rng.normal(0, s) for gi, s in zip(g, se)])
        m = S.reml(y, se, g); est.append((m["tau"], m["omega"], m["p_omega0"] < 0.05, m["omega_ci"] is not None and m["omega_ci"][0] <= om <= m["omega_ci"][1]))
    e = np.array(est, float)
    print(f"true tau {tau} omega {om}: median tau {np.median(e[:,0]):.3f} omega {np.median(e[:,1]):.3f} | P(reject omega=0) {e[:,2].mean():.2f} | omega CI coverage {e[:,3].mean():.2f}")
