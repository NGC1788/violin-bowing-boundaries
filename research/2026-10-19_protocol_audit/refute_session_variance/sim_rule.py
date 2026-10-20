"""Is the reading rule 'omega_hat >= tau_hat and omega=0 rejected' informative with this layout (8 groups, 13 sessions,
within-group df 5) and these SEs? Simulate truths where string type dominates. Post hoc; Claude 2026-10-20."""
import json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent)); import stats as ST
rng = np.random.default_rng(7)
out = {}
for var, truths in (("U_S1", [(0.06, 0.07), (0.10, 0.05), (0.12, 0.04), (0.15, 0.07), (0.10, 0.07)]),
                    ("L_A12", [(0.114, 0.158), (0.20, 0.10), (0.25, 0.12), (0.20, 0.158)])):
    S = ST.sessions(var); s = list(S); g = [ST.GRP[k] for k in s]; se = np.array([S[k][1] for k in s]); G = sorted(set(g))
    for tau, om in truths:
        R = []
        for _ in range(300):
            a = {k: rng.normal(0, tau) for k in G}
            y = np.array([a[gi] + rng.normal(0, om) + rng.normal(0, e) for gi, e in zip(g, se)])
            r = ST.reml(y, se, g, ci=False); R.append((r["tau"], r["omega"], r["p_omega0"] < 0.05))
        R = np.array(R, float)
        rule = np.mean((R[:, 1] >= R[:, 0]) & (R[:, 2] > 0))
        out[f"{var} tau{tau} om{om}"] = dict(med_tau=float(np.median(R[:, 0])), med_om=float(np.median(R[:, 1])), P_tau_hat0=float(np.mean(R[:, 0] < 1e-3)),
                                              P_om_ge_tau=float(np.mean(R[:, 1] >= R[:, 0])), P_rule=float(rule), tau_q=np.quantile(R[:, 0], [.1, .9]).tolist())
        o = out[f"{var} tau{tau} om{om}"]
        print(f"{var} true tau {tau} omega {om}: median tau_hat {o['med_tau']:.3f} (10-90% {o['tau_q'][0]:.3f}-{o['tau_q'][1]:.3f}, P tau_hat=0 {o['P_tau_hat0']:.2f}) "
              f"omega_hat {o['med_om']:.3f} | P(omega_hat>=tau_hat) {o['P_om_ge_tau']:.2f} | P(rule 'session>=type' fires) {o['P_rule']:.2f}", flush=True)
(Path(__file__).parent / "sim_rule.json").write_text(json.dumps(out, indent=1))
