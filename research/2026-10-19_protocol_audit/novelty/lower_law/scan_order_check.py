"""POST HOC: within-scan beta/time-order confound of the minimum-force beta exponent (the open item in paper §3.8).
Scans run in ascending beta except A1 (descending). If a time drift d per scan adds to the fitted exponent, ascending
sessions read b_true + delta and A1 reads b_true - delta. Assuming A1 (sample 1) shares the true exponent of the
other A T1 samples (A2-A4), delta = (mean_ascending - b_A1) / 2. Uncertainty: bootstrap SEs plus the between-session
SD of A2-A4 (treated as extra session variance). One reversed session only; A1's attack schedule also differs.
Claude, 2026-10-20."""
import json, math, statistics as st
from pathlib import Path
H = Path(__file__).resolve().parent
on = json.loads((H / "onset_axis.json").read_text())


def se(ci):
    return (ci[1] - ci[0]) / 3.92


out = {}
for ax in ("onset", "window"):
    if ax == "onset":
        b = {s: (on["onset"][s]["b_beta"], se(on["onset"][s]["b_beta_ci"])) for s in ("A1", "A2", "A3", "A4")}
    else:
        b = {s: (on["measured_se"][s][2], on["measured_se"][s][3]) for s in ("A1", "A2", "A3", "A4")}
    asc = [b[s] for s in ("A2", "A3", "A4")]
    w = [1 / x[1] ** 2 for x in asc]; m = sum(wi * x[0] for wi, x in zip(w, asc)) / sum(w); sm = math.sqrt(1 / sum(w))
    sdb = st.stdev([x[0] for x in asc])
    d = (m - b["A1"][0]) / 2; sd = math.sqrt(sm ** 2 + b["A1"][1] ** 2 + sdb ** 2 * (1 + 1 / 3)) / 2
    out[ax] = dict(b={k: v[0] for k, v in b.items()}, ascending_mean=m, delta=d, delta_se=sd, ci95=[d - 1.96 * sd, d + 1.96 * sd], between_session_sd=sdb)
    print(f"{ax}: delta {d:+.3f} ± {sd:.3f} (95% [{d - 1.96 * sd:+.2f}, {d + 1.96 * sd:+.2f}])")
(H / "scan_order_check.json").write_text(json.dumps(out, indent=1))
