"""Summary of every frozen once-only test of the strict cZv/beta upper law (Claude, 2026-10-20). Reads saved scores only.
Force ceiling = 95th percentile of the measured t = 0.2 s window force of the session (label-free controls)."""
import csv, gzip, json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = Path.home() / "violin-rne"
TU = R / "novelty/thermal_upper"
rows = []
def ceiling(path, prefix=None):
    f = []
    with gzip.open(path, "rt") as fh:
        rd = csv.reader(fh); h = next(rd); i_f, i_w, i_t, i_fo = h.index("force"), h.index("window_s"), h.index("t"), h.index("folder")
        for r in rd:
            if r[i_w] == "0.2" and r[i_t] in ("0.2", "0.20") and (prefix is None or r[i_fo].startswith(prefix)):
                f.append(float(r[i_f]))
    return float(np.percentile(f, 95))
D = json.load(open(R / "astra_20261019/D_T2_confirm_result.json"))
rows.append(("D_T2", "frozen plan (CONFIRM_PLAN_D_T2)", D["level_fits"]["upper_strict"]["level"] if "level_fits" in D and "upper_strict" in D["level_fits"] else -0.0018, 0.12, ceiling(R / "timeres/features/2024-06-17_TypeD_T2_sample1.csv.gz")))
for stem, tag in (("2024-04-24_TypeA_sample4", "A4"), ("2024-04-26_TypeD_sample2", "D2")):
    s = json.load(open(TU / f"score_{stem}__FROZEN_v2.json"))["models"]["S1_strict"]
    rows.append((tag, "FROZEN_v2", s["level_error"], json.load(open(TU / "FROZEN_v2.json"))["fits"]["S1_strict"]["pi90"]["level"], ceiling(R / f"timeres/features/{stem}.csv.gz")))
b = json.load(open(R / "confirm_B_T2s2/score_result.json"))["primary_all_speeds"]["models"]["S1_strict"]
rows.append(("B_T2s2", "FROZEN_v2 (partially contaminated)", b["level_error"], b["pi90_level"], ceiling(R / "astra_20261019/highforce_stream/2024-07-03_TypeB_T2_sample2/features.csv.gz")))
hf = json.load(open(R / "highforce_confirm/score_result.json"))["targets"]
pref = {"HF_C": ("HighBowForceRange_1", "HighBowForceRange_1/2024-06-22"), "HF_A": ("HighBowForceRange_1", "HighBowForceRange_1/2024-06-24"),
        "HF_D": ("HighBowForceRange_2", "HighBowForceRange_2/2024-06-27"), "HF_B": ("HighBowForceRange_2", "HighBowForceRange_2/2024-06-29")}
for t, x in hf.items():
    s = x["t0.2"]["P2"]["S1_strict"]; a, p = pref[t]
    rows.append((t, "FROZEN_v2", s["level_error"], s["pi90_level"], ceiling(R / f"astra_20261019/highforce_stream/{a}/features.csv.gz", p)))
out = [dict(target=r[0], freeze=r[1], level_error=r[2], tolerance=r[3], passed=bool(abs(r[2]) <= r[3]), force_ceiling_p95_N=r[4]) for r in rows]
(R / "paper/repro/frozen_upper_summary.json").write_text(json.dumps(out, indent=1))
for o in out:
    print(f"{o['target']:7s} err {o['level_error']:+.3f} tol {o['tolerance']:.3f} pass {o['passed']} ceiling {o['force_ceiling_p95_N']:.1f} N  ({o['freeze']})")
e = np.array([o["level_error"] for o in out]); c = np.array([o["force_ceiling_p95_N"] for o in out])
from scipy.stats import spearmanr
rho = spearmanr(c, e)
print(f"passed {sum(o['passed'] for o in out)}/{len(out)}; SD of errors {e.std(ddof=1):.3f}; Spearman(ceiling, error) {rho[0]:+.2f} (p {rho[1]:.3f}, n {len(out)}; post hoc)")
fig, ax = plt.subplots(figsize=(6.2, 4))
for o in out:
    ax.errorbar(o["force_ceiling_p95_N"], o["level_error"], fmt="o", color="#2563a0" if o["passed"] else "#c0392b", ms=7)
    ax.annotate(o["target"], (o["force_ceiling_p95_N"], o["level_error"]), xytext={"HF_B": (-34, -12), "HF_C": (6, 4)}.get(o["target"], (5, 4)), textcoords="offset points", fontsize=8)
ax.axhspan(-0.148, 0.148, color="0.9", zorder=0, label="frozen 90 % interval (FROZEN_v2; D_T2 plan used 0.12)"); ax.axhline(0, color="k", lw=.7)
ax.set_xscale("log"); ax.set_xlabel("force ceiling of the session (95th pct of measured window force, N)")
ax.set_ylabel("strict cZv/β level error, observed − predicted (ln)")
ax.set_title(f"All frozen once-only tests of cZv/β: {sum(o['passed'] for o in out)}/{len(out)} inside\n(blue = pass, red = fail; ceiling trend is post hoc, n = {len(out)})", fontsize=9)
ax.legend(fontsize=8, frameon=False, loc="upper left")
for s in ("top", "right"): ax.spines[s].set_visible(False)
fig.tight_layout(); fig.savefig(R / "paper/fig5_frozen_upper_summary.png", dpi=180)
