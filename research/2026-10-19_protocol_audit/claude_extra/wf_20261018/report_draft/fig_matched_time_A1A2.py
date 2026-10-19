"""Figure for the final-report draft (report_draft track, Claude 2026-10-18).

(a) Matched-time speed exponent of the lower (minimum-force) boundary vs window start t, A1 vs A2.
    Numbers: recomputed here with claude_extra/verify_astra_boundaries.fit (independent implementation)
    and cross-checked against astra_20261018/results/descriptive_boundaries.json (read-only).
(b) Speed-specific boundary levels at beta = 0.1 (separate intercept per speed, common beta slope,
    as in claude_extra/per_speed_levels.py), A1 vs A2, t = 0 and t = 0.3 s.
Intervals: beta-column resampling within each speed (conditional on the specimen/session, fixed classifier
and grid; NOT session-to-session uncertainty).
Reads only A1/A2 feature files. Writes only into this folder.
Usage: OPENBLAS_NUM_THREADS=1 python fig_matched_time_A1A2.py [--plot-only]
"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(Path.home() / "violin-rne/claude_extra"))
import verify_astra_boundaries as V  # noqa: E402

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager  # noqa: E402

font_manager.fontManager.addfont("/System/Library/Fonts/AppleSDGothicNeo.ttc")
plt.rcParams["font.family"] = "Apple SD Gothic Neo"
plt.rcParams["axes.unicode_minus"] = False

COL = {"A1": "#2a78d6", "A2": "#eb6834"}
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e6e5e1"
REPS = 100


def fit_levels(D):
    lv, lb, lo, hi = D.T
    levels = np.unique(lv)
    idx = np.searchsorted(levels, lv)

    def nll(x):
        mu = x[idx] + x[3] * lb
        s = np.exp(x[4])
        p = np.where(np.isfinite(hi), norm.cdf((hi - mu) / s), 1.0) - np.where(np.isfinite(lo), norm.cdf((lo - mu) / s), 0.0)
        return -np.sum(np.log(np.clip(p, 1e-300, None)))

    r = minimize(nll, [-2, -2, -2, -1, np.log(0.5)], method="Nelder-Mead",
                 options={"maxiter": 40000, "xatol": 1e-8, "fatol": 1e-10})
    return r.x, np.exp(levels)


def strat_resample(D, rng):
    """Resample beta columns within each speed (keeps 40 columns per speed)."""
    out = []
    for lv in np.unique(D[:, 0]):
        sub = D[D[:, 0] == lv]
        out.append(sub[rng.integers(0, len(sub), len(sub))])
    return np.vstack(out)


def main():
    res = {}
    for tag, stem in V.SPECIMENS.items():
        rows, beta, speed = V.load(stem)
        col = V.columns(beta)
        res[tag] = {"speeds": speed}
        for t in V.TIMES:
            D = V.intervals(rows, beta, col, speed, t)
            x = V.fit(D)
            lev, vs = fit_levels(D)
            F01 = np.exp(lev[:3] + lev[3] * np.log(0.1))
            rng = np.random.default_rng(1)
            bs_exp, bs_F = [], []
            for _ in range(REPS):
                Db = strat_resample(D, rng)
                bs_exp.append(V.fit(Db)[1])
                lb_, _ = fit_levels(Db)
                bs_F.append(np.exp(lb_[:3] + lb_[3] * np.log(0.1)))
            bs_F = np.array(bs_F)
            res[tag][f"{t:.1f}"] = {
                "speed_exp": float(x[1]), "beta_exp": float(x[2]), "sigma": float(np.exp(x[3])),
                "speed_exp_ci95": [float(np.percentile(bs_exp, 2.5)), float(np.percentile(bs_exp, 97.5))],
                "levels_speed_ms": [float(v) for v in vs],
                "F_at_beta0.1_N": [float(f) for f in F01],
                "F_at_beta0.1_ci95": [[float(np.percentile(bs_F[:, i], 2.5)), float(np.percentile(bs_F[:, i], 97.5))] for i in range(3)],
                "left_censored": int(np.sum(~np.isfinite(D[:, 2]))), "right_censored": int(np.sum(~np.isfinite(D[:, 3]))),
                "columns": int(len(D)),
            }
            r = res[tag][f"{t:.1f}"]
            print(f"{tag} t={t:.1f}: speed exp {r['speed_exp']:+.3f} {np.round(r['speed_exp_ci95'], 2)}  beta {r['beta_exp']:+.3f}  "
                  f"F(beta 0.1) {np.round(r['F_at_beta0.1_N'], 3)}  L/R cens {r['left_censored']}/{r['right_censored']}", flush=True)

    # cross-check against Astra's JSON (read-only)
    astra = json.loads((Path.home() / "violin-rne/astra_20261018/results/descriptive_boundaries.json").read_text())
    check = []
    for s in astra["settings"]:
        mine = res[s["specimen"]][f"{s['window_start_s']:.1f}"]["speed_exp"]
        check.append({"specimen": s["specimen"], "t": s["window_start_s"], "astra": s["fit"]["speed_exp"], "mine": mine,
                      "astra_ci95_position_cluster": s["position_cluster_sensitivity"]["speed_percentile_95"],
                      "abs_diff": abs(mine - s["fit"]["speed_exp"])})
    res["crosscheck_vs_astra"] = check
    res["note"] = ("Intervals: beta-column resampling within speed, 100 reps, conditional on specimen/session, "
                   "fixed classifier (repo regime_map.classify, A2 thresholds, f0 98.08 Hz) and force grid; "
                   "not session-to-session uncertainty. Boundary = first K=2 consecutive Helmholtz windows "
                   "(0.2 s windows) in the column sorted by current window force; interval-censored lognormal fit.")
    (HERE / "matched_time_A1A2.json").write_text(json.dumps(res, indent=1, ensure_ascii=False))
    plot(res)


def plot(res):
    from matplotlib.ticker import NullFormatter
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    ax = axes[0]
    ts = np.array(V.TIMES)
    for k, tag in enumerate(("A1", "A2")):
        e = np.array([res[tag][f"{t:.1f}"]["speed_exp"] for t in V.TIMES])
        lo = np.array([res[tag][f"{t:.1f}"]["speed_exp_ci95"][0] for t in V.TIMES])
        hi = np.array([res[tag][f"{t:.1f}"]["speed_exp_ci95"][1] for t in V.TIMES])
        off = (k - 0.5) * 0.012
        ax.errorbar(ts + off, e, yerr=[e - lo, hi - e], color=COL[tag], marker="o", ms=7, lw=2, capsize=0,
                    label=f"{tag} ({'2024-03-25' if tag == 'A1' else '2024-04-19'} TypeA)")
        ax.text(ts[-1] + 0.02, e[-1], tag, color=INK, va="center", fontsize=10)
    ax.axhline(1.0, color=INK2, lw=1, ls=":")
    ax.axhline(0.0, color=INK2, lw=1, ls=":")
    ax.text(0.222, 1.03, "+1 (Schelleng·Beigi)", color=INK2, fontsize=8, va="bottom", ha="left")
    ax.text(0.355, 0.03, "0 (Schoonderwaldt 2008: 속도 무관)", color=INK2, fontsize=8, va="bottom", ha="right")
    ax.set_xlim(-0.04, 0.36)
    ax.set_ylim(-0.15, 1.5)
    ax.set_xlabel("평탄 구간 시작 후 창 시작 시각 t (s), 0.2 s 창")
    ax.set_ylabel("최소 활 힘 경계의 속도 지수")
    ax.set_title("(a) 같은 시각 비교: 속도 지수", fontsize=11, color=INK)
    ax.legend(frameon=False, fontsize=8, loc="upper right")

    ax = axes[1]
    for tag in ("A1", "A2"):
        for t, ls, mk in (("0.0", "-", "o"), ("0.3", "--", "s")):
            r = res[tag][t]
            v = np.array(r["levels_speed_ms"]); F = np.array(r["F_at_beta0.1_N"]); ci = np.array(r["F_at_beta0.1_ci95"])
            ax.errorbar(v * (1.0 if tag == "A1" else 1.04), F, yerr=[F - ci[:, 0], ci[:, 1] - F], color=COL[tag], ls=ls,
                        marker=mk, ms=7, lw=2, capsize=0, label=f"{tag}, t = {t} s")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xticks([0.05, 0.1, 0.2], ["0.05", "0.10", "0.20"])
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.set_yticks([0.1, 0.2, 0.5, 1.0], ["0.1", "0.2", "0.5", "1.0"])
    ax.set_xlabel("활 속도 (m/s, 측정값)")
    ax.set_ylabel("β = 0.1에서의 경계 힘 (N)")
    ax.set_title("(b) 속도별 경계 수준: 차이는 0.05 m/s에 몰림", fontsize=11, color=INK)
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    for a in axes:
        a.grid(True, color=GRID, lw=0.8); a.set_axisbelow(True)
        for sp in ("top", "right"):
            a.spines[sp].set_visible(False)
    fig.suptitle("A1·A2 시간 맞춤 하한 경계 (K=2 연속 헬름홀츠, 현재 창 힘, 구간 검열). 오차 막대 = 같은 속도 안 β 열 재표본 95% (세션 변동 미포함)",
                 fontsize=9, color=INK2)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(HERE / "fig_matched_time_A1A2.png", dpi=160)
    print("saved", HERE / "fig_matched_time_A1A2.png")


if __name__ == "__main__":
    if "--plot-only" in sys.argv:
        plot(json.loads((HERE / "matched_time_A1A2.json").read_text()))
    else:
        main()
