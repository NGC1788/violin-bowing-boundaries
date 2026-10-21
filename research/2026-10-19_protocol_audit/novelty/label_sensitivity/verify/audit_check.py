"""Verifier (claims and fairness audit), 2026-10-21. Read-only w.r.t. the parent analysis; writes only verify/.
(1) Re-derive every number quoted in RESULT.md / draft_v2 3.8 / 최종보고서 10.8 from results.json + posthoc_amplitude.json.
(2) Check the post hoc mechanism claims: within-COLUMN correlation of amp1x-lost vs log force (the parent computed it
    across the whole session), and how base-Helmholtz amplitude actually scales with v and beta.
(3) A closer analogue of the simulation's amplitude check (amp >= 0.5 v/beta): a v/beta-SCALED floor,
    amp >= r * k_sess * v/beta, k_sess = median(amp * beta / v) over base-Helmholtz windows of the session."""
import json, sys
from collections import defaultdict
from pathlib import Path

import numpy as np

LSD = Path.home() / "violin-rne/novelty/label_sensitivity"; sys.path.insert(0, str(LSD))
import label_sweep as LS  # noqa: E402

R = json.loads((LSD / "results.json").read_text()); P = json.loads((LSD / "posthoc_amplitude.json").read_text())
out = {"numbers": {}, "within_column_corr": {}, "amp_scaling": {}, "vbeta_floor": {}, "vbeta_removal": {}}

# ---- (1) numbers
m = P["mechanism"]
out["numbers"]["tercile_small_beta"] = [min(m[s]["beta_tercile_0"] for s in m), max(m[s]["beta_tercile_0"] for s in m)]
out["numbers"]["tercile_large_beta"] = [min(m[s]["beta_tercile_2"] for s in m), max(m[s]["beta_tercile_2"] for s in m)]
out["numbers"]["corr_session_wide"] = [min(m[s]["corr_lost_vs_logforce_within_H"] for s in m), max(m[s]["corr_lost_vs_logforce_within_H"] for s in m)]
for ax in ("window", "onset"):
    rows = R["summary"][ax]["rows"]; means = {k: v["mean"] for k, v in rows.items()}
    means.update({k: v["mean"] for k, v in P["summary"][ax].items()})
    neutral = {k: v for k, v in means.items() if not k.startswith("amp")}
    fly = {k: v for k, v in means.items() if k.startswith("fly") or k == "base"}
    out["numbers"][ax] = dict(S_prereg=R["summary"][ax]["S"], means=means,
                              neutral_range=[min(neutral.values()), max(neutral.values()), max(neutral.values()) - min(neutral.values())],
                              flyback_span=max(fly.values()) - min(fly.values()),
                              shifts={k: v["shift"] for k, v in rows.items()} | {k: v["shift"] for k, v in P["summary"][ax].items()},
                              toward_minus2={k: v["toward_minus2"] for k, v in rows.items()})
out["numbers"]["gate"] = {k: R["gate"][k] for k in ("window_max_abs_diff", "onset_max_abs_diff", "pass_")}
out["numbers"]["gate_window_sessions"] = R["gate"]["window"]

# ---- (2)+(3)
for sess, st in LS.STEM.items():
    stem, rep = st if isinstance(st, tuple) else (st, None)
    raw = LS.read(stem, rep)
    d0 = LS.build(raw, LS.VARIANTS["base"]); d1 = LS.build(raw, LS.VARIANTS["amp1x"])
    amp = np.array([r[3] for r in raw[0]]); H = d0["H"]
    lost = (H & ~d1["H"]).astype(float)
    key = np.array([f"{a}_{b}" for a, b in zip(d0["folder"], d0["column"])])
    with np.errstate(invalid="ignore", divide="ignore"):
        lf = np.log(d0["force"])
    good = np.isfinite(lf)
    # within-column: demean both within (folder, column) among H windows
    Hg = H & good; lh, fh, kh = lost[Hg], lf[Hg], key[Hg]
    dl, dfz = lh.copy(), fh.copy()
    for k in np.unique(kh):
        s = kh == k; dl[s] -= lh[s].mean(); dfz[s] -= fh[s].mean()
    out["within_column_corr"][sess] = float(np.corrcoef(dl, dfz)[0, 1]) if dl.std() > 0 else None
    # amplitude scaling over base-H windows
    ok = H & good & np.isfinite(amp) & (amp > 0)
    X = np.column_stack([np.ones(ok.sum()), np.log(d0["speed"][ok]), np.log(d0["beta"][ok]), lf[ok]])
    co = np.linalg.lstsq(X, np.log(amp[ok]), rcond=None)[0]
    out["amp_scaling"][sess] = dict(p_v=float(co[1]), p_beta=float(co[2]), p_F=float(co[3]))
    # v/beta-scaled floor
    okk = H & np.isfinite(amp) & (amp > 0); vb = d0["speed"] / d0["beta"]; ksess = float(np.median(amp[okk] / vb[okk]))
    lb = np.log(d0["beta"]); terc = np.digitize(lb, np.quantile(lb, [1 / 3, 2 / 3]))
    e = {}
    for r in (0.3, 0.5, 0.7):
        d = dict(d0); d["H"] = H & np.isfinite(amp) & (amp >= r * ksess * vb)
        e[f"vb{r}"] = dict(window_b_beta=LS.fit(d)[1], changed=float(np.mean(d["H"] != H)))
        if sess in LS.ONSET:
            e[f"vb{r}"]["onset_b_beta"] = LS.fit(LS.to_onset(d, sess))[1]
        rem = H & ~d["H"]
        out["vbeta_removal"].setdefault(f"vb{r}", {})[sess] = [float(rem[terc == i].sum() / max(1, H[terc == i].sum())) for i in range(3)]
    out["vbeta_floor"][sess] = e
    print(sess, "wcorr", round(out["within_column_corr"][sess] or float("nan"), 2), "amp p_beta", round(co[2], 2), "p_v", round(co[1], 2),
          "| vb0.5 β", round(e["vb0.5"]["window_b_beta"], 2), "removal terciles", [round(x, 2) for x in out["vbeta_removal"]["vb0.5"][sess]], flush=True)

base = R["sessions"]; out["vbeta_summary"] = {}
for ax, ss in (("window", list(LS.STEM)), ("onset", LS.ONSET)):
    for r in ("vb0.3", "vb0.5", "vb0.7"):
        x = np.array([out["vbeta_floor"][s][r][f"{ax}_b_beta"] for s in ss]); b = np.array([base[s]["base"][f"{ax}_b_beta"] for s in ss])
        mm, sd, _ = LS.tci(x); dm, _, dci = LS.tci(x - b)
        out["vbeta_summary"].setdefault(ax, {})[r] = dict(mean=mm, sd=sd, shift=dm, shift_ci=dci,
                                                           changed=float(np.mean([out["vbeta_floor"][s][r]["changed"] for s in ss])))
        print(f"{ax} {r}: mean {mm:+.3f} shift {dm:+.3f} [{dci[0]:+.3f}, {dci[1]:+.3f}]")
(Path(__file__).resolve().parent / "audit_check.json").write_text(json.dumps(out, indent=1, default=float))
print(json.dumps(out["numbers"], indent=1, default=float))
