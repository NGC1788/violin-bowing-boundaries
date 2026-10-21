"""B1 corner-width labels (PLAN.md): gate against stored features, variants, paired column bootstrap, regression,
figure. Claude, 2026-10-21."""
import csv, json, math, sys
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE.parent))
import label_sweep as LS  # noqa: E402
C, LL = LS.C, LS.LL
STEM = "2024-04-10_TypeB_sample1"


def num(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return math.nan


def same(a, b):
    return (math.isnan(a) and math.isnan(b)) or abs(a - b) <= 1e-9


def keyed(d):
    rows = [r for r in C.intervals(d, 0.2) if np.isfinite(r["lower"][0]) or np.isfinite(r["lower"][1])]
    return {(round(r["lv"], 6), round(r["lb"], 6)): r for r in rows}


def paired_boot(kb, kv, nb=200, seed=11):
    keys = sorted(set(kb) & set(kv)); rng = np.random.default_rng(seed); out = []
    for _ in range(nb):
        kk = [keys[i] for i in rng.choice(len(keys), len(keys))]
        out.append(LL.fit([kv[k] for k in kk])[1] - LL.fit([kb[k] for k in kk])[1])
    return np.percentile(out, [2.5, 97.5]).tolist()


def main():
    rec = {(r["folder"], int(r["trial"])): r for r in csv.DictReader(open(HERE / "b1_flyback.csv"))}
    raw = LS.read(STEM, None)
    rows = raw[0]
    # gate
    ok = n = 0; errs = sum(1 for r in rec.values() if r["error"])
    for (k, f, v, a, p, f0, fpp, sfr) in rows:
        r = rec.get(k); n += 1
        if r and not r["error"] and same(num(r["periodicity"]), p) and same(num(r["flybacks_per_period"]), fpp) and same(num(r["second_flyback_ratio"]), sfr):
            ok += 1
    gate = dict(matched=ok, total=n, share=ok / n, stream_errors=errs, pass_=bool(ok / n >= 0.999))
    print("gate", gate, flush=True)
    res = {"gate": gate}
    d0 = LS.build(raw, LS.VARIANTS["base"])
    dfb = np.array([num(rec[k]["flyback_ms"]) if k in rec else math.nan for (k, *_rest) in rows])
    f0w = np.array([r[5] for r in rows]); T0ms = 1000.0 / f0w
    beta = d0["beta"]; Hb = d0["H"]
    colkey = list(zip(d0["folder"], d0["column"]))
    med = defaultdict(list)
    for ck, h, x in zip(colkey, Hb, dfb):
        if h and np.isfinite(x):
            med[ck].append(x)
    med = {ck: float(np.median(v)) for ck, v in med.items()}
    variants = {f"abs{k}": Hb & np.isfinite(dfb) & (dfb <= k * beta * T0ms) for k in (1, 2, 4)}
    variants.update({f"rel{q}": Hb & np.array([np.isfinite(x) and ck in med and x <= q * med[ck] for ck, x in zip(colkey, dfb)]) for q in (1.5, 2)})
    lb = np.log(beta); terc = np.digitize(lb, np.quantile(lb, [1 / 3, 2 / 3]))
    base_w = LS.fit(d0)[1]; base_o = LS.fit(LS.to_onset(d0, "B1"))[1]
    kb_w = keyed(d0); kb_o = keyed(LS.to_onset(d0, "B1"))
    res["base"] = dict(window_b_beta=base_w, onset_b_beta=base_o, H_share=float(Hb.mean()))
    res["variants"] = {}
    for name, Hv in variants.items():
        d = dict(d0, H=Hv)
        rem = [float((Hb & ~Hv)[terc == i].sum() / max(1, Hb[terc == i].sum())) for i in range(3)]
        bw = LS.fit(d)[1]; do = LS.to_onset(d, "B1"); bo = LS.fit(do)[1]
        ciw = paired_boot(kb_w, keyed(d)); cio = paired_boot(kb_o, keyed(do))
        sw, so = bw - base_w, bo - base_o
        e = dict(window_b_beta=bw, onset_b_beta=bo, shift_window=sw, shift_window_ci=ciw, shift_onset=so, shift_onset_ci=cio,
                 removed_by_tercile=rem, beta_confounded=bool(abs(rem[2] - rem[0]) > 0.20), H_share=float(Hv.mean()),
                 size_window="small" if abs(sw) < 0.1 else ("moderate" if abs(sw) < 0.3 else "large"),
                 size_onset="small" if abs(so) < 0.1 else ("moderate" if abs(so) < 0.3 else "large"),
                 moves_to_schelleng=bool(bw <= -1.8 or bo <= -1.8))
        res["variants"][name] = e
        print(f"{name}: window β {bw:+.3f} (shift {sw:+.3f} [{ciw[0]:+.3f}, {ciw[1]:+.3f}]) onset β {bo:+.3f} (shift {so:+.3f} [{cio[0]:+.3f}, {cio[1]:+.3f}]) "
              f"removed by β tercile {[round(x, 2) for x in rem]} flag {e['beta_confounded']}", flush=True)
    # descriptive regression among base-Helmholtz strokes
    m = Hb & np.isfinite(dfb) & (dfb > 0) & (d0["force"] > 0)
    X = np.c_[np.ones(m.sum()), np.log(beta[m]), np.log(d0["speed"][m]), np.log(d0["force"][m])]
    y = np.log(dfb[m]); coef, *_ = np.linalg.lstsq(X, y, rcond=None); r2 = 1 - np.sum((y - X @ coef) ** 2) / np.sum((y - y.mean()) ** 2)
    ratio = dfb[m] / (beta[m] * T0ms[m])
    res["regression"] = dict(n=int(m.sum()), intercept=float(coef[0]), log_beta=float(coef[1]), log_v=float(coef[2]), log_F=float(coef[3]), r2=float(r2),
                             dfb_over_betaT0_quartiles=np.percentile(ratio, [25, 50, 75]).tolist(), dfb_ms_quartiles=np.percentile(dfb[m], [25, 50, 75]).tolist())
    print("regression", res["regression"], flush=True)
    (HERE / "results_b1.json").write_text(json.dumps(res, indent=1, default=float))
    # figure
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.6))
    for v, col in ((0.05, "#1a73e8"), (0.1, "#188038"), (0.2, "#d93025")):
        s = m & (np.abs(d0["speed"] - v) < 0.02)
        ax[0].scatter(beta[s], dfb[s], s=3, alpha=.25, color=col, label=f"v = {v} m/s")
    bb = np.geomspace(beta.min(), beta.max(), 50); T0 = float(np.median(T0ms))
    for k, ls in ((1, "-"), (2, "--"), (4, ":")):
        ax[0].plot(bb, k * bb * T0, "k", ls=ls, lw=1, label=f"{k} x beta x T0")
    ax[0].set_xscale("log"); ax[0].set_yscale("log"); ax[0].set_xlabel("Relative bow position beta"); ax[0].set_ylabel("Flyback duration at the bridge (ms)")
    ax[0].set_title("B1, Helmholtz strokes at t = 0.2 s: corner width vs beta"); ax[0].legend(fontsize=7, markerscale=4); ax[0].grid(alpha=.25, which="both")
    names = ["base"] + list(res["variants"])
    for j, (axn, key, sh) in enumerate((("Window force", "window_b_beta", "shift_window_ci"), ("Onset force", "onset_b_beta", "shift_onset_ci"))):
        ys = [res["base"][key]] + [res["variants"][n][key] for n in names[1:]]
        lo = [0] + [res["base"][key] + res["variants"][n][sh][0] for n in names[1:]]
        hi = [0] + [res["base"][key] + res["variants"][n][sh][1] for n in names[1:]]
        x = np.arange(len(names)) + (j - 0.5) * 0.2
        ax[1].errorbar(x[1:], ys[1:], yerr=[np.array(ys[1:]) - np.array(lo[1:]), np.array(hi[1:]) - np.array(ys[1:])], fmt="o", capsize=3, label=axn + " axis", color=["#202124", "#1a73e8"][j])
        ax[1].plot([x[0]], [ys[0]], "s", color=["#202124", "#1a73e8"][j])
    ax[1].axhline(-2, color="#d93025", ls="--", lw=1); ax[1].text(0, -1.97, "Schelleng (-2)", color="#d93025", fontsize=7)
    ax[1].set_xticks(range(len(names))); ax[1].set_xticklabels([n + (" (beta-confounded)" if n != "base" and res["variants"][n]["beta_confounded"] else "") for n in names], rotation=20, fontsize=7)
    ax[1].set_ylabel("B1 minimum-force beta exponent"); ax[1].set_title("B1 exponent under corner-width labels\n(squares = default; bars = paired 95% bootstrap)", fontsize=9); ax[1].legend(fontsize=7); ax[1].grid(alpha=.25)
    fig.tight_layout(); fig.savefig(HERE / "fig_b1_flyback.png", dpi=150)


if __name__ == "__main__":
    main()
