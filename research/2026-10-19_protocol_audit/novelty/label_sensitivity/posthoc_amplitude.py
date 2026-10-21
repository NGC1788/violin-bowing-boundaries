"""POST HOC (decided after label_sweep.py: the fixed amplitude floor dominated S). Two checks, labelled post hoc:
(1) Mechanism: does the fixed floor (amp1x) remove Helmholtz windows by beta (Helmholtz amplitude scales with v/beta),
    independently of force? Base-H windows lost under amp1x, by beta tercile and speed.
(2) A relative amplitude criterion, the analogue of the simulation's amplitude check (0.5 v/beta): Helmholtz only if
    amplitude >= r x the median amplitude of base-Helmholtz windows in the same (folder, beta column), r in {0.5, 0.7}.
Claude, 2026-10-21."""
import json, sys
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE))
import label_sweep as LS  # noqa: E402


def main():
    out = {"mechanism": {}, "relative": {}}
    for sess, st in LS.STEM.items():
        stem, rep = st if isinstance(st, tuple) else (st, None)
        raw = LS.read(stem, rep)
        d0 = LS.build(raw, LS.VARIANTS["base"]); d1 = LS.build(raw, LS.VARIANTS["amp1x"])
        amp = np.array([r[3] for r in raw[0]])
        lb = np.log(d0["beta"]); terc = np.digitize(lb, np.quantile(lb, [1 / 3, 2 / 3]))
        lost = d0["H"] & ~d1["H"]
        out["mechanism"][sess] = {f"beta_tercile_{i}": float(lost[terc == i].sum() / max(1, d0["H"][terc == i].sum())) for i in range(3)}
        out["mechanism"][sess]["corr_lost_vs_logforce_within_H"] = float(np.corrcoef(lost[d0["H"]], np.log(d0["force"][d0["H"]]))[0, 1])
        # relative amplitude
        key = list(zip(d0["folder"], d0["column"]))
        med = defaultdict(list)
        for k, h, a in zip(key, d0["H"], amp):
            if h:
                med[k].append(a)
        med = {k: float(np.median(v)) for k, v in med.items()}
        e = {}
        for r in (0.5, 0.7):
            d = dict(d0)
            d["H"] = np.array([h and (k in med) and a >= r * med[k] for k, h, a in zip(key, d0["H"], amp)])
            bw = LS.fit(d); e[f"rel{r}"] = dict(window_b_beta=bw[1], changed=float(np.mean(d["H"] != d0["H"])))
            if sess in LS.ONSET:
                e[f"rel{r}"]["onset_b_beta"] = LS.fit(LS.to_onset(d, sess))[1]
        out["relative"][sess] = e
        m = out["mechanism"][sess]
        print(f"{sess:7s} amp1x removes base-H: β terciles {m['beta_tercile_0']:.2f}/{m['beta_tercile_1']:.2f}/{m['beta_tercile_2']:.2f}, "
              f"corr(lost, log F) {m['corr_lost_vs_logforce_within_H']:+.2f} | rel0.5 β {e['rel0.5']['window_b_beta']:+.2f} rel0.7 β {e['rel0.7']['window_b_beta']:+.2f}", flush=True)
    base = json.loads((HERE / "results.json").read_text())["sessions"]
    for ax, ss in (("window", list(LS.STEM)), ("onset", LS.ONSET)):
        for r in ("rel0.5", "rel0.7"):
            x = np.array([out["relative"][s][r][f"{ax}_b_beta"] for s in ss]); b = np.array([base[s]["base"][f"{ax}_b_beta"] for s in ss])
            m, sd, ci = LS.tci(x); dm, dsd, dci = LS.tci(x - b)
            out.setdefault("summary", {}).setdefault(ax, {})[r] = dict(mean=m, sd=sd, shift=dm, shift_ci=dci,
                                                                      changed=float(np.mean([out["relative"][s][r]["changed"] for s in ss])))
            print(f"{ax} {r}: mean {m:+.3f} shift {dm:+.3f} [{dci[0]:+.3f}, {dci[1]:+.3f}]")
    (HERE / "posthoc_amplitude.json").write_text(json.dumps(out, indent=1, default=float))


if __name__ == "__main__":
    main()
