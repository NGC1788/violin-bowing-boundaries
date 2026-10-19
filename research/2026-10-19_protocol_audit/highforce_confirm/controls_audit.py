"""Step 2 of HIGH_FORCE_COMPATIBILITY.md: control-only design audit of HighBowForceRange_1/2 (Claude, 2026-10-19).
Parses ONLY control columns (folder, trial, beta, window_s, t, plateau_s, force, speed, error) of the stream
features; the classifier columns (amplitude_std, periodicity, f0_hz, flybacks_*, flyback_height) are never
converted or stored. Also reads ramp_meta.csv (label-free stroke controls). Output: controls_audit.json."""
import csv
import gzip
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path.home() / "violin-rne/astra_20261019/highforce_stream"
OUT = Path(__file__).parent / "controls_audit.json"
CTRL = ("folder", "trial", "beta", "window_s", "t", "plateau_s", "force", "speed", "error")
TIMES = (0.0, 0.1, 0.2, 0.3)


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def read_controls(path):
    with gzip.open(path, "rt") as fh:
        rd = csv.reader(fh)
        head = next(rd)
        ix = [head.index(c) for c in CTRL]
        for row in rd:
            yield dict(zip(CTRL, (row[i] for i in ix)))


def beta_clusters(b):
    """Group sorted beta values at gaps larger than 10x the median within-step spacing."""
    s = np.sort(b)
    d = np.diff(np.log(s))
    thr = max(10 * np.median(d), 1e-3)
    cuts = np.flatnonzero(d > thr)
    groups = np.split(s, cuts + 1)
    return groups, thr


def main():
    out = dict(script_sha256=sha(__file__), parsed_columns=list(CTRL), archives={})
    for arch in ("HighBowForceRange_1", "HighBowForceRange_2"):
        d = ROOT / arch
        rec = json.loads((d / "receipt.json").read_text())
        hashes = {n: sha(d / n) for n in ("features.csv.gz", "ramp_meta.csv", "blocks.npz")}
        assert all(hashes[n] == rec["outputs"][n] for n in hashes), (arch, "output hash mismatch")
        win = defaultdict(lambda: defaultdict(list))          # folder -> t -> [(trial, force)]
        beta, speed, errs, nwin = {}, defaultdict(list), defaultdict(int), defaultdict(int)
        for r in read_controls(d / "features.csv.gz"):
            f = r["folder"]
            nwin[f] += 1
            if r["error"]:
                errs[f] += 1
                continue
            if r["window_s"] != "0.2":
                continue
            k = (f, int(r["trial"]))
            beta[k] = float(r["beta"])
            speed[f].append(float(r["speed"]))
            t = round(float(r["t"]), 2)
            if t in TIMES:
                win[f][t].append((int(r["trial"]), float(r["force"])))
        ramp = defaultdict(list)
        with open(d / "ramp_meta.csv") as fh:
            for r in csv.DictReader(fh):
                if not r["error"]:
                    ramp[r["folder"]].append((int(r["trial"]), float(r["beta"]), float(r["v_max"]), float(r["force_plateau"]),
                                              float(r["accel"])))
        A = dict(receipt_status=rec["status"], receipt_trials=rec["validation"]["trials"], output_sha256=hashes, folders={})
        for f in sorted({k[0] for k in beta} | set(ramp)):
            trials = sorted({k[1] for k in beta if k[0] == f})
            bvals = np.array([beta[(f, t)] for t in trials])
            groups, thr = beta_clusters(bvals)
            sizes = [len(g) for g in groups]
            gmed = [float(np.median(g)) for g in groups]
            within = max(float(np.ptp(np.log(g))) for g in groups)
            between = float(np.min(np.diff(np.log(gmed)))) if len(gmed) > 1 else None
            rm = np.array([x[1:] for x in sorted(ramp[f])]) if ramp[f] else np.zeros((0, 4))
            fx = {}
            for t in TIMES:
                w = win[f].get(t, [])
                fx[str(t)] = dict(n_windows=len(w), n_strokes=len({x[0] for x in w}),
                                  force_min=float(min(x[1] for x in w)) if w else None,
                                  force_max=float(max(x[1] for x in w)) if w else None)
            # force steps per beta column from the plateau force (label-free)
            col_of = {}
            sb = sorted((beta[(f, t)], t) for t in trials)
            gi, pos = 0, 0
            for i, (b, t) in enumerate(sb):
                if pos == sizes[gi]:
                    gi, pos = gi + 1, 0
                col_of[t] = gi
                pos += 1
            fp = {int(x[0]): x[3] for x in sorted(ramp[f])}
            per_col_n = [sizes[c] for c in range(len(sizes))]
            per_col_frange = []
            for c in range(len(sizes)):
                v = sorted(fp[t] for t in trials if col_of[t] == c and t in fp)
                per_col_frange.append([float(v[0]), float(v[-1]), len(v)] if v else None)
            # trial order vs beta/force (is the design force-major or beta-major?)
            tr = np.array(trials)
            A["folders"][f] = dict(
                n_strokes=len(trials), n_windows_total=nwin[f], n_error_windows=errs[f],
                speed_median=float(np.median(speed[f])), speed_iqr=np.percentile(speed[f], [25, 75]).tolist(),
                ramp_v_max_set=sorted(set(np.round(rm[:, 1], 4).tolist())) if len(rm) else None,
                beta_n_columns=len(groups), beta_column_sizes_set=sorted(set(sizes)), beta_column_medians=gmed,
                beta_range=[float(bvals.min()), float(bvals.max())], beta_gap_threshold_log=float(thr),
                beta_max_within_spread_log=within, beta_min_between_gap_log=between,
                plateau_force_range=[float(rm[:, 2].min()), float(rm[:, 2].max())] if len(rm) else None,
                per_column_plateau_force=[per_col_frange[0], per_col_frange[len(sizes) // 2], per_col_frange[-1]],
                window_force_by_t=fx,
                trial_range=[int(tr.min()), int(tr.max())],
                beta_spearman_with_trial=float(np.corrcoef(np.argsort(np.argsort(tr)), np.argsort(np.argsort(bvals)))[0, 1]),
                accel_median=float(np.median(rm[:, 3])) if len(rm) else None)
        out["archives"][arch] = A
    OUT.write_text(json.dumps(out, indent=1))
    for arch, A in out["archives"].items():
        print(arch, A["receipt_status"], A["receipt_trials"])
        for f, x in A["folders"].items():
            print(f"  {f}: strokes {x['n_strokes']} err {x['n_error_windows']} v {x['speed_median']:.4f} vmax {x['ramp_v_max_set']}"
                  f" | beta cols {x['beta_n_columns']} sizes {x['beta_column_sizes_set']} range {x['beta_range'][0]:.4f}-{x['beta_range'][1]:.4f}"
                  f" within {x['beta_max_within_spread_log']:.4f} gap {x['beta_min_between_gap_log']}"
                  f" | Fplateau {x['plateau_force_range'][0]:.2f}-{x['plateau_force_range'][1]:.2f}"
                  f" | t0.2 n {x['window_force_by_t']['0.2']['n_strokes']} F {x['window_force_by_t']['0.2']['force_min']}-{x['window_force_by_t']['0.2']['force_max']}")
            print("    t-availability:", {t: v["n_strokes"] for t, v in x["window_force_by_t"].items()}, "col plateau F (first/mid/last):", x["per_column_plateau_force"])


if __name__ == "__main__":
    main()
