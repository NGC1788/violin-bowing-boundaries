"""Control-only design audit of B_T2 sample 2 (2024-07-03; stream output). Claude, 2026-10-20.
Parses only folder/trial/beta/window_s/t/force/speed/error of the features and the label-free ramp_meta.csv.
Classifier columns are never parsed. Output: controls_audit.json"""
import csv, gzip, hashlib, json
from collections import defaultdict
from pathlib import Path
import numpy as np

D = Path.home() / "violin-rne/astra_20261019/highforce_stream/2024-07-03_TypeB_T2_sample2"
CTRL = ("folder", "trial", "beta", "window_s", "t", "force", "speed", "error")
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
rec = json.loads((D / "receipt.json").read_text())
hashes = {n: sha(D / n) for n in ("features.csv.gz", "ramp_meta.csv", "blocks.npz")}
assert all(hashes[n] == rec["outputs"][n] for n in hashes) and rec["status"] == "validated"
beta, speed, wf, err = {}, defaultdict(list), defaultdict(lambda: defaultdict(list)), defaultdict(int)
with gzip.open(D / "features.csv.gz", "rt") as fh:
    rd = csv.reader(fh); head = next(rd); ix = [head.index(c) for c in CTRL]
    for row in rd:
        r = dict(zip(CTRL, (row[i] for i in ix)))
        if r["error"]:
            err[r["folder"]] += 1; continue
        if r["window_s"] != "0.2":
            continue
        k = (r["folder"], int(r["trial"])); beta[k] = float(r["beta"]); speed[r["folder"]].append(float(r["speed"]))
        t = round(float(r["t"]), 2)
        if t in (0.0, 0.1, 0.2, 0.3):
            wf[r["folder"]][t].append((k[1], float(r["force"])))
ramp = defaultdict(list)
for r in csv.DictReader(open(D / "ramp_meta.csv")):
    if not r["error"]:
        ramp[r["folder"]].append((int(r["trial"]), float(r["beta"]), float(r["v_max"]), float(r["force_at_onset"]), float(r["force_plateau"]), float(r["accel"])))
out = dict(receipt=rec["validation"], output_sha256=hashes, folders={})
for f in sorted({k[0] for k in beta}):
    tr = sorted(t for (ff, t) in beta if ff == f); b = np.array([beta[(f, t)] for t in tr])
    s = np.sort(b); d = np.diff(np.log(s)); thr = max(10 * np.median(d), 1e-3); cuts = np.flatnonzero(d > thr)
    groups = np.split(s, cuts + 1); sizes = [len(g) for g in groups]
    rm = sorted(ramp[f]); fo = np.array([x[3] for x in rm]); fp = np.array([x[4] for x in rm])
    # force steps within the first beta column (onset force, label-free)
    col0 = sorted([x for x in rm], key=lambda x: x[1])[:sizes[0]]
    fs = np.sort([x[3] for x in col0])
    out["folders"][f] = dict(n_strokes=len(tr), trial_range=[tr[0], tr[-1]], speed_median=float(np.median(speed[f])),
        v_max_set=sorted(set(round(x[2], 4) for x in rm)), beta_n_columns=len(groups), beta_column_sizes=sorted(set(sizes)),
        beta_range=[float(b.min()), float(b.max())], beta_col_medians=[float(np.median(g)) for g in groups],
        within_spread_max=float(max(np.ptp(np.log(g)) for g in groups)), between_gap_min=float(np.min(np.diff(np.log([np.median(g) for g in groups])))) if len(groups) > 1 else None,
        onset_force_range=[float(fo.min()), float(fo.max())], plateau_force_range=[float(fp.min()), float(fp.max())],
        first_col_onset_force_log_step_median=float(np.median(np.diff(np.log(fs)))) if len(fs) > 1 else None,
        windows_by_t={str(t): len(v) for t, v in wf[f].items()}, error_windows=err[f],
        first_trials_beta=[round(beta[(f, t)], 4) for t in tr[:3]], last_trials_beta=[round(beta[(f, t)], 4) for t in tr[-3:]])
Path("controls_audit.json").write_text(json.dumps(out, indent=1))
for f, x in out["folders"].items():
    print(f.split('/')[-1], {k: x[k] for k in ("n_strokes", "speed_median", "v_max_set", "beta_n_columns", "beta_column_sizes", "beta_range", "within_spread_max", "between_gap_min", "onset_force_range", "first_col_onset_force_log_step_median", "windows_by_t", "error_windows", "first_trials_beta", "last_trials_beta")})
