"""Check trial order -> (beta column, force rank) direction per folder in ramp_meta. (Claude, 2026-10-19)"""
import csv, glob
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr
AC = Path.home() / "violin-rne/astra_20261018/archive_companion"
files = sorted(glob.glob(str(AC / "2024-*/ramp_meta_*.csv")))
for fn in files:
    rows = list(csv.DictReader(open(fn)))
    by = {}
    for r in rows:
        by.setdefault(r["folder"], []).append(r)
    for fo, rs in sorted(by.items()):
        tr = np.array([int(r["trial"]) for r in rs]); b = np.array([float(r["beta"]) for r in rs])
        fon = np.array([float(r["force_at_onset"]) if r["force_at_onset"] not in ("", "nan") else np.nan for r in rs])
        o = np.argsort(tr); tr, b, fon = tr[o], b[o], fon[o]
        rb = spearmanr(tr, b)[0]
        # within blocks of 50 consecutive trials
        rr = [spearmanr(np.arange(50), fon[k*50:(k+1)*50], nan_policy="omit")[0] for k in range(40)]
        # is the beta block structure (trial-1)//50 ?
        blk = (tr - 1) // 50
        within = np.mean([np.ptp(np.log(b[blk == k])) for k in range(40)])
        print(f"{fo:55s} n {len(rs)} spearman(trial,beta) {rb:+.3f} | median within-block spearman(order,F_on) {np.nanmedian(rr):+.3f} | mean within-block log-beta range {within:.3f}")
