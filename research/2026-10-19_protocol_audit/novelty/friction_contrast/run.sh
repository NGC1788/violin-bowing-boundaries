#!/bin/zsh
# Reproduce (Claude, 2026-10-19). Data: Zenodo 15879128 (md5 output_measurements.zip 0d6c318849943a817e0425268e186e0e,
# function.zip 3477ca5aa4aa3a919e2e75b0855ee57f) extracted into data_untrusted/ (zips deleted after extraction).
set -e
cd ~/violin-rne/novelty/friction_contrast
PY=~/violin-bowing-boundaries/.venv-mac/bin/python
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
REC=data_untrusted/output_measurements/reconstructions
$PY -I scripts/friction_metrics.py $REC results/friction_metrics.json 0.15 > results/friction_metrics.log
$PY -I scripts/friction_metrics.py $REC results/friction_metrics_thr0.30.json 0.30 > results/friction_metrics_thr0.30.log
$PY -I scripts/friction_metrics.py $REC results/friction_metrics_thr0.10.json 0.10 > results/friction_metrics_thr0.10.log
$PY -B scripts/bt2_boundaries.py results/boundary_laws.json > results/boundary_laws.log   # project caches only (trusted), -B: no bytecode in final/
$PY -I scripts/analysis.py > results/analysis.log
$PY -I scripts/fig_summary.py $REC results figs/friction_summary.png
