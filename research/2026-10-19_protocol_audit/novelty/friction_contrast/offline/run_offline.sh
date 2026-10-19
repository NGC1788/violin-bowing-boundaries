#!/bin/zsh
# Offline continuation (Claude, 2026-10-19): no network; uses only ../data_untrusted (already downloaded) and project caches.
set -e
cd ~/violin-rne/novelty/friction_contrast
PY=~/violin-bowing-boundaries/.venv-mac/bin/python
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
O=offline
$PY -I $O/scripts/coverage.py "$PWD/data_untrusted" "$PWD/results" "$PWD/$O/results/coverage.json" > $O/results/coverage.log
$PY -B $O/scripts/levels_offline.py "$PWD/$O/results" > $O/results/levels.log   # trusted project code; writes final/cache/{B_T2s1,C_T2}.npz once
$PY -I $O/scripts/friction_offline.py "$PWD/data_untrusted/output_measurements/reconstructions" "$PWD/$O/results" > $O/results/friction_offline.log 2>&1
$PY -I $O/scripts/fig_offline.py "$PWD/$O/results" "$PWD/$O/figs/friction_offline.png"
