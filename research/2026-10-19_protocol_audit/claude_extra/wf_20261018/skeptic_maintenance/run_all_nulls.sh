#!/bin/zsh
# sequential, single-threaded null runs for every A1/A2 speed folder
cd /Users/jeongjihoon/violin-rne/claude_extra/wf_20261018/skeptic_maintenance
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
P=~/violin-bowing-boundaries/.venv-mac/bin/python
$P -B nulls.py A2 0 1000 > nulls_A2_v1.log 2>&1
$P -B nulls.py A1 0 300 > nulls_A1_v1.log 2>&1
$P -B nulls.py A2 1 300 > nulls_A2_v2.log 2>&1
$P -B nulls.py A2 2 300 > nulls_A2_v3.log 2>&1
$P -B nulls.py A1 1 300 > nulls_A1_v2.log 2>&1
$P -B nulls.py A1 2 300 > nulls_A1_v3.log 2>&1
echo done > nulls_done.flag
