#!/bin/zsh
# Wait for D_T2 to be processed by replicate.py, then run Astra's frozen two-step confirmation once (Claude, 2026-10-19).
cd $HOME/violin-rne
PY=$HOME/violin-bowing-boundaries/.venv-mac/bin/python
until $PY -c "import json,sys; s=json.load(open('timeres/replicate_state.json')).get('2024-06-17_TypeD_T2_sample1.7z',{}).get('status',''); sys.exit(0 if s=='done' or s.startswith('error') else 1)"; do sleep 60; done
F=timeres/features/2024-06-17_TypeD_T2_sample1.csv.gz
echo "[$(date)] D_T2 처리 완료 → 1단계(동결)" 
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 $PY astra_20261019/confirm_d_t2.py freeze-controls --features $F --out astra_20261019/D_T2_controls_frozen.json || exit 1
echo "[$(date)] 2단계(채점)"
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 $PY astra_20261019/confirm_d_t2.py score --controls-receipt astra_20261019/D_T2_controls_frozen.json --out astra_20261019/D_T2_confirm_result.json
echo "[$(date)] 끝"
