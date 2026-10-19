#!/bin/zsh
# Waits for B_T2 sample 2 (done + regular frozen score written) or error, then runs score_v3.py --score once. Claude 2026-10-19.
PY=$HOME/violin-bowing-boundaries/.venv-mac/bin/python
S=$HOME/violin-rne/timeres/replicate_state.json
OUT=$HOME/violin-rne/novelty/thermal_upper/score_2024-07-03_TypeB_T2_sample2__FROZEN_v2.json
while true; do
  st=$($PY -c "import json; print(json.load(open('$S')).get('2024-07-03_TypeB_T2_sample2.7z',{}).get('status',''))")
  if [[ $st == done && -f $OUT ]]; then break; fi
  if [[ $st == error* ]]; then echo "[$(date)] B_T2s2 error: $st -> run incomplete"; break; fi
  sleep 60
done
sleep 30
cd $HOME/violin-rne/confirm_v3 && OPENBLAS_NUM_THREADS=1 $PY score_v3.py --score
echo "[$(date)] done"
