#!/bin/zsh
# Score the frozen predictions once per newly processed archive (Claude, 2026-10-19). A3 excluded (frozen after processing).
PY=$HOME/violin-bowing-boundaries/.venv-mac/bin/python
S=$HOME/violin-rne/timeres/replicate_state.json
F=$HOME/violin-rne/timeres/features
typeset -A TAG=( 2024-04-24_TypeA_sample4 A4 2024-04-26_TypeD_sample2 D2 2024-07-03_TypeB_T2_sample2 B_T2s2 )
done_list=()
while (( ${#done_list} < 3 )); do
  for stem tag in ${(kv)TAG}; do
    (( ${done_list[(Ie)$stem]} )) && continue
    st=$($PY -c "import json; print(json.load(open('$S')).get('$stem.7z',{}).get('status',''))")
    if [[ $st == done ]]; then
      echo "[$(date)] $stem 처리 완료 → 채점"
      cd $HOME/violin-rne/novelty/method_protocol && OPENBLAS_NUM_THREADS=1 $PY predict_from_controls.py --target $tag --features $F/$stem.csv.gz --out frozen/${tag}_controls_prediction.json && OPENBLAS_NUM_THREADS=1 $PY score_frozen.py --prediction frozen/${tag}_controls_prediction.json --out frozen/${tag}_score.json
      cd $HOME/violin-rne/novelty/thermal_upper && OPENBLAS_NUM_THREADS=1 $PY score_target.py $stem FROZEN_v2.json
      done_list+=$stem
    elif [[ $st == error* ]]; then done_list+=$stem; echo "[$(date)] $stem 오류: $st"; fi
  done
  sleep 120
done
echo "[$(date)] 끝"
