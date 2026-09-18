#!/usr/bin/env bash
# Unattended run: calibrate friction on one bow speed, then simulate whole diagrams with the best set.
# Usage: nohup bash scripts/overnight.sh >> logs/overnight.log 2>&1 < /dev/null &
set -Eeuo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs reports/calibration
DIAGRAM="${DIAGRAM:-2024-03-25_TypeA_sample1}"
TRAIN="${TRAIN:-2024-03-27_r_v2}"
ITER="${ITER:-1500}"
RATE="${RATE:-50000}"
FINAL_RATE="${FINAL_RATE:-100000}"
LEVELS="${LEVELS:-10}"
RANKS="${RANKS:-50}"
SETS="${SETS:-16}"

echo "=== [1/3] calibration  $(date -Is)  diagram $DIAGRAM, train $TRAIN, $ITER sets at $RATE Hz"
bash scripts/run.sh calibrate --diagram "$DIAGRAM" --conditions "$TRAIN" --hold-out \
  --rate "$RATE" --levels "$LEVELS" --ranks "$RANKS" --batch-params "$SETS" --iterations "$ITER"

BEST_ARGS="$(bash scripts/run.sh python - <<'PY'
import glob, json
rows = [json.loads(line) for path in glob.glob("reports/calibration/*.jsonl")
        for line in open(path, encoding="utf-8") if line.strip()]
best = max(rows, key=lambda r: r["helmholtz_iou"])
print(" ".join(f"--{key.replace('_', '-')} {value}" for key, value in best["params"].items()))
PY
)"
echo "=== best parameters: $BEST_ARGS"

echo "=== [2/3] full grid with calibrated parameters  $(date -Is)"
for path in reports/collection/*/regime_map; do
  name="$(basename "$(dirname "$path")")"
  echo "--- simulate $name"
  bash scripts/run.sh simulate --diagram "$name" --rate "$FINAL_RATE" $BEST_ARGS || echo "!! failed: $name"
done

echo "=== [3/3] full grid with literature parameters (baseline, training diagram only)  $(date -Is)"
bash scripts/run.sh simulate --diagram "$DIAGRAM" --rate "$FINAL_RATE" || echo "!! baseline failed"
echo "=== done  $(date -Is)"
