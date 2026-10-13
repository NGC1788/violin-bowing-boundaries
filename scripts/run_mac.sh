#!/usr/bin/env bash
set -Eeuo pipefail
TASK_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$TASK_ROOT"
TASK_PYTHON="$TASK_ROOT/.venv-mac/bin/python"
if [[ ! -x "$TASK_PYTHON" ]]; then
  echo 'Run bash scripts/bootstrap_mac.sh first.' >&2
  exit 1
fi
export PYTHONPATH="$TASK_ROOT/scripts${PYTHONPATH:+:$PYTHONPATH}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export NUMBA_NUM_THREADS="${NUMBA_NUM_THREADS:-2}"
if [[ -d "$TASK_ROOT/.local-tools/7zip-26.03" ]]; then
  export PATH="$TASK_ROOT/.local-tools/7zip-26.03:$PATH"
fi
TASK_COMMAND="${1:-help}"
if [[ $# -gt 0 ]]; then shift; fi
case "$TASK_COMMAND" in
  python) exec "$TASK_PYTHON" "$@" ;;
  test) exec "$TASK_PYTHON" -m pytest tests -q "$@" ;;
  benchmark) exec "$TASK_PYTHON" scripts/benchmark_sets.py "$@" ;;
  catalog) exec "$TASK_PYTHON" scripts/zenodo_catalog.py catalog "$@" ;;
  download) exec "$TASK_PYTHON" scripts/zenodo_catalog.py download "$@" ;;
  archive-audit) exec "$TASK_PYTHON" scripts/archive_audit.py "$@" ;;
  trial-audit) exec "$TASK_PYTHON" scripts/trial_audit.py "$@" ;;
  regime-map) exec "$TASK_PYTHON" scripts/regime_map.py "$@" ;;
  *) echo 'Usage: bash scripts/run_mac.sh {python|test|benchmark|catalog|download|archive-audit|trial-audit|regime-map} [arguments]' ;;
esac
