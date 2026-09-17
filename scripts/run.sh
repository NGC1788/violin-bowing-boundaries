#!/usr/bin/env bash
set -Eeuo pipefail
TASK_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$TASK_ROOT"
if command -v uv >/dev/null 2>&1; then
  TASK_UV="$(command -v uv)"
elif [[ -x "$HOME/.local/bin/uv" ]]; then
  TASK_UV="$HOME/.local/bin/uv"
else
  echo 'Run bash scripts/bootstrap_ubuntu.sh first.' >&2
  exit 1
fi
export UV_CACHE_DIR="$TASK_ROOT/.cache/uv"
export UV_PYTHON_INSTALL_DIR="$TASK_ROOT/.cache/python"
if [[ -d "$TASK_ROOT/.local-tools/7zip-26.03" ]]; then
  export PATH="$TASK_ROOT/.local-tools/7zip-26.03:$PATH"
fi
TASK_COMMAND="${1:-help}"
if [[ $# -gt 0 ]]; then shift; fi
case "$TASK_COMMAND" in
  setup-tools) exec "$TASK_UV" run --locked python scripts/install_7zip.py "$@" ;;
  doctor) exec "$TASK_UV" run --locked python scripts/doctor.py --require-cuda "$@" ;;
  smoke) exec "$TASK_UV" run --locked python scripts/smoke_test.py "$@" ;;
  catalog) exec "$TASK_UV" run --locked python scripts/zenodo_catalog.py catalog "$@" ;;
  download) exec "$TASK_UV" run --locked python scripts/zenodo_catalog.py download "$@" ;;
  archive-audit) exec "$TASK_UV" run --locked python scripts/archive_audit.py "$@" ;;
  prepare-first) exec "$TASK_UV" run --locked python scripts/prepare_first.py "$@" ;;
  trial-audit) exec "$TASK_UV" run --locked python scripts/trial_audit.py "$@" ;;
  jupyter) exec "$TASK_UV" run --locked jupyter lab --ip=127.0.0.1 --port=8888 --no-browser "$@" ;;
  python) exec "$TASK_UV" run --locked python "$@" ;;
  *) echo 'Usage: bash scripts/run.sh {setup-tools|doctor|smoke|catalog|download|archive-audit|prepare-first|trial-audit|jupyter|python} [arguments]' ;;
esac
