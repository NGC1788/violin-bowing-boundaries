#!/usr/bin/env bash
set -euo pipefail
root="$(cd "$(dirname "$0")/.." && pwd)"
if [[ -n "${VIOLIN_LOOP_PYTHON:-}" ]]; then
  python_bin="$VIOLIN_LOOP_PYTHON"
elif [[ -x "$root/../.venv-mac/bin/python" ]]; then
  python_bin="$root/../.venv-mac/bin/python"
elif [[ -x "$root/../.venv-loop/bin/python" ]]; then
  python_bin="$root/../.venv-loop/bin/python"
else
  python_bin="python3"
fi
mode="${1:-tick}"
if [[ $# -gt 0 ]]; then shift; fi
case "$mode" in
  tick) exec "$python_bin" "$root/forecast_v3/stage.py" "$@" ;;
  agent) exec "$python_bin" "$root/forecast_v3/planner.py" "$@" ;;
  *) exec "$python_bin" "$root/forecast_v3/cli.py" "$mode" "$@" ;;
esac
