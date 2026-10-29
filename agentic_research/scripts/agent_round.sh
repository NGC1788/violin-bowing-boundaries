#!/usr/bin/env bash
set -euo pipefail
loop_root="$(cd "$(dirname "$0")/.." && pwd)"
if [[ -n "${VIOLIN_LOOP_PYTHON:-}" ]]; then
  loop_python="$VIOLIN_LOOP_PYTHON"
elif [[ -x "$loop_root/../.venv-mac/bin/python" ]]; then
  loop_python="$loop_root/../.venv-mac/bin/python"
elif [[ -x "$loop_root/../.venv/bin/python" ]]; then
  loop_python="$loop_root/../.venv/bin/python"
else
  loop_python=python3
fi
exec "$loop_python" "$loop_root/planner.py" "$@"
