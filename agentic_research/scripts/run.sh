#!/usr/bin/env bash
set -euo pipefail
root="$(cd "$(dirname "$0")/.." && pwd)"
if [[ -n "${VIOLIN_LOOP_PYTHON:-}" ]]; then
  python_bin="$VIOLIN_LOOP_PYTHON"
elif [[ -x "$root/../.venv-mac/bin/python" ]]; then
  python_bin="$root/../.venv-mac/bin/python"
elif [[ -x "$root/../.venv/bin/python" ]]; then
  python_bin="$root/../.venv/bin/python"
else
  python_bin="python3"
fi
exec "$python_bin" "$root/cli.py" "$@"
