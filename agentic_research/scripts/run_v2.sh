#!/usr/bin/env bash
set -euo pipefail
root="$(cd "$(dirname "$0")/.." && pwd)"
python_bin="${VIOLIN_LOOP_PYTHON:-$root/../.venv-mac/bin/python}"
if [[ ! -x "$python_bin" ]]; then
  printf '%s\n' 'Set VIOLIN_LOOP_PYTHON to the tested Python environment.' >&2
  exit 2
fi
exec "$python_bin" "$root/engine_v2/cli.py" "$@"
