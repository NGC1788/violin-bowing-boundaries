#!/usr/bin/env bash
# CPU research environment, separate from the Linux/CUDA lockfile.
set -Eeuo pipefail
TASK_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$TASK_ROOT"
if [[ "$(uname -s)" != Darwin || "$(uname -m)" != arm64 ]]; then
  echo 'This environment was verified on Apple Silicon macOS.' >&2
  exit 1
fi
if command -v uv >/dev/null 2>&1; then
  TASK_UV="$(command -v uv)"
elif [[ -x "$HOME/.local/bin/uv" ]]; then
  TASK_UV="$HOME/.local/bin/uv"
else
  echo 'Install uv from https://docs.astral.sh/uv/getting-started/installation/ and rerun.' >&2
  exit 1
fi
if [[ ! -x .venv-mac/bin/python ]]; then
  "$TASK_UV" venv --python 3.11.15 .venv-mac
fi
"$TASK_UV" pip sync --python .venv-mac/bin/python requirements-mac.lock.txt
bash scripts/run_mac.sh python -c 'import platform, numpy, scipy, sklearn, numba, torch; print("MAC IMPORT PASS", platform.python_version(), platform.machine(), "torch", torch.__version__)'
echo 'Next: bash scripts/run_mac.sh test'
