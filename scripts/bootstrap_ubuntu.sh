#!/usr/bin/env bash
# Run as a normal user, inside the Ubuntu server's terminal.
set -Eeuo pipefail
trap 'printf "\nSETUP FAILED at line %s. Resolve that error before continuing.\n" "$LINENO" >&2' ERR
TASK_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$TASK_ROOT"
if [[ "$(uname -s)" != Linux || "$(uname -m)" != x86_64 ]]; then
  echo 'This environment targets Linux x86_64 with an NVIDIA GPU. Run it on the Ubuntu server.' >&2
  exit 1
fi
if [[ "$(id -u)" == 0 ]]; then
  echo 'Use your normal school account, without sudo, for this script.' >&2
  exit 1
fi
command -v python3 >/dev/null || { echo 'python3 is required for preflight.' >&2; exit 1; }
python3 scripts/doctor.py --preflight --require-cuda
mkdir -p work logs reports/environment reports/smoke data/raw/zenodo data/raw/violin \
  data/interim data/processed data/manifests data/splits models
if command -v uv >/dev/null 2>&1; then
  TASK_UV="$(command -v uv)"
elif [[ -x "$HOME/.local/bin/uv" ]]; then
  TASK_UV="$HOME/.local/bin/uv"
else
  command -v curl >/dev/null || { echo 'curl is required to install uv.' >&2; exit 1; }
  curl --fail --show-error --silent --location https://astral.sh/uv/install.sh -o work/uv-install.sh
  env UV_INSTALL_DIR="$HOME/.local/bin" UV_NO_MODIFY_PATH=1 sh work/uv-install.sh
  TASK_UV="$HOME/.local/bin/uv"
fi
"$TASK_UV" --version
# Cache on the same volume as the project: no surprise filling a separate home partition.
export UV_CACHE_DIR="$TASK_ROOT/.cache/uv"
export UV_PYTHON_INSTALL_DIR="$TASK_ROOT/.cache/python"
"$TASK_UV" sync --locked --python 3.11.15
"$TASK_UV" run --locked python scripts/doctor.py --require-cuda
"$TASK_UV" run --locked python scripts/smoke_test.py
TASK_STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
"$TASK_UV" pip freeze --python .venv/bin/python > "reports/environment/packages_${TASK_STAMP}.txt"
printf '\nSETUP PASS. GPU and synthetic IO checks passed. No research data downloaded.\n'
printf 'Next: bash scripts/run.sh catalog\n'
