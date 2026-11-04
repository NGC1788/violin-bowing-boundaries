#!/bin/bash
# Claude API as the strong mutator, on the Mac: same engine, evaluator and starting population as Colab.
#   export ANTHROPIC_API_KEY=...        (your own key; it is never written to disk by this script)
#   BUDGET=5 agentic_research/evolve_v4/claude_mutator.sh            # pilot, then BUDGET=100 continues
# Spend is cumulative per run dir (runs/claude_<task>/claude_spend.json) and the engine stops itself
# before the next call could push it past BUDGET. TASK=bundle (violin) or bundle_robust (AI audit).
set -euo pipefail
MODEL=${MODEL:-claude-opus-5-5}; BUDGET=${BUDGET:-5}; EFFORT=${EFFORT:-high}; TASK=${TASK:-bundle}
REPO=$(cd "$(dirname "$0")/../.." && pwd)
RUN=$REPO/agentic_research/evolve_v4/runs/claude_$TASK
mkdir -p "$RUN"
cd "$REPO/agentic_research/evolve_v4/colab/$TASK"
exec "$REPO/.venv-mac/bin/python" -m agentic_research.evolve_v4.colab_engine --run-dir "$RUN" --model "$MODEL" \
  --budget-usd "$BUDGET" --effort "$EFFORT" --concurrency 4 --max-tokens 45000 --hours 72 --seed 11
