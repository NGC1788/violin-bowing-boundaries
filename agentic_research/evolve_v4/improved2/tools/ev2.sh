#!/bin/bash
# usage: tools/ev2.sh NAME POLICY.py  -> scores POLICY on violin search (colab/bundle_noeps) and AI-audit search
# (colab/bundle_robust); raw JSON goes to tools/res/NAME.{V,R}.json, one summary line is printed.
T=$(cd "$(dirname "$0")" && pwd)
E=$(cd "$T/../.." && pwd)
name=$1; f=$(realpath "$2")
mkdir -p "$T/res"
(cd "$E/colab/bundle_noeps" && python -m agentic_research.evolve_v4.colab_eval "$f") > "$T/res/$name.V.json" 2> "$T/res/$name.V.err" &
(cd "$E/colab/bundle_robust" && EVOLVE_PARALLEL=2 python -m agentic_research.evolve_v4.colab_eval "$f") > "$T/res/$name.R.json" 2> "$T/res/$name.R.err"
wait
python - "$T/res/$name" "$name" <<'PY'
import json, sys
p, n = sys.argv[1], sys.argv[2]
out = [n]
for k in "VR":
    try:
        d = json.load(open(p + f".{k}.json"))
        out += [d['total'], d['totals']['eps0.2'], d['totals']['eps0.4'], d['compatible_total'], round(d['max_case_seconds'], 2)]
    except Exception:
        out += ['ERROR', open(p + f".{k}.err").read()[-300:].replace('\n', ' '), '', '', '']
print(*out, sep='\t')
PY
