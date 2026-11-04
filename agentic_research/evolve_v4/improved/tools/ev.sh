#!/bin/bash
# usage: ev.sh NAME FILE  -> scores FILE with the bundle evaluator, appends to results.tsv
W=$(cd "$(dirname "$0")" && pwd)/work; mkdir -p $W
B=$(cd "$(dirname "$0")/../../colab/bundle" && pwd)
name=$1; f=$(realpath $2)
mkdir -p $W/res
(cd $B && python -m agentic_research.evolve_v4.colab_eval $f) > $W/res/$name.json 2> $W/res/$name.err
python - "$W/res/$name.json" "$name" <<'PY' | tee -a $W/results.tsv
import json, sys
try:
    d = json.load(open(sys.argv[1]))
    print(sys.argv[2], d['total'], d['totals']['eps0.2'], d['totals']['eps0.4'], d['compatible_total'], round(d['max_case_seconds'], 2), sep='\t')
except Exception as e:
    print(sys.argv[2], 'ERROR', open(sys.argv[1].replace('.json', '.err')).read()[-300:].replace('\n', ' '), sep='\t')
PY
