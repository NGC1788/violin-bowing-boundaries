#!/bin/bash
# score a policy on the robust transfer search split: rcmp.sh NAME FILE
W=$(cd "$(dirname "$0")" && pwd)/work; mkdir -p $W/res
B=$(cd "$(dirname "$0")/../../colab/bundle_robust" && pwd)
f=$(realpath $2)
(cd $B && python -m agentic_research.evolve_v4.colab_eval $f) > $W/res/R_$1.json 2>$W/res/R_$1.err
python -c "import json,sys; d=json.load(open(sys.argv[1])); print(sys.argv[2], d['total'], d['totals']['eps0.2'], d['totals']['eps0.4'], d['compatible_total'], round(d['max_case_seconds'],2), sep='\t')" $W/res/R_$1.json R_$1 | tee -a $W/results_robust.tsv
