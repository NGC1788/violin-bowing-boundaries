#!/bin/bash
# score a policy on the robust transfer search split: rcmp.sh NAME FILE
W=/tmp/claude-0/-home-user-violin-bowing-boundaries/e50b1587-f063-5723-ad34-32ffcb1d001e/scratchpad/work
B=/home/user/violin-bowing-boundaries/agentic_research/evolve_v4/colab/bundle_robust
f=$(realpath $2)
(cd $B && python -m agentic_research.evolve_v4.colab_eval $f) > $W/res/R_$1.json 2>$W/res/R_$1.err
python -c "import json,sys; d=json.load(open(sys.argv[1])); print(sys.argv[2], d['total'], d['totals']['eps0.2'], d['totals']['eps0.4'], d['compatible_total'], round(d['max_case_seconds'],2), sep='\t')" $W/res/R_$1.json R_$1 | tee -a $W/results_robust.tsv
