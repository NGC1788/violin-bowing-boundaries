"""A third structural check: reachable first runs are strict record extrema.

This verifier is independent of the production sweep. For finite thresholds,
fixed positives have activation score +inf (>=) or -inf (<). Impossible K-windows
are omitted, rather than assigning a sentinel that could activate at infinity.
"""
from pathlib import Path
import math,json,itertools,hashlib
from oracle import literal_labels,first_run
HERE=Path(__file__).resolve().parent


def envelope_first_start(values,guard,fixed,threshold,k,direction):
    records=[];record=-math.inf if direction=='ge' else math.inf
    for s in range(len(values)-k+1):
        indices=range(s,s+k)
        if not all(fixed[j] or guard[j] for j in indices):continue
        values_here=[values[j] for j in indices if not fixed[j]]
        if direction=='ge':
            activation=min(values_here,default=math.inf)
            if activation>record:
                records.append((s,record,activation));record=activation
        else:
            activation=max(values_here,default=-math.inf)
            if activation<record:
                records.append((s,activation,record));record=activation
    matches=[s for s,low,high in records if low<threshold<=high]
    assert len(matches)<=1
    return matches[0] if matches else None


def main():
    checked=0;fixtures=0
    for n in range(1,7):
        # Scores include ties; the three bit statuses mean immutable zero,
        # immutable one and active score. All 3^n statuses are enumerated.
        values=[(.25,.5,.5,.75,.25,.9)[j] for j in range(n)]
        for statuses in itertools.product(range(3),repeat=n):
            guard=[int(s==2) for s in statuses];fixed=[int(s==1) for s in statuses]
            for k in (1,2,3):
                for direction in ('ge','lt'):
                    fixtures+=1
                    for threshold in [0.,.25,.3,.5,.6,.75,.8,.9,1.]:
                        direct=first_run(literal_labels(fixed,guard,values,threshold,direction),k)
                        via_record=envelope_first_start(values,guard,fixed,threshold,k,direction)
                        assert direct==via_record,(statuses,k,direction,threshold,direct,via_record)
                        checked+=1
    out=dict(passed=True,status_rule_k_fixtures=fixtures,threshold_comparisons=checked,
       source_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),HERE/'oracle.py']},
       interpretation='Structural record-extrema equivalence, not a novel general algorithm or empirical validation.')
    (HERE/'record_envelope.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
if __name__=='__main__':main()
