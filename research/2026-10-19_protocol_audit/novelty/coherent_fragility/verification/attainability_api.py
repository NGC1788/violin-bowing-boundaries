"""Independent point/open-cell replay of closed-form attainability API."""
from pathlib import Path
import sys,math,json,itertools,hashlib
from oracle import literal_labels,first_run,threshold_atoms
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'literature'))
from attainability import first_run_threshold_intervals,no_run_threshold_interval,contains


def main():
    checked=fixtures=0
    domains=[(0.,1.),(.5,.5),(.5,1.),(0.,math.inf),(math.inf,math.inf)]
    for n in range(0,6):
        values=[(.25,.5,.5,.75,.25)[j] for j in range(n)]
        for statuses in itertools.product(range(3),repeat=n):
            guard=[int(s==2) for s in statuses];fixed=[int(s==1) for s in statuses]
            for k in (1,2,3,7):
                for relation in ('ge','lt'):
                    for domain in domains:
                        intervals=first_run_threshold_intervals(values,guard,fixed,k,relation,domain)
                        no_run=no_run_threshold_interval(values,guard,fixed,k,relation,domain)
                        if no_run is not None:intervals.append(no_run)
                        probes=[x['threshold'] for x in threshold_atoms(values,domain)]
                        probes += [domain[0],domain[1]]
                        for threshold in probes:
                            actual=first_run(literal_labels(fixed,guard,values,threshold,relation),k)
                            matches=[x['start'] for x in intervals if contains(x,threshold)]
                            assert matches==[actual],(values,statuses,k,relation,domain,threshold,actual,matches,intervals)
                            checked+=1
                        fixtures+=1
    # Locked NaNs preserve fixed positives and never create guard events.
    values=[math.nan,.5,math.nan];guard=[0,1,0];fixed=[1,0,0]
    for relation in ('ge','lt'):
        intervals=first_run_threshold_intervals(values,guard,fixed,2,relation,(0.,math.inf))
        no_run=no_run_threshold_interval(values,guard,fixed,2,relation,(0.,math.inf))
        if no_run:intervals.append(no_run)
        for threshold in [0.,.5,.75,math.inf]:
            actual=first_run(literal_labels(fixed,guard,values,threshold,relation),2)
            assert [x['start'] for x in intervals if contains(x,threshold)]==[actual]
            checked+=1
    out=dict(passed=True,fixtures=fixtures,threshold_comparisons=checked,
        source_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),HERE/'oracle.py',HERE.parent/'literature/attainability.py']})
    (HERE/'attainability_api.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
if __name__=='__main__':main()
