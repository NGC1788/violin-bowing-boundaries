"""Independent literal replay of every coherent scalar-threshold atom.

Run from any working directory. The implementation under test is imported only
at main(); oracle.py never imports it. Counts distinguish generated fixtures,
threshold probes, individual profile comparisons and minima checks.
"""
from pathlib import Path
import sys, math, json, time, hashlib
import numpy as np
from oracle import first_run, literal_column, literal_folder, threshold_atoms
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent))


def original_eligible(force, baseline, k):
    s=first_run(baseline,k)
    return s is not None and s>0 and 0<force[s-1]<force[s]


def normalize(c):
    return dict(c,labels=c['baseline'])


def check_case(analyze, columns, relation, domain, k, counts):
    slopes=(-2.,-1.2,0.);epsilons=(0.,.2,.4)
    result=analyze(columns,relation,domain,slopes=slopes,epsilons=epsilons,k=k)
    groups=sorted({c['group'] for c in columns})
    assert result['groups']==groups
    atoms=result['atoms'];n_atoms=len(atoms['representative'])
    scores=[v for c in columns for v,g in zip(c['values'],c['guard']) if g]
    probes=threshold_atoms(scores,domain)
    probes += [{'threshold':t} for t in atoms['representative']]
    seen=set(); oracle_costs={}
    for probe in probes:
        threshold=probe['threshold']
        memberships=[i for i in range(n_atoms)
          if (threshold>atoms['lo'][i] or (threshold==atoms['lo'][i] and atoms['lo_closed'][i]))
          and (threshold<atoms['hi'][i] or (threshold==atoms['hi'][i] and atoms['hi_closed'][i]))]
        assert len(memberships)==1,(domain,threshold,atoms,memberships)
        ai=memberships[0];seen.add(ai);counts['threshold_probes']+=1
        for gi,group in enumerate(groups):
            cols=[normalize(c) for c in columns if c['group']==group]
            b=literal_folder(cols,threshold,relation,k,0.,0.)
            assert int(result['n_all_edits'][ai,gi])==b['cost_all']
            assert int(result['n_eligible_edits'][ai,gi])==b['cost_eligible']
            assert int(result['n_all_changed_columns'][ai,gi])==b['changed_columns']
            ec=sum(c['eligible'] and any(a!=d for a,d in zip(c['labels'],ed)) for c,ed in zip(cols,b['labels']))
            assert int(result['n_changed_columns'][ai,gi])==ec
            invalid=sum(not z['valid'] for z in b['boundaries'])
            assert int(result['invalid_eligible_columns'][ai,gi])==invalid
            for pi,p in enumerate(slopes):
                expected=literal_folder(cols,threshold,relation,k,p,0.)
                valid=expected['valid']
                for field in ('lo','hi','gap'):
                    actual_name={'lo':'lower','hi':'upper'}.get(field,field)
                    actual=float(result[actual_name][ai,gi,pi])
                    if valid:
                        assert math.isclose(actual,expected[field],rel_tol=1e-13,abs_tol=1e-13),(field,actual,expected)
                    else:
                        assert math.isnan(actual),(field,actual,expected)
                for ei,eps in enumerate(epsilons):
                    expected_compatible=valid and expected['lo']-eps<=expected['hi']+eps
                    # The production comparison is gap <= 2 epsilon. Random
                    # cases avoid exact machine-rounding disagreement at this
                    # algebraically equivalent boundary; fixed exact tests below
                    # include zero-gap point intersections.
                    assert bool(result['compatible'][ai,gi,pi,ei])==expected_compatible
                    oracle_costs.setdefault((gi,pi,ei),{})[ai]=(expected_compatible,b['cost_all'],b['cost_eligible'])
                counts['profile_comparisons']+=1
    assert seen==set(range(n_atoms)),(seen,n_atoms)
    counts['atoms']+=n_atoms
    for (gi,pi,ei),values in oracle_costs.items():
        for target in (False,True):
            # Invalid threshold decisions are not evidence of incompatibility;
            # exclude them from both target directions.
            rows=[(a,v) for a,v in values.items() if v[0]==target and math.isfinite(float(result['gap'][a,gi,pi]))]
            for cost_name,cost_ix in [('n_all_edits',1),('n_eligible_edits',2)]:
                expected=min((v[cost_ix] for a,v in rows),default=math.inf)
                mask=np.asarray(result['compatible'][:,gi,pi,ei])==target
                mask &= np.isfinite(result['gap'][:,gi,pi])
                actual=np.min(result[cost_name][:,gi][mask]) if mask.any() else math.inf
                assert actual==expected
                counts['minima_checks']+=1
    counts['fixtures']+=1


def cases():
    rng=np.random.default_rng(20261021)
    for k in (1,2,3):
        # Exhaust all baseline vectors up to length six. The global coherent
        # rule remains independent from these baselines, exercising both edit
        # directions and labels that the rule cannot individually change.
        for n in range(1,7):
            for bits in range(2**n):
                baseline=[(bits>>j)&1 for j in range(n)]
                force=np.exp(np.linspace(-3,1,n)).tolist()
                vals=[(.25,.5,.5,.75,math.nan,1.)[j%6] for j in range(n)]
                guard=[int(math.isfinite(x)) for x in vals];fixed=[0]*n
                cols=[dict(id='a',group='g',force=force,beta=.1,baseline=baseline,
                  values=vals,guard=guard,fixed=fixed,eligible=original_eligible(force,baseline,k))]
                yield cols, 'ge' if bits%2==0 else 'lt', (0.,1.), k
        for i in range(150):
            cols=[]
            for ci in range(int(rng.integers(1,7))):
                n=int(rng.integers(1,13));status=rng.integers(0,3,n)
                vals=rng.choice([0.,.25,.5,.5,.75,1.,math.nan],n).tolist()
                guard=[int(s==2 and math.isfinite(v)) for s,v in zip(status,vals)]
                fixed=[int(s==1) for s in status]
                baseline=rng.integers(0,2,n).tolist()
                force=np.sort(rng.choice([-.1,0.,.01,.02,.05,.1,.1,.2,.5,1.,2.,4.],n)).tolist()
                cols.append(dict(id=f'c{ci}',group=f'g{ci%2}',force=force,beta=float(rng.choice([.05,.1,.2])),
                    baseline=baseline,values=vals,guard=guard,fixed=fixed,
                    eligible=original_eligible(force,baseline,k)))
            yield cols, ('ge','lt')[i%2], [(0.,1.),(.25,.75),(.5,.5),(-math.inf,math.inf)][i%4], k


def main():
    from threshold_profile import analyze_thresholds
    started=time.monotonic();counts=dict(fixtures=0,threshold_probes=0,atoms=0,profile_comparisons=0,minima_checks=0)
    for ci,case in enumerate(cases()):
        try:check_case(analyze_thresholds,*case,counts)
        except Exception:
            print('FAILED FIXTURE',ci,repr(case));raise
    counts.update(passed=True,seconds=time.monotonic()-started,
       source_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),HERE/'oracle.py',HERE.parent/'threshold_profile.py']})
    (HERE/'exhaustive_results.json').write_text(json.dumps(counts,indent=2)+'\n')
    print(json.dumps(counts,indent=2))
if __name__=='__main__':main()
