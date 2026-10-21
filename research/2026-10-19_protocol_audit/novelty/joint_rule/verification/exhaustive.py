"""Independent full-grid small-fixture verification of the joint-rule solver."""
from pathlib import Path
import hashlib, itertools, json, math, sys, time
import numpy as np
from oracle import labels, evaluate, independent_probes, eligible, boundary
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent))
COUNT_FIELDS=('n_all_edits','n_eligible_edits','n_all_changed_columns','n_changed_columns',
              'invalid_eligible_columns','no_run_eligible_columns','left_censored_eligible_columns',
              'nonpositive_or_tied_eligible_columns')


def col(identifier, force, periodicity, ratio, fixed=None, guard=None, group='g', beta=.1):
    n=len(force)
    c=dict(id=identifier,group=group,force=list(force),beta=beta,periodicity=list(periodicity),ratio=list(ratio),
           fixed=list(fixed if fixed is not None else [0]*n),guard=list(guard if guard is not None else [1]*n))
    c['baseline']=labels(c,.8,.5);c['eligible']=eligible(c)
    return c


def named_cases():
    force=[1,2,4,8,16,32]
    a=col('u_gate',force,[.1,.82,.95,.95,.95,.95],[.1]*6)
    b=col('v_gate',force,[.1,.1,.95,.95,.95,.95],[.1,.1,.6,.1,.1,.1])
    c=col('fixed_reference',[1,2.5,3.5,8,16,32],[.1]*6,[.9]*6,fixed=[0,0,1,1,1,1],guard=[0]*6)
    yield 'joint_only',[a,b,c],2,(.7,.9),(.3,.7)
    # A different folder needs v <= .4; the first needs v > .6.
    d=col('opposing_gate',force,[.1,.95,.95,.95,.95,.95],[.1,.4,.1,.1,.1,.1],group='opposing')
    e=dict(c,id='opposing_reference',group='opposing')
    yield 'conflicting_local_optima',[a,b,c,d,e],2,(.7,.9),(.3,.7)
    # Raising v beyond .6 adds index zero, making another original boundary left-censored.
    f=col('collateral_left_censor',force,[.95]*6,[.6,.1,.1,.1,.1,.1],group='collateral')
    yield 'invalid_collateral',[a,b,c,f],2,(.7,.9),(.3,.7)
    tied=col('ties',[1,1,2,3,5,7],[.7,.8,.8,.9,.9,.95],[.3,.5,.5,.7,.7,.2])
    tied['baseline']=[0,0,0,1,1,0];tied['eligible']=True
    yield 'tied_scores_and_force',[tied],2,(.7,.9),(.3,.7)
    nan=col('guarded_nan',force,[math.nan,.85,.85,.95,.75,.75],[math.nan,.4,.4,.4,.6,.6],fixed=[1,0,0,0,0,0],guard=[0,1,1,1,1,1])
    yield 'guarded_nan_fixed',[nan],2,(.7,.9),(.3,.7)


def random_cases():
    rng=np.random.default_rng(20261022)
    # Exhaust ternary fixed/guard/off states for short columns; scores include ties.
    for k in (1,2,3):
        for n in range(1,5):
            for code in itertools.product(range(3),repeat=n):
                p=[(.7,.8,.8,.9)[j%4] for j in range(n)]
                r=[(.3,.5,.5,.7)[j%4] for j in range(n)]
                c=col('c',np.exp(np.arange(n)),p,r,fixed=[int(s==1) for s in code],guard=[int(s==2) for s in code])
                c['eligible']=eligible(c,k)
                yield f'exhaust_{k}_{n}_{code}',[c],k,(.7,.9),(.3,.7)
        for i in range(130):
            cols=[]
            for j in range(int(rng.integers(1,6))):
                n=int(rng.integers(1,11));status=rng.integers(0,3,n)
                p=rng.choice([.5,.7,.75,.8,.8,.85,.9,1.],n).tolist()
                r=rng.choice([.1,.3,.4,.5,.5,.6,.7,.9],n).tolist()
                force=np.sort(rng.choice([-.1,0.,.01,.02,.05,.1,.1,.2,.5,1.,2.,4.],n)).tolist()
                c=col(f'c{j}',force,p,r,fixed=[int(s==1) for s in status],guard=[int(s==2) for s in status],group=f'g{j%2}',beta=float(rng.choice([.05,.1,.2])))
                # Test generic baseline edit accounting, not only frozen-rule baselines.
                if i%2:c['baseline']=rng.integers(0,2,n).tolist()
                c['eligible']=eligible(c,k)
                cols.append(c)
            doms=[((.7,.9),(.3,.7)),((.8,.8),(.5,.5)),((0.,1.),(0.,1.))]
            yield f'random_{k}_{i}',cols,k,*doms[i%3]


def membership(atoms,t):
    indices=[i for i in range(len(atoms['representative']))
      if (t>atoms['lo'][i] or (t==atoms['lo'][i] and atoms['lo_closed'][i]))
      and (t<atoms['hi'][i] or (t==atoms['hi'][i] and atoms['hi_closed'][i]))]
    assert len(indices)==1,(t,atoms,indices)
    return indices[0]


def check(analyze,name,columns,k,ud,vd,counts):
    slopes=(-2.,-1.2,0.);epsilons=(0.,.1,.2,.4)
    groups=sorted({c['group'] for c in columns})
    outputs={g:analyze([c for c in columns if c['group']==g],u_domain=ud,v_domain=vd,slopes=slopes,epsilons=epsilons,k=k) for g in groups}
    us=independent_probes([x for c in columns for x in c['periodicity']],ud)
    vs=independent_probes([x for c in columns for x in c['ratio']],vd)
    us=sorted(set(us+[float(x) for o in outputs.values() for x in o['u_atoms']['representative']]))
    vs=sorted(set(vs+[float(x) for o in outputs.values() for x in o['v_atoms']['representative']]))
    seen={g:set() for g in groups};expected_cells={g:{} for g in groups};selected=[]
    for u,v in itertools.product(us,vs):
        counts['threshold_pairs']+=1
        row=[]
        for g,o in outputs.items():
            ui=membership(o['u_atoms'],u);vi=membership(o['v_atoms'],v);seen[g].add((ui,vi))
            cs=[c for c in columns if c['group']==g]
            ref=evaluate(cs,u,v,k,slopes,epsilons)
            for field in COUNT_FIELDS:
                assert int(o[field][ui,vi])==ref[field],(name,g,u,v,field,int(o[field][ui,vi]),ref[field])
                counts['scalar_comparisons']+=1
            for field in ('lower','upper','gap'):
                np.testing.assert_allclose(o[field][ui,vi],ref[field],rtol=1e-13,atol=1e-13,equal_nan=True,err_msg=f'{name}/{field}')
                counts['scalar_comparisons']+=len(slopes)
            assert np.array_equal(o['compatible'][ui,vi],ref['compatible']),(name,g,u,v,ref,o['compatible'][ui,vi])
            counts['scalar_comparisons']+=len(slopes)*len(epsilons)
            ec=[c for c in cs if c['eligible']]
            ids=o['metadata']['eligible_column_ids']
            assert ids==[c['id'] for c in ec]
            bs=[boundary(c,labels(c,u,v),k) for c in ec]
            invalid_indices=[j for j,b in enumerate(bs) if not b['valid']]
            expected_invalid=invalid_indices[0] if invalid_indices else -1
            assert int(o['invalid_witness'][ui,vi])==expected_invalid
            for pi,p in enumerate(slopes):
                if ec and not invalid_indices:
                    lower=[b['low']-p*b['log_beta'] for b in bs]
                    upper=[b['high']-p*b['log_beta'] for b in bs]
                    li=int(o['lower_witness'][ui,vi,pi]);hi=int(o['upper_witness'][ui,vi,pi])
                    assert lower[li]==max(lower) and upper[hi]==min(upper)
                else:
                    assert int(o['lower_witness'][ui,vi,pi])==-1
                    assert int(o['upper_witness'][ui,vi,pi])==-1
                counts['certificate_checks']+=1
            expected_cells[g][ui,vi]=ref
            row.append(ref)
        selected.append((u,v,row))
    for g,o in outputs.items():
        assert len(seen[g])==len(o['u_atoms']['representative'])*len(o['v_atoms']['representative'])
        counts['cells']+=len(seen[g])
        for pi in range(len(slopes)):
            for ei in range(len(epsilons)):
                for field in ('n_all_edits','n_eligible_edits'):
                    want=min((ref[field] for ref in expected_cells[g].values() if ref['compatible'][pi][ei]),default=math.inf)
                    mask=o['compatible'][:,:,pi,ei]
                    got=int(o[field][mask].min()) if mask.any() else math.inf
                    assert want==got,(name,g,pi,ei,field,want,got)
                    counts['minima_checks']+=1
    if name=='joint_only':
        assert not any(row[0]['compatible'][0][0] for u,v,row in selected if u==.8 or v==.5)
        assert any(row[0]['compatible'][0][0] for u,v,row in selected)
    if name=='conflicting_local_optima':
        assert all(any(row[j]['compatible'][0][0] for u,v,row in selected) for j in range(2))
        assert max(sum(z['compatible'][0][0] for z in row) for u,v,row in selected)==1
    if name=='invalid_collateral':
        gi=groups.index('g');ci=groups.index('collateral')
        assert any(row[gi]['compatible'][0][0] for u,v,row in selected)
        assert all(row[ci]['invalid_eligible_columns']>0 for u,v,row in selected if row[gi]['compatible'][0][0])
    counts['fixtures']+=1


def main():
    from solver import analyze_group
    started=time.monotonic();counts=dict(fixtures=0,threshold_pairs=0,cells=0,scalar_comparisons=0,minima_checks=0,certificate_checks=0)
    for case in itertools.chain(named_cases(),random_cases()):
        try:check(analyze_group,*case,counts)
        except Exception:
            print('FAILED',case[0],repr(case));raise
    counts.update(passed=True,seconds=time.monotonic()-started,source_sha256={str(p.relative_to(HERE.parent)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),HERE/'oracle.py',HERE.parent/'solver.py']})
    (HERE/'exhaustive_results.json').write_text(json.dumps(counts,indent=2)+'\n')
    print(json.dumps(counts,indent=2))
if __name__=='__main__':main()
