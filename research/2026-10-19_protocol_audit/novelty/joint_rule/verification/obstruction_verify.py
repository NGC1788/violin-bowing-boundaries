"""Independent minimal fixed-subset obstruction and rectangle-cover audit.

Uses literal window labels and first-run search, not theory rectangle routines
or production solver. All eligible column pairs are checked anew. Selected
fixed subsets are audited on a separate arrangement of raw feature events and
all supplied rectangle endpoints, including exact points and immediate-right
probes. Every rectangle's claimed boundary is checked against literal labels.
"""
from pathlib import Path
from collections import Counter
import gzip,hashlib,itertools,json,math,time
import numpy as np
from oracle import labels,first_run,boundary,independent_probes
HERE=Path(__file__).resolve().parent;ROOT=HERE.parent


def local_state_table(c,ur,vr,counts):
    us=np.array(sorted({x for x,g in zip(c['periodicity'],c['guard']) if g and .7<=x<.9})+[.9])
    vs=np.array(sorted({x for x,g in zip(c['ratio'],c['guard']) if g and .3<=x<.7})+[.7])
    table=np.empty((len(us),len(vs)),np.int16)
    for i,u in enumerate(us):
        for j,v in enumerate(vs):
            s=first_run(labels(c,u,v),2);table[i,j]=-1 if s is None else s
            counts['literal_column_patterns']+=1;counts['literal_window_classifications']+=len(c['baseline'])
    return table[np.searchsorted(us,ur)[:,None],np.searchsorted(vs,vr)[None,:]]


def bounds(c,states):
    force=np.array(c['force']);lo=force[np.maximum(states-1,0)];hi=force[np.maximum(states,0)]
    valid=(states>0)&(lo>0)&(hi>lo)
    low=np.full(states.shape,-math.inf);high=np.full(states.shape,math.inf)
    for s in np.unique(states):
        if s>0 and 0<force[s-1]<force[s]:
            low[states==s]=math.log(force[s-1]);high[states==s]=math.log(force[s])
    return valid,low,high


def inside(interval,points):
    assert interval['high_closed'] is True
    return ((points>=interval['low']) if interval['low_closed'] else (points>interval['low'])) & (points<=interval['high'])


def cover_audit(cert,columns,counts):
    subset=cert.get('chosen_pair',cert.get('chosen_triple'));ids=subset['ids'];cs=[columns[x] for x in ids]
    p,eps=cert['p'],cert['epsilon'];rectangles=cert['rectangles']
    us=[x for c in cs for x in c['periodicity']]
    vs=[x for c in cs for x in c['ratio']]
    for r in rectangles:
        us.extend([r['u']['low'],r['u']['high']]);vs.extend([r['v']['low'],r['v']['high']])
        for ax,domain in [('u',(.7,.9)),('v',(.3,.7))]:
            q=r[ax];assert domain[0]<=q['low']<=q['high']<=domain[1]
            assert q['low']<q['high'] or q['low_closed']
    ur=np.array(independent_probes(us,(.7,.9)));vr=np.array(independent_probes(vs,(.3,.7)))
    states=[local_state_table(c,ur,vr,counts) for c in cs]
    bs=[bounds(c,s) for c,s in zip(cs,states)]
    coverage=np.zeros((len(ur),len(vr)),np.int16)
    for r in rectangles:
        iu=np.flatnonzero(inside(r['u'],ur));iv=np.flatnonzero(inside(r['v'],vr));ix=np.ix_(iu,iv)
        assert len(iu) and len(iv)
        coverage[ix]+=1
        claims=r.get('boundaries') or [r['first'],r['second']]
        assert len(claims)==len(cs)
        low=[];high=[];good=[]
        for c,s,(valid,l,h),claim in zip(cs,states,bs,claims):
            expected=-1 if claim['start'] is None else claim['start']
            assert np.all(s[ix]==expected),(cert['group'],p,c['id'],claim)
            assert np.all(valid[ix]==claim['valid'])
            if claim['valid']:
                flo,fhi=c['force'][expected-1:expected+1]
                assert claim['force']==[flo,fhi]
                ll=math.log(flo)-p*math.log(c['beta']);hh=math.log(fhi)-p*math.log(c['beta'])
                assert math.isclose(claim['lower'],ll,rel_tol=1e-13,abs_tol=1e-13)
                assert math.isclose(claim['upper'],hh,rel_tol=1e-13,abs_tol=1e-13)
                low.append(ll);high.append(hh);good.append(True)
            else:good.append(False)
            counts['rectangle_boundary_claims']+=1
        assert not all(good) or max(low)-eps>min(high)+eps
        counts['certificate_rectangles']+=1
    assert np.all(coverage==1),(cert['group'],p,'cover missing/overlap')
    # Each individual column is satisfiable at the fixed baseline. Therefore
    # a proven globally obstructing pair is automatically minimum cardinality2.
    for c in cs:assert boundary(c,labels(c,.8,.5),2)['valid']
    counts['certificate_cover_probe_points']+=coverage.size;counts['certificates']+=1


def main():
    start=time.monotonic();raw=json.load(gzip.open(ROOT/'input_joint.json.gz','rt'))
    cols={c['id']:dict(c,periodicity=[math.nan if x is None else x for x in c['periodicity']],ratio=[math.nan if x is None else x for x in c['ratio']]) for c in raw['columns']}
    report=json.loads((ROOT/'theory/obstructions.json').read_text())
    certs=json.loads((ROOT/'theory/obstruction_certificates.json').read_text())
    references=json.loads((ROOT/'prior_arbitrary_reference.json').read_text())['records']
    geometry={(r['session']+'|'+r['folder'],r['p']):bool(r['restore']['feasible']) for r in references if r['epsilon']==.2 and r['mode']=='both'}
    counts=Counter();rows=[]
    groups=sorted({c['group'] for c in cols.values()})
    for gi,group in enumerate(groups):
        cs=sorted((c for c in cols.values() if c['group']==group and c['eligible']),key=lambda c:(c['column'],c['id']))
        z=np.load(ROOT/'profiles'/f'{gi:02d}.npz');ur=z['u_hi'];vr=z['v_hi'];z.close()
        states=[local_state_table(c,ur,vr,counts) for c in cs];b=[bounds(c,s) for c,s in zip(cs,states)]
        valid=np.array([x[0] for x in b]);low=np.array([x[1] for x in b]);high=np.array([x[2] for x in b])
        for c in cs:assert boundary(c,labels(c,.8,.5),2)['valid']
        for p in (-2.,-1.2):
            row=next(r for r in report['groups'] if r['group']==group and r['p']==p)
            lo=low-p*np.array([math.log(c['beta']) for c in cs])[:,None,None]
            hi=high-p*np.array([math.log(c['beta']) for c in cs])[:,None,None]
            full=valid.all(axis=0)&(lo.max(axis=0)-.2<=hi.min(axis=0)+.2)
            assert int(full.sum())==row['compatible_cells']
            assert row['arbitrary_geometry_feasible']==geometry[group,p]
            pair_ids=[]
            for i,j in itertools.combinations(range(len(cs)),2):
                feasible=valid[i]&valid[j]&(np.maximum(lo[i],lo[j])-.2<=np.minimum(hi[i],hi[j])+.2)
                if not feasible.any():
                    ids=[cs[i]['id'],cs[j]['id']]
                    pair_ids.append(ids)
                    recorded=next(z for z in row['all_certifying_pairs'] if z['ids']==ids)
                    both=valid[i]&valid[j]
                    assert recorded['numerical_cells']==int(both.sum())
                    assert recorded['invalid_cells']==int((~both).sum())
                counts['all_pairs_checked']+=1
            assert pair_ids==[x['ids'] for x in row['all_certifying_pairs']]
            assert len(pair_ids)==row['certifying_pair_count']
            if pair_ids:assert row['minimum_obstruction_cardinality']==2
            elif not full.any():
                triple_ids=[]
                for indices in itertools.combinations(range(len(cs)),3):
                    good=valid[list(indices)].all(axis=0)
                    feasible=good&(lo[list(indices)].max(axis=0)-.2<=hi[list(indices)].min(axis=0)+.2)
                    if not feasible.any():
                        ids=[cs[i]['id'] for i in indices];triple_ids.append(ids)
                        recorded=next(z for z in row['all_certifying_triples'] if z['ids']==ids)
                        assert recorded['numerical_cells']==int(good.sum())
                        assert recorded['invalid_cells']==int((~good).sum())
                    counts['all_triples_checked']+=1
                assert triple_ids==[x['ids'] for x in row['all_certifying_triples']]
                if triple_ids:assert row['minimum_obstruction_cardinality']==3
                else:assert row['minimum_obstruction_cardinality'] is None
            rows.append(dict(group=group,p=p,fixed_pairs=len(pair_ids),minimum_cardinality=row['minimum_obstruction_cardinality']))
        print('checked all subsets',gi+1,group,flush=True)
    for cert in certs['certificates']+certs.get('triple_certificates',[]):cover_audit(cert,cols,counts)
    files=[Path(__file__),HERE/'oracle.py',ROOT/'input_joint.json.gz',ROOT/'theory/obstructions.json',ROOT/'theory/obstruction_certificates.json',ROOT/'prior_arbitrary_reference.json']
    out=dict(counts,passed=True,seconds=time.monotonic()-start,groups=rows,
      source_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files})
    (HERE/'obstruction_verify.json').write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps({k:v for k,v in out.items() if k!='groups'},indent=2))
if __name__=='__main__':main()
