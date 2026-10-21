"""Independent all-cell empirical replay by column-local literal label tables.

No production solver/exporter/empirical import. The production algorithm sweeps
V labels while scanning U. This oracle instead reclassifies every observation
at every column-local Cartesian atom, records its first run, and broadcasts
those small tables onto each complete group arrangement. Endpoint/invalid
certificates are checked across every local group cell.
"""
from pathlib import Path
from collections import Counter
import gzip,hashlib,json,math,time
import numpy as np
from oracle import labels,boundary,evaluate
HERE=Path(__file__).resolve().parent;ROOT=HERE.parent
FIELDS=('n_all_edits','n_eligible_edits','n_all_changed_columns','n_changed_columns',
        'invalid_eligible_columns','no_run_eligible_columns','left_censored_eligible_columns',
        'nonpositive_or_tied_eligible_columns')


def cuts(columns,axis,domain):
    # Intentionally unpruned feature events: guards inapplicable on the other
    # axis produce duplicate states, retained to independently audit pruning.
    return np.array(sorted({float(x) for c in columns for x,g in zip(c[axis],c['guard'])
                  if g and math.isfinite(x) and domain[0]<=x<domain[1]})+[domain[1]])


def compare(got,want,name,counts):
    assert got.shape==want.shape,(name,got.shape,want.shape)
    if got.dtype.kind=='f':assert np.allclose(got,want,rtol=1e-13,atol=1e-13,equal_nan=True),name
    else:assert np.array_equal(got,want),name
    counts['profile_scalar_comparisons']+=int(got.size)


def independent_group(columns,ur,vr,slopes,eps,counts):
    shape=(len(ur),len(vr));out={key:np.zeros(shape,dtype=np.int64) for key in FIELDS}
    lo=np.full(shape+(len(slopes),),-math.inf);hi=np.full_like(lo,math.inf)
    lw=np.full(lo.shape,-1,np.int16);hw=np.full(lo.shape,-1,np.int16)
    iw=np.full(shape,-1,np.int16);eidx=0
    for c in columns:
        us=cuts([c],'periodicity',(.7,.9));vs=cuts([c],'ratio',(.3,.7))
        um=np.searchsorted(us,ur,side='left');vm=np.searchsorted(vs,vr,side='left')
        assert np.all(um<len(us)) and np.all(vm<len(vs))
        cost=np.zeros((len(us),len(vs)),np.int64)
        bad=np.zeros_like(cost,bool);nrun=bad.copy();left=bad.copy();tied=bad.copy()
        low=np.full(cost.shape+(len(slopes),),-math.inf);high=np.full_like(low,math.inf)
        for i,u in enumerate(us):
            for j,v in enumerate(vs):
                bits=labels(c,u,v);cost[i,j]=sum(a!=b for a,b in zip(bits,c['baseline']))
                counts['column_local_literal_patterns']+=1;counts['literal_window_classifications']+=len(bits)
                if c['eligible']:
                    b=boundary(c,bits,2);bad[i,j]=not b['valid']
                    if b['valid']:
                        low[i,j]=[b['low']-p*b['log_beta'] for p in slopes]
                        high[i,j]=[b['high']-p*b['log_beta'] for p in slopes]
                    else:
                        nrun[i,j]=b['cause']=='no_run';left[i,j]=b['cause']=='left_censored';tied[i,j]=b['cause']=='nonpositive_or_tied'
        mapped=lambda arr:arr[um[:,None],vm[None,:]]
        ed=mapped(cost);out['n_all_edits']+=ed;out['n_all_changed_columns']+=ed>0
        if c['eligible']:
            out['n_eligible_edits']+=ed;out['n_changed_columns']+=ed>0
            for field,arr in [('invalid_eligible_columns',bad),('no_run_eligible_columns',nrun),('left_censored_eligible_columns',left),('nonpositive_or_tied_eligible_columns',tied)]:out[field]+=mapped(arr)
            iw[(iw<0)&mapped(bad)]=eidx
            l=mapped(low);h=mapped(high)
            take=l>lo;lo[take]=l[take];lw[take]=eidx
            take=h<hi;hi[take]=h[take];hw[take]=eidx
            eidx+=1
    invalid=out['invalid_eligible_columns']>0
    if not eidx:invalid[:]=True
    lo[invalid]=np.nan;hi[invalid]=np.nan;lw[invalid]=-1;hw[invalid]=-1
    out.update(lower=lo,upper=hi,gap=lo-hi,compatible=(lo[...,None]-eps<=hi[...,None]+eps),lower_witness=lw,upper_witness=hw,invalid_witness=iw)
    return out


def check_witness(witness,cols,p,epsilon,counts):
    u,v=witness['u'],witness['v']
    result={};all_edits=eligible_edits=invalid=0;compatible=[];badgroups=[]
    for group in sorted({c['group'] for c in cols}):
        cs=[c for c in cols if c['group']==group]
        r=evaluate(cs,u,v,2,[p],[epsilon]);result[group]=r
        all_edits+=r['n_all_edits'];eligible_edits+=r['n_eligible_edits'];invalid+=r['invalid_eligible_columns']
        if r['compatible'][0][0]:compatible.append(group)
        if r['invalid_eligible_columns']:badgroups.append(group)
        counts['global_witness_window_classifications']+=sum(len(c['baseline']) for c in cs)
    assert witness['all_edits']==all_edits and witness['eligible_edits']==eligible_edits and witness['invalid_columns']==invalid
    assert witness['compatible_groups']==compatible and witness['invalid_groups']==badgroups
    for row in witness['groups']:
        r=result[row['group']]
        for field,rfield in [('all_edits','n_all_edits'),('eligible_edits','n_eligible_edits'),('changed_eligible_columns','n_changed_columns'),('invalid_columns','invalid_eligible_columns')]:assert row[field]==r[rfield]
        assert row['compatible']==r['compatible'][0][0]
        for f in ('lower','upper'):
            if row[f] is None:assert math.isnan(r[f][0])
            else:assert math.isclose(row[f],r[f][0],rel_tol=1e-13,abs_tol=1e-13)
    counts['global_witness_replays']+=1


def main():
    started=time.monotonic();raw=json.load(gzip.open(ROOT/'input_joint.json.gz','rt'))
    cols=[dict(c,periodicity=[math.nan if x is None else x for x in c['periodicity']],ratio=[math.nan if x is None else x for x in c['ratio']]) for c in raw['columns']]
    counts=Counter();files=[Path(__file__),HERE/'oracle.py',ROOT/'input_joint.json.gz'];groups=[]
    for meta_path in sorted((ROOT/'profiles').glob('[0-9][0-9].json')):
        meta=json.loads(meta_path.read_text());group=meta['group'];cs=[c for c in cols if c['group']==group]
        path=meta_path.with_suffix('.npz');z=np.load(path);a={k:z[k] for k in z.files};z.close()
        for axis,domain,score in [('u',(.7,.9),'periodicity'),('v',(.3,.7),'ratio')]:
            # Every original raw guarded score which can change a label has a
            # production atom boundary; this verifies event pruning completeness.
            relevant=set()
            for c in cs:
                for p,r,g in zip(c['periodicity'],c['ratio'],c['guard']):
                    if not g:continue
                    if axis=='u' and domain[0]<=p<domain[1] and r<.7:relevant.add(p)
                    if axis=='v' and domain[0]<=r<domain[1] and p>=.7:relevant.add(r)
            want=np.array(sorted(relevant)+[domain[1]])
            compare(a[axis+'_hi'],want,axis+'cuts',counts)
            compare(a[axis+'_representative'],want,axis+'representative',counts)
            compare(a[axis+'_lo'],np.r_[domain[0],want[:-1]],axis+'lower',counts)
            lc=np.zeros(len(want),bool);lc[0]=True
            compare(a[axis+'_lo_closed'],lc,axis+'lowerclosure',counts)
            compare(a[axis+'_hi_closed'],np.ones(len(want),bool),axis+'upperclosure',counts)
        assert meta['metadata']['eligible_column_ids']==[c['id'] for c in cs if c['eligible']]
        expected=independent_group(cs,a['u_hi'],a['v_hi'],a['slopes'],a['epsilons'],counts)
        for key,want in expected.items():compare(a[key],want,key,counts)
        counts['group_cells']+=int(a['n_all_edits'].size);counts['groups']+=1
        files.extend([path,meta_path]);groups.append(group)
        print('verified',counts['groups'],group,flush=True)
    assert len(groups)==39
    results=json.loads((ROOT/'results.json').read_text());assert groups==results['groups']
    seen=set()
    def unique_replay(w,p,e):
        if w is None:return
        key=(w['u'],w['v'],p,e)
        if key not in seen:check_witness(w,cols,p,e,counts);seen.add(key)
    for analysis in results['analyses']:
        p,e=analysis['p'],analysis['epsilon']
        for choices in analysis['policies'].values():
            for choice in choices:unique_replay(choice['witness'],p,e)
        for local in analysis['local']:
            for mode in local['modes'].values():
                if mode:unique_replay(mode['witness'],p,e)
    for row in results['minimum_tolerance']:unique_replay(row['witness'],row['p'],.2)
    files.append(ROOT/'results.json')
    out=dict(counts,passed=True,seconds=time.monotonic()-started,source_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files})
    (HERE/'real_profiles.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
if __name__=='__main__':main()
