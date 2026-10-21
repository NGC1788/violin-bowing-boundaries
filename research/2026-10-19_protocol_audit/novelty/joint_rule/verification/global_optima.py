"""Independent exhaustive dominance audit of all reported global/local optima.

Uses independently checked local profiles, not the production selection
functions. For every global row it asks whether ANY feasible candidate has a
lexicographically better objective than the published witness. Null witnesses
are checked for absence of any feasible cell. This is a certificate checker,
not a second implementation of the same sorting-based optimizer.
"""
from pathlib import Path
from collections import Counter
import hashlib,json,math,time
import numpy as np
HERE=Path(__file__).resolve().parent;ROOT=HERE.parent


def closest(lo,hi,closed,baseline):
    if lo<=baseline<=hi and (baseline!=lo or closed):return baseline
    if baseline>hi:return hi
    return lo if closed else math.nextafter(lo,math.inf)


def no_better(mask,arrays,target,counts,label):
    if target is None:
        assert not np.any(mask),('feasible null',label)
        counts['null_row_checks']+=1;return
    equal=mask.copy()
    for array,value in zip(arrays,target):
        if np.isscalar(array):
            assert not(array<value and equal.any()),('better scalar',label,array,value)
            if array!=value:return
        else:
            assert not np.any(equal & (array<value)),('better candidate',label)
            equal &= array==value
            if not equal.any():return
    counts['tie_row_checks']+=1


def main():
    started=time.monotonic();results=json.loads((ROOT/'results.json').read_text());counts=Counter()
    z=np.load(ROOT/'global_atoms.npz');a={k:z[k] for k in z.files};z.close()
    u=a['u_hi'];v=a['v_hi'];U,V=len(u),len(v)
    assert [U,V]==results['global_shape'] and U*V==results['n_global_cells']
    uc=np.array([closest(*row,.8) for row in zip(a['u_lo'],u,a['u_lo_closed'])])
    vc=np.array([closest(*row,.5) for row in zip(a['v_lo'],v,a['v_lo_closed'])])
    ud=((uc-.8)/.2)**2;vd=((vc-.5)/.4)**2;indices=np.arange(V)
    ub=int(np.searchsorted(u,.8));vb=int(np.searchsorted(v,.5))
    profiles=[];um=[];vm=[]
    for gi,group in enumerate(results['groups']):
        z=np.load(ROOT/'profiles'/f'{gi:02d}.npz');p={k:z[k] for k in z.files};z.close()
        profiles.append(p);um.append(np.searchsorted(p['u_hi'],u));vm.append(np.searchsorted(p['v_hi'],v))
    def rank(w,first):
        if w is None:return None
        i,j=w['global_cell'];assert w['u']==uc[i] and w['v']==vc[j]
        return (first,w['all_edits'],float(ud[i]+vd[j]),i,j)
    def at(w,field,gi=None):
        i,j=w['global_cell']
        vals=[p[field][um[g][i],vm[g][j]] for g,p in enumerate(profiles)]
        return vals if gi is None else vals[gi]
    granks={};lranks={};tranks={}
    for ai,an in enumerate(results['analyses']):
        for policy,choices in an['policies'].items():
            for bi,ch in enumerate(choices):
                w=ch['witness']
                if w is not None:
                    assert w['all_edits']==sum(at(w,'n_all_edits'))
                    assert w['all_edits']<=math.floor(78000*ch['budget_fraction'])
                    if policy=='preserve_all':assert sum(at(w,'invalid_eligible_columns'))==0
                granks[ai,policy,bi]=rank(w,-len(w['compatible_groups'])) if w else None
        for gi,row in enumerate(an['local']):
            assert row['group']==results['groups'][gi]
            for mode,choice in row['modes'].items():
                lranks[ai,gi,mode]=None if choice is None else rank(choice['witness'],choice['eligible_edits'])
                if choice:
                    w=choice['witness'];i,j=w['global_cell']
                    assert choice['eligible_edits']==at(w,'n_eligible_edits',gi)
                    si=list(profiles[gi]['slopes']).index(an['p']);ei=list(profiles[gi]['epsilons']).index(an['epsilon'])
                    assert bool(at(w,'compatible',gi)[si,ei])
                    if mode=='preserve_all':assert sum(at(w,'invalid_eligible_columns'))==0
                    if mode=='periodicity_only':assert j==vb
                    if mode=='ratio_only':assert i==ub
            assert row['joint_only']==(row['modes']['joint'] is not None and row['modes']['periodicity_only'] is None and row['modes']['ratio_only'] is None)
    for row in results['minimum_tolerance']:
        gi=results['groups'].index(row['group']);si=list(profiles[gi]['slopes']).index(row['p'])
        w=row['witness'];delta=max(0.,at(w,'gap',gi)[si]/2)
        assert math.isfinite(at(w,'gap',gi)[si]) and delta==row['minimum_epsilon']
        if row['policy']=='preserve_all':assert sum(at(w,'invalid_eligible_columns'))==0
        tranks[gi,si,row['policy']]=rank(w,row['minimum_epsilon'])
    for ui in range(U):
        allcost=np.array([p['n_all_edits'][um[g][ui],vm[g]] for g,p in enumerate(profiles)])
        ecost=np.array([p['n_eligible_edits'][um[g][ui],vm[g]] for g,p in enumerate(profiles)])
        invalid=np.array([p['invalid_eligible_columns'][um[g][ui],vm[g]] for g,p in enumerate(profiles)])
        total=allcost.sum(axis=0);preserve=invalid.sum(axis=0)==0;dist=ud[ui]+vd
        for ai,an in enumerate(results['analyses']):
            si=list(profiles[0]['slopes']).index(an['p']);ei=list(profiles[0]['epsilons']).index(an['epsilon'])
            ok=np.array([p['compatible'][um[g][ui],vm[g],si,ei] for g,p in enumerate(profiles)])
            coverage=ok.sum(axis=0)
            if ui==ub:assert [g for g,b in zip(results['groups'],ok[:,vb]) if b]==an['baseline_compatible_groups']
            for policy,choices in an['policies'].items():
                for bi,ch in enumerate(choices):
                    mask=total<=math.floor(78000*ch['budget_fraction'])
                    if policy=='preserve_all':mask &= preserve
                    no_better(mask,(-coverage,total,dist,ui,indices),granks[ai,policy,bi],counts,('global',ai,policy,bi,ui))
            for gi,row in enumerate(an['local']):
                for mode in row['modes']:
                    mask=ok[gi].copy()
                    if mode=='periodicity_only':mask &= indices==vb
                    elif mode=='ratio_only':mask &= ui==ub
                    elif mode=='preserve_all':mask &= preserve
                    no_better(mask,(ecost[gi],total,dist,ui,indices),lranks[ai,gi,mode],counts,('local',ai,gi,mode,ui))
        for gi,p in enumerate(profiles):
            for si in range(len(p['slopes'])):
                delta=np.maximum(0.,p['gap'][um[gi][ui],vm[gi],si]/2)
                for policy in ('unrestricted','preserve_all'):
                    mask=np.isfinite(delta)
                    if policy=='preserve_all':mask &= preserve
                    no_better(mask,(delta,total,dist,ui,indices),tranks[gi,si,policy],counts,('epsilon',gi,si,policy,ui))
        counts['global_cells']+=V
        if ui%250==0:print('checked row',ui,'/',U,flush=True)
    counts.update(global_budget_optima=len(granks),local_optima=len(lranks),minimum_tolerance_optima=len(tranks))
    out=dict(counts,passed=True,seconds=time.monotonic()-started,source_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),ROOT/'results.json',ROOT/'global_atoms.npz']})
    (HERE/'global_optima.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
if __name__=='__main__':main()
