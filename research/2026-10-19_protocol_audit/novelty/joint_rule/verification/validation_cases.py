"""Malformed-input rejection and floating-point endpoint regressions."""
from pathlib import Path
from copy import deepcopy
import hashlib,json,math,sys
import numpy as np
from exhaustive import col,check
HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE.parent))
from solver import analyze_group,threshold_atoms,map_atoms


def main():
    base=col('c',[1,2,4,8],[.1,.85,.85,.95],[.1,.4,.4,.4])
    cases=[]
    def edit(field,value):
        c=deepcopy(base);c[field]=value;cases.append(([c],{}))
    for field,value in [('id',''),('group',''),('beta',0),('beta',math.nan),
      ('force',[2,1,4,8]),('force',[1,2,math.inf,8]),('force',[]),
      ('baseline',[0,1,2,1]),('guard',[0,1]),('fixed',[0,0,.5,0]),
      ('fixed',[0,1,0,0]),('periodicity',[.1,math.nan,.85,.95]),
      ('ratio',[.1,math.inf,.4,.4]),('ratio',[.1,.4]),('eligible',1),
      ('baseline',[0,0,0,0])]:edit(field,value)
    cases.extend([([],{}),([base,deepcopy(base)],{}),([base,dict(base,id='other',group='x')],{})])
    for kwargs in [dict(k=0),dict(k=True),dict(k=1.2),dict(slopes=[]),dict(slopes=[math.nan]),
      dict(epsilons=[-.1]),dict(epsilons=[]),dict(u_domain=(.9,.7)),dict(v_domain=(.3,math.inf)),
      dict(u_domain=(.7,.8,.9)),dict(v_domain=(math.nan,.7))]:cases.append(([base],kwargs))
    for columns,kwargs in cases:
        try:analyze_group(columns,**kwargs)
        except (ValueError,TypeError):pass
        else:raise AssertionError(('did not reject',columns,kwargs))
    for scores,domain in [([math.nan],(.7,.9)),([math.inf],(.7,.9)),([.8],(.9,.7))]:
        try:threshold_atoms(scores,domain)
        except ValueError:pass
        else:raise AssertionError(('atoms did not reject',scores,domain))
    atoms=threshold_atoms([.7,.8,.8,.9],(.7,.9))
    assert atoms['lo'].tolist()==[.7,.7,.8]
    assert atoms['hi'].tolist()==[.7,.8,.9]
    assert map_atoms(atoms,[.7,math.nextafter(.7,math.inf),.8,math.nextafter(.8,math.inf),.9]).tolist()==[0,1,1,2,2]
    for values in [[.69],[.91],[math.nan]]:
        try:map_atoms(atoms,values)
        except ValueError:pass
        else:raise AssertionError(('mapping did not reject',values))
    # Adjacent representable values, endpoint singleton and empty guards.
    counts=dict(fixtures=0,threshold_pairs=0,cells=0,scalar_comparisons=0,minima_checks=0,certificate_checks=0)
    for a,b in [(.8,math.nextafter(.8,math.inf)),(.7,.7),(.9,.9)]:
        c=deepcopy(base);c['periodicity']=[a,b,.85,.95]
        check(analyze_group,'adjacent', [c],2,(.7,.9),(.3,.7),counts)
    c=deepcopy(base);c['fixed']=c['baseline'];c['guard']=[0]*4
    check(analyze_group,'no_guard',[c],2,(.7,.9),(.3,.7),counts)
    for dtype in (np.uint64,np.int64):
        got=analyze_group([base],k=dtype(2));want=analyze_group([base],k=2)
        assert np.array_equal(got['compatible'],want['compatible'])
    out=dict(integer_type_regressions=2,passed=True,invalid_group_inputs=len(cases),invalid_atoms=3,invalid_mappings=3,regressions=counts,
      source_sha256={str(p.relative_to(HERE.parent)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),HERE.parent/'solver.py']})
    (HERE/'validation_cases.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
if __name__=='__main__':main()
