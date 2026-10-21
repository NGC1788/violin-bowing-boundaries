"""API rejection cases and endpoint/guard regression checks."""
from pathlib import Path
import sys,copy,json,math,hashlib
import numpy as np
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent))
from threshold_profile import analyze_thresholds
from exhaustive import check_case

BASE=dict(id='a',group='g',beta=.1,force=[.1,.2,.3,.4],baseline=[0,1,1,0],
    guard=[1,1,1,1],fixed=[0,0,0,0],values=[.25,.5,.5,.75],eligible=True)

def main():
    cases=[]
    def add(name,**kwargs):cases.append((name,kwargs))
    for name,kwargs in [('relation',{'relation':'le'}),('Kbool',{'k':True}),('Kfloat',{'k':2.0}),('Kzero',{'k':0}),
        ('Knegative',{'k':-1}),('empty_slope',{'slopes':[]}),('nan_slope',{'slopes':[math.nan]}),
        ('empty_epsilon',{'epsilons':[]}),('negative_epsilon',{'epsilons':[-.1]}),('nan_epsilon',{'epsilons':[math.nan]}),
        ('reverse_domain',{'domain':(1,0)}),('nan_domain',{'domain':(0,math.nan)}),('long_domain',{'domain':(0,1,2)})]:add(name,**kwargs)
    changes=[('empty_id','id',''),('numeric_id','id',3),('empty_group','group',''),('beta_zero','beta',0.),
      ('beta_nan','beta',math.nan),('unsorted_force','force',[.2,.1,.3,.4]),('force_inf','force',[.1,.2,.3,math.inf]),
      ('force_length','force',[.1,.2,.3]),('bad_baseline','baseline',[0,2,1,0]),('bad_guard','guard',[0,1,.5,1]),
      ('bad_fixed','fixed',[0,0,0,math.nan]),('values_inf','values',[.25,.5,.5,math.inf]),('active_nan','values',[.25,math.nan,.5,.75]),
      ('overlap','fixed',[0,1,0,0]),('eligible_int','eligible',1),('invalid_eligible','baseline',[1,1,0,0])]
    for name,key,val in changes:
        c=copy.deepcopy(BASE);c[key]=val;add(name,columns=[c])
    add('duplicate_id',columns=[copy.deepcopy(BASE),copy.deepcopy(BASE)])
    results=[]
    for name,kw in cases:
        opts=dict(columns=[copy.deepcopy(BASE)],relation='ge',domain=(0.,1.));opts.update(kw)
        try:analyze_thresholds(**opts)
        except (ValueError,TypeError) as e:results.append(dict(name=name,rejected=True,error_type=type(e).__name__))
        else:raise AssertionError(('not rejected',name))
    counts=dict(fixtures=0,threshold_probes=0,atoms=0,profile_comparisons=0,minima_checks=0)
    # Simultaneous equal-score edits must never expose an intermediate label.
    tie=copy.deepcopy(BASE);tie['values']=[.5]*4
    for relation in ('ge','lt'):
        for domain in ((0.,1.),(.5,.5),(.5,1.),(0.,.5),(-math.inf,math.inf)):
            check_case(analyze_thresholds,[tie],relation,domain,2,counts)
    # A group with no eligible columns cannot support a law vacuously.
    no_eligible=copy.deepcopy(BASE);no_eligible['eligible']=False
    check_case(analyze_thresholds,[no_eligible],'ge',(0.,1.),2,counts)
    no_guards=copy.deepcopy(BASE);no_guards['guard']=[0]*4;no_guards['fixed']=no_guards['baseline']
    no_guards['values']=[math.nan]*4
    check_case(analyze_thresholds,[no_guards],'ge',(0.,math.inf),2,counts)
    empty=analyze_thresholds([], 'ge',(0.,1.))
    assert empty['groups']==[] and empty['compatible'].shape[1]==0
    out=dict(passed=True,rejection_cases=results,n_rejection_cases=len(results),regression_counts=counts,
        empty_input='Returns zero groups; no compatibility claims',
        source_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),HERE.parent/'threshold_profile.py']})
    (HERE/'validation_cases.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
if __name__=='__main__':main()
