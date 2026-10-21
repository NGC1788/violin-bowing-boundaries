"""Literal continuous-tolerance witnesses and machine-threshold endpoint audit.

The closed-form epsilon is a real-arithmetic geometric diagnostic evaluated
in float64. A shifted float64 comparison can switch a few ULPs away. For each
reported witness this checker independently obtains its actual L/U endpoints,
then finds the first nonnegative representable binary64 epsilon for which
L-epsilon <= U+epsilon is true. It does not redefine the scientific tolerance.
"""
from pathlib import Path
from collections import Counter,defaultdict
import gzip,hashlib,json,math,struct,time
from oracle import evaluate
HERE=Path(__file__).resolve().parent;ROOT=HERE.parent


def asint(value):return struct.unpack('>Q',struct.pack('>d',float(value)))[0]
def asfloat(value):return struct.unpack('>d',struct.pack('>Q',int(value)))[0]


def first_machine_epsilon(low,high):
    def accepts(e):return low-e<=high+e
    if accepts(0.):return 0.,None,0
    estimate=max(0.,(low-high)/2)
    right=max(1.,estimate+1.)
    assert accepts(right)
    left_i,right_i=asint(0.),asint(right);checks=0
    # Positive binary64 bit patterns have the same order as represented reals.
    # Monotonicity of each rounded subtraction/addition makes the predicate
    # monotone, so integer bisection certifies the first machine threshold.
    while right_i-left_i>1:
        mid=(left_i+right_i)//2;checks+=1
        if accepts(asfloat(mid)):right_i=mid
        else:left_i=mid
    first=asfloat(right_i);previous=math.nextafter(first,-math.inf)
    assert accepts(first) and not accepts(previous)
    assert previous==asfloat(left_i)
    return first,previous,checks


def main():
    start=time.monotonic();raw=json.load(gzip.open(ROOT/'input_joint.json.gz','rt'))
    groups=defaultdict(list)
    for c in raw['columns']:
        groups[c['group']].append(dict(c,periodicity=[math.nan if x is None else x for x in c['periodicity']],ratio=[math.nan if x is None else x for x in c['ratio']]))
    results=json.loads((ROOT/'results.json').read_text());counts=Counter();rows=[]
    for row in results['minimum_tolerance']:
        w=row['witness'];p=row['p'];eps=row['minimum_epsilon'];g=row['group']
        r=evaluate(groups[g],w['u'],w['v'],2,[p],[eps])
        assert r['invalid_eligible_columns']==0
        low,high=r['lower'][0],r['upper'][0];closed=max(0.,(low-high)/2)
        assert math.isclose(closed,eps,rel_tol=1e-13,abs_tol=1e-13)
        first,previous,nchecks=first_machine_epsilon(low,high)
        baseline=evaluate(groups[g],.8,.5,2,[p],[0.])
        baseline_required=max(0.,baseline['gap'][0]/2)
        rows.append(dict(group=g,p=p,policy=row['policy'],u=w['u'],v=w['v'],lower=low,upper=high,
          reported_closed_form_epsilon=eps,literal_closed_form_epsilon=closed,
          closed_form_passes_float_predicate=bool(low-eps<=high+eps),
          minimum_representable_epsilon=first,previous_representable_epsilon=previous,
          previous_is_incompatible=None if previous is None else bool(low-previous>high+previous),
          minimum_is_compatible=bool(low-first<=high+first),
          machine_minus_literal_closed_form=first-closed,baseline_epsilon_required=baseline_required))
        counts['tolerance_witnesses']+=1;counts['binary64_bisection_checks']+=nchecks
        counts['closed_form_false_in_float']+=not bool(low-eps<=high+eps)
    candidates=[r for r in rows if r['p']==-2. and r['policy']=='preserve_all']
    chosen=min(candidates,key=lambda r:r['reported_closed_form_epsilon'])
    assert chosen['group']=='A4|2024-04-24_TypeA_sample4/v3'
    assert chosen['u']==.8 and chosen['v']==.49326392706532574
    checks=[];all_edits=all_invalid=0
    for group,cs in sorted(groups.items()):
        r=evaluate(cs,chosen['u'],chosen['v'],2,[-2.],[.205552,.205553])
        all_edits+=r['n_all_edits'];all_invalid+=r['invalid_eligible_columns']
        checks.append(dict(group=group,invalid_columns=r['invalid_eligible_columns'],all_edits=r['n_all_edits'],
           compatible_at_0_205552=r['compatible'][0][0],compatible_at_0_205553=r['compatible'][0][1]))
    target=next(r for r in checks if r['group']==chosen['group'])
    assert not target['compatible_at_0_205552'] and target['compatible_at_0_205553']
    assert all_invalid==0 and all_edits==150 and target['all_edits']==1
    # Global geometric optimum was checked independently by global_optima.py;
    # here we check its concrete witness and distinguish mathematical rounding.
    out=dict(counts,passed=True,seconds=time.monotonic()-start,
      primary_no_loss_beta_minus2=dict(chosen,all_data_edits=all_edits,target_folder_edits=target['all_edits'],
         all_invalid_eligible_columns=all_invalid,epsilon_below=.205552,epsilon_above=.205553,
         target_compatible_below=False,target_compatible_above=True,
         groups=checks),witnesses=rows,
      interpretation='Epsilon is a log-force compatibility tolerance, not calibrated measurement uncertainty. Float endpoint tests do not turn post-hoc classifier sensitivity into physical confirmation.',
      source_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),HERE/'oracle.py',ROOT/'input_joint.json.gz',ROOT/'results.json']})
    (HERE/'tolerance_cutoff_checks.json').write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps({k:v for k,v in out.items() if k not in ('witnesses','primary_no_loss_beta_minus2')},indent=2))
    print(json.dumps({k:v for k,v in out['primary_no_loss_beta_minus2'].items() if k!='groups'},indent=2))
if __name__=='__main__':main()
