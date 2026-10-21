"""Independent full empirical profile oracle and selected-threshold replay.

Each column is solved literally at its own <=51 relevant score thresholds.
Global thresholds are independently reconstructed and mapped into those local
tables by binary search. This is not the production dirty-event update sweep.
No production solver, exporter or empirical driver is imported.
"""
from pathlib import Path
import gzip,json,math,hashlib,time
from collections import defaultdict,Counter
import numpy as np
from oracle import first_run,literal_labels,literal_column,literal_folder
HERE=Path(__file__).resolve().parent;ROOT=HERE.parent
FIELDS=('n_all_edits','n_eligible_edits','n_changed_columns','n_all_changed_columns',
        'invalid_eligible_columns','no_run_eligible_columns','left_censored_eligible_columns',
        'nonpositive_or_tied_eligible_columns')


def number(x):return math.inf if x=='Infinity' else -math.inf if x=='-Infinity' else x


def read_columns(raw,feature):
    columns=[]
    for c in raw['columns']:
        f=c['force'];s=first_run(c['labels'],2)
        features=c['features'][feature]
        columns.append(dict(c,eligible=s is not None and s>0 and 0<f[s-1]<f[s],
            fixed=features['fixed'],guard=features['guard'],values=[math.nan if v is None else v for v in features['values']]))
    return columns


def score_cuts(columns,domain):
    lo,hi=domain
    values={v for c in columns for v,g in zip(c['values'],c['guard']) if g and lo<=v<hi}
    return np.array(sorted(values)+[hi],float)


def check_numeric(actual,expected,label,counters):
    assert actual.shape==expected.shape,(label,actual.shape,expected.shape)
    if actual.dtype.kind=='f':
        assert np.allclose(actual,expected,rtol=1e-13,atol=1e-13,equal_nan=True),label
    else:assert np.array_equal(actual,expected),label
    counters['compared_profile_scalars']+=int(actual.size)


def independent_group(columns,thresholds,domain,relation,slopes,epsilons,counters):
    n=len(thresholds);result={key:np.zeros(n,dtype=np.int64) for key in FIELDS}
    lower=np.full((n,len(slopes)),-math.inf);upper=np.full_like(lower,math.inf)
    n_eligible=0
    for c in columns:
        cuts=score_cuts([c],domain)
        mapping=np.searchsorted(cuts,thresholds,side='left')
        assert np.all(mapping<len(cuts))
        costs=[];invalid=[];no_run=[];left=[];tied=[];lows=[];highs=[]
        for threshold in cuts:
            edited=literal_labels(c['fixed'],c['guard'],c['values'],threshold,relation)
            costs.append(sum(a!=b for a,b in zip(edited,c['labels'])))
            if c['eligible']:
                boundaries=[literal_column(c['force'],edited,c['beta'],2,p,0.) for p in slopes]
                b=boundaries[0];bad=not b['valid'];reason=b.get('reason')
                invalid.append(bad);no_run.append(reason=='no_run');left.append(reason=='left_censored');tied.append(reason=='invalid_force_bracket')
                lows.append([bb['lo'] if not bad else -math.inf for bb in boundaries])
                highs.append([bb['hi'] if not bad else math.inf for bb in boundaries])
            counters['local_literal_label_patterns']+=1
            counters['local_literal_label_evaluations']+=len(edited)
        cost=np.array(costs,dtype=np.int64)[mapping]
        result['n_all_edits']+=cost;result['n_all_changed_columns']+=cost>0
        if c['eligible']:
            n_eligible+=1;result['n_eligible_edits']+=cost;result['n_changed_columns']+=cost>0
            for field,values in [('invalid_eligible_columns',invalid),('no_run_eligible_columns',no_run),
              ('left_censored_eligible_columns',left),('nonpositive_or_tied_eligible_columns',tied)]:
                result[field]+=np.array(values,dtype=np.int64)[mapping]
            lower=np.maximum(lower,np.array(lows)[mapping]);upper=np.minimum(upper,np.array(highs)[mapping])
    bad=result['invalid_eligible_columns']>0
    if not n_eligible:bad[:]=True
    lower[bad]=np.nan;upper[bad]=np.nan
    result.update(lower=lower,upper=upper,gap=lower-upper,
                  compatible=lower[...,None]-epsilons<=upper[...,None]+epsilons)
    return result


def replay_threshold(columns,threshold,relation,p,epsilon):
    groups=defaultdict(list)
    for c in columns:groups[c['group']].append(c)
    out={'compatible_groups':[],'invalid_groups':[],'all_edits':0,'eligible_edits':0,'changed_eligible_columns':0,'invalid_eligible_columns':0}
    for group in sorted(groups):
        cols=groups[group];literal=literal_folder(cols,threshold,relation,2,p,epsilon)
        if literal['compatible']:out['compatible_groups'].append(group)
        if not literal['valid']:out['invalid_groups'].append(group)
        out['all_edits']+=literal['cost_all'];out['eligible_edits']+=literal['cost_eligible']
        out['changed_eligible_columns']+=sum(c['eligible'] and any(a!=b for a,b in zip(c['labels'],bits)) for c,bits in zip(cols,literal['labels']))
        out['invalid_eligible_columns']+=sum(not b['valid'] for b in literal['boundaries'])
    out['n_compatible']=len(out['compatible_groups']);out['n_invalid_groups']=len(out['invalid_groups'])
    out['excluded_column_edits']=out['all_edits']-out['eligible_edits']
    return out


def verify_choice(choice,arrays,rule,analysis,columns,counters):
    ai=choice['atom']['index'];p=analysis['p'];epsilon=analysis['epsilon']
    threshold=number(choice['atom']['threshold'])
    lo=number(choice['atom']['lo']);hi=number(choice['atom']['hi'])
    assert (lo<threshold or (lo==threshold and choice['atom']['lo_closed'])) and threshold<=hi
    replay=replay_threshold(columns,threshold,rule['relation'],p,epsilon)
    for key,val in replay.items():assert choice[key]==val,(rule['name'],p,epsilon,key,choice[key],val)
    counters['full_global_threshold_witnesses']+=1
    counters['full_global_witness_window_evaluations']+=sum(len(c['labels']) for c in columns)


def main():
    started=time.monotonic()
    with gzip.open(ROOT/'input_features.json.gz','rt') as fh:raw=json.load(fh)
    results=json.loads((ROOT/'results.json').read_text())
    refs=json.loads((ROOT/'arbitrary_reference.json').read_text())['records']
    ref={(r['session']+'|'+r['folder'],r['p'],r['epsilon'],r['mode']):r for r in refs}
    counts=Counter();rules=[]
    for rule in results['rules']:
        columns=read_columns(raw,rule['feature']);domain=tuple(map(number,rule['domain']))
        groups=sorted({c['group'] for c in columns});assert groups==rule['groups']
        thresholds=score_cuts(columns,domain)
        stored=np.load(ROOT/rule['profile_file']);arrays={key:stored[key] for key in stored.files};stored.close()
        assert len(thresholds)==rule['n_patterns']
        check_numeric(arrays['atom_hi'],thresholds,'independent_global_thresholds',counts)
        check_numeric(arrays['atom_representative'],thresholds,'representatives',counts)
        check_numeric(arrays['atom_lo'],np.r_[domain[0],thresholds[:-1]],'atom_lower',counts)
        expected_lc=np.zeros(len(thresholds),bool);expected_lc[0]=True
        check_numeric(arrays['atom_lo_closed'],expected_lc,'lower_closures',counts)
        check_numeric(arrays['atom_hi_closed'],np.ones(len(thresholds),bool),'upper_closures',counts)
        bygroup=defaultdict(list)
        for c in columns:bygroup[c['group']].append(c)
        for gi,group in enumerate(groups):
            expected=independent_group(bygroup[group],thresholds,domain,rule['relation'],arrays['slopes'],arrays['epsilons'],counts)
            for key,value in expected.items():check_numeric(arrays[key][:,gi],value,key,counts)
        total=arrays['n_all_edits'].sum(axis=1)
        dist=np.maximum(arrays['atom_lo']-rule['base_threshold'],0)+np.maximum(rule['base_threshold']-arrays['atom_hi'],0)
        def best(ii,cost,secondary=None):
            if not len(ii):return None
            if secondary is None:secondary=np.zeros(len(cost),int)
            return int(min(ii,key=lambda i:(int(cost[i]),int(secondary[i]),float(dist[i]),int(i))))
        for analysis in rule['analyses']:
            pi=list(arrays['slopes']).index(analysis['p']);ei=list(arrays['epsilons']).index(analysis['epsilon'])
            ok=arrays['compatible'][:,:,pi,ei];coverage=ok.sum(axis=1)
            optimum=best(np.flatnonzero(coverage==coverage.max()),total)
            assert analysis['global_best']['atom']['index']==optimum
            assert analysis['global_best']['n_compatible']==int(coverage.max())
            assert analysis['all_folders_jointly_feasible']==bool(coverage.max()==len(groups))
            assert analysis['baseline_compatible']==int(ok[rule['baseline_row']].sum())
            assert total[rule['baseline_row']]==0
            verify_choice(analysis['global_best'],arrays,rule,analysis,columns,counts)
            for b in analysis['budget_frontier']:
                ii=np.flatnonzero(total<=math.floor(78000*b['budget_fraction']));m=coverage[ii].max()
                optimum=best(ii[coverage[ii]==m],total)
                assert b['atom']['index']==optimum and b['n_compatible']==int(m)
                # Main witnesses are all replayed directly. Budget choices
                # have already passed every candidate profile and are checked
                # against its globally exact optimum without duplicate replay.
                counts['budget_optima']+=1
            local_feasible=0
            for gi,row in enumerate(analysis['local']):
                assert row['group']==groups[gi]
                ii=np.flatnonzero(ok[:,gi]);optimal=best(ii,arrays['n_eligible_edits'][:,gi],total)
                assert row['feasible']==(optimal is not None)
                expected_regions=[]
                if len(ii):
                    a=prev=int(ii[0])
                    for v in list(ii[1:])+[None]:
                        if v is None or v!=prev+1:
                            expected_regions.append(dict(lo=number(arrays['atom_lo'][a]),hi=number(arrays['atom_hi'][prev]),lo_closed=bool(arrays['atom_lo_closed'][a]),hi_closed=True))
                            if v is not None:a=int(v)
                        if v is not None:prev=int(v)
                actual_regions=[dict(x,lo=number(x['lo']),hi=number(x['hi'])) for x in row['feasible_threshold_regions']]
                assert actual_regions==expected_regions
                if optimal is not None:
                    local_feasible+=1;assert row['atom']['index']==optimal
                    costs={'eligible_edits':int(arrays['n_eligible_edits'][optimal,gi]),'folder_all_edits':int(arrays['n_all_edits'][optimal,gi]),
                        'global_all_edits':int(total[optimal]),'changed_eligible_columns':int(arrays['n_changed_columns'][optimal,gi])}
                    for key,val in costs.items():assert row[key]==val
                    literal=literal_folder(bygroup[row['group']],number(row['atom']['threshold']),rule['relation'],2,analysis['p'],analysis['epsilon'])
                    assert literal['compatible'] and literal['cost_eligible']==row['eligible_edits'] and literal['cost_all']==row['folder_all_edits']
                    bound=ref[row['group'],analysis['p'],analysis['epsilon'],'both']['restore']
                    assert bound['feasible'] and row['arbitrary_minimum']==bound['cost']<=row['eligible_edits']
                    if rule['feature'] in ('amplitude','flyback_height'):
                        delete=ref[row['group'],analysis['p'],analysis['epsilon'],'delete']['restore']
                        assert delete['feasible'] and row['arbitrary_delete_minimum']==delete['cost']<=row['eligible_edits']
                    counts['local_threshold_witnesses']+=1
                counts['local_optima']+=1
            assert local_feasible==analysis['local_feasible']
            counts['global_optima']+=1
        rules.append(dict(name=rule['name'],patterns=len(thresholds),passed=True,
          profile_sha256=hashlib.sha256((ROOT/rule['profile_file']).read_bytes()).hexdigest()))
        print(rule['name'],'passed',len(thresholds),'patterns',flush=True)
    out=dict(passed=True,rules=rules,counts=dict(counts),seconds=time.monotonic()-started,
      source_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),HERE/'oracle.py',ROOT/'input_features.json.gz',ROOT/'results.json',ROOT/'arbitrary_reference.json']})
    (HERE/'real_profiles.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
if __name__=='__main__':main()
