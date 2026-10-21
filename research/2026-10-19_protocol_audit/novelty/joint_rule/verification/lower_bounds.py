"""Compare joint-rule witnesses against previously certified arbitrary edits.

Costs are compared on the SAME original eligible columns within the SAME
folder. Global collateral edits are separately decomposed and never divided
by a target-folder lower bound. Directional bounds apply only when the actual
eligible changes have that direction (zero changes satisfy both).
"""
from pathlib import Path
from collections import Counter,defaultdict
import gzip,hashlib,json,math,time
from oracle import labels,evaluate
HERE=Path(__file__).resolve().parent;ROOT=HERE.parent


def main():
    started=time.monotonic();source=ROOT/'prior_arbitrary_reference.json'
    refs=json.loads(source.read_text())['records']
    lookup={(r['session']+'|'+r['folder'],r['p'],r['epsilon'],r['mode']):r for r in refs}
    assert len(lookup)==len(refs)
    raw=json.load(gzip.open(ROOT/'input_joint.json.gz','rt'))
    columns=[dict(c,periodicity=[math.nan if x is None else x for x in c['periodicity']],ratio=[math.nan if x is None else x for x in c['ratio']]) for c in raw['columns']]
    groups=defaultdict(list)
    for c in columns:groups[c['group']].append(c)
    results=json.loads((ROOT/'results.json').read_text());cache={};counts=Counter();records=[];global_records=[]
    def decompose(w):
        key=(w['u'],w['v'])
        if key in cache:return cache[key]
        whole=Counter();bygroup={}
        for group,cs in groups.items():
            cnt=Counter()
            for c in cs:
                changed=labels(c,*key)
                add=sum(not a and b for a,b in zip(c['baseline'],changed))
                delete=sum(a and not b for a,b in zip(c['baseline'],changed))
                cnt['all_additions']+=add;cnt['all_deletions']+=delete
                if c['eligible']:
                    cnt['eligible_additions']+=add;cnt['eligible_deletions']+=delete
                counts['direct_window_classifications']+=len(changed)
            cnt['all_edits']=cnt['all_additions']+cnt['all_deletions']
            cnt['eligible_edits']=cnt['eligible_additions']+cnt['eligible_deletions']
            bygroup[group]=dict(cnt);whole.update(cnt)
        cache[key]=(dict(whole),bygroup)
        return cache[key]
    def check_bound(group,p,eps,changes):
        applicable=['both']
        if changes['eligible_additions']==0:applicable.append('delete')
        if changes['eligible_deletions']==0:applicable.append('add')
        costs={}
        for direction in applicable:
            ref=lookup[group,p,eps,direction]
            cs=groups[group];ec=[c for c in cs if c['eligible']]
            assert ref['k']==2 and ref['n_columns']==len(ec)
            assert ref['n_windows']==sum(len(c['baseline']) for c in ec)
            bound=ref['restore'];assert bound['feasible'],(group,p,eps,direction,'contradicts certified impossibility')
            assert bound['cost'] is not None and changes['eligible_edits']>=bound['cost'],(group,p,eps,direction,changes,bound)
            costs[direction]=bound['cost'];counts[direction+'_bound_checks']+=1
        return costs
    for an in results['analyses']:
        p,eps=an['p'],an['epsilon']
        for row in an['local']:
            group=row['group']
            for mode,choice in row['modes'].items():
                if choice is None:continue
                w=choice['witness'];whole,parts=decompose(w);part=parts[group]
                assert part['eligible_edits']==choice['eligible_edits']
                assert whole['all_edits']==w['all_edits'] and whole['eligible_edits']==w['eligible_edits']
                literal=evaluate(groups[group],w['u'],w['v'],2,[p],[eps]);assert literal['compatible'][0][0]
                bounds=check_bound(group,p,eps,part)
                records.append(dict(group=group,p=p,epsilon=eps,mode=mode,u=w['u'],v=w['v'],target=part,whole_dataset=whole,applicable_arbitrary_bounds=bounds))
                counts['feasible_local_mode_witnesses']+=1
        for policy,choices in an['policies'].items():
            for choice in choices:
                w=choice['witness']
                if w is None:continue
                whole,parts=decompose(w)
                assert whole['all_edits']==w['all_edits'] and whole['eligible_edits']==w['eligible_edits']
                applicable={g:check_bound(g,p,eps,parts[g]) for g in w['compatible_groups']}
                global_records.append(dict(p=p,epsilon=eps,policy=policy,budget_fraction=choice['budget_fraction'],u=w['u'],v=w['v'],whole_dataset=whole,compatible_groups={g:dict(changes=parts[g],applicable_arbitrary_bounds=applicable[g]) for g in w['compatible_groups']}))
                counts['global_budget_witnesses']+=1
    out=dict(counts,distinct_classifier_pairs=len(cache),passed=True,seconds=time.monotonic()-started,
      note='Previously certified arbitrary restore costs are a reference, not re-derived here. Only identical eligible-folder costs are compared. One-sided bounds require matching observed direction.',
      local_witnesses=records,global_witnesses=global_records,
      source_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),HERE/'oracle.py',source,ROOT/'input_joint.json.gz',ROOT/'results.json']})
    (HERE/'lower_bounds.json').write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps({k:v for k,v in out.items() if k not in ('local_witnesses','global_witnesses')},indent=2))
if __name__=='__main__':main()
