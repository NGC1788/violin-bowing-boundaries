"""Independent compact-fixture alignment and baseline-rule reconstruction."""
from pathlib import Path
import gzip,json,hashlib,math
from collections import Counter
from oracle import literal_labels,first_run
HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
OLD=ROOT/'prior_input_columns.json.gz'
if not OLD.exists():OLD=ROOT.parent/'run_fragility/input_columns.json.gz'


def main():
    with gzip.open(OLD,'rt') as fh:old=json.load(fh)
    with gzip.open(ROOT/'input_features.json.gz','rt') as fh:new=json.load(fh)
    key=lambda c:(c['session'],c['folder'],c['column'])
    old_map={key(c):c for c in old['columns']};new_map={key(c):c for c in new['columns']}
    assert len(old_map)==len(old['columns'])==len(new_map)==len(new['columns'])
    assert old_map.keys()==new_map.keys()
    counts=Counter();mismatch=Counter();groups=set();window_ids=set()
    per_feature={name:Counter() for name in new['features']}
    for identity,c in new_map.items():
        before=old_map[identity]
        for field in ('force','labels','trial','beta','speed','session','folder','column'):
            assert c[field]==before[field],(identity,field)
        assert c['id']==f'{c["session"]}|{c["folder"]}|{c["column"]}'
        assert c['group']==f'{c["session"]}|{c["folder"]}'
        groups.add(c['group']);counts['columns']+=1;counts['windows']+=len(c['labels'])
        assert all(math.isfinite(x) for x in c['force'])
        assert all(a<b for a,b in zip(c['force'],c['force'][1:]))
        for trial in c['trial']:
            wid=(c['session'],c['folder'],trial)
            assert wid not in window_ids;window_ids.add(wid)
        s=first_run(c['labels'],2)
        eligible=s is not None and s>0 and 0<c['force'][s-1]<c['force'][s]
        counts['eligible_columns']+=eligible
        counts['eligible_windows']+=len(c['labels'])*eligible
        for name,features in c['features'].items():
            assert len(features['values'])==len(features['guard'])==len(features['fixed'])==len(c['labels'])
            values=[math.nan if x is None else x for x in features['values']]
            assert all(not(a and b) for a,b in zip(features['guard'],features['fixed']))
            assert all(not g or math.isfinite(v) for g,v in zip(features['guard'],values))
            threshold={'periodicity':.8,'second_flyback_ratio':.5,'amplitude':0.,'flyback_height':0.}[name]
            direction='lt' if name=='second_flyback_ratio' else 'ge'
            actual=literal_labels(features['fixed'],features['guard'],values,threshold,direction)
            mismatch[name]+=sum(a!=b for a,b in zip(actual,c['labels']))
            per_feature[name]['missing']+=sum(not math.isfinite(x) for x in values)
            per_feature[name]['guarded']+=sum(features['guard'])
            per_feature[name]['fixed_positive']+=sum(features['fixed'])
            per_feature[name]['negative']+=sum(math.isfinite(x) and x<0 for x in values)
    assert not any(mismatch.values())
    out=dict(counts,groups=len(groups),unique_window_ids=len(window_ids),baseline_mismatches=dict(mismatch),
        features={k:dict(v) for k,v in per_feature.items()},passed=True,
        source_sha256={str(p.relative_to(ROOT.parent)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [OLD,ROOT/'input_features.json.gz',Path(__file__),HERE/'oracle.py']})
    (HERE/'fixture_audit.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
if __name__=='__main__':main()
