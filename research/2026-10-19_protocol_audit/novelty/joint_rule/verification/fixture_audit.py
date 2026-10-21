"""Independent baseline and old-slice alignment of joint compact input.

No exporter or classifier imports. All old scalar slice feature events are
replayed literally, including their exact endpoints and open-cell values.
"""
from pathlib import Path
from collections import Counter
import gzip,json,hashlib,math
from oracle import labels,eligible,independent_probes
HERE=Path(__file__).resolve().parent;ROOT=HERE.parent


def main():
    prior=ROOT/'prior_input_features.json.gz'
    if not prior.exists():prior=ROOT.parent/'coherent_fragility/input_features.json.gz'
    old=json.load(gzip.open(prior,'rt'));new=json.load(gzip.open(ROOT/'input_joint.json.gz','rt'))
    oldmap={c['id']:c for c in old['columns']};newmap={c['id']:c for c in new['columns']}
    assert len(oldmap)==len(old['columns'])==len(newmap)==len(new['columns'])
    assert oldmap.keys()==newmap.keys()
    counts=Counter();window_ids=set()
    for identity,c0 in newmap.items():
        c=dict(c0,periodicity=[math.nan if v is None else v for v in c0['periodicity']],ratio=[math.nan if v is None else v for v in c0['ratio']])
        before=oldmap[identity]
        for field in ('force','trial','beta','speed','session','folder','column','id','group'):
            assert c[field]==before[field],(identity,field)
        assert c['baseline']==before['labels']
        assert c['baseline']==labels(c,.8,.5)
        assert c['eligible']==eligible(c)
        counts['columns']+=1;counts['windows']+=len(c['baseline'])
        counts['eligible_columns']+=c['eligible'];counts['eligible_windows']+=c['eligible']*len(c['baseline'])
        for trial in c['trial']:
            wid=(c['session'],c['folder'],trial);assert wid not in window_ids;window_ids.add(wid)
        for feature,axis,domain in [('periodicity','u',(.7,.9)),('second_flyback_ratio','v',(.3,.7))]:
            f=before['features'][feature]
            vals=[math.nan if x is None else x for x in f['values']]
            for t in independent_probes(vals,domain):
                oldbits=[int(bool(ff) or (bool(gg) and math.isfinite(x) and (x>=t if axis=='u' else x<t))) for ff,gg,x in zip(f['fixed'],f['guard'],vals)]
                newbits=labels(c,t,.5) if axis=='u' else labels(c,.8,t)
                assert oldbits==newbits,(identity,feature,t)
                counts['slice_patterns']+=1;counts['slice_window_comparisons']+=len(newbits)
        for i,(g,f,p,r) in enumerate(zip(c['guard'],c['fixed'],c['periodicity'],c['ratio'])):
            assert not(g and f)
            assert not g or (math.isfinite(p) and math.isfinite(r))
            counts['joint_only_admissible_windows']+=bool(g and .7<=p<.8 and .5<=r<.7)
            if not(math.isfinite(p) and math.isfinite(r)):
                assert not g and f==c['baseline'][i]
                counts['missing_locked_windows']+=1
    out=dict(counts,unique_window_ids=len(window_ids),groups=len({c['group'] for c in new['columns']}),passed=True,
      source_sha256={str(p.relative_to(ROOT.parent)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [prior,ROOT/'input_joint.json.gz',Path(__file__),HERE/'oracle.py']})
    (HERE/'fixture_audit.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
if __name__=='__main__':main()
