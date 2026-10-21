"""Predeclared, retrospective replay of two existing hard-rule witnesses in full MLE."""
from pathlib import Path
import gzip, hashlib, json, sys
from collections import Counter
import numpy as np
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE/'production'))
import model
RULES={'baseline':(.8,.5),'hard_restore_with_loss':(.8,.33218555281080747),'near_cutoff_no_loss':(.8,.49326392706532574)}

def classify(c,u,v):
    return [bool(f or (g and p is not None and r is not None and p>=u and r<v))
            for f,g,p,r in zip(c['fixed'],c['guard'],c['periodicity'],c['ratio'])]

def census(rows):
    counts=Counter('left' if np.isneginf(r['lower'][0]) else 'right' if np.isposinf(r['lower'][1]) else 'finite' for r in rows)
    return dict(counts)|{'n':len(rows)}

def main():
    src=HERE/'production/input_joint.json.gz'
    if not src.exists():src=HERE.parent/'joint_rule/input_joint.json.gz'
    data=json.loads(gzip.decompress(src.read_bytes()));cols=data['columns']
    sessions=sorted({c['session'] for c in cols})
    out={'post_hoc':True,'no_threshold_optimization':True,'input_sha256':hashlib.sha256(src.read_bytes()).hexdigest(),'rules':{}}
    bases={}
    for name,(u,v) in RULES.items():
        entry={'thresholds':[u,v],'sessions':{},'global_label_changes':0,'original_eligible_boundary_losses':0,'lost_boundary_groups':[]}
        lost_groups=set()
        for s in sessions:
            cc=[c for c in cols if c['session']==s]
            labels=[classify(c,u,v) for c in cc]
            if name=='baseline':assert all(h==[bool(x) for x in c['baseline']] for c,h in zip(cc,labels))
            # model labels accepts a map keyed by exact IDs (or read its API below).
            rr=model.extract_rows(cc,dict(zip([c['id'] for c in cc],labels)))
            free=model.fit(rr)
            assert free['success'] and free['grad_max']<2e-3 and -9+1e-5<free['params'][3]<3.5-1e-5
            fixed={str(p):model.fit(rr,fixed_beta=p,init=free['params']) for p in (-2.,-1.2)}
            for f in fixed.values():assert f['success'] and f['grad_max']<2e-3 and f['nll']>=free['nll']-1e-7
            changed=sum(sum(a!=b for a,b in zip(c['baseline'],h)) for c,h in zip(cc,labels))
            lost=0
            for c,h in zip(cc,labels):
                if not c['eligible']:continue
                starts=[i for i in range(len(h)-1) if h[i] and h[i+1]]
                valid=bool(starts) and starts[0]>0 and 0<c['force'][starts[0]-1]<c['force'][starts[0]]
                if not valid:lost+=1;lost_groups.add(c['group'])
            if name=='baseline':bases[s]=(free,rr)
            bf,br=bases[s]
            # Order is fixed by extract_rows for the same cc list.
            byid={r['id']:r for r in br}
            interval_changes=sum(tuple(r['lower'])!=tuple(byid[r['id']]['lower']) for r in rr)
            rec={'fit':free,'fixed':fixed,'T':{p:2*(f['nll']-free['nll']) for p,f in fixed.items()},
                 'delta_beta':free['params'][2]-bf['params'][2],'delta_speed':free['params'][1]-bf['params'][1],
                 'counts':census(rr),'label_changes':changed,'interval_changes':interval_changes,'original_eligible_boundary_losses':lost}
            entry['sessions'][s]=rec;entry['global_label_changes']+=changed;entry['original_eligible_boundary_losses']+=lost
        entry['lost_boundary_groups']=sorted(lost_groups)
        out['rules'][name]=entry
    assert out['rules']['hard_restore_with_loss']['global_label_changes']==3200
    assert len(out['rules']['hard_restore_with_loss']['lost_boundary_groups'])==9
    assert out['rules']['near_cutoff_no_loss']['global_label_changes']==150
    assert out['rules']['near_cutoff_no_loss']['original_eligible_boundary_losses']==0
    out['max_abs_beta_change']={k:max(abs(r['delta_beta']) for r in v['sessions'].values()) for k,v in out['rules'].items()}
    (HERE/'coherent_replay.json').write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps({k:{'edits':v['global_label_changes'],'losses':v['original_eligible_boundary_losses'],'max_beta_delta':out['max_abs_beta_change'][k]} for k,v in out['rules'].items()},indent=2))
if __name__=='__main__':main()
