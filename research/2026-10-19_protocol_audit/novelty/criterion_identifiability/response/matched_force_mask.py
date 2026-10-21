"""Post-hoc fair-mask audit, preserving original fixed-plan results."""
import copy,hashlib,json
import numpy as np
import replay as R


def main():
    old=json.loads((R.HERE/'results.json').read_text());out=copy.deepcopy(old)
    rng=np.random.default_rng(R.SEED);record={'post_hoc':True,'sessions':{},'failures':[],'n_fits':0}
    for s,e in old['sessions'].items():
        st=R.LS.STEM[s];stem,rep=st if isinstance(st,tuple) else (st,None)
        raw,fh=R.T.read2(stem,rep);d=R.LS.build(raw,R.T.BASE);cache=R.BoundaryCache(d)
        X={'amplitude':np.array([r[3] for r in raw[0]]),'flyback_height':fh,
           'periodicity':np.array([r[4] for r in raw[0]]),'second_flyback_ratio':np.array([r[7] for r in raw[0]])}
        if s=='B1':
            rec={(r['folder'],int(r['trial'])):R.LS.fnum(r['flyback_ms']) for r in R.csv.DictReader((R.ROOT/'novelty/label_sensitivity/b1_flyback/b1_flyback.csv').open())}
            X['B1_corner']=np.array([rec.get(r[0],np.nan) for r in raw[0]])
        for feature,x in X.items():
            zz=rng.standard_normal((R.N_REPLAY,len(x)))
            if s not in ('A2','B1','B_T2s1'):continue
            force_ok=np.isfinite(d['force']) & (d['force']>0)
            mu,mask,reg=R.regression(dict(d,H=d['H'] & force_ok),x,'M0')
            _,m1mask,_=R.regression(d,x,'M1');assert np.array_equal(mask,m1mask)
            aa=R.GRIDS[feature];curves=[]
            for j in range(R.N_REPLAY):
                sim=mu+reg['sigma']*zz[j];vals=[]
                for a in aa:
                    h,_=R.trim_log(d,sim,mask,a,feature in R.UPPER)
                    f=R.safe_fit(cache.rows(h),e['baseline']['params']);record['n_fits']+=1
                    if not f['success']:record['failures'].append({'s':s,'feature':feature,'j':j,'a':a,**f});vals.append(np.nan)
                    else:vals.append(f['params'][2]-e['baseline']['params'][2])
                curves.append(vals)
            curves=np.array(curves);mean=np.mean(curves,0).tolist()
            record['sessions'].setdefault(s,{})[feature]={'mean_delta':mean,'delta_replays':curves.tolist(),'regression':reg}
            out['sessions'][s]['features'][feature]['models']['M0']['mean_delta']=mean
            print(s,feature,flush=True)
    record['stored_target_summary']=R.aggregate(out)
    for e in out['sessions'].values():
        for f in e['features'].values():f['observed']=f['observed_refit']
    record['stable_target_summary']=R.aggregate(out)
    record['sha256']={str(p.relative_to(R.ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [R.Path(__file__),R.HERE/'PLAN_matched_force_mask.md',R.HERE/'results.json']}
    (R.HERE/'matched_force_mask.json').write_text(json.dumps(record,indent=2))
    print({k:{'models':record[k]['models'],'reduction':record[k]['mae_reduction'],'gate':record[k]['fixed_gate_pass']} for k in ('stored_target_summary','stable_target_summary')})


if __name__=='__main__':main()
