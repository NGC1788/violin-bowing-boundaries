"""Post-hoc independent likelihood + paired beta-block audit. No frozen rescoring."""
from pathlib import Path
import json, sys, hashlib
import numpy as np
from scipy.special import log_ndtr
from scipy.optimize import minimize
from scipy.stats import t as td
HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
sys.path.insert(0,str(ROOT/'novelty/lower_law'))
import onset_axis as OA
LL=OA.LL


def logprob(a,b):
    """Stable normal probability of (a,b], including left/right tails."""
    a,b=np.asarray(a,float),np.asarray(b,float)
    z=np.empty_like(a)
    left=np.isneginf(a); right=np.isposinf(b)
    z[left]=log_ndtr(b[left]); z[right]=log_ndtr(-a[right])
    finite=~(left|right)
    pos=finite&(a>0); neg=finite&~pos
    for m,u,v in ((neg,b,a),(pos,-a,-b)):
        top=log_ndtr(u[m]); bottom=log_ndtr(v[m])
        z[m]=top+np.log(-np.expm1(bottom-top))
    z[left&right]=0.
    return z


def objective(p,X,lo,hi):
    mu=X@p[:-1]; sig=np.exp(p[-1]); a=(lo-mu)/sig; b=(hi-mu)/sig
    lp=logprob(a,b)
    if not np.isfinite(lp).all(): return 1e100,np.zeros_like(p)
    ra=np.exp(-.5*a*a-.5*np.log(2*np.pi)-lp)
    rb=np.exp(-.5*b*b-.5*np.log(2*np.pi)-lp)
    dm=(rb-ra)/sig
    aa=np.where(np.isfinite(a),a,0); bb=np.where(np.isfinite(b),b,0)
    grad=np.r_[X.T@dm,np.sum(bb*rb-aa*ra)]
    return -lp.sum(),grad


def arrays(rows):
    X=np.array([[1.,r['lv'],r['lb']] for r in rows]); lo,hi=np.array([r['lower'] for r in rows]).T
    assert np.all(lo<hi), 'zero/reversed-width intervals require explicit sensitivity treatment'
    return X,lo,hi


def fit(rows,init=None):
    X,lo,hi=arrays(rows)
    if len(rows)<6 or np.linalg.matrix_rank(X)<3: raise ValueError('insufficient design')
    center=X[:,1:].mean(0); scale=X[:,1:].std(0)
    XX=np.c_[np.ones(len(X)),(X[:,1:]-center)/scale]
    fin=np.isfinite(lo)&np.isfinite(hi)
    if init is None:
        yy=np.where(fin,(np.where(np.isfinite(lo),lo,0)+np.where(np.isfinite(hi),hi,0))/2,
                    np.where(np.isfinite(hi),hi-.2,lo+.2))
        q=np.linalg.lstsq(XX,yy,rcond=None)[0]
        init_std=np.r_[q,np.log(max(.05,np.std(yy-XX@q)))]
    else:
        init=np.asarray(init); init_std=np.r_[init[0]+center@init[1:3],init[1:3]*scale,init[3]]
    opt=minimize(objective,init_std,args=(XX,lo,hi),jac=True,method='L-BFGS-B',
        bounds=[(None,None)]*3+[(-9.,3.5)],options={'maxiter':600,'ftol':1e-13,'gtol':1e-7})
    fallback=False
    if not opt.success or np.max(np.abs(opt.jac))>2e-3:
        alt=minimize(lambda p:objective(p,XX,lo,hi)[0],opt.x,method='Nelder-Mead',
                     options={'maxiter':3000,'xatol':1e-9,'fatol':1e-10})
        if alt.fun<=opt.fun+1e-8: opt=alt; fallback=True
    q=opt.x; p=np.r_[q[0]-center@(q[1:3]/scale),q[1:3]/scale,q[3]]
    return {'params':p.tolist(),'nll':float(opt.fun),'success':bool(opt.success),
            'grad_max':float(np.max(np.abs(objective(q,XX,lo,hi)[1]))),
            'sigma':float(np.exp(p[-1])),'fallback':fallback,'n':len(rows)}


def meta(y,se):
    y,se=np.asarray(y),np.asarray(se)
    w=1/se**2; k=len(y); Q=np.sum(w*(y-np.sum(w*y)/sum(w))**2)
    tau2=max(0,(Q-(k-1))/(sum(w)-sum(w*w)/sum(w)))
    ww=1/(se**2+tau2); mu=sum(ww*y)/sum(ww); q=sum(ww*(y-mu)**2)/(k-1)
    hse=np.sqrt(q/sum(ww)); mhse=np.sqrt(max(1,q)/sum(ww)); tc=td.ppf(.975,k-1)
    return {'mu':float(mu),'ci_hksj':[float(mu-tc*hse),float(mu+tc*hse)],
            'ci_modified_hksj':[float(mu-tc*mhse),float(mu+tc*mhse)],
            'tau':float(np.sqrt(tau2)),'q':float(q),'k':k}


def load_rows(s,axis):
    if s=='B_T2s2_std3': rr=OA.bt2s2_onset() if axis=='onset' else LL.bt2s2_rows(LL.BT.STD_SPEEDS)
    else: rr=OA.onset_rows(s) if axis=='onset' else LL.SV.rows_of(OA.TAG.get(s,s))
    rr=LL.with_bcol([dict(r) for r in rr])
    assert len(rr)==120
    assert all(sum(r['bcol']==j for r in rr)==3 for j in range(40))
    return sorted(rr,key=lambda r:(r['bcol'],r['lv']))


def counts(rr):
    lo,hi=np.array([r['lower'] for r in rr]).T
    return {'n':len(rr),'left':int(np.isneginf(lo).sum()),'right_no_band':int(np.isposinf(hi).sum()),
            'finite':int((np.isfinite(lo)&np.isfinite(hi)).sum()),'zero_width':int((lo==hi).sum())}


def censor(rows,c):
    out=[]; lc=np.log(c)
    for r in rows:
        lo,hi=r['lower']
        if hi<=lc: lo,hi=-np.inf,lc
        elif lo<lc: lo=-np.inf
        # A right-censored lower endpoint below c carries no remaining information.
        if np.isneginf(lo) and np.isposinf(hi): continue
        out.append(dict(r,lower=(lo,hi)))
    return out


def bootstrap_pair(w,o,length,nb=200,seed=20261020):
    rng=np.random.default_rng(seed); bw=[];bo=[]; diag={'failures':0,'fallbacks':0,'max_gradient':0.}
    iw=fit(w)['params'];io=fit(o)['params']
    groups=[np.arange(j,j+length) for j in range(0,40,length)]
    for _ in range(nb):
        kk=np.concatenate([groups[j] for j in rng.integers(len(groups),size=len(groups))])
        ind=np.concatenate([np.arange(j*3,j*3+3) for j in kk])
        a=fit([w[j] for j in ind],iw);b=fit([o[j] for j in ind],io)
        diag['failures']+=int(not a['success'])+int(not b['success'])
        diag['fallbacks']+=a['fallback']+b['fallback']
        diag['max_gradient']=max(diag['max_gradient'],a['grad_max'],b['grad_max'])
        bw.append(a['params'][1:3]);bo.append(b['params'][1:3])
    return np.array(bw),np.array(bo),diag


def main():
    source=json.loads((ROOT/'novelty/lower_law/onset_axis.json').read_text())
    names=list(source['onset']); sets={}; out={'post_hoc':True,'sessions':{},'bootstrap':{},'pooled':{}}
    for s in names:
        w,o=load_rows(s,'window'),load_rows(s,'onset'); sets[s]=(w,o)
        info={}
        for ax,rr in [('window',w),('onset',o)]:
            base=fit(rr); vals={}
            variants={'complete_case':[r for r in rr if np.isfinite(r['lower']).all()],
                      'omit_no_band':[r for r in rr if np.isfinite(r['lower'][1])],
                      'common_0.25':censor(rr,.25),'common_0.35':censor(rr,.35)}
            for key,rs in variants.items(): vals[key]={'fit':fit(rs),'counts':counts(rs)}
            old=source['onset'][s] if ax=='onset' else {'b_v':source['measured_se'][s][0],'b_beta':source['measured_se'][s][2]}
            info[ax]={'fit':base,'counts':counts(rr),'variants':vals,
                       'max_abs_coefficient_delta_from_claude':float(max(abs(base['params'][1]-old['b_v']),abs(base['params'][2]-old['b_beta'])))}
        out['sessions'][s]=info
        print('fit',s,info['window']['fit']['params'][1:3],info['onset']['fit']['params'][1:3],flush=True)
    (HERE/'audit_statistics.json').write_text(json.dumps(out,indent=2))
    for length in (1,5,10):
        boots={}; out['bootstrap'][str(length)]={}
        for s,(w,o) in sets.items():
            bw,bo,diag=bootstrap_pair(w,o,length); boots[s]=(bw,bo)
            rec={'diagnostics':diag}
            for ax,b in [('window',bw),('onset',bo),('onset_minus_window',bo-bw)]:
                rec[ax]={'se':np.std(b,axis=0,ddof=1).tolist(),'ci':np.percentile(b,[2.5,97.5],axis=0).T.tolist()}
            out['bootstrap'][str(length)][s]=rec
            print('bootstrap',length,s,flush=True)
        pooled={}
        for ax in ('window','onset','onset_minus_window'):
            pp={}
            for idx,key in enumerate(('v','beta')):
                y=[out['sessions'][s][ax]['fit']['params'][idx+1] if ax!='onset_minus_window' else
                   out['sessions'][s]['onset']['fit']['params'][idx+1]-out['sessions'][s]['window']['fit']['params'][idx+1] for s in names]
                se=[out['bootstrap'][str(length)][s][ax]['se'][idx] for s in names]
                pp[key]=meta(y,se)
            pooled[ax]=pp
        out['pooled'][str(length)]=pooled
        (HERE/'audit_statistics.json').write_text(json.dumps(out,indent=2))
    # Equally weighted construction×tension means: an explicit sensitivity, not a population model.
    groups={'A_T1':['A1','A2','A3','A4'],'C_T1':['C1'],'D_T1':['D1','D2'],
            'A_T2':['A_T2'],'B_T2':['B_T2s1','B_T2s2_std3'],'C_T2':['C_T2']}
    out['type_tension_equal_weight']={}
    for ax in ('window','onset'):
        y=np.array([np.mean([out['sessions'][s][ax]['fit']['params'][1:3] for s in ss],axis=0) for ss in groups.values()])
        m=y.mean(0);se=y.std(0,ddof=1)/np.sqrt(len(y));tc=td.ppf(.975,len(y)-1)
        out['type_tension_equal_weight'][ax]={'groups':list(groups),'means':y.tolist(),'mean':m.tolist(),'ci':np.c_[m-tc*se,m+tc*se].tolist()}
    out['provenance']={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),ROOT/'novelty/lower_law/onset_axis.py',ROOT/'novelty/lower_law/onset_axis.json']}
    (HERE/'audit_statistics.json').write_text(json.dumps(out,indent=2))
    print(json.dumps(out['pooled'],indent=2),flush=True)

if __name__=='__main__': main()
