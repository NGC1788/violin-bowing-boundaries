"""Independent offline sensitivity audit of the post-hoc load-decrease association."""
from __future__ import annotations
import json
import hashlib
import re
from datetime import date
from pathlib import Path
import sys
import numpy as np
from scipy.optimize import minimize

HERE=Path(__file__).resolve().parent
BASE=HERE.parent
sys.path.insert(0,str(BASE/'timeres/audit_20261017'))
from boundary_snapshot_audit import log_interval_probability

TAGS=('A1','A2','B1','C1','D1','A_T2','B_T2s1','B_T2r1','B_T2r2','C_T2')


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def validate(d):
    folders=d['folders']
    if len(folders)!=3:raise ValueError('Expected three acquired folders')
    for f in range(3):
        for t in (0.,.1,.2,.3):
            ids=np.sort(d['trial'][(d['folder']==f)&np.isclose(d['t'],t)])
            if not np.array_equal(ids,np.arange(1,2001)):
                raise ValueError('Missing or duplicate stroke in four-time acquisition inventory')
            levels=d['column'][(d['folder']==f)&np.isclose(d['t'],t)]
            if not np.array_equal(np.bincount(levels,minlength=40),np.full(40,50)):
                raise ValueError('Expected forty columns of fifty strokes')


def nominal_force(tag,trials):
    rank=(np.asarray(trials)-1)%50+1
    if tag=='A1':rank=51-rank
    return .1*40.**((rank-1)/49.)


def boundary_rows(d,tag,t,k,axis):
    force=d['force'] if axis=='measured' else nominal_force(tag,d['trial'])
    out=[]
    for folder in range(3):
        for col in range(40):
            ii=np.flatnonzero((d['folder']==folder)&(d['column']==col)&np.isclose(d['t'],t))
            if len(ii)!=50: raise ValueError('Incomplete force grid')
            ii=ii[np.argsort(force[ii],kind='stable')]
            lf=np.log(np.maximum(force[ii],1e-3));h=d['H'][ii]
            runs=np.flatnonzero(np.convolve(h.astype(int),np.ones(k,dtype=int),'valid')==k)
            if len(runs):
                s=int(runs[0]);lo=-np.inf if s==0 else lf[s-1];hi=lf[s]
            else:lo=lf[-1];hi=np.inf
            if not lo<hi:raise ValueError('Zero-width or reversed interval; do not hide it with probability clipping')
            out.append({'lv':float(np.log(np.median(d['speed'][ii]))),'lb':float(np.log(np.median(d['beta'][ii]))),
                        'lo':float(lo),'hi':float(hi),'folder':folder,'column':col})
    return out


def censored(rows,X,offset=None):
    lo=np.array([r['lo'] for r in rows]);hi=np.array([r['hi'] for r in rows])
    X=np.asarray(X,float);off=np.zeros(len(lo)) if offset is None else np.asarray(offset,float)
    if np.linalg.matrix_rank(X)!=X.shape[1]:raise ValueError('Rank-deficient fit')
    proxy=np.where(np.isfinite(lo),lo,hi)-off
    init=np.linalg.lstsq(X,proxy,rcond=None)[0]
    def nll(par):
        values=log_interval_probability(lo,hi,X@par[:-1]+off,np.exp(par[-1]))
        return -float(values.sum()) if np.isfinite(values).all() else 1e100
    fitted=[minimize(nll,np.r_[init,np.log(s)],method='L-BFGS-B',bounds=[(None,None)]*X.shape[1]+[(-6,4)],
                     options={'maxiter':1500,'ftol':1e-12,'gtol':1e-6}) for s in (.15,.5,1.5)]
    good=[r for r in fitted if r.success and r.fun<1e90]
    if not good:raise ValueError('All optimizer starts failed')
    best=min(good,key=lambda r:r.fun)
    return {'coef':best.x[:-1].tolist(),'sigma':float(np.exp(best.x[-1])),
            'nll':float(best.fun),'converged_starts':len(good),
            'objective_spread':float(max(r.fun for r in good)-best.fun),
            'sigma_bound_hit':bool(np.isclose(best.x[-1],-6) or np.isclose(best.x[-1],4))}


def design(rows):return np.array([[1.,r['lv'],r['lb']] for r in rows])


def cv_predict(X,y,groups):
    out=np.empty(len(y))
    for g in sorted(set(groups)):
        m=np.asarray(groups)!=g
        out[~m]=X[~m]@np.linalg.lstsq(X[m],y[m],rcond=None)[0]
    return out


def prediction_stats(y,p):
    e=y-p;sst=np.sum((y-y.mean())**2)
    return {'mae':float(np.mean(abs(e))),'rmse':float(np.sqrt(np.mean(e**2))),
            'cv_r2':float(1-np.sum(e**2)/sst),'predictions':p.tolist()}


def partial_corr(x,y,z):
    rx=x-z@np.linalg.lstsq(z,x,rcond=None)[0];ry=y-z@np.linalg.lstsq(z,y,rcond=None)[0]
    return {'r':float(np.corrcoef(rx,ry)[0,1]),'n':len(x),'nuisance_rank':int(np.linalg.matrix_rank(z)),
            'residual_dimension':int(len(x)-np.linalg.matrix_rank(z)),
            'scope':'작은 표본의 기술적 잔차 상관. 인과 효과나 확인 검정이 아니다.'}


def associations(x,y,prop,dates):
    types=np.array([t[0] for t in TAGS]);ones=np.ones((len(x),1))
    ztype=np.column_stack([types==k for k in sorted(set(types))]).astype(float)
    d=(dates-dates.mean())/30.
    models={'mean':ones,'dec5_pl':np.c_[ones,x],'log_Z':np.c_[ones,np.log(prop[:,0])],
            'log_EI':np.c_[ones,np.log(prop[:,1])],'log_Z_EI':np.c_[ones,np.log(prop)],
            'date_linear':np.c_[ones,d],'dec5_pl_date':np.c_[ones,x,d]}
    schemes={'leave_diagram_out':np.arange(len(x)), 'leave_type_out':types,
             'leave_date_out':dates,
             'leave_property_group_out':np.array(['A_T1' if t in ('A1','A2') else 'B_T2' if t.startswith('B_T2') else t for t in TAGS])}
    report={'r':float(np.corrcoef(x,y)[0,1]),'cv':{},
        'partial_date':partial_corr(x,y,np.c_[ones,d]),
        'partial_type':partial_corr(x,y,ztype),
        'partial_type_date':partial_corr(x,y,np.c_[ztype,d])}
    for scheme,groups in schemes.items():
        report['cv'][scheme]={name:prediction_stats(y,cv_predict(X,y,groups)) for name,X in models.items()}
    report['leave_one_diagram_corr']={t:float(np.corrcoef(np.delete(x,i),np.delete(y,i))[0,1]) for i,t in enumerate(TAGS)}
    m=np.array([t not in ('A1','A2') for t in TAGS])
    report['r_without_A1_A2']=float(np.corrcoef(x[m],y[m])[0,1])
    pc=np.array(report['cv']['leave_type_out']['dec5_pl']['predictions'])
    m=types=='C'
    report['C_type_unseen_prediction_residual']={'mean':float(np.mean(y[m]-pc[m])),
        'values':(y[m]-pc[m]).tolist(),'scope':'이미 선택된 C 대비의 사후 점검. 독립 확인이 아니다.'}
    return report


def main():
    out=HERE/'protocol_results';out.mkdir(exist_ok=False)
    desc_path=BASE/'novelty/lower_protocol/descriptors.json'
    desc=json.loads(desc_path.read_text())
    sys.path.insert(0,str(BASE/'novelty/lower_protocol'));import analysis as a
    datasets={};sources={str(desc_path):sha(desc_path)};dates=[];metadata={}
    for tag in TAGS:
        path=BASE/f'final/cache/{tag}.npz'
        d=dict(np.load(path,allow_pickle=False));validate(d);datasets[tag]=d;sources[str(path)]=sha(path)
        found=[re.findall(r'20\d\d-\d\d-\d\d',str(f))[-1] for f in d['folders']]
        if len(set(found))!=1:raise ValueError('Multiple folder dates; inspect instead of averaging')
        dates.append(date.fromisoformat(found[0]).toordinal())
        metadata[tag]={'folders':d['folders'].tolist(),'folder_date_proxy':found[0],
                       'unique_strokes':6000,'same_snapshot_rows':6000}
    x=np.array([desc['diagram'][t]['dec5_pl'] for t in TAGS])
    prop=np.array([[a.POMA[t][1],a.POMA[t][2]] for t in TAGS])
    report={'scope':'사후 반박 검증. 기존 10개 도표만 사용. D_T2 미열람. p값 선별 없음.',
            'tags':TAGS,'metadata':metadata,'sources_sha256':sources,'code_sha256':sha(__file__),
            'variants':{},'descriptor':x.tolist()}
    for axis in ('measured','nominal_rank'):
        for k in (2,3):
            for t in (.1,.2,.3):
                rs={tag:boundary_rows(d,tag,t,k,axis) for tag,d in datasets.items()}
                train=rs['A1']+rs['A2'];reference=censored(train,design(train))
                fits={tag:censored(rows,np.ones((len(rows),1)),design(rows)@reference['coef']) for tag,rows in rs.items()}
                y=np.array([fits[tag]['coef'][0] for tag in TAGS])
                key=f'{axis}_K{k}_t{t:.1f}'
                rec={'reference':reference,'levels':{tag:float(v) for tag,v in zip(TAGS,y)},'fits':fits,
                     'association':associations(x,y,prop,np.array(dates)),
                     'censor_counts':{tag:{'finite':sum(np.isfinite([r['lo'],r['hi']]).all() for r in rows),
                         'left':sum(np.isneginf(r['lo']) for r in rows),'right':sum(np.isposinf(r['hi']) for r in rows)} for tag,rows in rs.items()}}
                report['variants'][key]=rec
                print(key,'r',round(rec['association']['r'],4),'LODO',round(rec['association']['cv']['leave_diagram_out']['dec5_pl']['mae'],4),
                      'LOTO',round(rec['association']['cv']['leave_type_out']['dec5_pl']['mae'],4),flush=True)
    old=np.array([desc['diagram'][t]['lower'] for t in TAGS])
    new=np.array([report['variants']['measured_K2_t0.2']['levels'][t] for t in TAGS])
    report['legacy_level_max_abs_difference']=float(np.max(abs(old-new)))
    (out/'results.json').write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False,default=lambda x:x.item() if hasattr(x,'item') else list(x))+'\n')


if __name__=='__main__':main()
