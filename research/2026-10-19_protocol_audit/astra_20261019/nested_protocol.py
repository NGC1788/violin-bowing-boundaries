"""Fully refit the reference shape inside each held-out fold (post hoc)."""
from pathlib import Path
import json
import numpy as np
import protocol_falsification as p


def main():
    output=p.HERE/'nested_protocol_results.json'
    if output.exists():raise FileExistsError('Preserve the previous result')
    desc=json.loads((p.BASE/'novelty/lower_protocol/descriptors.json').read_text())
    import sys
    sys.path.insert(0,str(p.BASE/'novelty/lower_protocol'));import analysis as a
    tags=np.array(p.TAGS)
    x=np.array([desc['diagram'][t]['dec5_pl'] for t in tags])
    prop=np.log(np.array([[a.POMA[t][1],a.POMA[t][2]] for t in tags]))
    models={'mean':np.ones((10,1)),'dec5_pl':np.c_[np.ones(10),x],
            'log_Z':np.c_[np.ones(10),prop[:,0]],'log_EI':np.c_[np.ones(10),prop[:,1]],
            'log_Z_EI':np.c_[np.ones(10),prop]}
    datasets={t:dict(np.load(p.BASE/f'final/cache/{t}.npz',allow_pickle=False)) for t in tags}
    report={'scope':'사후 분석. 기준 모양과 회귀를 각 학습 fold에서 모두 재적합. 목표 자료는 dec5_pl 제어 기술자와 측정 범위만 예측에 사용.',
            'tags':tags.tolist(),'source_sha256':{str(p.BASE/f'final/cache/{t}.npz'):p.sha(p.BASE/f'final/cache/{t}.npz') for t in tags},
            'code_sha256':{str(f):p.sha(f) for f in (Path(__file__),Path(p.__file__))},'variants':{}}
    for axis in ('measured','nominal_rank'):
        rows={t:p.boundary_rows(datasets[t],t,.2,2,axis) for t in tags}
        for split,groups in [('leave_diagram_out',tags),('leave_type_out',np.array([t[0] for t in tags]))]:
            result={'folds':[],'models':{name:{'level_errors':np.empty(10),'column_median_errors':np.empty(10),'interval_distance_means':np.empty(10)} for name in models}}
            for group in sorted(set(groups)):
                train=np.flatnonzero(groups!=group);test=np.flatnonzero(groups==group)
                source=[r for i in train for r in rows[tags[i]]]
                reference=p.censored(source,p.design(source))
                levels=[]
                for i in train:
                    rr=rows[tags[i]]
                    levels.append(p.censored(rr,np.ones((len(rr),1)),p.design(rr)@reference['coef'])['coef'][0])
                # Freeze all prediction offsets before fitting/scoring held-out levels.
                offsets={name:X[test]@np.linalg.lstsq(X[train],levels,rcond=None)[0] for name,X in models.items()}
                fold={'held_out':tags[test].tolist(),'reference':reference,'source_levels':dict(zip(tags[train],levels)),
                      'predicted_offsets':{name:pred.tolist() for name,pred in offsets.items()}}
                for j,i in enumerate(test):
                    rr=rows[tags[i]];base=p.design(rr)@reference['coef']
                    true=p.censored(rr,np.ones((len(rr),1)),base)['coef'][0]
                    lo=np.array([r['lo'] for r in rr]);hi=np.array([r['hi'] for r in rr]);finite=np.isfinite(lo)&np.isfinite(hi)
                    for name in models:
                        pred=base+offsets[name][j]
                        result['models'][name]['level_errors'][i]=true-offsets[name][j]
                        result['models'][name]['column_median_errors'][i]=np.median(abs(pred[finite]-(lo[finite]+hi[finite])/2))
                        result['models'][name]['interval_distance_means'][i]=np.mean(np.maximum(np.maximum(lo-pred,pred-hi),0))
                result['folds'].append(fold)
                print(axis,split,group,'학습·채점 완료',flush=True)
            for name,r in result['models'].items():
                r.update(level_mae=float(np.mean(abs(r['level_errors']))),level_rmse=float(np.sqrt(np.mean(r['level_errors']**2))),
                         diagram_mean_column_median_error=float(r['column_median_errors'].mean()),
                         diagram_mean_interval_distance=float(r['interval_distance_means'].mean()))
                for k in ('level_errors','column_median_errors','interval_distance_means'):r[k]=r[k].tolist()
            report['variants'][axis+'_'+split]=result
    with output.open('x') as f:json.dump(report,f,ensure_ascii=False,indent=2,allow_nan=False)


if __name__=='__main__':main()
