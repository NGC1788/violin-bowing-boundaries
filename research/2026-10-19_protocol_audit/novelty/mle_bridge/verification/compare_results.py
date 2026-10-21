"""Compare production results with independent rows, likelihood and refits."""
from pathlib import Path
import argparse, gzip, hashlib, json, time
import numpy as np
import oracle as O
from independent_refits import fit_rows

HERE=Path(__file__).resolve().parent;BRIDGE=HERE.parent;PROD=BRIDGE/'production'


def run():
    start=time.monotonic();fit_comparisons=[];score_errors=[];decomp_errors=[]
    independent=json.load((HERE/'independent_refits.json').open())
    columns=json.load(gzip.open(PROD/'input_joint.json.gz','rt'))['columns']
    rows=O.original_rows_from_columns(columns);byid={r['id']:r for r in rows}
    prodrows=json.load((PROD/'rows.json').open())
    for r in prodrows:
        old=byid[r['id']]
        assert r['session']==old['session'] and r['group']==old['group']
        assert r['column']==old['column'] and r['eligible']==old['eligible']
        assert np.array_equal([r['lv'],r['lb']],old['x'][1:])
        assert np.array_equal(r['lower'],[old['lower'],old['upper']])
    def fit_same(a,b):
        assert a['success'] and b['success']
        e=float(max(abs(np.array(a['params'])-np.array(b['params']))))
        f=float(abs(a['nll']-b['nll']))
        assert e<2e-5 and f<1e-7,(e,f,a,b)
        fit_comparisons.append([e,f])
    base=json.load((PROD/'baselines.json').open())['sessions']
    for s,b in base.items():
        ref=independent['baseline'][s];fit_same(b['free'],ref['free'])
        assert b['row_ids']==ref['row_ids']
        for p in [-2.,-1.2]:fit_same(b['fixed'][str(p)],ref['fixed'][str(p)])
    selected=json.load((PROD/'selected.json').open())['cases']
    refcases={(r['group'],r['p']):r for r in independent['cases']}
    block_fits=[]
    for case in selected:
        ref=refcases[case['group'],case['p']]
        assert sorted(case['ids'])==ref['deleted_ids']
        fit_same(case['after']['free'],ref['free']);fit_same(case['after']['fixed'],ref['fixed'])
        rr=[byid[i] for i in base[case['session']]['row_ids']]
        drop=set(case['ids']);remaining=[r for r in rr if r['id'] not in drop]
        assert [r['id'] for r in remaining]==ref['remaining_ids']
        assert {rr[i]['id'] for i in case['deleted_row_indices']}==set(case['ids'])
        # Scores and decomposition from independent likelihood, preserving signs.
        def nll(params,rs):
            return O.negative_log_likelihood(params,[r['x'] for r in rs],
                                            [r['lower'] for r in rs],[r['upper'] for r in rs])
        before_free=case['before']['free'];before_fixed=case['before']['fixed'][str(case['p'])]
        deleted=[r for r in rr if r['id'] in drop]
        contrast=2*(nll(before_fixed['params'],deleted)-nll(before_free['params'],deleted))
        gain_free=nll(before_free['params'],remaining)-case['after']['free']['nll']
        gain_fixed=nll(before_fixed['params'],remaining)-case['after']['fixed']['nll']
        decomp=contrast+2*(gain_fixed-gain_free)
        error=abs(decomp-case['T_reduction']);decomp_errors.append(error)
        assert error<1e-7
        assert abs(contrast-case['decomposition']['removed_fullfit_contrast'])<1e-7
        assert abs(gain_free-case['decomposition']['free_nuisance_gain'])<1e-7
        assert abs(gain_fixed-case['decomposition']['fixed_nuisance_gain'])<1e-7
        # Repeat the secondary intervention for the same beta indices at all speeds.
        index={byid[i]['column'] for i in case['ids']}
        blockids={r['id'] for r in rr if r['column'] in index}
        assert blockids==set(case['block']['ids']) and len(blockids)==3*len(index)
        blockrows=[r for r in rr if r['id'] not in blockids]
        free=fit_rows(blockrows);fixed=fit_rows(blockrows,case['p'])
        fit_same(case['block']['free'],free);fit_same(case['block']['fixed'],fixed)
        block_fits.append({'group':case['group'],'p':case['p'],'free':free,'fixed':fixed})
    replay=json.load((BRIDGE/'coherent_replay.json').open());replay_fits={}
    for name,rule in replay['rules'].items():
        u,v=rule['thresholds'];changed_columns=[];changes=0
        for c in columns:
            copy=dict(c); labels=[]
            for fixed,guard,per,ratio in zip(c['fixed'],c['guard'],c['periodicity'],c['ratio']):
                per=float('nan') if per is None else per;ratio=float('nan') if ratio is None else ratio
                labels.append(bool(fixed or (guard and per>=u and ratio<v)))
            changes+=sum(bool(a)!=bool(b) for a,b in zip(labels,c['baseline']))
            copy['baseline']=labels;changed_columns.append(copy)
        newrows=O.original_rows_from_columns(changed_columns)
        assert changes==rule['global_label_changes']
        lost_ids=[];lost_groups=set()
        col_byid={c['id']:c for c in changed_columns}
        for row in newrows:
            c=col_byid[row['id']];idx=row['start']
            if c['eligible'] and (idx is None or idx==0 or not (0<c['force'][idx-1]<c['force'][idx])):
                lost_ids.append(row['id']);lost_groups.add(row['group'])
        assert len(lost_ids)==rule['original_eligible_boundary_losses']
        assert sorted(lost_groups)==rule['lost_boundary_groups']
        for s,old in rule['sessions'].items():
            rr=[r for r in newrows if r['session']==s]
            counts={'n':len(rr)}
            for r in rr:
                key='left' if np.isneginf(r['lower']) else 'right' if np.isposinf(r['upper']) else 'finite'
                counts[key]=counts.get(key,0)+1
            assert counts==old['counts']
            assert sum(r['id'] in lost_ids for r in rr)==old['original_eligible_boundary_losses']
            changes_here=sum((r['lower'],r['upper'])!=(byid[r['id']]['lower'],byid[r['id']]['upper']) for r in rr)
            assert changes_here==old['interval_changes']
        if name=='baseline':assert changes==0;continue
        fits={}
        for s,old in rule['sessions'].items():
            rr=[r for r in newrows if r['session']==s];assert len(rr)==120
            free=fit_rows(rr);fixed={str(p):fit_rows(rr,p) for p in [-2.,-1.2]}
            fit_same(old['fit'],free)
            for p in [-2.,-1.2]:fit_same(old['fixed'][str(p)],fixed[str(p)])
            fits[s]={'free':free,'fixed':fixed}
        replay_fits[name]=fits
    sources=[Path(__file__),HERE/'oracle.py',HERE/'independent_refits.json',PROD/'rows.json',
             PROD/'baselines.json',PROD/'selected.json',BRIDGE/'coherent_replay.json']
    return {'passed':True,'n_rows_exactly_reconstructed':len(rows),
            'numerical_fit_comparisons':len(fit_comparisons),
            'max_params_absolute_difference':max(a[0] for a in fit_comparisons),
            'max_nll_absolute_difference':max(a[1] for a in fit_comparisons),
            'decomposition_identities':len(decomp_errors),'max_decomposition_error':max(decomp_errors),
            'block_refits':block_fits,'coherent_rule_refits':replay_fits,
            'source_sha256':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
            'seconds':time.monotonic()-start}


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=HERE/'compare_results.json');a=p.parse_args()
    result=run();a.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ['block_refits','coherent_rule_refits','source_sha256']},indent=2))
