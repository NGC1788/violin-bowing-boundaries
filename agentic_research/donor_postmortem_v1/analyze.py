"""Posthoc additive diagnostics from sealed saved forecasts; no model imports."""
from pathlib import Path
import hashlib,json
import numpy as np

HERE=Path(__file__).resolve().parent
PARENT=HERE.parent/'donor_deletion_learning_v1'
REPO=HERE.parents[1]
METHODS=['retrieval_erm','retrieval_random_delete','retrieval_adversarial_delete','retrieval_query_group_dro']
COMPARISONS=[('retrieval_erm','markov'),('retrieval_query_group_dro','markov'),
             ('retrieval_adversarial_delete','retrieval_erm'),
             ('retrieval_adversarial_delete','retrieval_random_delete'),
             ('retrieval_adversarial_delete','retrieval_query_group_dro')]


def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(4*1024**2),b''):h.update(b)
    return h.hexdigest()


def load(p,expected):
    if p.is_symlink() or sha(p)!=expected:raise ValueError('saved input seal mismatch')
    with np.load(p,allow_pickle=False) as z:out={k:z[k] for k in z.files}
    if sha(p)!=expected:raise ValueError('saved input changed during read')
    return out


def losses(y,p):
    q=np.clip(p,1e-6,1-1e-6)
    return -(y*np.log(q)+(1-y)*np.log1p(-q))


def entropy(p):
    return -(p*np.log(np.maximum(p,1e-300))).sum(-1)


def corr(x,y,w):
    w=w/w.sum();dx=x-np.dot(w,x);dy=y-np.dot(w,y)
    den=np.sqrt(np.dot(w,dx*dx)*np.dot(w,dy*dy))
    return float(np.dot(w,dx*dy)/den) if den>1e-20 else None


def main():
    result=json.loads((PARENT/'RESULTS.json').read_text())
    if sha(PARENT/'RESULTS.json')!=json.loads((PARENT/'RESULTS_SEAL.json').read_text())['results_sha256']:raise ValueError('result seal')
    freeze=json.loads((PARENT/'FREEZE.json').read_text())
    if sha(PARENT/'FREEZE.json')!=result['freeze_sha256']:raise ValueError('freeze link')
    if any(sha(REPO/n)!=h for n,h in freeze['sources'].items()):raise ValueError('frozen code changed')
    seal=json.loads((PARENT/'OUTPUT_SEAL.json').read_text())
    if sha(PARENT/'OUTPUT_SEAL.json')!=result['batch_output_seal_sha256']:raise ValueError('output seal')
    for job,files in seal['jobs'].items():
        for name,h in files.items():
            if sha(PARENT/'local'/job/name)!=h:raise ValueError('parent pinned artifact changed')
    saved=load(PARENT/'local/all_predictions.npz',result['all_predictions_sha256'])
    gate=HERE.parent/'path_source_gate_v1/local'
    meta=load(gate/'source_metadata.npz',freeze['inputs']['source_metadata.npz'])
    # Only the already saved guard is used; future descriptors are not analyzed.
    guard_file=gate/'future_labels.npz'
    if sha(guard_file)!=freeze['inputs']['future_labels.npz']:raise ValueError('guard input')
    with np.load(guard_file,allow_pickle=False) as z:guard=z['guard']
    people,fold,y=meta['people'],meta['fold'],saved['truth'];unique=np.unique(people)
    if len(unique)!=34 or len(y)!=1442 or int((~guard).sum())!=15:raise ValueError('source population changed')
    counts={g:int((people==g).sum()) for g in unique}
    weights=np.array([1/(len(unique)*counts[g]) for g in people])
    assert abs(weights.sum()-1)<1e-14
    outcome={'invalid':~guard,'valid_all_zero':guard&~y.any(1),'valid_any_positive':guard&y.any(1)}
    assert np.stack(list(outcome.values())).sum(0).min()==1
    score={m:losses(y,p) for m,p in saved.items() if m!='truth'}
    for m,ell in score.items():
        assert abs(np.dot(weights,ell.mean(1))-result['aggregate'][m]['log_loss'])<1e-12
    def person_values(values):return np.array([values[people==g].mean(0) for g in unique])
    comparisons={}
    for method,baseline in COMPARISONS:
        delta=score[method]-score[baseline];row=delta.mean(1);total=float(np.dot(weights,row))
        by_outcome={name:float(np.dot(weights[mask],row[mask])) for name,mask in outcome.items()}
        by_fold={str(k):float(np.dot(weights[fold==k],row[fold==k])) for k in range(3)}
        by_event=(weights[:,None]*delta).sum(0)/9
        cells={str(k):{name:((weights[(fold==k)&mask,None]*delta[(fold==k)&mask]).sum(0)/9).tolist()
                       for name,mask in outcome.items()} for k in range(3)}
        assert abs(sum(by_outcome.values())-total)<1e-12
        assert abs(sum(by_fold.values())-total)<1e-12
        assert abs(by_event.sum()-total)<1e-12
        assert abs(sum(sum(sum(v) for v in group.values()) for group in cells.values())-total)<1e-12
        per=person_values(row)
        label_parts={}
        for truth_value in [0,1]:
            mask=y==truth_value;w=weights[:,None]*mask/9;mass=float(w.sum())
            label_parts[str(truth_value)]={'event_cell_macro_mass':mass,
                'loss_gap_contribution':float((w*delta).sum()),
                'conditional_mean_probability_change':float((w*(saved[method]-saved[baseline])).sum()/mass)}
        assert abs(sum(x['loss_gap_contribution'] for x in label_parts.values())-total)<1e-12
        comparisons[method+'__minus__'+baseline]={'loss_gap':total,'sign':'negative is improvement',
            'fold_additive_contributions':by_fold,'outcome_additive_contributions':by_outcome,
            'event_additive_contributions':by_event.tolist(),'fold_outcome_event_additive_contributions':cells,
            'true_label_partition':label_parts,'people_better':int((per<0).sum()),
            'people_worse':int((per>0).sum()),'people_equal':int((per==0).sum()),
            'most_improved_person_gap':float(per.min()),'most_worsened_person_gap':float(per.max())}
    concentration={m:{key:np.zeros(len(y)) for key in ['entropy','max_mass','effective_donors','deletion_risk','counterfactual_event_sensitivity']} for m in METHODS}
    traces={};support={};resources={}
    for k in range(3):
        ix=np.flatnonzero(fold==k);traces[str(k)]={};support[str(k)]={};resources[str(k)]={}
        for method in METHODS:
            job=f'job_{k}_{method}';pins=seal['jobs'][job]
            arr=load(PARENT/'local'/job/'predictions.npz',pins['predictions.npz'])
            assert np.array_equal(arr['row_indices'],ix)
            assert np.array_equal(arr['probability'],saved[method][ix])
            mass=arr['group_mass'];h=entropy(mass);cf=arr['counterfactual_probability']
            d=concentration[method];d['entropy'][ix]=h;d['effective_donors'][ix]=np.exp(h);d['max_mass'][ix]=mass.max(1)
            d['deletion_risk'][ix]=losses(y[ix,None,:],cf).mean(-1).max(1)
            d['counterfactual_event_sensitivity'][ix]=np.abs(cf-cf[:,0:1]).mean(-1).max(1)
            support[str(k)][method]={name:float((arr[name]==0).mean()) for name in ['positive_support','negative_support']}
            trace=json.loads((PARENT/'local'/job/'trace.json').read_text())
            traces[str(k)][method]={'checkpoints':[trace[i] for i in [0,9,19,29,39]],
                 'last_minus_first_primary_diagnostic':trace[-1]['online_person_macro_primary_diagnostic']-trace[0]['online_person_macro_primary_diagnostic'],
                 'last_minus_first_auxiliary':trace[-1]['online_weighted_auxiliary']-trace[0]['online_weighted_auxiliary'],
                 'last_optimized_minus_diagnostic_primary':trace[-1]['online_optimized_weighted_primary']-trace[-1]['online_person_macro_primary_diagnostic']}
            receipt=json.loads((PARENT/'local'/job/'receipt.json').read_text())
            resources[str(k)][method]={name:receipt[name] for name in ['memory_artifact_bytes','inference_memory_array_bytes','weights_artifact_bytes','predictions_artifact_bytes','held_full_plus_all_counterfactual_inference_seconds']}
    associations={}
    method='retrieval_adversarial_delete'
    for baseline in ['retrieval_erm','retrieval_random_delete','retrieval_query_group_dro']:
        dl=(score[method]-score[baseline]).mean(1);per_loss=person_values(dl)
        details={}
        for key in ['entropy','max_mass','effective_donors','counterfactual_event_sensitivity']:
            dx=concentration[method][key]-concentration[baseline][key];per_delta=person_values(dx)
            xcenter=dx.copy();ycenter=dl.copy()
            for g in unique:
                mask=people==g;xcenter[mask]-=dx[mask].mean();ycenter[mask]-=dl[mask].mean()
            details[key]={'person_macro_change':float(np.dot(weights,dx)),
                'record_weighted_pearson_with_loss_change':corr(dx,dl,weights),
                'within_person_centered_weighted_pearson':corr(xcenter,ycenter,weights),
                'person_mean_pearson':corr(per_delta,per_loss,np.ones(len(unique))),
                'people_delta_positive':int((per_delta>0).sum()),'people_delta_negative':int((per_delta<0).sum())}
        details['folds']={str(k):{key:float(np.dot(weights[fold==k],(concentration[method][key]-concentration[baseline][key])[fold==k])/weights[fold==k].sum()) for key in ['entropy','max_mass','effective_donors','deletion_risk']} for k in range(3)}
        details['deletion_risk_macro_change']=float(np.dot(weights,concentration[method]['deletion_risk']-concentration[baseline]['deletion_risk']))
        associations[method+'__minus__'+baseline]=details
    concentration_aggregate={m:{key:float(np.dot(weights,v)) for key,v in vals.items()} for m,vals in concentration.items()}
    for m in METHODS:assert abs(concentration_aggregate[m]['deletion_risk']-result['aggregate'][m]['deletion_risk_person_macro'])<1e-12
    output={'status':'SAVED_ARTIFACT_DIAGNOSIS_COMPLETE','scope':'posthoc exposed-source descriptive diagnosis, not new predictions or causal evidence',
        'rows':len(y),'people':len(unique),'weight_rule':'w_i=1/(34*n_person), event weights1/9; same fixed weights in every additive contribution',
        'comparisons':comparisons,'concentration_aggregates':concentration_aggregate,'matched_associations':associations,
        'saved_trace_summaries':traces,'support_zero_fractions':support,'saved_resource_costs':resources,
        'additive_reconstruction_tolerance':1e-12,'new_fits':0,'new_model_forwards':0,'new_gradients':0,
        'new_standalone_numerical_audits':0,'old_dev_test_truth_access_or_hash':False,
        'results_sha256':sha(PARENT/'RESULTS.json'),'output_seal_sha256':sha(PARENT/'OUTPUT_SEAL.json'),
        'all_predictions_sha256':result['all_predictions_sha256'],'metadata_sha256':freeze['inputs']['source_metadata.npz'],
        'guard_file_sha256':freeze['inputs']['future_labels.npz'],'analysis_source_sha256':sha(Path(__file__))}
    with (HERE/'RESULTS.json').open('x') as f:json.dump(output,f,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps({'status':output['status'],'comparisons':len(comparisons),'reconstruction_checks':'PASS','new_fits_or_forwards':0}))


if __name__=='__main__':main()
