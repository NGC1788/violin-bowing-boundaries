"""Exact finite deletion-reference ranks, not p-values."""
from pathlib import Path
import json,itertools,hashlib
import numpy as np
import run as R
HERE=Path(__file__).resolve().parent
TIE_ATOL=1e-9

def rank(values,target):
    v=np.asarray(values);ties=np.abs(v-target)<=TIE_ATOL;below=(v<target)&~ties;above=(v>target)&~ties
    assert np.all(ties|below|above)
    return {'n':len(v),'below':int(below.sum()),'tied':int(ties.sum()),'above':int(above.sum()),'midrank_percentile':float((below.sum()+.5*ties.sum())/len(v)),'target':float(target),'control_median':float(np.median(v)),'control_min':float(v.min()),'control_max':float(v.max())}

def aggregate(cases):
    if not cases:return {'n':0}
    d=np.array([c['delta_beta'] for c in cases]);toward=np.array([c['toward_target'] for c in cases]);t=np.array([c['T_reduction'] for c in cases]);block=np.array([c['block']['delta_beta'] for c in cases])
    return {'n':len(cases),'median_abs_delta_beta':float(np.median(abs(d))),'max_abs_delta_beta':float(max(abs(d))),'toward_target_positive':int((toward>TIE_ATOL).sum()),'toward_target_negative':int((toward<-TIE_ATOL).sum()),'T_reduction_positive':int((t>TIE_ATOL).sum()),'T_reduction_negative':int((t<-TIE_ATOL).sum()),'median_T_reduction':float(np.median(t)),'T_reduction_range':[float(min(t)),float(max(t))],'median_T_before':float(np.median([c['T_before'] for c in cases])),'median_T_after':float(np.median([c['T_after'] for c in cases])),'block_sign_agreement':int((np.sign(d)==np.sign(block)).sum()),'block_median_abs_delta_beta':float(np.median(abs(block))),'metrics':{m:{'median_percentile':float(np.median([c['controls']['all'][m]['midrank_percentile'] for c in cases])),'at_least_95_percentile':sum(c['controls']['all'][m]['midrank_percentile']>=.95 for c in cases),'span_matched_median_percentile':float(np.median([c['controls']['span_matched'][m]['midrank_percentile'] for c in cases if c['controls']['span_matched']]))} for m in ('abs_delta_beta','toward_target','T_reduction')}}

def main():
    obj=json.load(open(HERE/'selected.json'));jobs=R.jobs();lookup={(j['group'],j['k']):j for j in jobs};fail=[];Nfit=0;Ncontrol=0;maxgrad=0.;maxparity=0.;attemptcounts={}
    for j in jobs:
        meta=json.load(open(HERE/'controls'/f"{j['number']:02}.json"));npz=np.load(HERE/'controls'/f"{j['number']:02}.npz");Nfit+=meta['fit_count'];Ncontrol+=meta['n_controls'];fail+=meta['failures']
        expect=len(list(itertools.combinations(j['eligible_indices'],j['k'])));assert meta['n_controls']==expect
        assert np.array_equal(npz['combinations'],list(itertools.combinations(j['eligible_indices'],j['k'])))
        for lab in ['free']+[str(p) for p in j['p']]:
            maxgrad=max(maxgrad,float(np.max(npz[lab+'_grad_max'])));assert npz[lab+'_success'].all() and not npz[lab+'_scale_hit'].any(),meta
            for a,n in zip(*np.unique(npz[lab+'_attempts'],return_counts=True)):attemptcounts[str(a)]=attemptcounts.get(str(a),0)+int(n)
    assert not fail,fail
    for c in obj['cases']:
        for part in ('after','block'):
            assert c[part]['free']['success'] and c[part]['fixed']['success'],(c['group'],part)
            assert c[part]['fixed']['nll']>=c[part]['free']['nll']-1e-7,(c['group'],part)
        assert len(c['block']['ids'])==3*c['k']
        j=lookup[(c['group'],c['k'])];z=np.load(HERE/'controls'/f"{j['number']:02}.npz");p=str(c['p']);base=c['before']['free']['params'][2];b=z['free_params'][:,2];T=2*(z[p+'_nll']-z['free_nll']);assert T.min()>=-1e-7
        metrics={'abs_delta_beta':abs(b-base),'toward_target':abs(base-c['p'])-abs(b-c['p']),'T_reduction':c['T_before']-T};target={'abs_delta_beta':abs(c['delta_beta']),'toward_target':c['toward_target'],'T_reduction':c['T_reduction']}
        found=np.flatnonzero(np.all(z['combinations']==c['deleted_row_indices'],axis=1));assert len(found)==1;i=int(found[0]);span=float(z['log_beta_span'][i]);mask=(z['log_beta_span']>=.8*span)&(z['log_beta_span']<=1.2*span)
        maxparity=max(maxparity,max(abs(z['free_params'][i]-c['after']['free']['params'])),max(abs(z[p+'_params'][i]-c['after']['fixed']['params'])))
        assert maxparity<2e-4
        c['controls']={'job_number':j['number'],'selected_index':i,'n_all':len(b),'span':span,'n_span_matched':int(mask.sum()),'all':{m:rank(v,target[m]) for m,v in metrics.items()},'span_matched':{m:rank(v[mask],target[m]) for m,v in metrics.items()} if mask.any() else None}
    obj['status']='PASS';obj['tie_definition']={'absolute_tolerance':TIE_ATOL,'formula':'(number strictly below + 0.5 number tied)/number controls; certificate itself included; descriptive finite-population rank, not p-value'};R.write(HERE/'selected.json',obj)
    summary={'status':'PASS','control_jobs':len(jobs),'distinct_control_subsets':Ncontrol,'control_fits':Nfit,'unresolved_fit_failures':len(fail),'max_control_gradient':maxgrad,'selected_control_max_parameter_delta':maxparity,'optimizer_attempt_counts':attemptcounts,'max_decomposition_identity_error':max(abs(c['decomposition']['identity_error']) for c in obj['cases']),'all':aggregate(obj['cases']),'classifier_specific_52':aggregate([c for c in obj['cases'] if c['arbitrary_geometry_feasible']]),'geometry_infeasible_25':aggregate([c for c in obj['cases'] if not c['arbitrary_geometry_feasible']]),'by_slope':{str(p):aggregate([c for c in obj['cases'] if c['p']==p]) for p in R.PS}}
    R.write(HERE/'summary.json',summary)
    print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
