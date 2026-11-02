"""One bounded calibration feasibility audit on historically exposed DEV only."""
from pathlib import Path
import argparse
import datetime
import fcntl
import json
import time
import numpy as np
import torch
from .methods import calibrate,cap_predictions,person_loss,betting_evalue,hb_pvalue,math_controls
from ..violin_raw_v1.models import patterns,transition_prior,participant_weights
from ..violin_diagnosis_v1.diagnose import ORIGINAL,restore_model,check_original,sha

HERE=Path(__file__).resolve().parent
DIAG=HERE.parent/'violin_diagnosis_v1'


def inputs():
    summary=json.loads((DIAG/'SUMMARY.json').read_text())
    if (sha(DIAG/'diagnose.py')!=summary['diagnostic_code_sha256'] or
            sha(DIAG/'local/diagnostics.json')!=summary['diagnostics_sha256']):
        raise ValueError('diagnostic executable/output identity changed')
    names=[ORIGINAL/'execution_freeze_manifest.json',ORIGINAL/'data_freeze_manifest.json',
           ORIGINAL/'RESULTS.json',DIAG/'SUMMARY.json',DIAG/'FINAL_AUDIT.json',DIAG/'diagnose.py',DIAG/'local/diagnostics.json']
    names += [ORIGINAL/'local/prepared'/n for n in ['features.npz','train_labels.npz','dev_labels.npz']]
    for seed in (0,1,2):
        names += [ORIGINAL/f'local/campaign_v1/mlp_offset_s{seed}'/n for n in ['weights.npz','receipt.json','predictions.npz']]
    return {str(p.relative_to(HERE.parent)):sha(p) for p in names}


def source_hashes():
    return {n:sha(HERE/n) for n in ['AGENTS.md','PLAN.json','methods.py','audit.py','test_methods.py','REVIEW.json']}


def freeze():
    path=HERE/'FREEZE.json'
    if path.exists():raise ValueError('freeze exists; preserve it')
    check_original(ORIGINAL)
    review=json.loads((HERE/'REVIEW.json').read_text())
    if review['verdict']!='APPROVE':raise ValueError('source/math review missing')
    for n,h in review['reviewed_source_sha256'].items():
        if sha(HERE/n)!=h:raise ValueError('reviewed source changed')
    value={'scope':'one posthoc calibration audit; zero base training/test truth access',
           'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
           'sources':source_hashes(),'inputs':inputs()}
    path.write_text(json.dumps(value,indent=2)+'\n')
    print(json.dumps({'freeze_sha256':sha(path),'source_files':len(value['sources']),'input_files':len(value['inputs'])}))


def run():
    started=time.monotonic();torch.set_num_threads(1)
    frozen=json.loads((HERE/'FREEZE.json').read_text())
    if frozen['sources']!=source_hashes() or frozen['inputs']!=inputs():raise ValueError('frozen audit changed')
    check_original(ORIGINAL)
    local=HERE/'local';local.mkdir(exist_ok=True)
    with (local/'audit.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        marker=local/'started.json'
        if marker.exists():raise ValueError('audit was started; preserve and inspect rather than repeat')
        marker.write_text(json.dumps({'started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'freeze_sha256':sha(HERE/'FREEZE.json')})+'\n')
        try:_run(started,frozen,local)
        except Exception as error:
            (local/'failure.json').write_text(json.dumps({'type':type(error).__name__,'reason':str(error)})+'\n')
            raise


def _run(started,frozen,local):
    prepared=ORIGINAL/'local/prepared';jobs=ORIGINAL/'local/campaign_v1'
    with np.load(prepared/'features.npz',allow_pickle=False) as z:
        split=z['split'];train=split=='train';dev=split=='dev';held=split!='train'
        x=z['X'][dev];state=z['prefix_states'][dev];people=z['participant_ids'][dev]
        train_state=z['prefix_states'][train];train_people=z['participant_ids'][train]
    with np.load(prepared/'train_labels.npz',allow_pickle=False) as z:train_y=z['Y9']
    with np.load(prepared/'dev_labels.npz',allow_pickle=False) as z:y=z['Y9']
    _,prior=transition_prior(train_y,train_state,participant_weights(train_people))
    b=prior[state]@patterns();diagnostic=json.loads((DIAG/'local/diagnostics.json').read_text())
    p=[];residual=[];reproduction=[]
    for seed in (0,1,2):
        job=jobs/f'mlp_offset_s{seed}';meta=json.loads((job/'receipt.json').read_text())
        if sha(job/'weights.npz')!=diagnostic['weights'][job.name]:raise ValueError('weight changed')
        prediction,_,r,_=restore_model(job/'weights.npz',meta,x,state,prior,'mlp_offset')
        with np.load(job/'predictions.npz',allow_pickle=False) as z:saved=z['probability'][split[held]=='dev']
        error=float(np.max(np.abs(saved-prediction)));reproduction.append(error)
        if error>1e-12:raise ValueError('endpoint replay differs')
        p.append(prediction);residual.append(r)
    p=np.asarray(p);residual=np.asarray(residual)
    unique=np.unique(people);base_p=np.broadcast_to(b,p.shape)
    baseline=person_loss(y,base_p,people);original=person_loss(y,p,people)
    oracle={};held_out={};fits_all=[];private={}
    for name,statewise in [('global',False),('statewise',True)]:
        prediction,fits=calibrate(y,b,p,people,state,np.ones(len(y),bool),statewise)
        oracle_fits=fits
        fits_all.extend(fits);oracle[name]=person_loss(y,prediction,people)
        combined=np.full_like(p,np.nan);folds=[]
        for person in unique:
            take=people!=person;pred,fits=calibrate(y,b,p,people,state,take,statewise)
            combined[:,~take]=pred[:,~take];fits_all.extend(fits);folds.append(fits)
        if not np.isfinite(combined).all():raise ValueError('missing held-person prediction')
        held_out[name]=person_loss(y,combined,people)
        private[name]={'oracle_fits':oracle_fits,'fold_fits':folds,
            'held_person_loss':held_out[name].tolist(),'oracle_person_loss':oracle[name].tolist()}
    if oracle['statewise'].mean()>oracle['global'].mean()+1e-10:raise ValueError('nested oracle family check failed')
    if len(fits_all)>120:raise ValueError('calibration budget exceeded')
    elapsed=time.monotonic()-started
    if elapsed>600:raise ValueError('audit time budget exceeded')
    def summary(v):
        return {'macro_log_loss':float(v.mean()),'maximum_person_loss':float(v.max()),
            'relative_gain_vs_markov_percent':float(100*(baseline.mean()-v.mean())/baseline.mean()),
            'people_better_than_markov':int((v<baseline-1e-10).sum()),'people':len(unique)}
    cap_results={};cap_private={}
    for cap in [.025,.1]:
        mass=cap_predictions(prior[state],residual,cap);pred=mass@patterns()
        score=person_loss(y,pred,people);difference=score-baseline
        if np.abs(difference).max()>cap+1e-10:raise ValueError('span score bound failed')
        cap_results[str(cap)]={**summary(score),'span_cap':cap,
            'hypothetical_fixed_independent_person_hb':hb_pvalue(difference,cap),
            'hypothetical_fixed_independent_person_betting_e':betting_evalue(-difference/cap),
            'valid_risk_certificate_on_these_data':False}
        cap_private[str(cap)]={'person_loss':score.tolist(),'difference':difference.tolist()}
    g,s=held_out['global'],held_out['statewise']
    relative=float((g.mean()-s.mean())/g.mean());better=int((s<g-1e-10).sum())
    max_ok=bool(s.max()<=g.max()+1e-10)
    screen=bool(relative>=.02 and better>=5 and max_ok)
    result={
        'scope':'one posthoc dev-only feasibility audit, not a new forecast performance study',
        'new_base_model_training_jobs':0,'test_labels_accessed':False,
        'calibration_fit_requests':len(fits_all),'nonempty_scalar_optimizations':sum(f['status']!='empty_fallback' for f in fits_all),
        'dev_people':len(unique),'dev_rows':len(y),'source_people':len(np.unique(train_people)),
        'max_saved_dev_reproduction_error':max(reproduction),
        'baseline':summary(baseline),'original_offset':summary(original),
        'restricted_dev_oracle':{k:summary(v) for k,v in oracle.items()},
        'leave_one_dev_person_out_calibration':{k:summary(v) for k,v in held_out.items()},
        'fixed_caps':cap_results,
        'conditional_over_global_exploratory_screen':{
            'relative_gain_percent':100*relative,'people_strictly_better':better,
            'maximum_person_loss_no_worse':max_ok,'screen_passed':screen,
            'criterion':'at least2percent relative gain, at least5of8 people better, maximum person loss no worse; not confirmation'},
        'solver':{'interval':[0.,.95],'maximum_root_bracket_width':max(f['bracket_width'] for f in fits_all),
                  'maximum_iterations':max(f['iterations'] for f in fits_all),
                  'no_clipping_in_optimized_intervals':True,'nested_oracle_check_passed':True},
        'math_controls':math_controls(),
        'promotion_gates':{'valid_new_independent_calibration_or_dependence_theorem':False,
            'substantive_new_method_beyond_existing_controls':False,
            'conditional_exploratory_screen':screen,'new_independent_confirmation_design':False},
        'decision':'STOP_CORRECTION_CERTIFICATION_BRANCH',
        'limitations':['original dev informed method design; no independent confirmation',
            'LOO calibrators overlap, hence their eight losses are not iid calibration data',
            'hypothetical HB/e values are sensitivity diagnostics, not valid p/e evidence for these posthoc-selected predictors',
            'statewise mixture is conventional conditional stacking; oracle is a within-family lower loss bound only',
            'failure of these gates does not rule out arbitrary conditional corrections or new data'],
        'freeze_sha256':sha(HERE/'FREEZE.json'),'cpu_wall_seconds':time.monotonic()-started}
    if result['cpu_wall_seconds']>600:raise ValueError('audit time budget exceeded')
    if frozen['sources']!=source_hashes() or frozen['inputs']!=inputs():raise ValueError('frozen artifacts changed during audit')
    check_original(ORIGINAL);result['original_artifacts_unchanged']=True
    private.update({'baseline_person_loss':baseline.tolist(),'original_person_loss':original.tolist(),'caps':cap_private})
    (local/'details.json').write_text(json.dumps(private,indent=2,allow_nan=False)+'\n')
    result['private_details_sha256']=sha(local/'details.json')
    (HERE/'RESULTS.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    (HERE/'RESULTS_SEAL.json').write_text(json.dumps({'results_sha256':sha(HERE/'RESULTS.json'),'freeze_sha256':sha(HERE/'FREEZE.json')},indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['freeze','run'])
    args=parser.parse_args();freeze() if args.action=='freeze' else run()
