"""Export aggregate diagnostics and revision provenance; never export individuals."""
from pathlib import Path
import json
import numpy as np
from .diagnose import HERE,ORIGINAL,sha,check_original


def write(name,value):
    (HERE/name).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def main():
    check_original(ORIGINAL)
    summary=json.loads((HERE/'SUMMARY.json').read_text())
    if sha(HERE/'local/diagnostics.json')!=summary['diagnostics_sha256']:
        raise ValueError('diagnostic identity changed')
    d=json.loads((HERE/'local/diagnostics.json').read_text())
    m=json.loads((HERE/'local/mechanisms.json').read_text())
    if sha(HERE/'mechanisms.py')!=m['code_sha256']:
        raise ValueError('mechanism identity changed')
    out={'scope':m['scope'],'new_models_fitted':0,'test_labels_accessed':False,
         'diagnostics_sha256':sha(HERE/'local/diagnostics.json'),
         'mechanisms_sha256':sha(HERE/'local/mechanisms.json'),
         'mechanism_code_sha256':m['code_sha256'],'admitted_counts':m['admitted_counts'],
         'raw_audits':{'count':sum(len(x) for x in m['gain_raw_audit'].values()),
             'all_originals_match_frozen_prepared':all(a['original_matches_frozen_prepared'] for v in m['gain_raw_audit'].values() for a in v),
             'all_guards_preserved':all(a['guard_unchanged'] for v in m['gain_raw_audit'].values() for a in v),
             'all_rules_preserved':all(a['rules_unchanged'] for v in m['gain_raw_audit'].values() for a in v),
             'max_feature_action_error':max(a['feature_max_error'] for v in m['gain_raw_audit'].values() for a in v),
             'max_continuous_target_error':max(a['target_max_error'] for v in m['gain_raw_audit'].values() for a in v)},
         'offset_geometry':d['offset_geometry'],
         'feature_excursions':{part:{k:v for k,v in d['feature_shift'][part].items() if k!='people'} for part in ['train','dev']},
         'confidence_tails':{},'gain_sensitivity':{},'gain_associations':m['associations'],
         'caveats':['posthoc, train/dev only','all337dev rows admitted at both gains; no realized dev admission-subset mismatch',
             'future support verifies diagnostic transformation only, forbidden as an inference gate',
             'gain action verified on16audits, not continuous observer recomputation on every row',
             'no new predictor, causal claim or improvement certificate']}
    for method in ['mlp_ind','mlp_joint','mlp_offset']:
        v=d['models'][method]['dev_mean']
        out['confidence_tails'][method]={'dev_wrong_95_fraction':v['wrong_95_fraction'],
            'dev_seed_mean_loss_share':v['wrong_95_loss_share'],
            'dev_aggregate_loss_share':v['wrong_95_loss_contribution']/v['log_loss'],
            'largest_person_share_of_total_excess':v['largest_person_share_of_total_excess']}
        out['gain_sensitivity'][method]={gain:float(np.mean([p['gain_sensitivity'][method][gain]['mean_absolute_probability_change'] for p in m['dev_people']])) for gain in ['0.5','2.0']}
    write('MECHANISM_SUMMARY.json',out)
    initial=json.loads((HERE/'local/diagnostics_initial.json').read_text())
    boundary_count=0;max_change=0.
    for method in ['mlp_ind','mlp_joint','mlp_offset']:
        for part in ['train','dev']:
            for old,new in zip(initial['models'][method][part],d['models'][method][part]):
                boundary_count+=new['probability_exact_clip_boundary_count']
                max_change=max(max_change,max(abs(value-new[key]) for key,value in old.items() if isinstance(value,(int,float))))
    write('DIAGNOSTIC_REVISION.json',{
        'initial_diagnosis_code_sha256':sha(HERE/'local/diagnose_initial.py'),
        'initial_diagnostics_sha256':sha(HERE/'local/diagnostics_initial.json'),
        'initial_summary_sha256':sha(HERE/'local/SUMMARY_initial.json'),
        'current_diagnosis_code_sha256':sha(HERE/'diagnose.py'),
        'current_diagnostics_sha256':sha(HERE/'local/diagnostics.json'),
        'exact_clip_boundary_count_across_neural_train_dev':boundary_count,
        'max_original_scalar_change_after_boundary_fix':max_change,
        'reasons_for_forward_only_rechecks':[
            'independent review found softening one-sided derivative omitted exact clip boundaries; corrected and tested',
            'strengthened inventory/PCM, original-X/prefix-state/target/label identity checks for gain audits',
            'prefix invalid state explicitly uses prefix=True',
            'algebra checks use actual clipped BCE and interior-baseline no-go assumption'],
        'initial_snapshots_preserved_privately':True,'new_model_fits':0,'old_test_reopened':False,
        'review':'separate AI source/math review approved corrected revision; no human or new empirical confirmation'})
    data=json.loads((ORIGINAL/'data_freeze_manifest.json').read_text());root=ORIGINAL.parents[1]
    for name,expected in data['source_hashes'].items():
        if sha(root/name)!=expected:raise ValueError('old frozen data source changed')
    artifacts=['AGENTS.md','PLAN.json','MECHANISM_PLAN.json','SUMMARY.json','MECHANISM_SUMMARY.json',
               'RISK_FEASIBILITY.json','DIAGNOSTIC_REVISION.json','CAUSE_ANALYSIS.md','NEXT_METHOD.md',
               'LITERATURE_RESIDUAL_REVIEW.md','REVIEW.json','diagnose.py','mechanisms.py','risk_feasibility.py',
               'export_summary.py','test_diagnose.py','test_mechanisms.py','test_risk_feasibility.py']
    write('FINAL_AUDIT.json',{'scope':'posthoc diagnosis, not training or test evaluation',
        'verdict':'DIAGNOSIS_COMPLETE_NEW_METHOD_NOT_ESTABLISHED',
        'new_models_fitted':0,'test_labels_reopened':False,'max_sealed_dev_reproduction_error':summary['max_dev_reproduction_error'],
        'old_data_sources_verified':len(data['source_hashes']),
        'old_result_sha256':sha(ORIGINAL/'RESULTS.json'),
        'old_artifacts_verified_unchanged':True,'synthetic_tests_passed':9,
        'source_math_review':'approved by separate AI after listed corrections; not external empirical validation',
        'next_training_authorized_by_this_audit':False,
        'artifact_sha256':{name:sha(HERE/name) for name in artifacts}})
    print(json.dumps({'exported':['MECHANISM_SUMMARY.json','DIAGNOSTIC_REVISION.json','FINAL_AUDIT.json'],
                      'old_data_sources_verified':len(data['source_hashes']),'max_original_scalar_change':max_change}))


if __name__=='__main__':main()
