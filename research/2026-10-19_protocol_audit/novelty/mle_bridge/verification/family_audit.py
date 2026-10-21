"""Independent ID joins and rank/selection verification for all old minimal cores.

This accepts the previously verified geometric-certificate list as input. It
does not claim a second new exhaustive proof of the waveform-rule geometry.
"""
from pathlib import Path
import argparse,hashlib,json,time
import numpy as np
from control_audit import rank_summary

HERE=Path(__file__).resolve().parent;BRIDGE=HERE.parent;PROD=BRIDGE/'production'


def check_description(given,values):
    x=np.asarray(values,float)
    assert given['minimum']==float(min(x))
    assert given['maximum']==float(max(x))
    assert given['median']==float(np.median(x))


def run():
    start=time.monotonic();record=json.load((BRIDGE/'theory/family.json').open())
    prior=json.load((PROD/'prior_obstructions.json').open())
    previous={(r['group'],r['p']):r for r in prior['groups'] if not r['compatible_cells']}
    raw=json.load((PROD/'rows.json').open());base=json.load((PROD/'baselines.json').open())['sessions']
    selected={(c['group'],c['p']):c for c in json.load((PROD/'selected.json').open())['cases']}
    n_members=n_ranks=0;cases=[];sources=[BRIDGE/'theory/family.json',PROD/'prior_obstructions.json',PROD/'rows.json',PROD/'baselines.json',PROD/'selected.json']
    for case in record['cases']:
        old=previous[case['group'],case['p']];k=old['minimum_obstruction_cardinality']
        assert k==case['k'] and k in (2,3)
        original=old['all_certifying_pairs' if k==2 else 'all_certifying_triples']
        wanted={tuple(sorted(r['ids'])):r for r in original}
        assert len(wanted)==len(original)==case['family_size']==len(case['members'])
        assert wanted.keys()=={tuple(m['ids']) for m in case['members']}
        path=PROD/'controls'/f"{case['control_job_number']:02}.npz";sources.append(path);z=np.load(path)
        rr=[r for r in raw if r['session']==case['session']];index={r['id']:i for i,r in enumerate(rr)}
        b0=base[case['session']]['free']['params'][2]
        t0=2*(base[case['session']]['fixed'][str(case['p'])]['nll']-base[case['session']]['free']['nll'])
        assert b0==case['beta_before'] and t0==case['T_before']
        beta=z['free_params'][:,2];tt=2*(z[str(case['p'])+'_nll']-z['free_nll'])
        allvals={'abs_delta_beta':abs(beta-b0),'toward_target':abs(b0-case['p'])-abs(beta-case['p']),
                 'T_reduction':t0-tt}
        assert case['control_population_size']==len(beta)
        for m in case['members']:
            ids=tuple(m['ids']);ix=sorted(index[i] for i in ids)
            assert len(ids)==k and ix==m['row_indices']
            assert all(rr[i]['group']==case['group'] and rr[i]['eligible'] for i in ix)
            found=[i for i,v in enumerate(z['combinations']) if list(v)==ix]
            assert found==[m['control_index']];j=found[0]
            assert m['numerical_only_certificate']==(wanted[ids]['invalid_cells']==0)
            assert m['beta']==[float(np.exp(rr[i]['lb'])) for i in ix]
            assert m['beta_after']==float(beta[j]) and m['T_after']==float(tt[j])
            assert m['delta_beta']==float(beta[j]-b0)
            assert m['sigma_after']==float(np.exp(z['free_params'][j,3]))
            for name,values in allvals.items():
                assert m[name]==float(values[j])
                actual=rank_summary(values,m[name]);given=m['ranks'][name]
                assert {a:actual[a] for a in ['n','below','tied','above']}=={a:given[a] for a in ['n','below','tied','above']}
                assert actual['midrank_percentile']==given['midrank'];n_ranks+=1
            n_members+=1
        members=case['members'];chosen=case['best_for_T'];maxval=max(m['T_reduction'] for m in members)
        candidates=sorted([m for m in members if abs(m['T_reduction']-maxval)<=1e-9],key=lambda m:m['ids'])
        assert chosen==candidates[0]
        assert case['best_T_gap_to_all_control_max']==float(max(allvals['T_reduction'])-chosen['T_reduction'])
        old_ids=sorted(selected[case['group'],case['p']]['ids'])
        assert case['previous_choice']['ids']==old_ids
        assert case['previous_choice']==next(m for m in members if m['ids']==old_ids)
        counts={'away_from_target':sum(m['toward_target'] < -1e-9 for m in members),
                'toward_target':sum(m['toward_target'] > 1e-9 for m in members),
                'T_increase':sum(m['T_reduction'] < -1e-9 for m in members),
                'T_decrease':sum(m['T_reduction'] > 1e-9 for m in members),
                'numerical_only':sum(m['numerical_only_certificate'] for m in members)}
        assert counts==case['counts']
        for metric in allvals:
            check_description(case['metrics'][metric],[m[metric] for m in members])
            check_description(case['metrics'][metric]['all_controls'],allvals[metric])
            check_description(case['metrics'][metric]['rank_among_all_controls'],[m['ranks'][metric]['midrank'] for m in members])
        cases.append(case)
    def summary_check(given,cs):
        assert given['n_cases']==len(cs) and given['n_certificates']==sum(c['family_size'] for c in cs)
        check_description(given['family_size'],[c['family_size'] for c in cs])
        for label,countname in [('away_from_target','away_from_target'),('T_increase','T_increase')]:
            key='away_from_target' if label=='away_from_target' else 'T_increase'
            assert given['cases_with_some_'+key]==sum(c['counts'][countname]>0 for c in cs)
            assert given['cases_with_all_'+key]==sum(c['counts'][countname]==c['family_size'] for c in cs)
        assert given['cases_with_both_toward_directions']==sum(c['counts']['away_from_target']>0 and c['counts']['toward_target']>0 for c in cs)
        assert given['cases_with_both_T_directions']==sum(c['counts']['T_increase']>0 and c['counts']['T_decrease']>0 for c in cs)
        assert given['member_direction_counts']=={k:sum(c['counts'][k] for c in cs) for k in ['away_from_target','toward_target','T_increase','T_decrease']}
        assert given['cases_best_T_reduction_positive']==sum(c['best_for_T']['T_reduction']>1e-9 for c in cs)
        assert given['cases_best_T_matches_unrestricted_control_max']==sum(c['best_T_gap_to_all_control_max']<=1e-9 for c in cs)
        assert given['cases_lex_choice_matches_best_T']==sum(abs(c['best_for_T']['T_reduction']-c['previous_choice']['T_reduction'])<=1e-9 for c in cs)
        check_description(given['best_T_improvement_over_previous'],[c['best_for_T']['T_reduction']-c['previous_choice']['T_reduction'] for c in cs])
        check_description(given['best_T_gap_to_all_control_max'],[c['best_T_gap_to_all_control_max'] for c in cs])
        check_description(given['best_T_rank_among_all_controls'],[c['best_for_T']['ranks']['T_reduction']['midrank'] for c in cs])
        check_description(given['previous_choice_T_rank_among_all_controls'],[c['previous_choice']['ranks']['T_reduction']['midrank'] for c in cs])
        assert given['best_T_rank_at_least_95_percentile']==sum(c['best_for_T']['ranks']['T_reduction']['midrank']>=.95 for c in cs)
        for metric in ['abs_delta_beta','toward_target','T_reduction']:
            check_description(given['best_T_selected_metrics'][metric],[c['best_for_T'][metric] for c in cs])
    categories={'all':cases,'classifier_specific_52':[c for c in cases if c['arbitrary_geometry_feasible']],
                'geometry_infeasible_25':[c for c in cases if not c['arbitrary_geometry_feasible']],
                'triples':[c for c in cases if c['k']==3]}
    for label,cs in categories.items():summary_check(record['summary'][label],cs)
    for p in [-2.,-1.2]:summary_check(record['summary']['by_slope'][str(p)],[c for c in cases if c['p']==p])
    assert record['counts']=={'cases':77,'minimum_pairs':7718,'minimum_triples':24}
    triples=categories['triples']
    assert len(triples)==2 and sum(c['family_size'] for c in triples)==24
    assert all(c['counts']['away_from_target']==c['family_size'] and c['counts']['T_increase']==c['family_size'] for c in triples)
    return {'passed':True,'scope':'independent join/ranks/selection against previously certified geometric lists; no new geometric proof',
            'case_families':len(cases),'minimum_certificates_joined':n_members,
            'member_rank_comparisons':n_ranks,'selected_maximizers_checked':len(cases),
            'aggregate_groups_checked':6,'both_triple_families_all_away_and_T_increase':True,
            'source_sha256':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in set(sources)},
            'seconds':time.monotonic()-start}


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=HERE/'family_audit.json');a=p.parse_args()
    out=run();a.output.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({k:v for k,v in out.items() if k!='source_sha256'},indent=2))
