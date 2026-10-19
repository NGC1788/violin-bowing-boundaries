"""Synthetic-only checks. Never opens target features, caches, archives or network."""
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pytest
from scipy.stats import norm

spec=importlib.util.spec_from_file_location("confirm_d_t2",Path(__file__).with_name("confirm_d_t2.py"))
C=importlib.util.module_from_spec(spec);sys.modules[spec.name]=C;spec.loader.exec_module(C)


def parameters():
    # Frozen model constants are permitted; this file contains no D_T2 outcome.
    return json.loads((C.BASE/"astra_20261019/d_t2_freeze/frozen_parameters.json").read_text())


def test_import_selected_function_does_not_execute_top_level(tmp_path):
    src=tmp_path/"danger.py"
    src.write_text('raise RuntimeError("must not execute")\ndef f(x):\n    return np.asarray(x)+1\n')
    np.testing.assert_array_equal(C.selected_function(src,"f")([1,2]),[2,3])


def test_stable_likelihood_matches_moderate_cdf_and_handles_extreme_tails():
    lo=np.array([-np.inf,-1.,0.,5.,30.]);hi=np.array([0.,1.,np.inf,6.,31.])
    actual=C.log_interval_probability(lo,hi,0.,1.)
    expected=np.log(norm.cdf(hi[:4])-norm.cdf(lo[:4]))
    np.testing.assert_allclose(actual[:4],expected,atol=1e-8)
    assert np.isfinite(actual[4]) and actual[4]<-450


@pytest.mark.parametrize('iv,reason', [
    ([(0.,np.inf),(1.,np.inf)],'one_sided'),
    ([(-np.inf,0.),(-np.inf,1.)],'one_sided'),
    ([(0.,1.),(.2,.8)],'sigma_collapse'),
    ([(np.nan,np.nan)],'no_available'),
    ([(0.,0.),(1.,2.)],'zero_width'),
])
def test_nonidentified_levels_refuse_instead_of_fabricating(iv,reason):
    r=C.fit_level(iv,np.zeros(len(iv)))
    assert not r['success'] and reason in r['reason']


def test_single_start_and_optimizer_failure_are_recorded(monkeypatch):
    class Result:
        success=False;x=np.array([0.,-.7]);fun=5.;nit=400;message='iteration limit'
    calls=[]
    def fake(fun,start,**kwargs):
        calls.append((start,kwargs));return Result()
    monkeypatch.setattr(C,'minimize',fake)
    r=C.fit_level([(-1.,-.9),(.9,1.)],[0.,0.])
    assert len(calls)==1 and calls[0][0]==[0.,np.log(.5)]
    assert calls[0][1]=={'method':'Nelder-Mead'}
    assert not r['success'] and r['reason']=='optimizer_failed'


def test_identifiable_location_shift_and_sigma_are_finite():
    centers=np.array([-.8,-.4,-.1,.2,.5,.9])+0.3
    iv=np.c_[centers-.05,centers+.05]
    a=C.fit_level(iv,np.zeros(len(iv)))
    b=C.fit_level(iv,np.full(len(iv),.7))
    assert a['success'] and b['success']
    assert abs(a['level']-b['level']-.7)<2e-4
    assert a['sigma']>0


def rows40():
    return [{'folder':f'synthetic_speed{f}','column':c} for f in range(3) for c in range(40)]


def test_2000_bootstrap_draws_preserve_beta_clusters_and_seed():
    rows=rows40();draws=list(C.cluster_resamples(rows))
    assert len(draws)==2000
    for idx in draws:
        assert len(idx)==120
        multiplicity={c:[sum(rows[i]['column']==c and rows[i]['folder']==f'synthetic_speed{f}' for i in idx) for f in range(3)] for c in range(40)}
        assert all(len(set(n))==1 for n in multiplicity.values())
    again=next(C.cluster_resamples(rows))
    np.testing.assert_array_equal(draws[0],again)
    with pytest.raises(C.ConfirmationError):next(C.cluster_resamples(rows[:-1]))


def test_fixed_prediction_equations_include_empirical_intercept():
    p=parameters();r=[{'lv':np.log(.1),'lb':np.log(.08)}]
    pred=C.frozen_predictions(p,{'dec5_pl_N':-.08},r)
    e=p['upper_empirical_reference']
    expected=p['upper_empirical_log_c']+e['c']+e['a_v']*r[0]['lv']+e['b_beta']*r[0]['lb']+np.log(1.2021)
    assert pred['upper_empirical_secondary'][0]==expected
    assert np.isclose(pred['upper_strict'][0]-pred['upper_strict_no_Z'][0],np.log(1.2021))
    assert not pred['descriptor_extrapolation']
    assert C.frozen_predictions(p,{'dec5_pl_N':-.5},r)['descriptor_extrapolation']


def test_verdicts_use_only_fixed_margins_and_failure_not_bootstrap_rescue():
    pred={'lower_levels':{'protocol':0.,'log_Z_EI':.5,'mean':.3},'descriptor_extrapolation':False,'Z':1.2021}
    th={'lower_tolerance':.23,'upper_tolerance':.12,'lower_comparison_margin':.23,'upper_comparison_margin':.12}
    fits={'lower':{'success':True,'level':.1},'upper_strict':{'success':True,'level':.02}}
    d=C.decisions(fits,pred,th)
    assert '지지' in d['lower']['verdict'] and d['upper_strict']['verdict']=='허용오차 안'
    fits['lower']['level']=.5
    assert '반증' in C.decisions(fits,pred,th)['lower']['verdict']
    fits['lower']={'success':False,'reason':'one_sided'}
    assert C.decisions(fits,pred,th)['lower']['verdict']=='검증 불가'


def test_boundary_scoring_censors_and_nan_are_not_false_hits():
    s=C.boundary_score([(0.,1.),(-np.inf,0.),(1.,np.inf),(np.nan,np.nan)],[.5,-1.,np.nan,.1])
    assert s['hits']==2 and s['counts']['prediction_failures']==1
    assert s['counts']['right_censored']==1 and s['counts']['unavailable']==1
    assert s['n_finite_errors']==1 and s['median_abs_log_error']==0


def synthetic_controls():
    rec=[]
    for j,v in enumerate((.05,.1,.2)):
        folder=f'synthetic/{C.STEM}/v{j+1}'
        for trial in range(1,2001):
            col,rank=divmod(trial-1,50)
            beta=float(np.geomspace(.02,.2,40)[col]);force=.1*40**(rank/49)
            for window,t in ((.1,0.),(.1,1.),(.2,0.),(.2,.1),(.2,.2),(.2,.3)):
                f=force-(.05 if t>0 else 0.)
                rec.append((folder,trial,window,t,f,v,beta,''))
    return {key:np.array([r[i] for r in rec]) for i,key in enumerate(C.CONTROL_FIELDS)}


@pytest.fixture
def fixture(tmp_path,monkeypatch):
    params=parameters();d=synthetic_controls()
    source=tmp_path/'synthetic.csv';source.write_text('opaque synthetic bytes; no actual D_T2 data\n')
    deps={}
    texts={
        'descriptor':(C.BASE/'astra_20261019/descriptor_only.py').read_text(),
        'columns':(C.BASE/'claude_extra/verify_astra_boundaries.py').read_text(),
        'classifier':'raise RuntimeError("top-level must not run")\ndef classify(amplitude, periodicity, f0, flybacks, reference_f0, thresholds):\n    return "helmholtz" if amplitude == 1. else "multiple_slip"\n',
        'thresholds':json.dumps({'thresholds_applied':{'synthetic':True}}),
        'runner':Path(C.__file__).read_text()}
    for name,text in texts.items():
        path=tmp_path/(name+'.txt');path.write_text(text)
        deps[name]={'path':str(path),'sha256':C.sha(path)}
    monkeypatch.setattr(C,'verify_original_freeze',lambda base: (params,{}))
    monkeypatch.setattr(C,'runtime_dependencies',lambda base,repo:deps)
    reads=[]
    def reader(path,fields):
        assert Path(path)==source
        reads.append(tuple(fields))
        data={k:v.copy() for k,v in d.items()}
        if set(fields)&set(C.OUTCOME_FIELDS):
            data.update(amplitude_std=((data['force']>.4)&(data['force']<2.)).astype(float),
                        periodicity=np.full(len(data['t']),.95),f0_hz=np.full(len(data['t']),98.08),
                        flybacks_per_period=np.ones(len(data['t'])))
        return {k:data[k] for k in fields}
    monkeypatch.setattr(C,'read_columns',reader)
    return source,d,reads,deps


def test_controls_freeze_never_requests_H_or_outcome_and_is_exclusive(fixture,tmp_path):
    source,d,reads,deps=fixture;out=tmp_path/'controls.json'
    receipt=C.freeze_controls(source,out)
    assert reads==[C.CONTROL_FIELDS] and receipt['outcomes_parsed'] is False
    assert receipt['inventory']['n_strokes']==6000 and len(receipt['rows'])==120
    assert C.seal_path(out).exists()
    with pytest.raises(FileExistsError):C.freeze_controls(source,out)


def test_missing_duplicate_error_and_identity_controls_are_rejected(fixture):
    _,d,_,deps=fixture
    cols=C.selected_function(deps['columns']['path'],'columns')
    bad={k:v.copy() for k,v in d.items()};bad['error'][0]='X'
    with pytest.raises(C.ConfirmationError,match='error rows'):C.control_inventory(bad,cols)
    bad={k:v[1:].copy() for k,v in d.items()}
    # First row is descriptor-only; primary inventory still complete. Remove primary snapshot.
    bad={k:np.delete(v,2) for k,v in d.items()}
    with pytest.raises(C.ConfirmationError,match='Missing'):C.control_inventory(bad,cols)
    bad={k:np.r_[v,v[2:3]] for k,v in d.items()}
    with pytest.raises(C.ConfirmationError,match='Duplicate'):C.control_inventory(bad,cols)


def test_hash_tampering_is_rejected_before_any_outcome_read(fixture,tmp_path):
    source,_,reads,_=fixture;out=tmp_path/'controls.json'
    C.freeze_controls(source,out);source.write_text('changed synthetic bytes')
    with pytest.raises(C.ConfirmationError,match='input changed'):C.score(out,tmp_path/'results.json')
    assert reads==[C.CONTROL_FIELDS]


def test_receipt_tampering_is_rejected_before_outcome_read(fixture,tmp_path):
    source,_,reads,_=fixture;out=tmp_path/'controls.json'
    C.freeze_controls(source,out)
    value=json.loads(out.read_text());value['predictions']['lower_levels']['protocol']=999
    out.write_text(C.encode(value))
    with pytest.raises(C.ConfirmationError,match='sealing'):C.score(out,tmp_path/'results.json')
    assert reads==[C.CONTROL_FIELDS]


def test_two_stage_synthetic_scoring_and_failure_receipt(fixture,tmp_path,monkeypatch):
    source,_,reads,_=fixture;out=tmp_path/'controls.json';result=tmp_path/'results.json'
    C.freeze_controls(source,out)
    original=C.calculate_scores
    monkeypatch.setattr(C,'calculate_scores',lambda o,r,p,pa:original(o,r,p,pa,n_boot=2))
    value=C.score(out,result)
    assert value['stage']=='score_complete' and value['n_primary_windows']==6000
    assert reads==[C.CONTROL_FIELDS,C.CONTROL_FIELDS+C.OUTCOME_FIELDS]
    assert value['bootstrap']['lower']['requested']==2
    with pytest.raises(FileExistsError):C.score(out,result)
    with pytest.raises(FileExistsError):C.score(out,tmp_path/'different_name.json')
    # Tests only: a separate synthetic receipt to simulate first-access failure.
    second=tmp_path/'second_synthetic_controls.json'
    C.freeze_controls(source,second)
    def broken(*args,**kwargs):raise C.ConfirmationError('synthetic classifier failure')
    monkeypatch.setattr(C,'observed_boundaries',broken)
    fail=tmp_path/'failed.json'
    with pytest.raises(C.ConfirmationError):C.score(second,fail)
    assert json.loads(fail.read_text())['stage']=='score_failed'


def test_original_freeze_hashes_are_verified_without_data_access(tmp_path):
    b=tmp_path/'astra_20261019/d_t2_freeze';b.mkdir(parents=True)
    (tmp_path/'final').mkdir();plan=tmp_path/'final/CONFIRM_PLAN_D_T2.md';plan.write_text('synthetic frozen plan')
    source=tmp_path/'source.py';source.write_text('original source')
    params=b/'frozen_parameters.json';params.write_text(json.dumps({'source_sha256':{'source.py':C.sha(source)}}))
    (b/'freeze_receipt.json').write_text(json.dumps({'plan_sha256':C.sha(plan),'parameters_sha256':C.sha(params)}))
    C.verify_original_freeze(tmp_path)
    source.write_text('changed source')
    with pytest.raises(C.ConfirmationError,match='dependency hash'):C.verify_original_freeze(tmp_path)
    plan.write_text('changed plan')
    with pytest.raises(C.ConfirmationError,match='plan or parameters'):C.verify_original_freeze(tmp_path)


def test_actual_csv_reader_requests_controls_only(tmp_path):
    p=tmp_path/'synthetic_controls_only_check.csv'
    p.write_text('folder,trial,window_s,t,force,speed,beta,error,amplitude_std,H\nsynthetic,1,0.1,0,1,0.1,0.08,,must_not_parse_as_number,not_a_boolean\n')
    d=C.read_columns(p,C.CONTROL_FIELDS)
    assert set(d)==set(C.CONTROL_FIELDS) and d['force'][0]==1
