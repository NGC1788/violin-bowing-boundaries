"""Timeline/leakage and baseline invariants, not golden model scores."""
import numpy as np
import pytest
import benchmark as b


@pytest.fixture
def data(monkeypatch):
    n = 8
    times = list(b.PAST) + [(.2,t) for t in b.TARGETS]
    stroke = np.tile(np.arange(n),len(times))
    d = {'stroke_index':stroke,'first':np.arange(n),'folder_names':np.array(['fixture/v1']),
         'folder':np.repeat('fixture/v1',len(stroke)),
         'window_s':np.repeat([w for w,_ in times],n),
         't':np.repeat([t for _,t in times],n),
         'beta':np.tile(np.linspace(.02,.2,n),len(times)),
         'force':np.tile(np.linspace(-.05,.5,n),len(times)),
         'speed':np.full(len(stroke),.05),
         'amplitude_std':np.full(len(stroke),2.),
         'periodicity':np.tile([.9,.4]*4,len(times)),
         'f0_hz':np.full(len(stroke),98.08),
         'flybacks_per_period':np.ones(len(stroke)),
         'second_flyback_ratio':np.full(len(stroke),.1),
         'flyback_height':np.ones(len(stroke))}
    monkeypatch.setattr(b.audit,'recover_grid',lambda x:(x['stroke_index'].copy(),{}))
    return d


def test_future_signal_and_controls_never_enter_inputs(data):
    old,_ = b.make_dataset(data)
    future = data['t'] >= b.ORIGIN
    for field in ('force','speed','beta','amplitude_std','periodicity','f0_hz',
                  'flybacks_per_period','second_flyback_ratio','flyback_height'):
        data[field][future] = 0.
    new,_ = b.make_dataset(data)
    np.testing.assert_array_equal(old['X'],new['X'])
    np.testing.assert_array_equal(old['H'],new['H'])
    assert any(np.any(old['y'][t] != new['y'][t]) for t in b.TARGETS)


def test_missing_target_fails_without_survivor_selection(data):
    keep = np.ones(len(data['t']),bool)
    keep[b.window_indices(data,.2,.3)[-1]] = False
    for k,v in list(data.items()):
        if len(v)==len(keep): data[k]=v[keep]
    with pytest.raises(ValueError,match='support'): b.make_dataset(data)


def test_duplicates_fail(data):
    idx=b.window_indices(data,.2,.25)[0]
    for k,v in list(data.items()):
        if len(v)==len(data['t']) and k != 't': data[k]=np.concatenate((v,v[idx:idx+1]))
    data['t']=np.r_[data['t'],data['t'][idx]]
    with pytest.raises(ValueError,match='support'): b.make_dataset(data)


def test_row_reordering_preserves_stroke_alignment(data):
    expected,_=b.make_dataset(data)
    order=np.random.default_rng(4).permutation(len(data['t']))
    for k,v in list(data.items()):
        if len(v)==len(order): data[k]=v[order]
    actual,_=b.make_dataset(data)
    np.testing.assert_array_equal(actual['X'],expected['X'])
    for t in b.TARGETS: np.testing.assert_array_equal(actual['y'][t],expected['y'][t])


def test_negative_force_is_not_dropped(data):
    result,_=b.make_dataset(data)
    assert len(result['X'])==8
    assert result['X'][0,0]<0


def test_source_transition_calibration_is_exact():
    h=np.array([0,0,1,1],bool); y=np.array([0,1,1,1],bool)
    np.testing.assert_allclose(b.markov_prob(h,y,h),[.5,.5,2.5/3,2.5/3])
    speed=np.array([0,0,1,1])
    np.testing.assert_allclose(b.markov_prob(h,y,h,speed,speed),[.5,.5,2.5/3,2.5/3])
    with pytest.raises(ValueError,match='support'): b.markov_prob(h,y,h,speed,np.full(4,2))


def test_identical_predictions_have_zero_paired_delta():
    yy=np.array([0,1,0,1],bool); p=np.array([.2,.6,.3,.8])
    result=b.paired(yy,p,p,np.array([0,0,1,1]))
    assert result['candidate_minus_baseline_log_loss']==0
    assert result['position_cluster_percentile95']==[0.,0.]


def test_paired_loss_matches_headline_even_at_saturated_predictions():
    y=np.array([0,1,1,0],bool)
    p0=np.array([0.,1.,1.,0.]);p1=np.array([1.,0.,.5,.5])
    delta=b.paired(y,p0,p1,np.array([0,0,1,1]))['candidate_minus_baseline_log_loss']
    assert np.isclose(delta,b.metrics(y,p1)['log_loss']-b.metrics(y,p0)['log_loss'],atol=1e-12)
