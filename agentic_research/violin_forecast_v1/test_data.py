"""Synthetic data-boundary tests only: no MAT data or held-out targets accessed."""
from pathlib import Path
import sys
import copy
import itertools
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from agentic_research.violin_forecast_v1 import data as D


def participants():
    return [dict(participant_id=f'{g}_{i+1:03d}',cohort=g) for g,counts in D.SPLIT_COUNTS.items() for i in range(sum(counts))]


def records():
    rr=[]
    for i,p in enumerate(participants()):
        rng=np.random.default_rng(i)
        a=rng.normal(size=(3,2200))*np.array([.1+i*.003,2+i*.01,.01])[:,None]
        rr.append({**p,'source_id':p['participant_id']+'/pre/0',
                   'channels':dict(zip(D.CHANNELS,a))})
    return rr


def test_exact_stratified_split_is_content_order_independent():
    pp=participants();a=D.participant_split(pp);b=D.participant_split(pp[::-1]);assert a==b
    assert {s:list(a.values()).count(s) for s in D.SPLITS}=={'train':49,'dev':10,'test':13}
    for g,counts in D.SPLIT_COUNTS.items():
        assert tuple(sum(a[p['participant_id']]==s for p in pp if p['cohort']==g) for s in D.SPLITS)==counts


def test_future_mutation_keeps_inputs_bitwise_unchanged():
    rr=records();original=D.sequence_features(rr[0]['channels']);target=D.sequence_target(rr[0]['channels'])
    changed=copy.deepcopy(rr[0]['channels'])
    for channel in D.CHANNELS:changed[channel][D.PREFIX_END:]*=30
    after=D.sequence_features(changed)
    assert original[0].tobytes()==after[0].tobytes()
    assert original[1].tobytes()==after[1].tobytes()
    assert not np.array_equal(target,D.sequence_target(changed))


def test_prefix_mutation_cannot_change_target():
    raw=records()[0]['channels'];changed=copy.deepcopy(raw)
    for v in changed.values():v[:1024]*=2
    np.testing.assert_array_equal(D.sequence_target(raw),D.sequence_target(changed))
    assert not np.array_equal(D.sequence_features(raw)[0],D.sequence_features(changed)[0])


def test_heldout_participants_cannot_change_fitted_thresholds():
    rr=records();mapping=D.participant_split(participants());a=D.assemble_dataset(rr,mapping)
    changed=copy.deepcopy(rr)
    for r in changed:
        if mapping[r['participant_id']]!='train':
            for v in r['channels'].values():v[D.TARGET_START:D.TARGET_END]*=1000
    b=D.assemble_dataset(changed,mapping)
    np.testing.assert_array_equal(a['thresholds'],b['thresholds'])
    np.testing.assert_array_equal(a['X'],b['X'])
    fit=set(a['metadata']['threshold_fit']['training_participant_ids'])
    assert fit=={pid for pid,s in mapping.items() if s=='train'}
    assert not fit.intersection(a['participant_ids'][a['split']=='test'])
    # Even unusable held-out values must never be inspected by the fitting function.
    y=a['targets'].copy();y[a['split']!='train']=np.nan
    q,_=D.fit_thresholds(y,a['split'],a['participant_ids']);np.testing.assert_array_equal(q,a['thresholds'])


def test_exact_aliases_deduplicated_and_cross_participant_copy_rejected():
    rr=records();alias=copy.deepcopy(rr[0]);alias['source_id']+='alias'
    unique=D.deduplicate(rr+[alias]);assert len(unique)==72
    assert sum(len(r['aliases']) for r in unique)==73
    alias['participant_id']=rr[1]['participant_id']
    with pytest.raises(ValueError,match='crosses participant'):D.deduplicate(rr+[alias])


def test_target_rules_have_ten_suffix_rectangle_patterns_and_include_equality():
    thresholds=np.array([[1.,2.,3.],[1.,2.,3.]])
    targets=np.array(list(itertools.product([.5,1.5,2.5,3.5],repeat=2)))
    labels,ids=D.labels_and_patterns(targets,thresholds)
    assert len(np.unique(ids))==10 and D.admissible_patterns().shape==(10,9)
    for x,row in zip(targets,labels):
        np.testing.assert_array_equal(row,[int(x[0]<=a and x[1]<=b) for a in thresholds[0] for b in thresholds[1]])
    y,_=D.labels_and_patterns([[2.,2.]],thresholds)
    np.testing.assert_array_equal(y.reshape(3,3),[[0,0,0],[0,1,1],[0,1,1]])
    invalid=sum(not any(np.array_equal(row,p) for p in D.admissible_patterns()) for row in itertools.product((0,1),repeat=9))
    assert invalid==502


def test_no_feature_identity_or_full_duration_and_expected_shapes():
    rr=records();mapping=D.participant_split(participants());d=D.assemble_dataset(rr,mapping)
    assert d['X'].shape==(72,56) and d['targets'].shape==(72,2)
    np.testing.assert_array_equal(d['X'][:,-2:],d['prefix_targets'])
    raw=rr[0]['channels'];extended={k:np.r_[v,np.arange(700)] for k,v in raw.items()}
    np.testing.assert_array_equal(D.sequence_features(raw)[0],D.sequence_features(extended)[0])
    assert all(not any(word in name for word in ('participant','cohort','duration','trial')) for name in d['feature_names'])


def test_repeated_quantiles_fail_closed_and_mixed_split_participant_rejected():
    with pytest.raises(ValueError,match='Repeated'):D.fit_thresholds(np.ones((3,2)),['train']*3,['a','b','c'])
    with pytest.raises(ValueError,match='crosses splits'):D.fit_thresholds(np.arange(6).reshape(3,2),['train','dev','test'],['a','a','b'])


def test_private_cache_roundtrip_and_corruption(tmp_path):
    d=D.assemble_dataset(records(),D.participant_split(participants()))
    p=D.save_prepared_dataset(d,tmp_path/'private.npz');z=D.load_prepared_dataset(p)
    for name in ('X','targets','thresholds','Y9','participant_ids','split'):np.testing.assert_array_equal(d[name],z[name])
    assert d['alias_manifest']==z['alias_manifest']
    with p.open('ab') as f:f.write(b'changed')
    with pytest.raises(ValueError,match='hash mismatch'):D.load_prepared_dataset(p)


def test_actual_build_requires_explicit_agreed_scope_before_reading_source(tmp_path):
    with pytest.raises(ValueError,match='approval'):D.build_dataset(tmp_path,{})
    with pytest.raises(ValueError,match='timing'):D.build_dataset(tmp_path,{'data_build_approved':True,'timing_scope':'online_causal'})
