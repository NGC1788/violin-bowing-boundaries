"""Synthetic-only model/evaluator checks; no historical C/D evaluation."""
from pathlib import Path
import importlib.util
import json
import sys
import tempfile
import numpy as np
import pytest

# Resolve the package relative to this file, independent of pytest's cwd.
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from forecast_v3 import models as M, experiment as E, rule_patterns as R


def synthetic_dataset(n_train=40,n_dev=20):
    rng=np.random.default_rng(21);n=n_train+n_dev
    ids=np.arange(n)%10;prefix=(ids+1)%10
    x=rng.normal(size=(n,6)).astype(np.float32);x[::7,2]=np.nan
    # Prefix missingness is allowed, but preprocessing must learn only source statistics.
    groups=np.where(np.arange(n)<n_train,'A','C')
    return {'X':x,'Y9':R.admissible_patterns()[ids],'pattern_ids':ids,
            'prefix_Y9':R.admissible_patterns()[prefix],
            'ids':np.array([f'stroke-{i}' for i in range(n)]),
            'sessions':np.array([g+'1' for g in groups]),'groups':groups,
            'columns':np.array([f'{g}-col-{i//5}' for i,g in enumerate(groups)]),
            'split':np.where(np.arange(n)<n_train,'train','dev'),
            'feature_names':[f'prefix_{i}' for i in range(6)],
            'metadata':{'synthetic':True,'window':{'prefix_end':.2,'target_start':.25}}}


def test_preprocessing_uses_source_statistics_and_does_not_update_on_transform():
    source=np.array([[1,np.nan,np.nan],[3,4,np.nan],[5,8,np.nan]],float)
    transform=M.SourcePreprocessor().fit(source)
    np.testing.assert_array_equal(transform.median_,[3,6,0])
    before=[a.copy() for a in (transform.median_,transform.mean_,transform.scale_)]
    result=transform.transform([[1000000,np.nan,-1000000],[np.inf,4,np.nan]])
    assert np.isfinite(result).all()
    for actual,expected in zip((transform.median_,transform.mean_,transform.scale_),before):
        np.testing.assert_array_equal(actual,expected)
    np.testing.assert_array_equal(transform.transform([[3,6,0]]),np.zeros((1,3)))
    assert transform.n_source_rows_==3
    with pytest.raises(ValueError):M.SourcePreprocessor().transform(source)


def test_latent_coordinates_invert_every_pattern_and_random_mixture():
    patterns=R.admissible_patterns();np.testing.assert_allclose(M.signed_pattern_masses(patterns),np.eye(10),atol=1e-14)
    weights=np.random.default_rng(0).dirichlet(np.ones(10),200)
    p=weights@patterns
    np.testing.assert_allclose(M.signed_pattern_masses(p),weights,atol=1e-14)
    projected,masses=M.project_marginals(p)
    np.testing.assert_allclose(projected,p,atol=1e-14)
    np.testing.assert_allclose(masses,weights,atol=1e-14)


def test_projection_satisfies_simplex_kkt_and_restores_coherence_without_labels():
    rng=np.random.default_rng(5);v=rng.normal(size=(300,10));q=M.simplex_projection(v)
    np.testing.assert_allclose(q.sum(1),1,atol=1e-14);assert (q>=0).all()
    for a,b in zip(v,q):
        theta=(a-b)[b>1e-12]
        np.testing.assert_allclose(theta,theta[0],atol=1e-14)
        assert np.all(a[b<=1e-12]-theta[0]<=1e-12)
    p=rng.random((300,9));assert M.coherence(p)['incoherent_rows']>0
    projected,_=M.project_marginals(p)
    assert M.coherence(projected)['incoherent_rows']==0
    assert M.coherence(projected)['order_violation_rows']==0


def test_neural_backbone_initialization_is_identical_between_heads():
    torch=pytest.importorskip('torch')
    independent=M.make_network(6,32,9,2);joint=M.make_network(6,32,10,2)
    for index in (0,2):
        assert torch.equal(independent[index].weight,joint[index].weight)
        assert torch.equal(independent[index].bias,joint[index].bias)


def test_neural_training_matches_batches_updates_and_loss_definition():
    pytest.importorskip('torch');d=synthetic_dataset();train=d['split']=='train';dev=~train
    common={'seed':1,'epochs':2,'hidden':8,'batch_size':16}
    a=M.fit_predict(d['X'][train],d['Y9'][train],d['X'][dev],{**common,'method':'mlp_ind'})
    b=M.fit_predict(d['X'][train],d['Y9'][train],d['X'][dev],{**common,'method':'mlp_joint'})
    assert a.info['optimizer_updates']==b.info['optimizer_updates']==6
    assert a.info['minibatch_order_sha256']==b.info['minibatch_order_sha256']
    assert a.info['loss']==b.info['loss']=='mean marginal binary cross-entropy across nine rules'
    assert a.info['source_training_sha256']==b.info['source_training_sha256']
    assert np.isfinite(a.info['train_loss_by_epoch']).all() and np.isfinite(b.info['train_loss_by_epoch']).all()
    assert b.pattern_probabilities.shape==(20,10)
    np.testing.assert_allclose(b.pattern_probabilities.sum(1),1,atol=1e-14)
    assert M.coherence(b.probabilities)['incoherent_rows']==0
    repeated=M.fit_predict(d['X'][train],d['Y9'][train],d['X'][dev],{**common,'method':'mlp_joint'})
    np.testing.assert_array_equal(b.probabilities,repeated.probabilities)


def test_source_frequency_and_persistence_are_exact_and_do_not_use_eval_targets():
    d=synthetic_dataset();train=d['split']=='train';dev=~train
    result=M.fit_predict(d['X'][train],d['Y9'][train],d['X'][dev],{'method':'source_frequency'})
    expected=(d['Y9'][train].sum(0)+.5)/(train.sum()+1)
    np.testing.assert_allclose(result.probabilities,np.tile(expected,(dev.sum(),1)),atol=1e-14)
    result=M.fit_predict(d['X'][train],d['Y9'][train],d['X'][dev],{'method':'persistence'},eval_prefix_labels=d['prefix_Y9'][dev])
    np.testing.assert_array_equal(result.probabilities,d['prefix_Y9'][dev])
    with pytest.raises(TypeError):M.fit_predict(d['X'][train],d['Y9'][train],d['X'][dev],{'method':'persistence'},eval_y=d['Y9'][dev])


def test_hgb_handles_constant_source_labels_without_access_to_dev_labels():
    d=synthetic_dataset();train=d['split']=='train';dev=~train
    y=np.tile(R.admissible_patterns()[4],(train.sum(),1))
    result=M.fit_predict(d['X'][train],y,d['X'][dev],{'method':'hgb_history','hgb_max_iter':2})
    expected=(y.sum(0)+.5)/(len(y)+1)
    np.testing.assert_allclose(result.probabilities,np.tile(expected,(dev.sum(),1)))
    assert result.info['hgb_early_stopping'] is False


def test_primary_score_is_exact_mean_nine_rule_log_loss_and_raw_brier():
    y=R.admissible_patterns()[[0,9]];p=np.array([[.1]*9,[.8]*9])
    metrics=E.probability_metrics(y,p)
    assert metrics['mean_log_loss']==pytest.approx(( -np.log(.9)-np.log(.8))/2)
    assert metrics['mean_brier']==pytest.approx((.1**2+.2**2)/2)
    saturated=E.probability_metrics(y,y)
    assert saturated['mean_log_loss']==pytest.approx(-np.log1p(-1e-6))
    assert saturated['mean_brier']==0
    with pytest.raises(ValueError):E.probability_metrics(y,np.full((2,9),np.nan))


def test_evaluator_preserves_all_rows_artifacts_and_rejects_overwrite(tmp_path):
    d=synthetic_dataset();m=E.evaluate({'method':'source_frequency'},d,tmp_path/'run')
    assert m['source_rows']==40 and m['development_rows']==20
    assert m['verification_ok'] and m['verification_error_count']==0
    assert m['development_groups']==['C'] and m['development_sessions']==['C1']
    assert m['target'].startswith('future output')
    with np.load(tmp_path/'run/predictions.npz',allow_pickle=False) as saved:
        np.testing.assert_array_equal(saved['ids'],d['ids'][40:])
        np.testing.assert_array_equal(saved['targets'],d['Y9'][40:])
    assert (tmp_path/'run/model.npz').exists()
    with pytest.raises(FileExistsError):E.evaluate({'method':'source_frequency'},d,tmp_path/'run')
    changed={**d,'Y9':d['Y9'].copy(),'pattern_ids':d['pattern_ids'].copy()}
    changed['Y9'][40:]=R.admissible_patterns()[0];changed['pattern_ids'][40:]=0
    other=E.evaluate({'method':'source_frequency'},changed,tmp_path/'other')
    assert m['model']['source_training_sha256']==other['model']['source_training_sha256']
    assert m['mean_log_loss']!=other['mean_log_loss']
    with np.load(tmp_path/'run/predictions.npz') as a,np.load(tmp_path/'other/predictions.npz') as b:
        np.testing.assert_array_equal(a['probabilities'],b['probabilities'])


def test_neural_artifacts_and_prespecified_projection_are_saved(tmp_path):
    pytest.importorskip('torch');d=synthetic_dataset()
    m=E.evaluate({'method':'mlp_ind','epochs':1,'hidden':8,'batch_size':16},d,tmp_path/'ind')
    assert (tmp_path/'ind/model.pt').exists() and (tmp_path/'ind/preprocessor.npz').exists()
    assert m['auxiliary']['latent_mass_projection']['coherence']['incoherent_rows']==0
    assert 'NOT marginal-space optimal' in m['auxiliary']['latent_mass_projection']['definition']
    assert m['auxiliary']['forward_kl_coherent_repair']['mean_log_loss']<=m['mean_log_loss']
    assert m['auxiliary']['forward_kl_coherent_repair']['repair_audit']['max_certified_delta']<=0
    with np.load(tmp_path/'ind/predictions.npz') as saved:
        assert saved['projected_probabilities'].shape==(20,9)


@pytest.mark.parametrize('problem',['ids','groups','pattern','prefix','split'])
def test_evaluator_rejects_duplicate_leaking_or_misaligned_data(problem,tmp_path):
    d=synthetic_dataset()
    if problem=='ids':d['ids'][1]=d['ids'][0]
    elif problem=='groups':d['groups'][:]='same'
    elif problem=='pattern':d['pattern_ids'][0]=3
    elif problem=='prefix':d.pop('prefix_Y9')
    else:d['split'][:]='train'
    with pytest.raises(ValueError):E.evaluate({'method':'source_frequency'},d,tmp_path/problem)


def test_source_markov_uses_only_source_transitions_and_absent_row_fallback(tmp_path):
    patterns=R.admissible_patterns()
    train_prefix=patterns[[0,0,0,4,4]]
    train_y=patterns[[1,1,2,4,9]]
    eval_prefix=patterns[[0,4,7]]
    result=M.fit_predict(np.zeros((5,2)),train_y,np.zeros((3,2)),{'method':'source_markov'},
                         train_prefix_labels=train_prefix,eval_prefix_labels=eval_prefix,output_dir=tmp_path)
    expected0=np.full(10,.1);expected0[1]+=2;expected0[2]+=1;expected0/=4
    expected4=np.full(10,.1);expected4[4]+=1;expected4[9]+=1;expected4/=3
    fallback=(np.bincount([1,1,2,4,9],minlength=10)+.1)/6
    expected=np.array([expected0,expected4,fallback])
    np.testing.assert_allclose(result.pattern_probabilities,expected,atol=1e-14)
    np.testing.assert_allclose(result.probabilities,expected@patterns,atol=1e-14)
    assert result.info['learned_from_source'] and result.info['dirichlet_total_mass']==1.
    assert M.coherence(result.probabilities)['incoherent_rows']==0
    with np.load(tmp_path/'model.npz') as artifact:
        assert artifact['source_transition_counts'].sum()==5
    with pytest.raises(ValueError):M.fit_predict(np.zeros((5,2)),train_y,np.zeros((3,2)),{'method':'source_markov'},eval_prefix_labels=eval_prefix)
