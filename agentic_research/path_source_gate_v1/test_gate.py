import datetime
import json
import numpy as np
import pytest
from .core import (prefix_inputs, future_labels, reconstruct, build_catalogue,
                   catalogue_counts, mutation_ids, budget_check)
from .runner import input_paths, reserve_attempt, verify_discovery_chain, sha


def waveform(sr=8000):
    t = np.arange(round(sr*1.9))/sr
    return (.2*np.sin(2*np.pi*440*t) + .03*np.sin(2*np.pi*880*t)).astype(float)


def test_future_values_and_total_length_cannot_change_prefix_inputs():
    sr=8000; wave=waveform(sr)
    x, seq=prefix_inputs(wave,sr,'DOUBLE')
    assert x.shape==(60,) and seq.shape==(96,4)
    for changed in [wave.copy(), wave[:sr].copy(), np.pad(wave,(0,3000),constant_values=np.nan)]:
        if len(changed)>sr: changed[sr:]=np.nan
        xx,ss=prefix_inputs(changed,sr,'DOUBLE')
        np.testing.assert_array_equal(x,xx);np.testing.assert_array_equal(seq,ss)


def test_future_frames_ignore_prefix_gap_and_trailing_samples():
    sr=8000;wave=waveform(sr);base=future_labels(wave,sr,'DOUBLE')
    changed=np.full_like(wave,np.nan)
    changed[10000:14000]=wave[10000:14000]
    alt=future_labels(changed,sr,'DOUBLE')
    for k in base: np.testing.assert_array_equal(base[k],alt[k])
    assert base['support'][0,0]==10000 and base['support'][-1,1]==14000
    assert base['support'].shape==(46,2)


def test_guard_22_23_ddof_and_unvoiced_values():
    pitch=np.tile(np.arange(46,dtype=float),(2,1));power=np.tile(np.arange(46,dtype=float)/3,(2,1))
    masks=np.zeros((2,46),bool);masks[0,:22]=True;masks[1,:23]=True
    pitch[~masks]=np.nan
    result,guard=reconstruct(pitch,power,masks)
    np.testing.assert_array_equal(guard,[False,True]);assert np.isnan(result[0,0])
    assert result[1,0]==pytest.approx(np.std(np.arange(23),ddof=1))
    np.testing.assert_allclose(result[:,1],np.std(np.arange(46)/3,ddof=1))


def test_invalid_records_are_counted_and_never_added_to_catalogue():
    train=np.ones((3,46),bool);train[1,:]=False
    catalogue=build_catalogue(train);before=catalogue
    held=np.ones((4,46),bool);held[1,:22]=False;held[2,:]=False;held[3,:30]=False
    counts=catalogue_counts(catalogue,held)
    assert catalogue==before and len(catalogue)==2
    assert counts['held_rows']==4 and counts['held_valid_rows']==2
    assert counts['held_unknown_rows']==2 and counts['held_valid_unknown_rows']==1
    assert counts['held_unknown_fraction']==.5 and counts['held_valid_unknown_fraction']==.5


def test_catalogue_is_inner_train_only_even_after_arbitrary_held_mutation():
    train=np.ones((5,46),bool);train[1,0]=False
    original=build_catalogue(train)
    rng=np.random.default_rng(1)
    held=rng.random((12,46))<.5
    catalogue_counts(original,held);catalogue_counts(original,~held)
    assert build_catalogue(train)==original
    assert catalogue_counts(original,np.empty((0,46),bool))['held_valid_unknown_fraction'] is None


def test_shape_and_mask_dtype_fail_closed():
    with pytest.raises(ValueError):reconstruct(np.ones((2,45)),np.ones((2,45)),np.ones((2,45),bool))
    with pytest.raises(ValueError):build_catalogue(np.ones((2,46),int))
    power=np.ones((1,46));power[0,0]=np.nan
    with pytest.raises(ValueError):reconstruct(np.ones((1,46)),power,np.ones((1,46),bool))


def test_allowed_inputs_and_mutation_selection_do_not_use_held_truth():
    names=[p.name for p in input_paths().values()]
    assert 'train_labels.npz' in names
    assert not {'dev_labels.npz','test_labels.npz'} & set(names)
    ids=['c','a','b'];assert mutation_ids(ids,2)==mutation_ids(ids[::-1],2)


def test_attempt_marker_survives_failure_and_forbids_duplicate(tmp_path):
    reserve_attempt(tmp_path,'abc')
    first=(tmp_path/'started.json').read_bytes()
    with pytest.raises(FileExistsError):reserve_attempt(tmp_path,'different')
    assert (tmp_path/'started.json').read_bytes()==first
    assert json.loads(first)['max_attempts']==1


def test_budget_and_deadline_fail_without_relaxing_limits():
    utc=datetime.timezone.utc;now=datetime.datetime(2026,11,2,tzinfo=utc)
    deadline=datetime.datetime(2026,11,5,15,tzinfo=utc)
    budget_check(0,600,clock=lambda:599,now=lambda:now,deadline=deadline)
    with pytest.raises(TimeoutError):budget_check(0,600,clock=lambda:601,now=lambda:now,deadline=deadline)
    with pytest.raises(TimeoutError):budget_check(0,600,clock=lambda:0,now=lambda:deadline,deadline=deadline)


def provenance_fixture(tmp_path):
    paths={k:tmp_path/k for k in ['source_prepared','source_receipt','source_freeze','source_result','source_seal']}
    paths['source_prepared'].write_bytes(b'sealed source sequence')
    paths['source_freeze'].write_text('{}')
    receipt={'prepared_sha256':sha(paths['source_prepared']),'freeze_sha256':sha(paths['source_freeze'])}
    paths['source_receipt'].write_text(json.dumps(receipt))
    result={**receipt,'prepared_receipt_sha256':sha(paths['source_receipt'])}
    paths['source_result'].write_text(json.dumps(result))
    expected=sha(paths['source_result'])
    paths['source_seal'].write_text(json.dumps({'results_sha256':expected}))
    return paths,expected


def test_replaced_cached_sequence_and_self_reported_receipt_fail_prior_seal(tmp_path):
    paths,expected=provenance_fixture(tmp_path)
    verify_discovery_chain(paths,expected)
    paths['source_prepared'].write_bytes(b'replaced sequence')
    d=json.loads(paths['source_receipt'].read_text());d['prepared_sha256']=sha(paths['source_prepared'])
    paths['source_receipt'].write_text(json.dumps(d))
    with pytest.raises(ValueError):verify_discovery_chain(paths,expected)


def test_replacing_result_and_its_seal_still_fails_pinned_historical_result(tmp_path):
    paths,expected=provenance_fixture(tmp_path)
    d=json.loads(paths['source_result'].read_text());d['changed']=True
    paths['source_result'].write_text(json.dumps(d))
    paths['source_seal'].write_text(json.dumps({'results_sha256':sha(paths['source_result'])}))
    with pytest.raises(ValueError):verify_discovery_chain(paths,expected)
