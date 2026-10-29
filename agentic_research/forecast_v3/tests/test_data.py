"""Leakage, observer parity, identity, and cache integrity tests; no model fit."""
from pathlib import Path
import sys
import copy
import itertools
import math
import numpy as np
import pytest
import pyarrow as pa

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import data as fd


def synthetic():
    sessions = ['A1','B_T2r1','B_T2r2','C1','D2']
    records = [dict(session=s,folder=s+'/v1',trial=1,column=s+'|0',id=s+'|trial1') for s in sessions]
    windows = {}
    for key in (*fd.PREFIX_WINDOWS,fd.TARGET_WINDOW,fd.AUDIT_WINDOW):
        windows[key] = dict(beta=np.full(5,.1),force=np.arange(5)*.1+.1,speed=np.full(5,.05),
            amplitude_std=np.full(5,.5),periodicity=np.linspace(.65,.95,5),f0_hz=np.full(5,98.08),
            flybacks_per_period=np.ones(5),second_flyback_ratio=np.linspace(.1,.7,5),flyback_height=np.ones(5))
    return windows,records


def test_future_mutation_cannot_change_inputs_or_missing_flags_bitwise():
    windows,records = synthetic()
    before = fd.assemble_dataset(windows,records)
    changed = copy.deepcopy(windows)
    for key in (fd.TARGET_WINDOW,fd.AUDIT_WINDOW):
        for field in fd.NUMERIC_FIELDS: changed[key][field][:]=np.nan
    after = fd.assemble_dataset(changed,records)
    assert before['X'].tobytes()==after['X'].tobytes()
    np.testing.assert_array_equal(before['prefix_Y9'],after['prefix_Y9'])
    assert not np.array_equal(before['Y9'],after['Y9'])


def test_prefix_mutation_cannot_change_targets_and_prefix_nan_is_local():
    windows,records = synthetic()
    before = fd.assemble_dataset(windows,records)
    changed = copy.deepcopy(windows)
    changed[fd.LATEST_PREFIX]['periodicity'][0]=np.nan
    changed[fd.LATEST_PREFIX]['force'][1]=8.
    after = fd.assemble_dataset(changed,records)
    np.testing.assert_array_equal(before['Y9'],after['Y9'])
    np.testing.assert_array_equal(before['pattern_ids'],after['pattern_ids'])
    assert before['X'].tobytes()!=after['X'].tobytes()
    assert np.isfinite(after['X']).all()
    name='w0.10_t0.10_periodicity_missing'
    assert after['X'][0,after['feature_names'].index(name)]==1


def test_identity_fields_are_not_features_and_repeated_trial_numbers_are_valid():
    windows,records=synthetic(); d=fd.assemble_dataset(windows,records)
    assert d['X'].shape==(5,60)
    assert d['split'].tolist()==['train','train','train','dev','dev']
    assert not(set(d['groups'][d['split']=='train']) & set(d['groups'][d['split']=='dev']))
    assert len(np.unique(d['ids']))==5
    changed=copy.deepcopy(records)
    for r in changed: r['id']+='changed';r['folder']+='changed';r['trial']+=100
    np.testing.assert_array_equal(d['X'],fd.assemble_dataset(windows,changed)['X'])
    changed[0]['id']=changed[1]['id']
    with pytest.raises(ValueError,match='Duplicate'): fd.assemble_dataset(windows,changed)


def scalar_baseline(a,p,f0,fpp,r):
    if not(a>=.4063327834010124) and not(p>=.8): return False
    if not math.isfinite(p) or p<.5 or p<.8: return False
    if math.isfinite(f0) and f0<.75*98.08: return False
    return math.isfinite(fpp) and fpp>=1 and math.isfinite(r) and r<.5


def test_exact_observer_parity_at_threshold_edges_and_nonfinite_values():
    cases=list(itertools.product([0.,1.,np.nan],[.49,.5,.7,.8,.9,np.nan],
        [0.,.75*98.08,98.08,np.nan],[0.,1.,2.,np.nan],[.1,.3,.5,.7,np.nan]))
    arr=np.asarray(cases)
    rows=dict(zip(('amplitude_std','periodicity','f0_hz','flybacks_per_period','second_flyback_ratio'),arr.T))
    observed=fd.observer(rows)
    np.testing.assert_array_equal(observed['baseline'],[scalar_baseline(*c) for c in cases])
    assert not observed['fixed'].any()
    assert len(np.unique(observed['pattern_ids']))==10


def test_selected_rows_preserve_recording_repetitions_and_reject_duplicate_keys():
    windows,records=synthetic(); rows=[]
    for key,fields in windows.items():
        for i,r in enumerate(records):
            rows.append(dict(folder=r['folder'],trial=r['trial'],window_s=key[0],t=key[1],error='',
                **{f:float(v[i]) for f,v in fields.items()}))
    table=pa.Table.from_pylist(rows)
    loaded=fd._select_rows(table,records)
    np.testing.assert_array_equal(loaded[fd.TARGET_WINDOW]['periodicity'],windows[fd.TARGET_WINDOW]['periodicity'])
    with pytest.raises(ValueError,match='Duplicate'): fd._select_rows(pa.Table.from_pylist(rows+[rows[0]]),records)
    with pytest.raises(ValueError,match='Missing'): fd._select_rows(pa.Table.from_pylist(rows[1:]),records)


def test_cache_roundtrip_and_corruption_rejected(tmp_path):
    windows,records=synthetic(); original=fd.assemble_dataset(windows,records)
    path=fd.save_dataset(original,tmp_path/'cache.npz')
    loaded=fd.load_dataset(path)
    for key in ('X','Y9','ids','sessions','groups','split','prefix_Y9'):
        np.testing.assert_array_equal(original[key],loaded[key])
    assert loaded['feature_names']==original['feature_names']
    with path.open('ab') as f: f.write(b'corrupt')
    with pytest.raises(ValueError,match='hash'): fd.load_dataset(path)


def test_historical_observer_audit_fails_closed():
    windows,records=synthetic(); rows=windows[fd.AUDIT_WINDOW]; o=fd.observer(rows)
    for i,r in enumerate(records):
        r['expected']={k:int(o[k][i]) for k in ('baseline','guard','fixed')}
        r['expected'].update(periodicity=float(rows['periodicity'][i]),ratio=float(rows['second_flyback_ratio'][i]),force=float(rows['force'][i]))
    fd._audit_observer(rows,records)
    records[0]['expected']['baseline']^=1
    with pytest.raises(ValueError,match='audit failed'): fd._audit_observer(rows,records)
