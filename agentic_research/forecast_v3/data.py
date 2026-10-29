"""Pinned retrospective landmark dataset: prefix <= .20 s, target [.25,.45].

The landmark was extracted using the entire speed trace. This is therefore
offline future-window prediction, NOT causal online onset detection. Targets
are the existing mechanical-signal observer, not verified physical truth.
No estimator, imputer, scaler, or selection criterion is fitted here.
"""
from __future__ import annotations
import gzip
import hashlib
import json
from pathlib import Path
import numpy as np

try:
    from . import rule_patterns as rp
except ImportError:
    import rule_patterns as rp

PREFIX_WINDOWS = ((.2, 0.), (.1, 0.), (.1, .05), (.1, .1))  # (width,start)
TARGET_WINDOW = (.2, .25)
AUDIT_WINDOW = (.2, .2)
LATEST_PREFIX = (.1, .1)
NUMERIC_FIELDS = ('beta', 'force', 'speed', 'amplitude_std', 'periodicity',
                  'f0_hz', 'flybacks_per_period', 'second_flyback_ratio', 'flyback_height')
SIGNAL_FIELDS = ('periodicity', 'f0_hz', 'amplitude_std', 'flybacks_per_period',
                 'second_flyback_ratio', 'flyback_height')
OBSERVER = dict(min_amplitude=.4063327834010124, min_periodicity=.8,
                max_aperiodic=.5, subharmonic_ratio=.75, reference_f0=98.08)
TRAIN_SESSIONS = ('A1','A2','A3','A4','A_T2','B1','B_T2s1','B_T2r1','B_T2r2')
DEV_SESSIONS = ('C1','C_T2','D1','D2')
SCHEMA_VERSION = 'retrospective_prefix_v3_1'


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as fh:
        for block in iter(lambda: fh.read(1024*1024), b''): h.update(block)
    return h.hexdigest()


def _json_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode()).hexdigest()


def observer(rows):
    """Exact vector copy of label_sweep.helm(base) + joint_rule export guard.

    The amplitude OR periodicity first gate is retained, even though a
    qualifying periodicity makes the amplitude gate redundant for base.
    A missing f0 does NOT veto the old observer; this is intentional parity.
    The fixed override is retained and audited even though base requires
    finite P/R, so this observer's positive-missing lock is always false.
    """
    a, p, f0, fpp, r = (np.asarray(rows[k], dtype=float) for k in
        ('amplitude_std','periodicity','f0_hz','flybacks_per_period','second_flyback_ratio'))
    finite = np.isfinite(p) & np.isfinite(r)
    not_subharmonic = ~(np.isfinite(f0) & (f0 < OBSERVER['subharmonic_ratio']*OBSERVER['reference_f0']))
    physical = finite & (p >= OBSERVER['max_aperiodic']) & not_subharmonic & np.isfinite(fpp) & (fpp >= 1)
    baseline = ((a >= OBSERVER['min_amplitude']) | (p >= .8)) & physical & (p >= .8) & (r < .5)
    fixed = baseline & ~finite
    guard = physical & ~fixed
    labels = rp.labels_from_features(p, r, guard, fixed).astype(np.uint8)
    if not np.array_equal(labels[..., 4].astype(bool), baseline):
        raise AssertionError('Baseline reconstruction mismatch')
    return dict(Y9=labels, pattern_ids=rp.pattern_ids(labels), guard=guard, fixed=fixed, baseline=baseline)


def prefix_features(rows):
    """Transform ONLY one provided prefix; missing flags never inspect future."""
    force, beta, speed = (np.asarray(rows[k], dtype=float) for k in ('force','beta','speed'))
    if not np.isfinite(np.column_stack((force,beta,speed))).all() or np.any(beta <= 0) or np.any(speed <= 0):
        raise ValueError('Nonfinite prefix controls or nonpositive beta/speed')
    vals = [np.arcsinh(force/.1), np.log(beta), np.log(speed)]
    names = ['force_asinh_0.1N','log_beta','log_measured_speed']
    for name in SIGNAL_FIELDS:
        raw = np.asarray(rows[name], dtype=float)
        missing = ~np.isfinite(raw)
        v = np.where(missing, 0., raw)
        if name == 'f0_hz': v = np.arcsinh(v/98.08)
        elif name in ('amplitude_std','flyback_height'): v = np.arcsinh(v/.1)
        vals.extend((v, missing.astype(float)))
        names.extend((name, name+'_missing'))
    X = np.column_stack(vals).astype(np.float32)
    if not np.isfinite(X).all(): raise ValueError('Nonfinite prefix feature')
    return X, names


def assemble_dataset(windows, records, metadata=None):
    """Pure aligned-window assembly, also used by leakage mutation tests.

    records contain session/folder/trial/column/id; no identity is put into X.
    Each window dictionary contains arrays in that same deterministic order.
    """
    if any(w+t > .2+1e-12 for w,t in PREFIX_WINDOWS): raise AssertionError('Prefix overlaps future')
    parts = [prefix_features(windows[key]) for key in PREFIX_WINDOWS]
    X = np.column_stack([x for x,_ in parts])
    names = [f'w{w:.2f}_t{t:.2f}_{name}' for (w,t),(_,nn) in zip(PREFIX_WINDOWS,parts) for name in nn]
    # Target is not accessed until X has been constructed.
    target = observer(windows[TARGET_WINDOW])
    prefix = observer(windows[LATEST_PREFIX])
    sessions = np.asarray([r['session'] for r in records])
    if not set(sessions).issubset(set(TRAIN_SESSIONS+DEV_SESSIONS)):
        raise ValueError('Unknown session outside the original frozen thirteen')
    split = np.where(np.isin(sessions, TRAIN_SESSIONS), 'train', 'dev')
    ids = np.asarray([r['id'] for r in records])
    if len(np.unique(ids)) != len(ids): raise ValueError('Duplicate stroke identity')
    if len(X) != len(records): raise ValueError('Unaligned feature and identity rows')
    groups = np.asarray([s[0] for s in sessions])
    if set(groups[split=='train']) & set(groups[split=='dev']): raise AssertionError('Family leakage')
    info = dict(metadata or {})
    info.update(schema_version=SCHEMA_VERSION, rows=len(X), feature_count=len(names),
        prefix_windows_width_start=[list(k) for k in PREFIX_WINDOWS],
        target_window_start_end=[.25,.45], persistence_window_start_end=[.1,.2],
        origin_s=.2, minimum_gap_s=.05, scope='retrospective full-trace landmark; historically exposed development',
        target_kind='deterministic mechanical-signal observer, not physical ground truth',
        observer=OBSERVER, train_sessions=list(TRAIN_SESSIONS), dev_sessions=list(DEV_SESSIONS),
        split_counts={k:int((split==k).sum()) for k in ('train','dev')},
        grouping='conservative string family A/B/C/D; sessions are diagram IDs, not asserted independent specimens',
        input_exclusions=['future features','future force/speed','plateau_s','IDs','session/folder/trial','target labels'],
        normalization='none fitted; fixed transforms and prefix-only zero+missing flags; caller fits training-only scaler',
        ordered_ids_sha256=_json_sha(ids.tolist()), feature_names_sha256=_json_sha(names),
        split_manifest_sha256=_json_sha(list(zip(ids.tolist(),split.tolist(),groups.tolist()))))
    return dict(X=X,Y9=target['Y9'],pattern_ids=target['pattern_ids'],
        prefix_Y9=prefix['Y9'],prefix_pattern_ids=prefix['pattern_ids'],persistence_Y9=prefix['Y9'],
        ids=ids,sessions=sessions,groups=groups,split=split,
        columns=np.asarray([r['column'] for r in records]),
        measurement_sessions=np.asarray([r['session']+'|'+r['folder'] for r in records]),
        feature_names=names,metadata=info)


def _select_rows(table, records):
    """Align selected windows, rejecting missing/duplicate rows, retaining repeats.

    Different folders/recording repetitions may reuse trial numbers. Their
    (folder,trial) keys remain distinct; identical full keys are errors.
    """
    columns = {name:table[name].to_numpy(zero_copy_only=False) for name in table.column_names}
    index = {(r['folder'],r['trial']): i for i,r in enumerate(records)}
    if len(index) != len(records): raise ValueError('Ambiguous folder/trial identity in source')
    windows = {}
    for w,t in (*PREFIX_WINDOWS,TARGET_WINDOW,AUDIT_WINDOW):
        selected = np.flatnonzero(np.isclose(columns['window_s'],w,atol=1e-9,rtol=0)
                                  & np.isclose(columns['t'],t,atol=1e-9,rtol=0))
        aligned = np.full(len(records), -1, dtype=np.int64)
        for row in selected:
            key = (str(columns['folder'][row]),int(columns['trial'][row]))
            if key not in index: continue
            i = index[key]
            if aligned[i] >= 0: raise ValueError(f'Duplicate selected window: {key}, {w,t}')
            if columns['error'][row] not in ('',None): raise ValueError(f'Feature error: {key}')
            aligned[i] = row
        if np.any(aligned < 0): raise ValueError(f'Missing {int((aligned<0).sum())} required windows {w,t}')
        windows[(w,t)] = {name:np.asarray(columns[name][aligned],dtype=float) for name in NUMERIC_FIELDS}
    beta = windows[PREFIX_WINDOWS[0]]['beta']
    for win in windows.values():
        if not np.array_equal(beta,win['beta']): raise ValueError('Beta changed between windows')
    return windows


def _records_from_fixture(fixture):
    records = []
    seen = set()
    for c in fixture['columns']:
        for j,trial in enumerate(c['trial']):
            sid = f"{c['session']}|{c['folder']}|trial={int(trial)}"
            if sid in seen: raise ValueError('Duplicate fixture stroke')
            seen.add(sid)
            records.append(dict(session=c['session'],folder=c['folder'],trial=int(trial),id=sid,column=c['id'],
                expected={key:c[key][j] for key in ('baseline','guard','fixed','periodicity','ratio','force')}))
    return sorted(records,key=lambda r:(r['session'],r['folder'],r['trial']))


def _audit_observer(rows, records):
    observed = observer(rows)
    for key in ('baseline','guard','fixed'):
        if not np.array_equal(observed[key],np.asarray([r['expected'][key] for r in records],bool)):
            raise ValueError(f'Observer audit failed for {key}')
    for key,field in (('periodicity','periodicity'),('ratio','second_flyback_ratio'),('force','force')):
        expected = np.asarray([np.nan if r['expected'][key] is None else r['expected'][key] for r in records],float)
        if not np.array_equal(rows[field],expected,equal_nan=True):
            raise ValueError(f'Historical window audit failed for {key}')


def save_dataset(dataset,path):
    path = Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    arrays = {k:v for k,v in dataset.items() if k not in ('metadata','feature_names')}
    metadata = {k:v for k,v in dataset['metadata'].items() if k != 'artifact_sha256'}
    np.savez_compressed(path,**arrays,feature_names=np.asarray(dataset['feature_names']),
                        metadata_json=np.asarray(json.dumps(metadata,sort_keys=True,allow_nan=False)))
    receipt = dict(npz_sha256=sha256(path),metadata=metadata)
    path.with_suffix('.json').write_text(json.dumps(receipt,indent=2,sort_keys=True)+'\n')
    return path


def load_dataset(path):
    path = Path(path)
    receipt = json.loads(path.with_suffix('.json').read_text())
    if sha256(path) != receipt['npz_sha256']: raise ValueError('Cached dataset hash mismatch')
    with np.load(path,allow_pickle=False) as z:
        out = {k:z[k] for k in z.files if k not in ('metadata_json','feature_names')}
        out['metadata'] = json.loads(str(z['metadata_json']))
        out['feature_names'] = z['feature_names'].tolist()
    if out['metadata'] != receipt['metadata']: raise ValueError('Metadata receipt mismatch')
    out['metadata']['artifact_sha256'] = receipt['npz_sha256']
    return out


# Explicit controller-facing aliases: the same receipt validation is mandatory.
save_prepared_dataset = save_dataset
load_prepared_dataset = load_dataset


def load_or_build_dataset(features_dir,joint_input_path,cache_dir=None):
    """Build a hashed 78k dataset; raw CSVs and their provenance remain external.

    Source CSV hashes MUST match the existing joint-rule fixture. This avoids
    silently changing the historically audited observer or row selection.
    An existing cache is reusable only with the same source/code identities.
    """
    import pyarrow as pa
    import pyarrow.csv as pc
    features_dir, joint_input_path = Path(features_dir),Path(joint_input_path)
    fixture = json.load(gzip.open(joint_input_path,'rt'))
    records = _records_from_fixture(fixture)
    if len(records)!=78000 or set(r['session'] for r in records)!=set(TRAIN_SESSIONS+DEV_SESSIONS):
        raise ValueError('Expected the original thirteen diagrams and 78,000 strokes')
    sources = {Path(k).name:v for k,v in fixture['source_sha256'].items() if k.startswith('timeres/features/')}
    if len(sources)!=12: raise ValueError('Expected twelve archives (one contains two repetitions)')
    for name,expected in sorted(sources.items()):
        if sha256(features_dir/name)!=expected: raise ValueError(f'Source hash mismatch: {name}')
    lineage = dict(joint_input_sha256=sha256(joint_input_path),source_csv_sha256=sources,
                   loader_sha256=sha256(__file__),rule_patterns_sha256=sha256(rp.__file__))
    fingerprint = _json_sha(lineage)
    cache_path = Path(cache_dir)/f'prefix_future_{fingerprint[:16]}.npz' if cache_dir is not None else None
    if cache_path is not None and cache_path.exists(): return load_dataset(cache_path)
    all_windows = {key:{field:np.full(len(records),np.nan) for field in NUMERIC_FIELDS}
                   for key in (*PREFIX_WINDOWS,TARGET_WINDOW,AUDIT_WINDOW)}
    covered = np.zeros(len(records),bool)
    selected_columns = ['folder','trial','window_s','t','error',*NUMERIC_FIELDS]
    types = {k:pa.float64() for k in ['window_s','t',*NUMERIC_FIELDS]}
    types.update(folder=pa.string(),trial=pa.int64(),error=pa.string())
    for name in sorted(sources):
        stem = name.removesuffix('.csv.gz')
        selected = [i for i,r in enumerate(records) if r['folder'].split('/')[0]==stem]
        if not selected: raise ValueError(f'No selected records for source {name}')
        table = pc.read_csv(features_dir/name,convert_options=pc.ConvertOptions(include_columns=selected_columns,
            column_types=types,null_values=['','nan','NaN','None'],strings_can_be_null=True))
        rr = [records[i] for i in selected]
        ww = _select_rows(table,rr)
        _audit_observer(ww[AUDIT_WINDOW],rr)
        for key in all_windows:
            for field in NUMERIC_FIELDS: all_windows[key][field][selected] = ww[key][field]
        covered[selected]=True
        del table,ww
    if not covered.all(): raise ValueError('Missing source strokes')
    info = dict(lineage,input_sha256=fingerprint,observer_audit=dict(strokes=78000,
        baseline_mismatches=0,guard_mismatches=0,fixed_mismatches=0,historical_feature_mismatches=0,
        audit_window_start_end=[.2,.4]),source_records=fixture.get('source_records',[]))
    dataset = assemble_dataset(all_windows,records,info)
    if cache_path is not None:
        save_dataset(dataset,cache_path)
        return load_dataset(cache_path)
    return dataset
