"""Participant-separated retrospective forecasting of stored violin descriptors.

No waveform-causality, Hz/dB-unit, or human-quality-ground-truth claim. This
module accesses performance properties only and never reads questionnaire fields.
Actual target export requires explicit protocol approval; synthetic functions do not.
"""
from __future__ import annotations
import hashlib
import importlib.metadata
import json
from pathlib import Path
import sys
import numpy as np

CHANNELS=('f0_raw_data','power_raw_data','ap_raw_data')
STAGES=('Results_pre','Results_feed','Results_post')
FIELDS=('Violin_Trials_GS','Violin_Trials_GS_online','Violin_Trials_GS_online_MC_Task','Violin_Trials_GS_online_GS_Task')
SPLIT_COUNTS={'CG':(19,4,5),'FG':(20,4,5),'EG':(10,2,3)}
SPLITS=('train','dev','test')
SEED=20261030
PREFIX_END=1024
TARGET_START=1536
TARGET_END=2048
MAT_SHA='fef94d704500624c8391e1caaf3c20ec764f606ae17f48d81a7e15932db31822'
TARGET_NAMES=('stored_pitch_coordinate_std','stored_power_coordinate_std')


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def file_sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def participant_split(participants,seed=SEED,counts=SPLIT_COUNTS):
    """Hash-rank stable pseudonymous array-position IDs; never use outcomes."""
    if isinstance(seed,bool) or not isinstance(seed,int):raise ValueError('Integer split seed required')
    if len({p['participant_id'] for p in participants})!=len(participants):raise ValueError('Duplicate participant identity')
    mapping={}
    if set(p['cohort'] for p in participants)!=set(counts):raise ValueError('Cohort count mismatch')
    for cohort,ns in sorted(counts.items()):
        ids=[p['participant_id'] for p in participants if p['cohort']==cohort]
        if len(ids)!=sum(ns):raise ValueError('Participant count mismatch for '+cohort)
        ranked=sorted(ids,key=lambda pid:(digest({'seed':seed,'cohort':cohort,'participant':pid}),pid))
        start=0
        for name,n in zip(SPLITS,ns):
            for pid in ranked[start:start+n]:mapping[pid]=name
            start+=n
    return mapping


def content_sha(channels):
    h=hashlib.sha256()
    for name in CHANNELS:
        a=np.asarray(channels[name],dtype='<f8').ravel()
        h.update(name.encode());h.update(len(a).to_bytes(8,'little'));h.update(a.tobytes())
    return h.hexdigest()


def deduplicate(records):
    """Merge exact three-channel aliases within a participant; reject cross-person copies."""
    unique={}
    for r in sorted(records,key=lambda r:r['source_id']):
        h=content_sha(r['channels'])
        if h in unique:
            if unique[h]['participant_id']!=r['participant_id']:
                raise ValueError('Identical sequence crosses participant identities')
            unique[h]['aliases'].append(r['source_id'])
        else:unique[h]={**r,'content_sha256':h,'aliases':[r['source_id']]}
    return sorted(unique.values(),key=lambda r:r['source_id'])


def sequence_features(channels):
    """X depends only on [0:1024]; two persistence summaries use [512:1024]."""
    parts=[];names=[]
    prefix={name:np.asarray(channels[name],dtype=float).ravel()[:PREFIX_END] for name in CHANNELS}
    if any(len(a)!=PREFIX_END or not np.isfinite(a).all() for a in prefix.values()):
        raise ValueError('Missing/nonfinite prefix')
    for width in (64,256,1024):
        t=np.arange(width,dtype=float);t-=t.mean();denom=float(np.sum(t*t,dtype=np.float64))
        for channel in CHANNELS:
            a=prefix[channel][-width:]
            values=(a.mean(),a.std(ddof=1),*np.quantile(a,[.1,.9],method='linear'),
                    np.abs(np.diff(a)).mean(),float(np.sum(t*a,dtype=np.float64)/denom))
            parts.extend(values)
            names.extend(f'last{width}_{channel}_{stat}' for stat in ('mean','std_ddof1','p10','p90','mean_abs_diff','slope_per_frame'))
    persistent=np.array([prefix[name][-512:].std(ddof=1) for name in CHANNELS[:2]])
    parts.extend(persistent);names.extend('last512_'+n for n in TARGET_NAMES)
    return np.asarray(parts,dtype=np.float64),persistent,names


def sequence_target(channels):
    selected=[np.asarray(channels[name],dtype=float).ravel()[TARGET_START:TARGET_END] for name in CHANNELS[:2]]
    if any(len(a)!=512 or not np.isfinite(a).all() for a in selected):raise ValueError('Missing/nonfinite target window')
    return np.asarray([a.std(ddof=1) for a in selected],dtype=float)


def fit_thresholds(targets,split,participant_ids):
    """Only source/train rows are passed to quantile; held-out values are not inspected."""
    targets=np.asarray(targets,float);split=np.asarray(split);participant_ids=np.asarray(participant_ids)
    if targets.shape!=(len(split),2) or len(participant_ids)!=len(split):raise ValueError('Unaligned thresholds input')
    for pid in np.unique(participant_ids):
        if len(set(split[participant_ids==pid]))!=1:raise ValueError('Participant crosses splits')
    train=split=='train'
    if not train.any():raise ValueError('No training targets')
    y=targets[train]
    if not np.isfinite(y).all() or (y<0).any():raise ValueError('Invalid training targets')
    thresholds=np.quantile(y,[.25,.5,.75],axis=0,method='linear').T
    if not (np.diff(thresholds,axis=1)>0).all():raise ValueError('Repeated training quantiles require a new reviewed rule bank')
    return thresholds,dict(training_rows=int(train.sum()),training_participant_ids=sorted(set(participant_ids[train].tolist())),
                           training_participant_ids_sha256=digest(sorted(set(participant_ids[train].tolist()))),
                           quantile_method='linear',quantiles=[.25,.5,.75])


def admissible_patterns():
    # Both rule axes increase the threshold; positive entries form a suffix×suffix.
    patterns=[np.zeros(9,dtype=np.uint8)]
    for a in range(3):
        for b in range(3):
            grid=np.zeros((3,3),dtype=np.uint8);grid[a:,b:]=1;patterns.append(grid.ravel())
    return np.unique(patterns,axis=0)


def labels_and_patterns(targets,thresholds):
    targets=np.asarray(targets,float);thresholds=np.asarray(thresholds,float)
    if targets.ndim!=2 or targets.shape[1]!=2 or thresholds.shape!=(2,3):raise ValueError('Expected Nx2 targets and2x3 thresholds')
    if not np.isfinite(targets).all() or (targets<0).any() or not np.isfinite(thresholds).all() or not(np.diff(thresholds,axis=1)>0).all():
        raise ValueError('Invalid targets/thresholds')
    a=targets[:,0,None]<=thresholds[0];b=targets[:,1,None]<=thresholds[1]
    labels=(a[:,:,None]&b[:,None,:]).reshape(-1,9).astype(np.uint8)
    match=(labels[:,None,:]==admissible_patterns()[None,:,:]).all(axis=2)
    if not match.any(axis=1).all():raise AssertionError('Invalid suffix-rectangle pattern')
    return labels,match.argmax(axis=1).astype(np.int64)


def assemble_dataset(records,split_manifest,metadata=None):
    """Partition identities before computing targets or fitting thresholds."""
    if any(r['participant_id'] not in split_manifest for r in records):raise ValueError('Unassigned participant')
    unique=deduplicate(records)
    ids=np.asarray([r['source_id'] for r in unique]);participants=np.asarray([r['participant_id'] for r in unique])
    split=np.asarray([split_manifest[pid] for pid in participants])
    if not set(split).issubset(SPLITS):raise ValueError('Unknown split')
    # Finish every prefix before touching future target windows.
    features=[sequence_features(r['channels']) for r in unique]
    X=np.stack([f[0] for f in features]);prefix_targets=np.stack([f[1] for f in features])
    targets=np.stack([sequence_target(r['channels']) for r in unique])
    thresholds,fit_receipt=fit_thresholds(targets,split,participants)
    Y9,pattern_ids=labels_and_patterns(targets,thresholds)
    prefix_Y9,prefix_pattern_ids=labels_and_patterns(prefix_targets,thresholds)
    info=dict(metadata or {})
    info.update(rows=len(unique),raw_cells=len(records),deduplicated_extra_cells=len(records)-len(unique),
        cross_participant_content_duplicates=0,input_frames=[0,1024],gap_frames=[1024,1536],target_frames=[1536,2048],
        persistence_frames=[512,1024],ddof=1,feature_count=X.shape[1],target_names=list(TARGET_NAMES),
        split_manifest=split_manifest,split_manifest_sha256=digest(split_manifest),threshold_fit=fit_receipt,
        participant_counts={s:len(set(participants[split==s])) for s in SPLITS},
        row_counts={s:int((split==s).sum()) for s in SPLITS},
        ordered_sequence_ids_sha256=digest(ids.tolist()),patterns=admissible_patterns().tolist(),
        scope='retrospective stored-descriptor forecasting; waveform-causality and physical units unresolved',
        excluded_inputs=['participant/cohort/phase/trial IDs','full duration','future summaries','questionnaires','motion','stored aggregate scores'],
        normalization='none fitted; caller must fit training-only preprocessing')
    return dict(X=X,targets=targets,prefix_targets=prefix_targets,thresholds=thresholds,Y9=Y9,pattern_ids=pattern_ids,
        prefix_Y9=prefix_Y9,prefix_pattern_ids=prefix_pattern_ids,participant_ids=participants,groups=participants,
        cohorts=np.asarray([r['cohort'] for r in unique]),split=split,sequence_ids=ids,ids=ids,
        content_sha256=np.asarray([r['content_sha256'] for r in unique]),
        alias_manifest={r['source_id']:r['aliases'] for r in unique},feature_names=features[0][2],metadata=info)


def _cells(value):
    a=np.asarray(value)
    if not a.size:return []
    return [np.asarray(x,dtype=float).ravel() for x in a.ravel()] if a.dtype.kind=='O' else [np.asarray(a,dtype=float).ravel()]


def read_performance_records(data_root):
    """Parse the MAT container, access ONLY permitted performance properties."""
    root=Path(data_root);mat=root/'extracted/Sound_and_Motion_Data/participants_raw_Sound_Motion.mat'
    if file_sha(mat)!=MAT_SHA:raise ValueError('Source MAT identity changed')
    sys.path.insert(0,str(root/'python_packages'))
    import matio
    obj=matio.load_from_mat(mat,raw_data=True)
    participants=[];records=[]
    for cohort in SPLIT_COUNTS:
        for index,person in enumerate(obj['participants_'+cohort].ravel()):
            pid=f'{cohort}_{index+1:03d}';participants.append(dict(participant_id=pid,cohort=cohort))
            for stage in STAGES:
                result=person.properties[stage]
                if not hasattr(result,'properties'):continue
                for field in FIELDS:
                    arr=result.properties[field]
                    if not isinstance(arr,np.ndarray) or not arr.dtype.names or not arr.size:continue
                    gs=arr.ravel()[0];channels={name:_cells(gs[name]) for name in CHANNELS}
                    if len(set(len(a) for a in channels.values()))!=1:raise ValueError('Channel trial-count mismatch')
                    for trial in range(len(channels[CHANNELS[0]])):
                        values={name:channels[name][trial] for name in CHANNELS}
                        lengths=[len(a) for a in values.values()]
                        if len(set(lengths))!=1 or min(lengths)<TARGET_END:raise ValueError('Incomplete or unaligned sequence')
                        records.append(dict(source_id=f'{pid}/{stage}/{field}/{trial:03d}',participant_id=pid,
                                            cohort=cohort,channels=values))
    return participants,records,dict(mat_sha256=MAT_SHA,parser='mat-io',parser_version=importlib.metadata.version('mat-io'),
        matlab_declarations_sha256=file_sha(root/'extracted/Sound_and_Motion_Data/participant_results.m'))


def build_dataset(data_root,protocol):
    if protocol.get('data_build_approved') is not True:raise ValueError('Protocol approval required before actual target export')
    if protocol.get('timing_scope')!='retrospective_stored_descriptor':raise ValueError('Strict waveform-causal timing is unverified')
    seed=protocol.get('split_seed',SEED)
    participants,records,lineage=read_performance_records(data_root)
    manifest=participant_split(participants,seed)
    result=assemble_dataset(records,manifest,{**lineage,'protocol_sha256':digest(protocol),
        'loader_sha256':file_sha(__file__),'split_seed':seed,'split_algorithm':'cohort-stratified SHA256 rank of seed/cohort/pseudonymous array-position ID'})
    if result['metadata']['rows']!=2117 or result['metadata']['raw_cells']!=3161:raise ValueError('Deduplicated source count differs from schema audit')
    return result


def save_prepared_dataset(dataset,path):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    private={'metadata','alias_manifest','feature_names'}
    np.savez_compressed(path,**{k:v for k,v in dataset.items() if k not in private},
        metadata_json=np.asarray(json.dumps(dataset['metadata'],sort_keys=True)),
        alias_manifest_json=np.asarray(json.dumps(dataset['alias_manifest'],sort_keys=True)),feature_names=np.asarray(dataset['feature_names']))
    receipt={'npz_sha256':file_sha(path),'metadata':dataset['metadata']}
    path.with_suffix('.json').write_text(json.dumps(receipt,indent=2,sort_keys=True)+'\n')
    return path


def load_prepared_dataset(path):
    path=Path(path);receipt=json.loads(path.with_suffix('.json').read_text())
    if file_sha(path)!=receipt['npz_sha256']:raise ValueError('Dataset hash mismatch')
    with np.load(path,allow_pickle=False) as z:
        result={k:z[k] for k in z.files if k not in ('metadata_json','alias_manifest_json','feature_names')}
        result.update(metadata=json.loads(str(z['metadata_json'])),alias_manifest=json.loads(str(z['alias_manifest_json'])),feature_names=z['feature_names'].tolist())
    if result['metadata']!=receipt['metadata']:raise ValueError('Metadata mismatch')
    return result
