"""Prefix-only source data and inner-person-fold preprocessing."""
from pathlib import Path
import hashlib,json,time,fcntl
import numpy as np
from ..violin_raw_v1.features import interval_frames,PREFIX
from ..violin_raw_v1.prepare import read_supported_samples
from ..violin_raw_v1.data import labels_and_states
from ..violin_raw_v1.models import participant_weights,transition_prior

HERE=Path(__file__).resolve().parent
OLD=HERE.parent/'violin_raw_v1'


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify_old_sources():
    root=OLD.parents[1]
    for manifest in ['data_freeze_manifest.json','execution_freeze_manifest.json']:
        m=json.loads((OLD/manifest).read_text())
        for name,h in m['source_hashes'].items():
            if sha(root/name)!=h:raise ValueError('old executable changed: '+name)
    m=json.loads((OLD/'execution_freeze_manifest.json').read_text())
    for name in ['features.npz','train_labels.npz']:
        if sha(OLD/'local/prepared'/name)!=m['prepared_files'][name]:raise ValueError('old source input changed')
    # Deliberately do not open or hash any dev/test truth file.


def tokens(samples,sr,subtype):
    # Cut before any descriptor, dtype conversion, finite check or normalization.
    cut=samples[:round(sr)].copy()
    f=interval_frames(cut,sr,subtype,PREFIX)
    return np.column_stack([f['pitch_cents'],f['power_dbfs'],f['periodicity'],f['voiced']]).astype(np.float32)


def source_folds(people):
    people=np.asarray(people);lookup={}
    for cohort in ['CG','EG','FG']:
        ids=[p for p in np.unique(people) if str(p).rsplit('_',1)[-1]==cohort]
        ids.sort(key=lambda p:hashlib.sha256(('20261102:'+str(p)).encode()).hexdigest())
        lookup.update({pid:i%3 for i,pid in enumerate(ids)})
    if len(lookup)!=len(np.unique(people)):raise ValueError('unknown source cohort')
    return np.asarray([lookup[p] for p in people])


def shuffle_tokens(sequence,source_ids):
    result=[]
    for row,sid in zip(sequence,source_ids):
        seed=int.from_bytes(hashlib.sha256(('order-control:'+str(sid)).encode()).digest()[:8],'big')
        result.append(row[np.random.default_rng(seed).permutation(len(row))])
    return np.asarray(result)


def fit_fold(x,seq,targets,guard,people,mask):
    """All target/scaler/prior fitting slices INNER train before inspecting data."""
    mask=np.asarray(mask,bool);valid=mask&guard
    t=targets[valid].astype(float)
    if len(t)<4 or not np.isfinite(t).all() or (t<0).any():raise ValueError('invalid inner training targets')
    thresholds=np.quantile(t,[.25,.5,.75],axis=0).T
    if not np.all(np.diff(thresholds,axis=1)>0):raise ValueError('degenerate inner thresholds')
    xm=x[mask].mean(0);xs=x[mask].std(0);xs=np.where(xs>1e-8,xs,1.)
    sm=seq[mask].mean((0,1));ss=seq[mask].std((0,1));ss=np.where(ss>1e-8,ss,1.)
    cm=np.log1p(t).mean(0);cs=np.log1p(t).std(0);cs=np.where(cs>1e-8,cs,1.)
    y,_=labels_and_states(targets[mask],guard[mask],thresholds)
    _,s=labels_and_states(x[mask,57:59],x[mask,59].astype(bool),thresholds,prefix=True)
    frequency,prior=transition_prior(y,s,participant_weights(people[mask]))
    return {'thresholds':thresholds,'x_mean':xm,'x_scale':xs,'seq_mean':sm,'seq_scale':ss,
            'continuous_mean':cm,'continuous_scale':cs,'frequency':frequency,'prior':prior}


def prepare():
    from .runner import check
    check();verify_old_sources();local=HERE/'local';local.mkdir(exist_ok=True)
    with (local/'prepare.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        _prepare_locked()


def _prepare_locked():
    from .runner import check
    frozen=check();local=HERE/'local'
    if (local/'prepared.npz').exists() or (local/'prepare_started.json').exists():raise ValueError('preparation already started; preserve it')
    (local/'prepare_started.json').write_text(json.dumps({'scope':'source-only prefix preparation'})+'\n')
    started=time.monotonic()
    with np.load(OLD/'local/prepared/features.npz',allow_pickle=False) as z:
        mask=z['split']=='train';x=z['X'][mask];people=z['participant_ids'][mask];ids=z['source_ids'][mask]
    with np.load(OLD/'local/prepared/train_labels.npz',allow_pickle=False) as z:
        targets=z['targets'];guard=z['target_guard']
    root=Path.home()/'violin-rne/data/external/blanco2021_raw'
    partition=json.loads((OLD/'data_partition_receipt.json').read_text())
    if sha(root/'inventory_records.json')!=partition['original_inventory_sha256']:raise ValueError('inventory changed')
    records={r['source_id']:r for r in json.loads((root/'inventory_records.json').read_text())['records']}
    audited=set(sorted(ids,key=lambda sid:hashlib.sha256(str(sid).encode()).hexdigest())[:8])
    sequence=[];checks=0
    for i,sid in enumerate(ids):
        row=records[str(sid)];samples=read_supported_samples(row,root)
        seq=tokens(samples,row['sample_rate'],row['subtype'])
        if seq.shape!=(96,4) or not np.isfinite(seq).all():raise ValueError('prefix token contract failed')
        if sid in audited:
            changed=samples.astype(float)/32768;changed[row['sample_rate']:]=np.nan
            if not np.array_equal(seq,tokens(changed,row['sample_rate'],'DOUBLE')):raise ValueError('future affects prefix tokens')
            checks+=1
        sequence.append(seq)
        if time.monotonic()-started>600:raise TimeoutError('source preparation budget reached')
    seq=np.asarray(sequence);fold=source_folds(people)
    check()
    np.savez_compressed(local/'prepared.npz',X=x,sequence=seq,targets=targets,guard=guard,people=people,source_ids=ids,fold=fold)
    summary={'scope':'source-only exploratory preparation; no new dev/test truth access',
        'freeze_sha256':sha(HERE/'FREEZE.json'),
        'source_inputs':frozen['source_inputs'],'source_hashes':frozen['sources'],
        'rows':len(x),'people':len(np.unique(people)),'sequence_shape':list(seq.shape),
        'fold_people':[len(np.unique(people[fold==i])) for i in range(3)],
        'fold_rows':[int((fold==i).sum()) for i in range(3)],'future_mutation_checks':checks,
        'prepared_sha256':sha(local/'prepared.npz'),'seconds':time.monotonic()-started}
    (HERE/'PREPARED.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary,indent=2))


if __name__=='__main__':prepare()
