"""Auditable source-only arrays and train/held separation. No automatic IO."""
import hashlib,json
from pathlib import Path
import numpy as np
from ..exact_count_learning_v1.preprocess import fit_training_stats,transform_prefix,transform_labels,normalized_thresholds
from ..violin_raw_v1.data import labels_and_states
from ..violin_raw_v1.models import participant_weights,transition_prior,patterns


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(4*1024**2),b''):h.update(b)
    return h.hexdigest()


def source_folds(people):
    lookup={}
    for cohort in ['CG','EG','FG']:
        ids=[p for p in np.unique(people) if str(p).rsplit('_',1)[-1]==cohort]
        ids.sort(key=lambda p:hashlib.sha256(('20261102:'+str(p)).encode()).hexdigest())
        lookup.update({p:i%3 for i,p in enumerate(ids)})
    if len(lookup)!=len(np.unique(people)):raise ValueError('unknown source cohort')
    return np.array([lookup[p] for p in people],int)


def validate_arrays(prefix,meta,labels,contract):
    if set(prefix)!={'X','sequence'} or set(meta)!={'people','source_ids','fold'}:
        raise ValueError('unexpected prefix/meta keys')
    expected={'pitch_cents','power_dbfs','periodicity','voiced','targets','guard','support'}
    if set(labels)!=expected:raise ValueError('unexpected future-label keys')
    x=np.asarray(prefix['X']);people=np.asarray(meta['people']);ids=np.asarray(meta['source_ids']);fold=np.asarray(meta['fold'])
    n=contract['expected_rows'];t=contract['model_config']['future_frames']
    if x.shape!=(n,60) or prefix['sequence'].shape!=(n,96,4) or not np.isfinite(x).all() or not np.isfinite(prefix['sequence']).all():
        raise ValueError('prefix shape or finite support')
    if (x[:,57:59]<0).any() or not np.isin(x[:,59],[0,1]).all():raise ValueError('invalid prefix moments/guard')
    if people.shape!=(n,) or ids.shape!=(n,) or fold.shape!=(n,) or people.dtype.kind not in 'US' or ids.dtype.kind not in 'US':
        raise ValueError('row metadata shape/type')
    if len(np.unique(ids))!=n or not all(str(i) for i in ids) or not all(str(p) for p in people):raise ValueError('duplicate/missing identities')
    if len(np.unique(people))!=contract['expected_people'] or not np.array_equal(fold,source_folds(people)):
        raise ValueError('person split/provenance changed')
    if [int((fold==i).sum()) for i in range(3)]!=contract['fold_rows'] or [len(np.unique(people[fold==i])) for i in range(3)]!=contract['fold_people']:
        raise ValueError('fold counts changed')
    for k in ['pitch_cents','power_dbfs','periodicity','voiced']:
        if labels[k].shape!=(n,t):raise ValueError('future shape')
    m=labels['voiced'];g=labels['guard'];target=labels['targets']
    if m.dtype!=np.bool_ or g.dtype!=np.bool_ or g.shape!=(n,) or target.shape!=(n,2):raise ValueError('mask/guard/target types')
    if not np.isfinite(labels['pitch_cents'][m]).all() or not np.isfinite(labels['power_dbfs']).all() or not np.isfinite(labels['periodicity']).all():
        raise ValueError('nonfinite observed future')
    expected_guard=m.sum(1)>=contract['model_config']['minimum_voiced']
    if not np.array_equal(g,expected_guard) or int((~g).sum())!=contract['expected_invalid']:raise ValueError('invalid rows/guard changed')
    rebuilt=np.column_stack((np.array([np.std(p[mask],ddof=1) if valid else np.nan for p,mask,valid in zip(labels['pitch_cents'],m,g)]),
                              labels['power_dbfs'].std(1,ddof=1)))
    if not np.allclose(rebuilt,target,atol=1e-10,rtol=1e-10,equal_nan=True):raise ValueError('continuous target reconstruction changed')
    support=labels['support'];starts=55125+np.arange(t)*441
    expected_support=np.column_stack((starts,starts+2205))
    if support.shape!=(n,t,2) or not np.array_equal(support,np.broadcast_to(expected_support,(n,t,2))):raise ValueError('future timestamp support changed')
    return {'X':x,'people':people,'source_ids':ids,'fold':fold,'pitch':labels['pitch_cents'],
            'power':labels['power_dbfs'],'voiced':m,'targets':target,'guard':g}


def load_arrays(paths,expected,contract):
    """Explicit paths/hashes only; no glob or old-dev/test truth access."""
    keys=['prefix_inputs.npz','source_metadata.npz','future_labels.npz']
    out=[]
    for name in keys:
        p=Path(paths[name])
        if p.is_symlink() or sha(p)!=expected[name]:raise ValueError('sealed input changed: '+name)
        with np.load(p,allow_pickle=False) as z:out.append({k:z[k] for k in z.files})
        if sha(p)!=expected[name]:raise ValueError('input changed during load')
    return validate_arrays(*out,contract)


def fit_inner(data,fold,*,fit_markov=False):
    """Slice every supervised array BEFORE fitting any statistic."""
    train=data['fold']!=fold
    args=[data[k][train] for k in ['X','pitch','power','voiced','targets','guard']]
    stats=fit_training_stats(*args)
    pitch,power=transform_labels(args[1],args[2],args[3],stats)
    y,_=labels_and_states(args[4],args[5],stats['thresholds'])
    weights=participant_weights(data['people'][train]);prior=None
    if fit_markov:
        _,states=labels_and_states(args[0][:,57:59],args[0][:,59].astype(bool),stats['thresholds'],prefix=True)
        _,prior=transition_prior(y,states,weights)
    return {'prefix':transform_prefix(args[0],stats),'pitch':pitch,'power':power,'voiced':args[3],
            'labels':y.astype(float),'weights':weights,'thresholds':normalized_thresholds(stats)},stats,prior


def prediction_inputs(data,fold,stats):
    held=data['fold']==fold
    # No held future target, mask, center, guard or support is touched here.
    return transform_prefix(data['X'][held],stats),np.flatnonzero(held)


def markov_prediction(data,fold,stats,prior):
    held=data['fold']==fold
    _,state=labels_and_states(data['X'][held,57:59],data['X'][held,59].astype(bool),stats['thresholds'],prefix=True)
    return prior[state]@patterns()


def verify_predictions(probability,row_indices,expected_indices):
    p=np.asarray(probability);i=np.asarray(row_indices)
    if not np.array_equal(i,expected_indices) or p.shape!=(len(expected_indices),9) or not np.isfinite(p).all():
        raise ValueError('prediction row order/shape/finite mismatch')
    if (p<0).any() or (p>1).any():raise ValueError('probabilities outside[0,1]')
    return p


def score(y,p,people):
    from ..violin_raw_v1.metrics import row_scores,coherence_rate
    loss,brier=row_scores(y,p);unique=np.unique(people)
    means=np.array([loss[people==u].mean() for u in unique]);bs=np.array([brier[people==u].mean() for u in unique])
    return {'log_loss':float(means.mean()),'brier':float(bs.mean()),'maximum_person_loss':float(means.max()),
            'people':len(unique),'rows':len(y),'coherence_rate':coherence_rate(p)},means


def training_memory(data,fold,train):
    selected=np.flatnonzero(data['fold']!=fold)
    people=data['people'][selected];unique=np.unique(people)
    groups=np.searchsorted(unique,people).astype(np.int64)
    return {'prefix':train['prefix'],'labels':train['labels'],'groups':groups,'source_rows':selected}


def training_support_summary(memory):
    """Raw support on source training queries only, before any model fitting."""
    groups=memory['groups'];labels=memory['labels'];unique=np.unique(groups)
    missing_positive=missing_negative=cells=0
    for query_group in unique:
        for removed in [-1,*[int(g) for g in unique if g!=query_group]]:
            keep=(groups!=query_group)&(groups!=removed)
            if not keep.any():raise ValueError('empty training donor remainder')
            missing_positive+=int((labels[keep].sum(0)==0).sum())
            missing_negative+=int(((1-labels[keep]).sum(0)==0).sum());cells+=9
    return {'query_group_macro_donor_event_cells':cells,'zero_positive_support_cells':missing_positive,
            'zero_negative_support_cells':missing_negative,'groups':len(unique),'memory_rows':len(groups),
            'scope':'training only; raw empirical support before fixed common uniform prior'}
