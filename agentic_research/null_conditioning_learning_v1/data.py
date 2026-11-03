"""Source-only loaders; inner-train cuts only; no IO at import."""
import numpy as np
from ..donor_deletion_learning_v1.data import (
    sha,source_folds,validate_arrays,load_arrays,prediction_inputs,
    verify_predictions,score,fit_inner as common_fit_inner)
from ..violin_raw_v1.models import patterns,pattern_ids


def fine_to_coarse():
    bank=patterns();out=[]
    for a in range(8):
        for b in range(8):
            event=np.array([a//2<=i and b//2<=j for i in range(3) for j in range(3)])
            out.append(int(np.flatnonzero((bank==event).all(1))[0]))
    out.append(int(np.flatnonzero(~bank.any(1))[0]))
    return np.asarray(out,dtype=np.int64)


def fit_fine_cuts(targets,guard,coarse_cuts):
    targets=np.asarray(targets,float);guard=np.asarray(guard);coarse=np.asarray(coarse_cuts,float)
    if targets.shape!=(len(guard),2) or guard.dtype!=np.bool_ or coarse.shape!=(2,3):
        raise ValueError('fine training schema')
    if not np.isfinite(coarse).all() or not (np.diff(coarse,axis=1)>0).all():
        raise ValueError('original coarse cuts must be strictly increasing')
    valid=targets[guard]
    if len(valid)<4 or not np.isfinite(valid).all() or (valid<0).any():raise ValueError('fine train support')
    cuts=np.quantile(valid,np.arange(1,8)/8,axis=0,method='linear').T
    cuts[:,[1,3,5]]=coarse
    if (np.diff(cuts,axis=1)<0).any():raise ValueError('octile/coarse mismatch')
    return cuts


def fine_labels(targets,guard,cuts):
    y=np.asarray(targets,float);g=np.asarray(guard);t=np.asarray(cuts,float)
    if g.dtype!=np.bool_ or y.shape!=(len(g),2) or t.shape!=(2,7):raise ValueError('fine label schema')
    if not np.isfinite(t).all() or (np.diff(t,axis=1)<0).any() or not np.isfinite(y[g]).all():
        raise ValueError('fine labels require finite valid targets and ordered cuts')
    ids=np.full(len(y),64,dtype=np.int64)
    r=(y[g,:,None]>t[None]).sum(2)
    ids[g]=8*r[:,0]+r[:,1]
    return ids


def fit_inner(data,fold):
    train,stats,_=common_fit_inner(data,fold,fit_markov=False)
    selected=data['fold']!=fold
    cuts=fit_fine_cuts(data['targets'][selected],data['guard'][selected],stats['thresholds'])
    train['fine_labels']=fine_labels(data['targets'][selected],data['guard'][selected],cuts)
    train['coarse_labels']=pattern_ids(train['labels'])
    if not np.array_equal(fine_to_coarse()[train['fine_labels']],train['coarse_labels']):
        raise ValueError('fine/coarse target inconsistency')
    return train,stats,{'cuts':cuts.tolist(),'train_fine_counts':np.bincount(train['fine_labels'],minlength=65).tolist()}
