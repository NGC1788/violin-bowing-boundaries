"""Inner-train-only statistics; prefix transform accepts no future labels."""
import numpy as np
from ..violin_raw_v1.data import labels_and_states


def fit_training_stats(prefix_train,pitch_train,power_train,voiced_train,targets_train,guard_train):
    """Caller must slice inner train FIRST. This API has no held-out selector."""
    x=np.asarray(prefix_train,float);pitch=np.asarray(pitch_train,float);power=np.asarray(power_train,float)
    voiced=np.asarray(voiced_train);targets=np.asarray(targets_train,float);guard=np.asarray(guard_train)
    n=len(x)
    if x.shape!=(n,60) or pitch.ndim!=2 or pitch.shape!=power.shape or pitch.shape!=voiced.shape or len(pitch)!=n:
        raise ValueError('bad inner-train arrays')
    if targets.shape!=(n,2) or guard.shape!=(n,) or guard.dtype!=np.bool_ or voiced.dtype!=np.bool_:
        raise ValueError('bad inner-train labels/masks')
    if not np.isfinite(x).all() or not np.isfinite(power).all() or not np.isfinite(pitch[voiced]).all():
        raise ValueError('inner-train observed arrays must be finite')
    if (x[:,57:59]<0).any() or not np.isin(x[:,59],[0,1]).all():raise ValueError('invalid prefix moments/guard')
    valid=targets[guard]
    if len(valid)<4 or not np.isfinite(valid).all() or (valid<0).any():raise ValueError('insufficient valid inner-train targets')
    threshold=np.quantile(valid,[.25,.5,.75],axis=0).T
    if not (np.diff(threshold,axis=1)>0).all():raise ValueError('degenerate inner-train thresholds')
    if not np.array_equal(guard,voiced.sum(1)>=max(2,int(np.ceil(pitch.shape[1]/2)))):
        raise ValueError('target guard differs from future mask')
    moments=np.log1p(x[:,57:59]);nobs=voiced.sum(1)
    means=np.divide(np.where(voiced,pitch,0).sum(1),nobs,out=np.zeros(n),where=nobs>0)
    pe=np.where(voiced,pitch-means[:,None],0);de=power-power.mean(1,keepdims=True)
    degrees=np.array([np.maximum(nobs-1,0).sum(),n*(power.shape[1]-1)])
    if (degrees<1).any():raise ValueError('insufficient contrast degrees of freedom')
    path_scale=np.maximum(np.sqrt(np.array([np.square(pe).sum(),np.square(de).sum()])/degrees),1e-6)
    def scale(values,axis=0):return np.maximum(np.std(values,axis=axis),1e-6)
    return {'x_mean':x.mean(0),'x_scale':scale(x),'moment_mean':moments.mean(0),'moment_scale':scale(moments),
            'path_scale':path_scale,
            'thresholds':threshold,'fit_rows':n,'fit_valid_rows':int(guard.sum())}


def transform_prefix(prefix,stats):
    x=np.asarray(prefix,float)
    if x.ndim!=2 or x.shape[1]!=60 or not np.isfinite(x).all() or (x[:,57:59]<0).any():
        raise ValueError('finite60dim prefix matrix required')
    if not np.isin(x[:,59],[0,1]).all():raise ValueError('invalid prefix guard')
    _,state=labels_and_states(x[:,57:59],x[:,59].astype(bool),stats['thresholds'],prefix=True)
    summary=np.clip((x-stats['x_mean'])/stats['x_scale'],-10,10)
    anchors=(np.log1p(x[:,57:59])-stats['moment_mean'])/stats['moment_scale']
    return np.column_stack((summary,np.eye(11)[state],anchors))


def transform_labels(pitch,power,voiced,stats):
    """Returns auxiliary LABELS, not additional inference features."""
    pitch,power,voiced=np.asarray(pitch,float),np.asarray(power,float),np.asarray(voiced)
    if voiced.dtype!=np.bool_ or pitch.shape!=power.shape or pitch.shape!=voiced.shape:raise ValueError('label shapes')
    if not np.isfinite(power).all() or not np.isfinite(pitch[voiced]).all():raise ValueError('observed label values')
    n=voiced.sum(1);mean=np.divide(np.where(voiced,pitch,0).sum(1),n,out=np.zeros(len(n)),where=n>0)
    normalized_pitch=np.where(voiced,pitch-mean[:,None],0)/stats['path_scale'][0]
    return normalized_pitch,(power-power.mean(1,keepdims=True))/stats['path_scale'][1]


def normalized_thresholds(stats):return stats['thresholds']/stats['path_scale'][:,None]
