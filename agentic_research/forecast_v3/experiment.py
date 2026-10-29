"""Trusted evaluation of future observer forecasts, never physical ground truth.

The controller must freeze code, input manifest, split, methods, and acceptance
criteria before calling evaluate on development data. This module accepts a
fixed dataset and writes only into the supplied run directory. It does not
select models, tune thresholds, or authorize evaluation on its own.
"""
from __future__ import annotations
from pathlib import Path
import hashlib
import importlib.metadata
import json
import platform
import time
import numpy as np
from . import models, rule_patterns

CLIP = 1e-6


def _serializable(value):
    if isinstance(value,dict):return {str(k):_serializable(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [_serializable(v) for v in value]
    if isinstance(value,np.ndarray):return value.tolist()
    if isinstance(value,np.generic):return value.item()
    if isinstance(value,Path):return str(value)
    return value


def _canonical(value):
    return json.dumps(_serializable(value),sort_keys=True,ensure_ascii=False,separators=(',',':'),allow_nan=False)


def _json_file(path,value):
    # New outputs are exclusive. A failed partial attempt is preserved for inspection.
    with Path(path).open('x',encoding='utf-8') as f:f.write(_canonical(value)+'\n')


def loss_rows(labels,probabilities):
    y=np.asarray(labels)
    p=models.validate_probabilities(probabilities)
    if y.shape!=p.shape or not np.isin(y,[0,1]).all():raise ValueError('Forecast targets must be aligned binary labels')
    clipped=np.clip(p,CLIP,1-CLIP)
    loss=-(y*np.log(clipped)+(1-y)*np.log1p(-clipped))
    return loss.mean(axis=1),((p-y)**2).mean(axis=1),loss


def probability_metrics(labels,probabilities):
    row_loss,row_brier,rule_loss=loss_rows(labels,probabilities)
    return {'rows':len(row_loss),'mean_log_loss':float(row_loss.mean()),
            'mean_brier':float(row_brier.mean()),
            'per_rule_log_loss':rule_loss.mean(axis=0).tolist(),
            'coherence':models.coherence(probabilities)}


def _group_metrics(labels,probabilities,group_ids):
    group_ids=np.asarray(group_ids).astype(str)
    if group_ids.shape!=(len(labels),):raise ValueError('Invalid group metadata')
    return {g:probability_metrics(labels[group_ids==g],probabilities[group_ids==g])
            for g in sorted(set(group_ids.tolist()))}


def _validate_dataset(dataset):
    required={'X','Y9','pattern_ids','ids','sessions','groups','columns','split','feature_names','metadata'}
    if not isinstance(dataset,dict) or not required.issubset(dataset):
        raise ValueError('Dataset missing required fields: '+str(sorted(required-set(dataset))))
    x=np.asarray(dataset['X']);y=np.asarray(dataset['Y9']);n=len(x)
    prefix=dataset.get('prefix_Y9',dataset.get('persistence_Y9'))
    if prefix is None:raise ValueError('Observed prefix rule labels are required')
    prefix=np.asarray(prefix)
    if x.ndim!=2 or not n or x.shape[1]!=len(dataset['feature_names']) or y.shape!=(n,9) or prefix.shape!=(n,9):
        raise ValueError('Inconsistent dataset dimensions')
    actual=rule_patterns.pattern_ids(y);rule_patterns.pattern_ids(prefix)
    if not np.array_equal(actual,np.asarray(dataset['pattern_ids'])):raise ValueError('Pattern IDs disagree with future labels')
    ids=np.asarray(dataset['ids']).astype(str)
    if ids.shape!=(n,) or len(set(ids.tolist()))!=n:raise ValueError('Stroke IDs must be unique')
    split=np.asarray(dataset['split']).astype(str)
    if split.shape!=(n,) or set(split.tolist())!={'train','dev'}:raise ValueError('Expected nonempty fixed train/dev splits')
    train=split=='train';dev=split=='dev'
    for key in ('sessions','groups','columns'):
        values=np.asarray(dataset[key]).astype(str)
        if values.shape!=(n,):raise ValueError('Metadata not aligned: '+key)
        if set(values[train]) & set(values[dev]):raise ValueError('Training/development group leakage: '+key)
    return x,y,prefix,train,dev


def evaluate(config,dataset,output_dir):
    """Evaluate one frozen configuration; write complete metrics and predictions.

    Development labels are used only after fit_predict returns. The labels,
    full stroke metadata, and targets never enter the learner's fit interface.
    No iid-row, seed-based population CI, or physical accuracy claim is made.
    """
    started=time.perf_counter();config=models.normalize_config(config)
    x,y,prefix,train,dev=_validate_dataset(dataset)
    output=Path(output_dir);output.mkdir(parents=True,exist_ok=True)
    for name in ('metrics.json','run_config.json','predictions.npz','data_receipt.json'):
        if (output/name).exists():raise FileExistsError(f'Never overwrite prior run output: {name}')
    _json_file(output/'run_config.json',config)
    metadata=_serializable(dataset['metadata'])
    _json_file(output/'data_receipt.json',metadata)
    fitted=models.fit_predict(x[train],y[train],x[dev],config,
                              train_prefix_labels=prefix[train],eval_prefix_labels=prefix[dev],output_dir=output)
    # A fixed, label-free auxiliary inference baseline; no future target is passed.
    repaired=None;repair_info=None
    if config['method'] in ('hgb_history','mlp_ind'):
        from .coherent_repair import repair
        repaired,repair_info=repair(fitted.probabilities,max_iterations=64,chunk_size=2048)
    target=y[dev];observed=prefix[dev];p=fitted.probabilities
    metrics=probability_metrics(target,p)
    ids=np.asarray(dataset['ids']).astype(str)[dev]
    groups=np.asarray(dataset['groups']).astype(str)[dev]
    sessions=np.asarray(dataset['sessions']).astype(str)[dev]
    metrics.update(schema_version=1,method=config['method'],seed=config['seed'],
        config=config,primary_metric='mean_log_loss',probability_clip=CLIP,
        source_rows=int(train.sum()),development_rows=int(dev.sum()),rules=9,
        target='future output of fixed shared-guard observer rules, not physical state truth',
        scope='historically exposed retrospective development; not independent confirmation',
        groups=_group_metrics(target,p,groups),sessions=_group_metrics(target,p,sessions),
        source_groups=sorted(set(np.asarray(dataset['groups']).astype(str)[train])),
        development_groups=sorted(set(groups)),development_sessions=sorted(set(sessions)),
        evaluation_ids_sha256=hashlib.sha256(_canonical(ids.tolist()).encode()).hexdigest(),
        input_manifest_sha256=hashlib.sha256(_canonical(metadata).encode()).hexdigest(),
        feature_names=list(dataset['feature_names']),model=fitted.info,
        verification_ok=True,verification_error_count=0,
        checks={'finite_probabilities':True,'identical_fixed_target_rows':True,
                'source_only_fit_interface':True,'target_pattern_ids_verified':True})
    transitions={'changed':np.any(target!=observed,axis=1),
                 'unchanged':np.all(target==observed,axis=1),
                 'entry':np.any(target>observed,axis=1),
                 'exit':np.any(target<observed,axis=1)}
    metrics['transition_subsets']={name:probability_metrics(target[mask],p[mask]) if mask.any()
                                  else {'rows':0,'mean_log_loss':None,'mean_brier':None}
                                  for name,mask in transitions.items()}
    metrics['transition_subset_note']='Entry/exit may overlap; descriptive subsets, not selected primary endpoints.'
    if fitted.pattern_probabilities is not None:
        ids_y=np.asarray(dataset['pattern_ids'])[dev]
        metrics['joint_pattern_log_loss']=float(-np.log(np.clip(fitted.pattern_probabilities[np.arange(len(target)),ids_y],CLIP,1.)).mean())
    predictions={'probabilities':p,'targets':target,'prefix_labels':observed,
                 'ids':ids,'groups':groups,'sessions':sessions,
                 'columns':np.asarray(dataset['columns']).astype(str)[dev],
                 'row_log_loss':loss_rows(target,p)[0],'row_brier':loss_rows(target,p)[1]}
    if fitted.pattern_probabilities is not None:predictions['pattern_probabilities']=fitted.pattern_probabilities
    if config['method']=='mlp_ind':
        projected,masses=models.project_marginals(p)
        auxiliary=probability_metrics(target,projected)
        auxiliary['groups']=_group_metrics(target,projected,groups)
        auxiliary['sessions']=_group_metrics(target,projected,sessions)
        auxiliary['definition']='Euclidean projection of signed latent pattern masses onto simplex; NOT marginal-space optimal projection; no fitting or target access.'
        metrics['auxiliary']={'latent_mass_projection':auxiliary}
        predictions['projected_probabilities']=projected
        predictions['projected_pattern_probabilities']=masses
        predictions['projected_row_log_loss']=loss_rows(target,projected)[0]
    if repaired is not None:
        repaired_metrics=probability_metrics(target,repaired)
        repaired_metrics['groups']=_group_metrics(target,repaired,groups)
        repaired_metrics['sessions']=_group_metrics(target,repaired,sessions)
        repaired_metrics['repair_audit']=repair_info
        metrics.setdefault('auxiliary',{})['forward_kl_coherent_repair']=repaired_metrics
        predictions['repaired_probabilities']=repaired
        predictions['repaired_row_log_loss']=loss_rows(target,repaired)[0]
    if config['method']=='mlp_joint' and metrics['coherence']['incoherent_rows']:
        raise AssertionError('Joint model violated its probability coherence construction')
    with (output/'predictions.npz').open('xb') as f:np.savez_compressed(f,**predictions)
    metrics['total_evaluation_seconds']=time.perf_counter()-started
    metrics['versions']={'python':platform.python_version(),**{name:importlib.metadata.version(name)
                        for name in ('numpy','scikit-learn','torch')}}
    _json_file(output/'metrics.json',metrics)
    return metrics


def run_experiment(config,data_path,output_dir):
    """Convenience adapter for a controller holding an already audited NPZ path."""
    from .data import load_dataset
    return evaluate(config,load_dataset(data_path),output_dir)
