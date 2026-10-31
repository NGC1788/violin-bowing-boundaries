"""Freeze and execute one label-free, hash-selected numerical solver audit."""
from __future__ import annotations
import argparse
import fcntl
import hashlib
from datetime import datetime, timezone
import json
import signal
import sys
import time
from pathlib import Path
import numpy as np
from scipy.special import rel_entr

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0,str(ROOT))
from agentic_research.projection_audit_v2.solver import patterns, reference_project, all_losses
from agentic_research.forecast_v3.coherent_repair import _solve


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_new(path, value):
    with path.open('x') as file:
        json.dump(value,file,indent=2,allow_nan=False)
        file.write('\n')


def selected_indices(ids, count):
    if len(set(ids.tolist())) != len(ids):
        raise ValueError('Prediction IDs are not unique')
    return np.array(sorted(range(len(ids)), key=lambda i:(
        hashlib.sha256(('projection_audit_v1:'+str(ids[i])).encode()).hexdigest(),
        str(ids[i])))[:count],dtype=int)


def load_allowed(path):
    # NPZ is lazy: do not dereference targets/row metrics/other forbidden arrays.
    with np.load(path,allow_pickle=False) as source:
        return {key:source[key].copy() for key in
                ('ids','probabilities','repaired_probabilities')}


def prediction_paths(protocol):
    return {name:ROOT/'agentic_research/forecast_v3/runs/forecast_20261030_v3'/run/'predictions.npz'
            for name,run in protocol['models'].items()}


def freeze():
    protocol=json.loads((HERE/'protocol.json').read_text())
    if datetime.now(timezone.utc) >= datetime.fromisoformat(protocol['end_date']):
        raise RuntimeError('Audit protocol expired')
    audit_snapshot=ROOT/'agentic_research/published_results/forecast_20261030_v3/post_run_artifact_checksums.json'
    prior=json.loads(audit_snapshot.read_text())['hashes']
    paths=prediction_paths(protocol)
    inputs={}
    for name,path in paths.items():
        digest=sha(path)
        key='runs/forecast_20261030_v3/'+protocol['models'][name]+'/predictions.npz'
        if digest != prior[key]: raise ValueError('Prior artifact hash mismatch: '+name)
        inputs[str(path.relative_to(ROOT))]=digest
    dependencies=[ROOT/'agentic_research/forecast_v3'/file for file in
                  ('coherent_repair.py','models.py','rule_patterns.py')]
    sources=list(HERE.glob('*.py'))+[HERE/'protocol.json']+dependencies+[ROOT/'agentic_research/projection_audit_v1/solver.py']
    manifest={'schema':1,'frozen_unix':time.time(),
              'source_hashes':{str(p.relative_to(ROOT)):sha(p) for p in sources},
              'input_hashes':inputs,
              'prior_snapshot_hash':sha(audit_snapshot),
              'scope':'One numerical audit; source/selection locked before sample evaluation; no observed target access.'}
    write_new(HERE/'freeze_manifest.json',manifest)
    print(json.dumps({'frozen':True,'sha256':sha(HERE/'freeze_manifest.json')}))


def diagnostics(q,p,bank):
    masses=np.linalg.solve(np.vstack((np.ones(len(bank)),bank.T)),np.r_[1.,p])
    feasible=bool(masses.min()>=-1e-9 and np.isfinite(p).all() and np.all((p>=0)&(p<=1)))
    value=float(np.mean(rel_entr(p,np.clip(q,1e-6,1-1e-6))+
                        rel_entr(1-p,1-np.clip(q,1e-6,1-1e-6))))
    return {'coherent':feasible,'minimum_signed_mass':float(masses.min()),
            'kl_objective_if_feasible':value if feasible else None,
            'worst_feasible_target_delta':float((all_losses(p,bank)-all_losses(q,bank)).max())}


def run():
    protocol=json.loads((HERE/'protocol.json').read_text())
    if datetime.now(timezone.utc) >= datetime.fromisoformat(protocol['end_date']):
        raise RuntimeError('Audit protocol expired')
    manifest=json.loads((HERE/'freeze_manifest.json').read_text())
    for field in ('source_hashes','input_hashes'):
        for path,digest in manifest[field].items():
            if sha(ROOT/path)!=digest:raise ValueError('Frozen hash changed: '+path)
    with (HERE/'audit.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        write_new(HERE/'run_receipt.json',{'started_unix':time.time(),
                  'manifest_sha256':sha(HERE/'freeze_manifest.json'),'max_audit_runs':1})
        def timeout(*_): raise TimeoutError('Frozen 600-second audit deadline')
        prior_handler=signal.signal(signal.SIGALRM,timeout)
        signal.alarm(protocol['budget']['max_seconds'])
        started=time.perf_counter()
        active_model=None;active_id=None;rows=[];result={}
        try:
            result={'scope':'Label-free numerical audit; no observed-label performance or model ranking',
                    'manifest_sha256':sha(HERE/'freeze_manifest.json'), 'models':{}}
            bank=patterns();common_ids=None
            for name,path in prediction_paths(protocol).items():
                active_model=name;active_id=None;rows=[]
                data=load_allowed(path)
                if any(data[key].shape!=(len(data['ids']),9) for key in ('probabilities','repaired_probabilities')):raise ValueError('Shape mismatch')
                if len(set(data['ids'].tolist())) != len(data['ids']):raise ValueError('Duplicate prediction IDs')
                if common_ids is None:
                    indices=selected_indices(data['ids'],protocol['sample_size'])
                    common_ids=data['ids'][indices]
                else:
                    positions={v:i for i,v in enumerate(data['ids'])}
                    indices=np.array([positions[v] for v in common_ids])
                q=data['probabilities'][indices]
                stored=data['repaired_probabilities'][indices]
                base_start=time.perf_counter()
                raw,_,_= _solve(q,64)
                batch_seconds=time.perf_counter()-base_start
                rows=[]
                for i,row in enumerate(q):
                    active_id=str(common_ids[i])
                    ref=reference_project(row,bank)
                    raw_d=diagnostics(row,raw[i],bank)
                    stored_d=diagnostics(row,stored[i],bank)
                    original_d=diagnostics(row,row,bank)
                    ref_d=diagnostics(row,ref['probabilities'],bank)
                    returned_d=diagnostics(row,ref['returned_probabilities'],bank)
                    rows.append({'id':str(common_ids[i]),'original':original_d,
                        'stored_guarded_repair':stored_d,'raw_64_iteration_candidate':raw_d,
                        'reference':{k:v for k,v in ref.items() if k not in
                                     ('probabilities','returned_probabilities','weights')},
                        'reference_candidate':ref_d,'reference_guarded_return':returned_d,
                        'raw_objective_minus_reference':(raw_d['kl_objective_if_feasible']-ref['objective'])
                            if raw_d['kl_objective_if_feasible'] is not None else None})
                result['models'][name]={'rows':rows,'summary':{
                    'n':len(rows),
                    'original_coherent':sum(r['original']['coherent'] for r in rows),
                    'stored_repair_coherent':sum(r['stored_guarded_repair']['coherent'] for r in rows),
                    'raw64_coherent':sum(r['raw_64_iteration_candidate']['coherent'] for r in rows),
                    'reference_accepted':sum(r['reference']['accepted'] for r in rows),
                    'reference_return_coherent':sum(r['reference_guarded_return']['coherent'] for r in rows),
                    'reference_gap_pass':sum(r['reference']['gap_pass'] for r in rows),
                    'reference_identity':sum(r['reference']['identity'] for r in rows),
                    'identity_outside_clip':sum(r['reference']['identity_outside_clip'] for r in rows),
                    'reference_candidate_coherent':sum(r['reference_candidate']['coherent'] for r in rows),
                    'reference_score_guard_pass':sum(r['reference']['worst_feasible_target_delta']<=0 for r in rows),
                    'reference_max_fw_gap':max((r['reference']['fw_gap'] for r in rows if r['reference']['fw_gap'] is not None),default=None),
                    'reference_max_feasible_target_delta':max(r['reference_candidate']['worst_feasible_target_delta'] for r in rows),
                    'reference_return_max_delta':max(r['reference_guarded_return']['worst_feasible_target_delta'] for r in rows),
                    'median_raw_objective_minus_reference':float(np.median([r['raw_objective_minus_reference'] for r in rows if r['raw_objective_minus_reference'] is not None])),
                    'max_raw_objective_minus_reference':max(r['raw_objective_minus_reference'] for r in rows if r['raw_objective_minus_reference'] is not None),
                    'reference_seconds_total':sum(r['reference']['seconds'] for r in rows),
                    'raw64_batch_seconds':batch_seconds}}
            for field in ('source_hashes','input_hashes'):
                for frozen_path,digest in manifest[field].items():
                    if sha(ROOT/frozen_path)!=digest:raise ValueError('Hash changed during audit: '+frozen_path)
            result['elapsed_seconds']=time.perf_counter()-started
            write_new(HERE/'results.json',result)
            write_new(HERE/'result_seal.json',{'manifest_sha256':sha(HERE/'freeze_manifest.json'),
                      'results_sha256':sha(HERE/'results.json'),'scope':'Same-user integrity, not external attestation'})
            print(json.dumps({k:v['summary'] for k,v in result['models'].items()},indent=2))
        except BaseException as exc:
            write_new(HERE/'failure.json',{'type':type(exc).__name__,'error':str(exc),
                      'elapsed_seconds':time.perf_counter()-started,
                      'active_model':active_model,'failed_id':active_id,
                      'completed_models':result.get('models',{}),'completed_rows_in_active_model':rows})
            raise
        finally:
            signal.alarm(0);signal.signal(signal.SIGALRM,prior_handler)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('command',choices=('freeze','run'))
    args=parser.parse_args()
    (freeze if args.command=='freeze' else run)()
