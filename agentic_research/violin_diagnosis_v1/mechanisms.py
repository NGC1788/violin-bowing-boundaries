"""Frozen-endpoint measurement probes. No fitting or test-truth access."""
from pathlib import Path
import hashlib
import json
import numpy as np
import torch
from ..violin_raw_v1.features import feature_names, interval_frames, PREFIX, TARGET, extract_example
from ..violin_raw_v1.data import labels_and_states
from ..violin_raw_v1.models import participant_weights, transition_prior
from ..violin_raw_v1.prepare import read_supported_samples
from .diagnose import ORIGINAL, HERE, sha, check_original, restore_model, offset_geometry


def rms_frames(samples, sr, interval):
    start, stop = (round(t * sr) for t in interval)
    segment = samples[start:stop]; width = round(.05 * sr)
    starts = [round(i * .01 * sr) for i in range(round((interval[1]-interval[0])/.01)+1)]
    result = []
    for index in starts:
        if index + width > len(segment): continue
        frame = segment[index:index+width]; frame = frame - frame.mean()
        result.append(np.sqrt(np.mean(frame * frame)))
    return np.asarray(result)


def gain_admissible(samples, sr, gain):
    if np.max(np.abs(samples)) * gain > 1:
        return False
    rms = np.r_[rms_frames(samples, sr, PREFIX), rms_frames(samples, sr, TARGET)]
    return bool(np.all(np.minimum(rms, gain*rms) > 1e-12) and
                np.array_equal(rms >= 1e-4, gain * rms >= 1e-4))


def gain_shift(x, gain):
    result = np.asarray(x).copy()
    chosen = [i for i, name in enumerate(feature_names()) if name.startswith('power_dbfs_')
              and (name.endswith('_mean') or name.endswith('_p10') or name.endswith('_p90'))]
    assert len(chosen) == 9
    result[:, chosen] += 20 * np.log10(gain)
    return result


def rank_correlation(a, b):
    # Descriptive average-rank Spearman; no hypothesis test or causal claim.
    def ranks(values):
        values = np.asarray(values); return np.array([((values < v).sum() + ((values == v).sum()-1)/2) for v in values])
    a, b = ranks(a), ranks(b)
    return float(np.corrcoef(a, b)[0, 1]) if a.std() and b.std() else None


def main():
    torch.set_num_threads(1); check_original(ORIGINAL)
    prepared = ORIGINAL/'local/prepared'; run = ORIGINAL/'local/campaign_v1'
    with np.load(prepared/'features.npz', allow_pickle=False) as z:
        selected = z['split'] != 'test'; split = z['split'][selected]
        x = z['X'][selected]; states = z['prefix_states'][selected]
        people = z['participant_ids'][selected]; ids = z['source_ids'][selected]
    train = split == 'train'; dev = split == 'dev'
    with np.load(prepared/'train_labels.npz', allow_pickle=False) as z: y = z['Y9']
    _, prior = transition_prior(y, states[train], participant_weights(people[train]))
    inventory_root = Path.home()/'violin-rne/data/external/blanco2021_raw'
    inventory_path = inventory_root/'inventory_records.json'
    partition = json.loads((ORIGINAL/'data_partition_receipt.json').read_text())
    if sha(inventory_path) != partition['original_inventory_sha256']:
        raise ValueError('raw inventory identity changed')
    records = json.loads(inventory_path.read_text())['records']
    by_id = {r['source_id']:r for r in records}
    admitted = {g: np.zeros(len(ids), bool) for g in (.5, 2.)}
    audits = {g: [] for g in (.5, 2.)}
    receipt = json.loads((prepared/'receipt.json').read_text()); thresholds = np.array(receipt['train_thresholds'])
    target_by_part = {}
    for part in ('train','dev'):
        with np.load(prepared/f'{part}_labels.npz',allow_pickle=False) as z:
            target_by_part[part] = {key:z[key].copy() for key in ('Y9','targets','target_guard')}
    local_index = np.empty(len(ids),int)
    for part in ('train','dev'): local_index[split==part] = np.arange((split==part).sum())
    for i, sid in enumerate(ids):
        r = by_id[str(sid)]
        samples = read_supported_samples(r, inventory_root).astype(float)/32768
        for gain in admitted: admitted[gain][i] = gain_admissible(samples, r['sample_rate'], gain)
    for gain in admitted:
        audit_indices = sorted(np.flatnonzero(admitted[gain]), key=lambda i:hashlib.sha256(str(ids[i]).encode()).hexdigest())[:8]
        for i in audit_indices:
            r = by_id[str(ids[i])]
            samples = read_supported_samples(r, inventory_root).astype(float)/32768
            original = extract_example(samples,r['sample_rate'],'DOUBLE')
            changed = extract_example(samples*gain,r['sample_rate'],'DOUBLE')
            before, _ = labels_and_states(original['targets'][None], np.array([original['target_guard']]), thresholds)
            after, _ = labels_and_states(changed['targets'][None], np.array([changed['target_guard']]), thresholds)
            _, prefix_state = labels_and_states(original['prefix_targets'][None], np.array([original['prefix_guard']]), thresholds, prefix=True)
            reference = target_by_part[str(split[i])]; row = local_index[i]
            if not (np.allclose(original['X'],x[i],rtol=0,atol=1e-10) and
                    int(prefix_state[0]) == int(states[i]) and
                    original['target_guard'] == reference['target_guard'][row] and
                    np.array_equal(before[0],reference['Y9'][row]) and
                    np.allclose(original['targets'],reference['targets'][row],rtol=0,atol=1e-10,equal_nan=True)):
                raise ValueError('recomputed original differs from frozen prepared record')
            audits[gain].append({'feature_max_error':float(np.max(np.abs(gain_shift(original['X'][None],gain)[0]-changed['X']))),
                'original_matches_frozen_prepared':True,
                'target_max_error':float(np.nanmax(np.abs(original['targets']-changed['targets']))),
                'guard_unchanged':original['target_guard']==changed['target_guard'], 'rules_unchanged':bool(np.array_equal(before,after))})
    summary = json.loads((HERE/'SUMMARY.json').read_text())
    if (sha(HERE/'local/diagnostics.json') != summary['diagnostics_sha256'] or
            sha(HERE/'diagnose.py') != summary['diagnostic_code_sha256']):
        raise ValueError('diagnosis source/result identity changed')
    diagnostic = json.loads((HERE/'local/diagnostics.json').read_text())
    result = {'scope':'post-hoc train/dev measurement probes; no test truth, fit or alternative performance score',
              'code_sha256':sha(Path(__file__)), 'inventory_sha256':sha(inventory_path),
              'new_models_fitted':0,'test_labels_accessed':False,'gain_raw_audit':{str(g):audits[g] for g in audits},
              'admitted_counts':{str(g):{s:int((admitted[g]&(split==s)).sum()) for s in ['train','dev']} for g in admitted},
              'dev_people':[], 'state_support':diagnostic['state_support'],
              'admission_warning':'Uses future support only to verify diagnostic invariance; not usable as an inference gate.'}
    reference_receipt = json.loads((run/'mlp_ind_s0/receipt.json').read_text())
    z = (x-np.array(reference_receipt['scaler_mean']))/np.array(reference_receipt['scaler_scale'])
    for i,pid in enumerate(np.unique(people[dev])):
        mask=dev&(people==pid); magnitudes=np.abs(z[mask]).mean(0); largest=np.argsort(magnitudes)[-6:][::-1]
        result['dev_people'].append({'anonymous_index':i+1,'cohort':str(pid).rsplit('_',1)[-1],
            'top_standardized_feature_magnitudes':[{ 'feature':feature_names()[j], 'mean_abs_z':float(magnitudes[j])} for j in largest],
            'gain_sensitivity':{}})
    sensitivity={method:{g:[] for g in admitted} for method in ['mlp_ind','mlp_joint','mlp_offset']}
    geometry=[]
    for method in sensitivity:
        for seed in (0,1,2):
            job=run/f'{method}_s{seed}';meta=json.loads((job/'receipt.json').read_text())
            expected=diagnostic['weights'][job.name]
            if sha(job/'weights.npz')!=expected:raise ValueError('diagnostic weight identity changed')
            original, mass, residual, _=restore_model(job/'weights.npz',meta,x,states,prior,method)
            if method=='mlp_offset':
                geometry.append([offset_geometry(mass[dev&(people==pid)],residual[dev&(people==pid)],
                    states[dev&(people==pid)],prior,people[dev&(people==pid)]) for pid in np.unique(people[dev])])
            for gain in sensitivity[method]:
                changed,_,_,_=restore_model(job/'weights.npz',meta,gain_shift(x,gain),states,prior,method)
                diff=np.abs(changed-original).mean(1)
                sensitivity[method][gain].append([float(diff[(people==pid)&dev&admitted[gain]].mean())
                    if ((people==pid)&dev&admitted[gain]).any() else None for pid in np.unique(people[dev])])
    result['associations']={}
    for i,person in enumerate(result['dev_people']):
        person['offset_geometry']={k:float(np.mean([g[i][k] for g in geometry])) for k in geometry[0][i]}
        for method in sensitivity:
            person['gain_sensitivity'][method]={}
            for gain in sensitivity[method]:
                vals=[s[i] for s in sensitivity[method][gain]]
                person['gain_sensitivity'][method][str(gain)]={'mean_absolute_probability_change':float(np.mean(vals)) if vals[0] is not None else None,
                    'admitted_rows':int((dev&(people==np.unique(people[dev])[i])&admitted[gain]).sum())}
    for method in sensitivity:
        excess=diagnostic['models'][method]['dev_mean']['per_person_excess_vs_markov']
        result['associations'][method]={}
        for gain in sensitivity[method]:
            a=[p['gain_sensitivity'][method][str(gain)]['mean_absolute_probability_change'] for p in result['dev_people']]
            valid=[i for i,v in enumerate(a) if v is not None]
            result['associations'][method][str(gain)]={'people':len(valid),'spearman_sensitivity_vs_excess':rank_correlation([a[i] for i in valid],[excess[i] for i in valid])}
    check_original(ORIGINAL)
    result['original_artifacts_verified_unchanged']=True
    (HERE/'local/mechanisms.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'admitted_counts':result['admitted_counts'],'raw_audit_count':{str(g):len(audits[g]) for g in audits},
        'max_raw_feature_action_error':max(a['feature_max_error'] for records in audits.values() for a in records),
        'raw_rule_preservation':all(a['rules_unchanged'] for records in audits.values() for a in records),
        'associations':result['associations']},indent=2))


if __name__=='__main__':main()
