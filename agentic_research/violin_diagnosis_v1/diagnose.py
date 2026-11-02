"""Forward-only post-hoc diagnosis. Never load test truth or fit a new model."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from torch import nn
from ..violin_raw_v1.models import patterns, pattern_ids, participant_weights, transition_prior

HERE = Path(__file__).resolve().parent
ORIGINAL = HERE.parent / 'violin_raw_v1'
EPS = 1e-6


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def macro(values, people):
    v, people = np.asarray(values), np.asarray(people)
    return float(np.mean([v[people == pid].mean() for pid in np.unique(people)]))


def losses(y, p):
    c = np.clip(np.asarray(p, float), EPS, 1 - EPS)
    return -(y * np.log(c) + (1 - y) * np.log1p(-c))


def local_directions(y, p, baseline):
    """Derivatives at epsilon=0, not a calibration fit or mixture search."""
    c = np.clip(p, EPS, 1 - EPS)
    delta = baseline - p
    interior = (p > EPS) & (p < 1 - EPS)
    active = interior | ((p == EPS) & (delta > 0)) | ((p == 1 - EPS) & (delta < 0))
    mixture = np.where(active, (c - y) / (c * (1 - c)) * delta, 0.0)
    # At either clip boundary attenuation moves inward: retain that right derivative.
    soft_active = (p >= EPS) & (p <= 1 - EPS)
    attenuation = np.where(soft_active, -(p - y) * (np.log(c) - np.log1p(-c)), 0.0)
    return mixture, attenuation


def statistics(y, p, people, baseline=None):
    ll = losses(y, p)
    wrong = ((y == 1) & (p < .05)) | ((y == 0) & (p > .95))
    loss = macro(ll, people)
    result = {'log_loss': loss, 'brier': macro((p - y) ** 2, people),
              'wrong_95_fraction': macro(wrong, people),
              'wrong_95_loss_contribution': macro(ll * wrong, people),
              'wrong_95_loss_share': macro(ll * wrong, people) / loss,
              'probability_clipped_fraction': macro((p <= EPS) | (p >= 1 - EPS), people),
              'probability_exact_clip_boundary_count': int(((p == EPS) | (p == 1 - EPS)).sum()),
              'mean_binary_entropy': macro(-(np.clip(p, EPS, 1-EPS) * np.log(np.clip(p, EPS, 1-EPS)) +
                 (1-np.clip(p, EPS, 1-EPS)) * np.log1p(-np.clip(p, EPS, 1-EPS))), people),
              'per_person_loss': [float(ll[people == pid].mean()) for pid in np.unique(people)]}
    if baseline is not None:
        mix, attenuation = local_directions(y, p, baseline)
        diff = ll.mean(1) - losses(y, baseline).mean(1)
        per_person = np.array([diff[people == pid].mean() for pid in np.unique(people)])
        result.update({'macro_excess_vs_markov': float(per_person.mean()),
            'per_person_excess_vs_markov': per_person.tolist(),
            'people_worse_than_markov': int((per_person > 0).sum()),
            'largest_person_share_of_total_excess': float(per_person.max() / per_person.sum()) if per_person.sum() > 0 else None,
            'gradient_toward_markov_at_zero': macro(mix, people),
            'gradient_soften_marginal_logits_at_zero': macro(attenuation, people),
            'people_with_negative_markov_gradient': int(sum(mix[people == pid].mean() < 0 for pid in np.unique(people)))})
    return result


def restore_model(weight_path, receipt, x, state, prior, method):
    """No optimizer is constructed; all saved parameters load strictly."""
    with np.load(weight_path, allow_pickle=False) as z:
        weights = {k: torch.from_numpy(z[k].copy()) for k in z.files}
    hidden = weights['0.weight'].shape[0]; outputs = weights['2.weight'].shape[0]
    model = nn.Sequential(nn.Linear(60, hidden), nn.GELU(), nn.Linear(hidden, outputs))
    model.load_state_dict(weights, strict=True); model.eval()
    mean, scale = np.array(receipt['scaler_mean']), np.array(receipt['scaler_scale'])
    standardized = (np.asarray(x, float) - mean) / scale
    tx = torch.tensor(np.clip(standardized, -8, 8), dtype=torch.float32)
    with torch.inference_mode():
        raw_logits = model(tx)
        if method == 'mlp_ind':
            p = torch.sigmoid(raw_logits).numpy().astype(float)
            return p, None, None, standardized
        logits = raw_logits
        if method == 'mlp_offset': logits = logits + torch.tensor(np.log(prior[state]), dtype=torch.float32)
        mass = torch.softmax(logits, 1).numpy().astype(float)
        mass /= mass.sum(1, keepdims=True)
        return mass @ patterns(), mass, raw_logits.numpy().astype(float), standardized


def offset_geometry(mass, residual, state, prior, people):
    pi = prior[state]
    centered_r = residual - residual.mean(1, keepdims=True)
    log_prior = np.log(pi); centered_prior = log_prior - log_prior.mean(1, keepdims=True)
    rn = np.linalg.norm(centered_r, axis=1); pn = np.linalg.norm(centered_prior, axis=1)
    cosine = np.sum(centered_r * (-centered_prior), axis=1) / np.maximum(rn * pn, 1e-12)
    kl = np.sum(np.where(mass > 0, mass * (np.log(np.maximum(mass, 1e-300)) - log_prior), 0), axis=1)
    qh = -np.sum(mass * np.log(np.maximum(mass, 1e-300)), axis=1)
    ph = -np.sum(pi * log_prior, axis=1)
    return {'centered_residual_norm': macro(rn, people), 'centered_log_prior_norm': macro(pn, people),
            'cosine_against_centered_log_prior': macro(cosine, people),
            'fraction_residual_norm_exceeds_prior_norm': macro(rn > pn, people),
            'KL_pattern_posterior_to_prior': macro(kl, people),
            'pattern_entropy': macro(qh, people), 'prior_pattern_entropy': macro(ph, people)}


def check_original(original):
    root = original.parents[1]
    freeze = json.loads((original / 'execution_freeze_manifest.json').read_text())
    for name, expected in freeze['source_hashes'].items():
        if sha(root / name) != expected: raise ValueError('original source changed')
    prepared = original / 'local/prepared'
    # Do not even open test_labels.npz in this diagnostic stage.
    for name in ('features.npz', 'train_labels.npz', 'dev_labels.npz'):
        if sha(prepared / name) != freeze['prepared_files'][name]: raise ValueError('original prepared input changed')
    seal = json.loads((original / 'local/campaign_v1/predictions_seal.json').read_text())
    for name, expected in seal['files'].items():
        if sha(original / 'local/campaign_v1' / name) != expected: raise ValueError('original prediction changed')
    results_seal = json.loads((original / 'RESULTS_SEAL.json').read_text())
    if sha(original / 'RESULTS.json') != results_seal['results_sha256']: raise ValueError('original results changed')
    return freeze, seal


def main():
    torch.set_num_threads(1)
    original = ORIGINAL; prepared = original / 'local/prepared'; run = original / 'local/campaign_v1'
    before, _ = check_original(original)
    with np.load(prepared / 'features.npz', allow_pickle=False) as z:
        split = z['split']; people_all = z['participant_ids']; states_all = z['prefix_states']
        train = split == 'train'; dev = split == 'dev'; held = split != 'train'
        x = {'train': z['X'][train], 'dev': z['X'][dev]}
    people = {'train': people_all[train], 'dev': people_all[dev]}
    states = {'train': states_all[train], 'dev': states_all[dev]}
    truth = {}
    for name in ('train', 'dev'):
        with np.load(prepared / f'{name}_labels.npz', allow_pickle=False) as z: truth[name] = z['Y9']
    _, prior = transition_prior(truth['train'], states['train'], participant_weights(people['train']))
    baseline = {name: prior[states[name]] @ patterns() for name in ('train', 'dev')}
    report = {'scope': 'post-hoc train/dev endpoint diagnosis only; no new fit or test truth access',
              'new_models_fitted': 0, 'test_labels_accessed': False,
              'original_result_sha256': sha(original / 'RESULTS.json'), 'diagnostic_code_sha256': sha(__file__),
              'source_counts': {'train_people': len(set(people['train'])), 'dev_people': len(set(people['dev'])),
                                'train_rows': len(x['train']), 'dev_rows': len(x['dev'])},
              'weights': {}, 'dev_reproduction_max_abs': {}, 'models': {}, 'offset_geometry': {},
              'feature_shift': {}, 'state_support': {}}
    for method in ('frequency', 'markov', 'hgb'):
        with np.load(run / f'{method}_s0/predictions.npz', allow_pickle=False) as z:
            p = z['probability'][split[held] == 'dev']
        if method == 'markov':
            error = float(np.max(np.abs(p - baseline['dev'])))
            if error > 1e-12: raise ValueError('source Markov reconstruction differs')
            report['dev_reproduction_max_abs']['markov'] = error
        report['models'][method] = {'dev': statistics(truth['dev'], p, people['dev'], baseline['dev'])}
    report['models']['markov']['train'] = statistics(truth['train'], baseline['train'], people['train'])
    reconstructed = {}; geometry = {'train': [], 'dev': []}
    for method in ('mlp_ind', 'mlp_joint', 'mlp_offset'):
        result = {'train': [], 'dev': []}
        for seed in (0, 1, 2):
            job = run / f'{method}_s{seed}'; receipt = json.loads((job / 'receipt.json').read_text())
            weight = job / 'weights.npz'; report['weights'][job.name] = sha(weight)
            predictions = {}
            for part in ('train', 'dev'):
                p, mass, residual, standardized = restore_model(weight, receipt, x[part], states[part], prior, method)
                predictions[part] = p
                result[part].append(statistics(truth[part], p, people[part], baseline[part]))
                if method == 'mlp_offset':
                    geometry[part].append(offset_geometry(mass, residual, states[part], prior, people[part]))
                if seed == 0 and method == 'mlp_ind':
                    per_person = []
                    for i, pid in enumerate(np.unique(people[part])):
                        z = standardized[people[part] == pid]
                        per_person.append({'anonymous_index': i + 1, 'cohort': str(pid).rsplit('_', 1)[-1],
                            'fraction_features_abs_z_gt_3': float((np.abs(z) > 3).mean()),
                            'fraction_features_clipped_at_8': float((np.abs(z) > 8).mean()),
                            'max_abs_z': float(np.abs(z).max())})
                    report['feature_shift'][part] = {'macro_fraction_abs_z_gt_3': macro(np.abs(standardized) > 3, people[part]),
                        'macro_fraction_clipped_at_8': macro(np.abs(standardized) > 8, people[part]), 'people': per_person}
            with np.load(job / 'predictions.npz', allow_pickle=False) as z:
                original_dev = z['probability'][split[held] == 'dev']
            error = float(np.max(np.abs(predictions['dev'] - original_dev)))
            report['dev_reproduction_max_abs'][job.name] = error
            if error > 1e-6: raise ValueError('saved weights cannot reproduce sealed development predictions')
        report['models'][method] = result
        for part in ('train', 'dev'):
            scalar = ['log_loss', 'brier', 'wrong_95_fraction', 'wrong_95_loss_contribution', 'wrong_95_loss_share',
                      'probability_clipped_fraction', 'mean_binary_entropy', 'macro_excess_vs_markov',
                      'gradient_toward_markov_at_zero', 'gradient_soften_marginal_logits_at_zero']
            report['models'][method][part + '_mean'] = {k: float(np.mean([s[k] for s in result[part]])) for k in scalar}
            per_person = np.mean([s['per_person_excess_vs_markov'] for s in result[part]], axis=0)
            report['models'][method][part + '_mean']['per_person_excess_vs_markov'] = per_person.tolist()
            report['models'][method][part + '_mean']['largest_person_share_of_total_excess'] = float(per_person.max() / per_person.sum()) if per_person.sum() > 0 else None
    for part in ('train', 'dev'):
        report['offset_geometry'][part] = {k: float(np.mean([g[k] for g in geometry[part]])) for k in geometry[part][0]}
        report['state_support'][part] = {str(s): {'rows': int((states[part] == s).sum()),
            'people': len(set(people[part][states[part] == s]))} for s in range(11)}
    train_ids, dev_ids = pattern_ids(truth['train']), pattern_ids(truth['dev'])
    report['pattern_distribution'] = {}
    for part, ids in [('train', train_ids), ('dev', dev_ids)]:
        w = participant_weights(people[part]); freq = np.bincount(ids, weights=w, minlength=10) / w.sum()
        report['pattern_distribution'][part] = freq.tolist()
    p, q = (np.asarray(report['pattern_distribution'][s]) for s in ['train', 'dev']); m = (p + q) / 2
    def kl(a, b): return float(np.sum(a[a > 0] * np.log(a[a > 0] / b[a > 0])))
    report['pattern_distribution']['JS_nats'] = (kl(p, m) + kl(q, m)) / 2
    after, _ = check_original(original)
    if before != after: raise ValueError('original manifest changed during diagnosis')
    report['original_frozen_artifacts_verified_unchanged'] = True
    local = HERE / 'local'; local.mkdir(exist_ok=True)
    (local / 'diagnostics.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    summary = {'scope': report['scope'], 'new_models_fitted': 0, 'test_labels_accessed': False,
        'original_result_sha256': report['original_result_sha256'], 'diagnostic_code_sha256': sha(__file__),
        'diagnostics_sha256': sha(local / 'diagnostics.json'),
        'max_dev_reproduction_error': max(report['dev_reproduction_max_abs'].values()),
        'original_frozen_artifacts_verified_unchanged': True,
        'train_log_loss': {'markov': report['models']['markov']['train']['log_loss'], **{
            method: report['models'][method]['train_mean']['log_loss'] for method in ('mlp_ind', 'mlp_joint', 'mlp_offset')}},
        'dev_log_loss': {method: report['models'][method]['dev_mean']['log_loss'] for method in ('mlp_ind', 'mlp_joint', 'mlp_offset')}}
    (HERE / 'SUMMARY.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__': main()
