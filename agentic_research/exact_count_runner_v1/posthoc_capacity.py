"""Read-only postmortem of the completed batch; no fitting or model forward.

An analytic Chernoff cap, evaluated at SAVED normalization statistics only.
This is not another standalone numerical/setting audit or candidate selection.
Writes only public aggregates, without person IDs, rows, or predictions.
"""
import hashlib
import json
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_pinned(path, expected):
    if path.is_symlink() or sha(path) != expected:
        raise ValueError("postmortem input seal mismatch")
    if path.suffix == '.npz':
        with np.load(path, allow_pickle=False) as z:
            value = {k: z[k] for k in z.files}
    else:
        value = json.loads(path.read_text())
    if sha(path) != expected:
        raise ValueError("postmortem input changed during read")
    return value


def chernoff_cap(ratio, minimum_df):
    # For X~chi2_d and a<1: P(X<=da)<=exp(d/2*(log(a)+1-a)).
    # For d>=minimum_df, the right side is <= its value at minimum_df.
    ratio = np.asarray(ratio)
    out = np.ones_like(ratio, dtype=float)
    low = (ratio > 0) & (ratio < 1)
    a = ratio[low] ** 2
    out[low] = np.exp(minimum_df / 2 * (np.log(a) + 1 - a))
    out[ratio <= 0] = 0
    return out


def macro(y, cap, people, eps):
    # A deliberately optimistic box relaxation: choose q=cap for positives,
    # q=0 for negatives, even if no common model can attain those choices.
    pc = np.clip(cap, eps, 1 - eps)
    loss = (-y * np.log(pc) - (1 - y) * np.log1p(-eps)).mean(1)
    means = [float(loss[people == p].mean()) for p in np.unique(people)]
    return float(np.mean(means)), len(means)


def main():
    freeze = json.loads((HERE / 'FREEZE.json').read_text())
    result = read_pinned(HERE / 'RESULTS.json',
                         json.loads((HERE / 'RESULTS_SEAL.json').read_text())['results_sha256'])
    seal = read_pinned(HERE / 'OUTPUT_SEAL.json', result['batch_output_seal_sha256'])
    contract = read_pinned(HERE / 'CONTRACT.json',
                          freeze['sources']['agentic_research/exact_count_runner_v1/CONTRACT.json'])
    gate = HERE.parent / 'path_source_gate_v1/local'
    meta = read_pinned(gate / 'source_metadata.npz', freeze['inputs']['source_metadata.npz'])
    labels = read_pinned(gate / 'future_labels.npz', freeze['inputs']['future_labels.npz'])
    saved = read_pinned(HERE / 'local/all_predictions.npz', result['all_predictions_sha256'])
    config = contract['model_config']
    sigma_min = config['scale_range_normalized_units'][0]
    eps = config['probability_clipping']
    records = []
    weighted = 0.
    total_people = 0
    for k in range(3):
        stat_path = HERE / f'local/job_{k}_exact_count/stats.json'
        stats = read_pinned(stat_path, seal['jobs'][f'job_{k}_exact_count']['stats.json'])
        thresholds = np.asarray(stats['thresholds'])
        normalized = thresholds / np.asarray(stats['path_scale'])[:, None]
        pitch = chernoff_cap(normalized[0] / sigma_min, config['minimum_voiced'] - 1)
        power = chernoff_cap(normalized[1] / sigma_min, config['future_frames'] - 1)
        cap = (pitch[:, None] * power[None, :]).reshape(9)
        # Reconstruct each fold's train/held event labels using that fold's
        # SAVED train-only thresholds. Do not reuse other folds' OOF labels.
        target, guard = labels['targets'], labels['guard']
        y = ((target[:, 0, None, None] <= thresholds[0, :, None]) &
             (target[:, 1, None, None] <= thresholds[1, None, :]) &
             guard[:, None, None]).reshape(-1, 9).astype(float)
        held = meta['fold'] == k
        assert np.array_equal(y[held], saved['truth'][held])
        actual = saved['exact_count'][held]
        assert (actual <= cap[None, :] + 1e-12).all()
        train_bound, train_people = macro(y[~held], cap, meta['people'][~held], eps)
        held_bound, held_people = macro(y[held], cap, meta['people'][held], eps)
        weighted += held_bound * held_people
        total_people += held_people
        records.append({'fold': k, 'normalized_thresholds': normalized.tolist(),
                        'pitch_chernoff_upper_caps': pitch.tolist(),
                        'power_chernoff_upper_caps': power.tolist(),
                        'queries_entirely_below_clip': int((cap < eps).sum()),
                        'innertrain_box_oracle_lower_bound': train_bound,
                        'held_box_oracle_lower_bound': held_bound,
                        'innertrain_people': train_people, 'held_people': held_people})
    aggregate = weighted / total_people
    assert all(r['queries_entirely_below_clip'] == 6 for r in records)
    assert aggregate > result['aggregate']['markov']['log_loss']
    assert aggregate <= result['aggregate']['exact_count']['log_loss'] + 1e-12
    output = {'status': 'POSTHOC_STRUCTURAL_CAPACITY_FAILURE',
              'scope': 'completed exact_count parameterization only; not moment approximation',
              'method': 'closed-form Chernoff upper cap, no scipy CDF, model forward or fitting',
              'folds': records, 'held_macro_loss_lower_bound': aggregate,
              'observed_exact_macro_loss': result['aggregate']['exact_count']['log_loss'],
              'reference_markov_macro_loss': result['aggregate']['markov']['log_loss'],
              'limitation': 'ordinary float evaluation of a mathematical bound, not interval arithmetic; posthoc exposed source diagnosis, not novelty or independent accuracy evidence',
              'new_fits': 0, 'new_model_predictions': 0, 'old_dev_test_truth_access_or_hash': False,
              'results_sha256': sha(HERE / 'RESULTS.json'),
              'output_seal_sha256': sha(HERE / 'OUTPUT_SEAL.json'),
              'script_sha256': sha(Path(__file__)),
              'frozen_inputs_unchanged': True}
    destination = HERE / 'CAPACITY_DIAGNOSIS.json'
    with destination.open('x') as f:
        json.dump(output, f, indent=2, allow_nan=False)
        f.write('\n')
    print(json.dumps({'status': output['status'], 'held_macro_loss_lower_bound': aggregate,
                      'six_of_nine_queries_below_clip_every_fold': True}))


if __name__ == '__main__':
    main()
