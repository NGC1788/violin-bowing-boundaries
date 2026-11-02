"""Bounded, sealed twelve-trial study with separate train/dev/test access.

Same-user integrity, not a security sandbox. Failures and interrupted attempts
stop the campaign. Sources/data are never re-frozen to excuse a failed run.
"""
from __future__ import annotations
import argparse
import datetime
import fcntl
import functools
import hashlib
import importlib.metadata
import json
from pathlib import Path
import signal
import sys
import time
import numpy as np
from .inventory import file_sha
from .prepare import verify_freeze as verify_data_freeze
from .models import fit_predict, pattern_ids
from .metrics import summarize, select_baseline, paired_bootstrap

HERE = Path(__file__).resolve().parent
FEATURE_KEYS = {'X', 'prefix_states', 'prefix_Y9', 'prefix_guard', 'participant_ids', 'split', 'source_ids'}
TRIALS = [(m, 0) for m in ('frequency', 'markov', 'hgb')] + [
    (m, s) for m in ('mlp_ind', 'mlp_joint', 'mlp_offset') for s in (0, 1, 2)]


def fail_closed(action):
    """Integrity failures are terminal, including failures before test opening.

    A competing process that fails to acquire the lock cannot stop its owner.
    """
    @functools.wraps(action)
    def wrapped(self, *args, **kwargs):
        try:
            return action(self, *args, **kwargs)
        except BaseException as error:
            self.record_halt(error)
            raise
    return wrapped


def write_json(path, value, *, once=False):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n'
    if once:
        with path.open('x') as f: f.write(text)
    else:
        temporary = path.with_suffix(path.suffix + '.tmp')
        temporary.write_text(text); temporary.replace(path)


def check_hashes(root, expected):
    root = Path(root).resolve()
    for name, wanted in expected.items():
        path = (root / name).resolve()
        if not path.is_relative_to(root) or file_sha(path) != wanted:
            raise ValueError('integrity mismatch: ' + name)


def environment():
    return {'python': sys.version.split()[0], **{p: importlib.metadata.version(p)
            for p in ('numpy', 'scipy', 'scikit-learn', 'torch')}}


def load_features(prepared):
    with np.load(Path(prepared) / 'features.npz', allow_pickle=False) as z:
        if set(z.files) != FEATURE_KEYS: raise ValueError('unexpected predictor fields')
        x = {k: z[k] for k in z.files}
    n = len(x['X'])
    if x['X'].shape != (n, 60) or not np.isfinite(x['X']).all():
        raise ValueError('invalid predictor matrix')
    if not np.isin(x['split'], ['train', 'dev', 'test']).all(): raise ValueError('invalid split')
    if x['prefix_states'].shape != (n,) or not np.isin(x['prefix_states'], range(11)).all():
        raise ValueError('invalid prefix states')
    if x['prefix_Y9'].shape != (n, 9): raise ValueError('invalid prefix labels')
    pattern_ids(x['prefix_Y9'])
    for pid in np.unique(x['participant_ids']):
        if len(np.unique(x['split'][x['participant_ids'] == pid])) != 1:
            raise ValueError('participant crosses partitions')
    return x


def load_training_inputs(prepared):
    """This function never opens dev_labels.npz or test_labels.npz."""
    features = load_features(prepared)
    with np.load(Path(prepared) / 'train_labels.npz', allow_pickle=False) as z:
        y = z['Y9']
    if y.shape != (int((features['split'] == 'train').sum()), 9):
        raise ValueError('unaligned train labels')
    pattern_ids(y)
    return features, y


def score_family(labels, predictions, participants, guard):
    """Average seed LOSSES; never replace them with ensemble probability loss."""
    seed_scores = [summarize(labels, p, participants, guard) for p in predictions]
    result = {'seeds': len(seed_scores), 'seed_log_losses': [r['log_loss'] for r in seed_scores]}
    for key in ('log_loss', 'brier', 'coherence_rate'):
        result[key] = float(np.mean([r[key] for r in seed_scores]))
    result['participant_loss'] = np.mean([r['participant_loss'] for r in seed_scores], axis=0).tolist()
    result['participants'] = seed_scores[0]['participants']; result['rows'] = seed_scores[0]['rows']
    for name in ('valid', 'invalid'):
        first = seed_scores[0][name]
        result[name] = {**first, 'log_loss': float(np.mean([r[name]['log_loss'] for r in seed_scores]))
                        if first['rows'] else None}
    result['cohort_log_loss'] = {}
    ids = np.asarray(participants)
    for cohort in ('CG', 'FG', 'EG'):
        mask = np.array([str(pid).endswith('_' + cohort) for pid in ids])
        if mask.any():
            result['cohort_log_loss'][cohort] = float(np.mean([
                summarize(labels[mask], p[mask], ids[mask], guard[mask])['log_loss'] for p in predictions]))
    return result


class Campaign:
    def __init__(self, base=HERE):
        self.base = Path(base).resolve(); self.repository = self.base.parents[1]
        self.prepared = self.base / 'local/prepared'
        self.run = self.base / 'local/campaign_v1'
        self.manifest_path = self.base / 'execution_freeze_manifest.json'

    def record_halt(self, error):
        if not self.manifest_path.exists(): return
        self.run.mkdir(parents=True, exist_ok=True)
        with (self.run / 'campaign.lock').open('a') as lock:
            try: fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError: return
            state_path = self.run / 'state.json'
            try: state = json.loads(state_path.read_text()) if state_path.exists() else {}
            except (ValueError, OSError): state = None  # Preserve malformed original state as evidence.
            if state and state.get('status') == 'EVALUATED': return
            halt = self.run / 'halt.json'
            if not halt.exists():
                write_json(halt, {'error_type': type(error).__name__, 'error': str(error),
                           'policy': 'terminal halt; restoring a file does not permit retry'}, once=True)
            if state is not None:
                state['stopped'] = True
                if state.get('status') not in ('FAILED', 'EVALUATION_FAILED'): state['status'] = 'SAFETY_HALT'
                write_json(state_path, state)

    def freeze(self):
        verify_data_freeze(self.base / 'data_freeze_manifest.json')
        if self.manifest_path.exists() or self.run.exists(): raise ValueError('existing freeze/campaign preserved')
        review = json.loads((self.base / 'EXECUTION_REVIEW.json').read_text())
        snapshot = json.loads((self.base / 'local/reviews/execution_snapshot.json').read_text())
        snapshot_digest = hashlib.sha256(json.dumps(snapshot['source_hashes'], sort_keys=True).encode()).hexdigest()
        if snapshot_digest != snapshot['snapshot_sha256']:
            raise ValueError('review snapshot hash does not bind its source map')
        if (review['verdict'] != 'approve' or review['blockers'] or review['scope'] != 'training_evaluation'
                or review['snapshot_sha256'] != snapshot['snapshot_sha256']):
            raise ValueError('execution review not approved for this snapshot')
        mandatory = {str((self.base / f).relative_to(self.repository)) for f in
                     ('runner.py', 'models.py', 'metrics.py', 'protocol.json', 'test_runner.py')}
        if not mandatory.issubset(snapshot['source_hashes']): raise ValueError('review lacks critical execution source')
        check_hashes(self.repository, snapshot['source_hashes'])
        protocol = json.loads((self.base / 'protocol.json').read_text())
        if protocol['models']['trials'] != [list(t) for t in TRIALS]: raise ValueError('unexpected trial menu')
        prepared_receipt = json.loads((self.prepared / 'receipt.json').read_text())
        summary = json.loads((self.base / 'PREPARED_SUMMARY.json').read_text())
        data_sha = file_sha(self.base / 'data_freeze_manifest.json')
        if prepared_receipt['data_freeze_sha256'] != data_sha or summary['data_freeze_sha256'] != data_sha:
            raise ValueError('prepared data references a different data freeze')
        if file_sha(self.prepared / 'receipt.json') != summary['prepared_receipt_sha256']:
            raise ValueError('prepared receipt differs from the reviewed public summary')
        if prepared_receipt['files'] != summary['file_sha256']:
            raise ValueError('prepared file identities differ from the reviewed public summary')
        check_hashes(self.prepared, prepared_receipt['files'])
        source = dict(snapshot['source_hashes'])
        source[str((self.base / 'EXECUTION_REVIEW.json').relative_to(self.repository))] = file_sha(self.base / 'EXECUTION_REVIEW.json')
        manifest = {'version': 1, 'scope': 'bounded_twelve_trial_forecast', 'source_hashes': source,
                    'review_snapshot_sha256': review['snapshot_sha256'],
                    'data_freeze_sha256': file_sha(self.base / 'data_freeze_manifest.json'),
                    'prepared_receipt_sha256': file_sha(self.prepared / 'receipt.json'),
                    'prepared_files': prepared_receipt['files'], 'environment': environment(),
                    'frozen_utc': datetime.datetime.now(datetime.timezone.utc).isoformat()}
        write_json(self.manifest_path, manifest, once=True)
        return manifest

    def verify(self):
        manifest = json.loads(self.manifest_path.read_text())
        check_hashes(self.repository, manifest['source_hashes'])
        verify_data_freeze(self.base / 'data_freeze_manifest.json')
        if file_sha(self.base / 'data_freeze_manifest.json') != manifest['data_freeze_sha256']:
            raise ValueError('data freeze changed')
        if file_sha(self.prepared / 'receipt.json') != manifest['prepared_receipt_sha256']:
            raise ValueError('prepared receipt changed')
        check_hashes(self.prepared, manifest['prepared_files'])  # byte hashes only, not target parsing
        if environment() != manifest['environment']: raise ValueError('execution environment changed')
        protocol = json.loads((self.base / 'protocol.json').read_text())
        if datetime.datetime.now(datetime.timezone.utc) >= datetime.datetime.fromisoformat(protocol['budget']['end_utc']):
            raise ValueError('campaign deadline reached')
        return manifest, protocol

    @fail_closed
    def train(self, fit=None):
        manifest, protocol = self.verify(); fit = fit or fit_predict
        self.run.mkdir(parents=True, exist_ok=True)
        with (self.run / 'campaign.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            state_path = self.run / 'state.json'
            if (self.run / 'halt.json').exists(): raise ValueError('campaign halted; no retry')
            if state_path.exists():
                old = json.loads(state_path.read_text())
                if old['status'] in ('TRAINED', 'EVALUATED'): return old
                # An earlier RUNNING state is an interrupted attempt, never a retry invitation.
                raise ValueError('failed, incomplete, or interrupted campaign preserved; no retry')
            state = {'status': 'RUNNING', 'attempts': 0, 'completed': [], 'worker_seconds': 0.0,
                     'manifest_sha256': file_sha(self.manifest_path)}
            write_json(state_path, state)
            previous_alarm = signal.getsignal(signal.SIGALRM)
            try:
                features, y = load_training_inputs(self.prepared)
                train = features['split'] == 'train'; held = ~train; indices = np.flatnonzero(held)
                for method, seed in TRIALS:
                    self.verify()
                    remaining = protocol['budget']['worker_seconds'] - state['worker_seconds']
                    until_deadline = (datetime.datetime.fromisoformat(protocol['budget']['end_utc']) -
                                      datetime.datetime.now(datetime.timezone.utc)).total_seconds()
                    if state['attempts'] >= protocol['budget']['max_training_attempts'] or min(remaining, until_deadline) <= 0:
                        raise TimeoutError('campaign budget exhausted')
                    job_id = f'{method}_s{seed}'; job = self.run / job_id; job.mkdir()
                    state['attempts'] += 1; state['active_job'] = job_id; write_json(state_path, state)
                    write_json(job / 'started.json', {'attempt': state['attempts'], 'method': method, 'seed': seed}, once=True)
                    def timeout(signum, frame): raise TimeoutError('training job time budget reached')
                    signal.signal(signal.SIGALRM, timeout)
                    started = time.perf_counter()
                    limit = min(float(protocol['budget']['trial_seconds']), remaining, until_deadline)
                    try:
                        signal.setitimer(signal.ITIMER_REAL, limit)
                        probability, meta = fit(method, seed, features['X'][train], y,
                            features['prefix_states'][train], features['participant_ids'][train],
                            features['X'][held], features['prefix_states'][held], protocol['models']['parameters'])
                    finally:
                        signal.setitimer(signal.ITIMER_REAL, 0)
                        elapsed = time.perf_counter() - started
                        state['worker_seconds'] += elapsed
                    if elapsed > limit:
                        raise TimeoutError('training call exceeded its measured time limit')
                    if state['worker_seconds'] > protocol['budget']['worker_seconds']:
                        raise TimeoutError('campaign worker budget exceeded')
                    if probability.shape != (len(indices), 9) or not np.isfinite(probability).all() or ((probability < 0) | (probability > 1)).any():
                        raise ValueError('invalid held-out predictions')
                    weights = meta.pop('state_dict', None)
                    if weights is not None: np.savez_compressed(job / 'weights.npz', **weights)
                    np.savez_compressed(job / 'predictions.npz', probability=probability, row_indices=indices)
                    meta.update({'worker_seconds_cumulative': state['worker_seconds'],
                                 'predictions_sha256': file_sha(job / 'predictions.npz')})
                    write_json(job / 'receipt.json', meta, once=True)
                    state['completed'].append(job_id); state.pop('active_job', None); write_json(state_path, state)
                    print(f'Completed {len(state["completed"])}/12: {job_id}', flush=True)
                self.verify()
                sealed = {f'{job}/{name}': file_sha(self.run / job / name) for job in state['completed']
                          for name in ('predictions.npz', 'receipt.json')}
                write_json(self.run / 'predictions_seal.json', {'manifest_sha256': file_sha(self.manifest_path),
                    'files': sealed, 'completed_jobs': state['completed'], 'test_labels_parsed': False}, once=True)
                state['status'] = 'TRAINED'; write_json(state_path, state)
                return state
            except BaseException as error:
                state.update(status='FAILED', error_type=type(error).__name__, error=str(error))
                write_json(state_path, state)
                raise
            finally:
                signal.signal(signal.SIGALRM, previous_alarm)

    def predictions(self, features):
        seal = json.loads((self.run / 'predictions_seal.json').read_text())
        expected = [f'{m}_s{s}' for m, s in TRIALS]
        if seal['completed_jobs'] != expected or seal['manifest_sha256'] != file_sha(self.manifest_path):
            raise ValueError('incomplete or mismatched prediction seal')
        expected_files = {f'{job}/{name}' for job in expected for name in ('predictions.npz', 'receipt.json')}
        if set(seal['files']) != expected_files: raise ValueError('prediction seal does not cover every job')
        check_hashes(self.run, seal['files'])
        held_indices = np.flatnonzero(features['split'] != 'train'); result = {}
        for method, seed in TRIALS:
            with np.load(self.run / f'{method}_s{seed}' / 'predictions.npz', allow_pickle=False) as z:
                if not np.array_equal(z['row_indices'], held_indices): raise ValueError('prediction row alignment changed')
                p = z['probability']
            result.setdefault(method, []).append(p)
        return result, held_indices

    @fail_closed
    def evaluate(self):
        _, protocol = self.verify()
        with (self.run / 'campaign.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            if (self.run / 'halt.json').exists(): raise ValueError('campaign halted; no repeated evaluation')
            state_path = self.run / 'state.json'; state = json.loads(state_path.read_text())
            if state['status'] == 'EVALUATED': raise ValueError('test already evaluated; inspect preserved results')
            if state['status'] != 'TRAINED': raise ValueError('all twelve jobs must be completed first')
            if (self.run / 'test_opened.json').exists(): raise ValueError('test has already been opened; no repeated evaluation')
            features = load_features(self.prepared); prediction, indices = self.predictions(features)
            held_split = features['split'][indices]; held_ids = features['participant_ids'][indices]
            def score(split, label):
                mask = held_split == split
                y, guard = label['Y9'], label['target_guard']
                if y.shape != (int(mask.sum()), 9): raise ValueError('label row count mismatch')
                return {name: score_family(y, [p[mask] for p in ps], held_ids[mask], guard)
                        for name, ps in prediction.items()}
            with np.load(self.prepared / 'dev_labels.npz', allow_pickle=False) as label:
                dev = score('dev', label)
            choice = select_baseline({name: dev[name]['log_loss'] for name in ('frequency', 'markov', 'hgb')})
            selection = {'baseline': choice, 'dev_scores': {name: r['log_loss'] for name, r in dev.items()},
                         'predictions_seal_sha256': file_sha(self.run / 'predictions_seal.json'),
                         'manifest_sha256': file_sha(self.manifest_path)}
            write_json(self.run / 'selection.json', selection, once=True)
            selection_sha = file_sha(self.run / 'selection.json')
            self.verify(); self.predictions(features)
            write_json(self.run / 'test_opened.json', {'selection_sha256': selection_sha,
                       'opened_utc': datetime.datetime.now(datetime.timezone.utc).isoformat()}, once=True)
            try:
                with np.load(self.prepared / 'test_labels.npz', allow_pickle=False) as label:
                    test = score('test', label)
                    mask = held_split == 'test'
                    persistence = score_family(label['Y9'], [features['prefix_Y9'][indices][mask]], held_ids[mask], label['target_guard'])
                comparisons = {}
                for name in (choice, 'mlp_joint', 'mlp_ind'):
                    candidate, comparator = test['mlp_offset'], test[name]
                    comparisons[name] = {'relative_reduction': 1 - candidate['log_loss'] / comparator['log_loss'],
                        **paired_bootstrap(candidate['participant_loss'], comparator['participant_loss'])}
                result = {'scope': 'recording-relative descriptor prediction; exploratory nine-participant test',
                          'dev': dev, 'test': test, 'selected_baseline': choice,
                          'persistence_untrained_auxiliary': persistence,
                          'primary_offset_comparisons': comparisons,
                          'two_percent_screen_passed': all(c['relative_reduction'] >= protocol['evaluation']['minimum_relative_improvement'] for c in comparisons.values()),
                          'all_comparison_intervals_below_zero': all(c['percentile_95_interval'][1] < 0 for c in comparisons.values()),
                          'training_attempts': state['attempts'], 'worker_seconds': state['worker_seconds'],
                          'manifest_sha256': file_sha(self.manifest_path),
                          'predictions_seal_sha256': file_sha(self.run / 'predictions_seal.json'),
                          'selection_sha256': file_sha(self.run / 'selection.json'),
                          'test_opened_marker_sha256': file_sha(self.run / 'test_opened.json'),
                          'novelty': 'joint heads, prior offsets and coherence are conventional; accuracy alone is not method novelty'}
                self.verify()
                self.predictions(features)
                if (file_sha(self.run / 'selection.json') != selection_sha or
                        file_sha(self.run / 'predictions_seal.json') != selection['predictions_seal_sha256']):
                    raise ValueError('prediction or comparator seal changed during evaluation')
                write_json(self.run / 'results.json', result, once=True)
                write_json(self.run / 'results_seal.json', {'results_sha256': file_sha(self.run / 'results.json'),
                           'manifest_sha256': file_sha(self.manifest_path)}, once=True)
                state['status'] = 'EVALUATED'; state['stopped'] = True; write_json(state_path, state)
                return result
            except BaseException as error:
                state.update(status='EVALUATION_FAILED', stopped=True, error_type=type(error).__name__, error=str(error))
                write_json(state_path, state); raise


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('action', choices=['freeze', 'train', 'evaluate', 'status'])
    args = parser.parse_args(); campaign = Campaign()
    if args.action == 'freeze': result = campaign.freeze()
    elif args.action == 'train': result = campaign.train()
    elif args.action == 'evaluate': result = campaign.evaluate()
    else:
        path = campaign.run / 'state.json'; result = json.loads(path.read_text()) if path.exists() else {'status': 'NOT_STARTED'}
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__': main()
