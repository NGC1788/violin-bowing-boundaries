"""Reviewed, frozen, one-attempt source-only preparation. No model training."""
from pathlib import Path
import argparse, datetime, fcntl, hashlib, json, platform, time
import numpy as np
from .core import (prefix_inputs, future_labels, reconstruct, build_catalogue,
                   catalogue_counts, mutation_ids, mask_keys, budget_check)
from ..violin_raw_v1.prepare import read_supported_samples
from ..discovery_v1.data import source_folds, verify_old_sources

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OLD = HERE.parent / 'violin_raw_v1'
DISCOVERY = HERE.parent / 'discovery_v1'
RAW = Path.home() / 'violin-rne/data/external/blanco2021_raw'
LEDGER = HERE.parent / 'discovery_v2/LOOP_BUDGET.json'
SOURCE_FILES = ['AGENTS.md', 'PLAN.json', 'core.py', 'runner.py', 'test_gate.py']
DEPENDENCIES = ['discovery_v1/data.py', 'violin_raw_v1/features.py', 'violin_raw_v1/prepare.py',
                'violin_raw_v1/data.py', 'violin_raw_v1/inventory.py', 'violin_raw_v1/models.py']


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(4 * 1024 ** 2), b''):
            h.update(block)
    return h.hexdigest()


def dump(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, allow_nan=False) + '\n')


def sources():
    paths = [HERE/n for n in SOURCE_FILES] + [HERE.parent/n for n in DEPENDENCIES]
    return {str(p.relative_to(ROOT)): sha(p) for p in paths}


def input_paths():
    # Explicit allowlist: never glob, open or hash old dev/test truth.
    return {'prefix_features': OLD/'local/prepared/features.npz',
            'train_labels': OLD/'local/prepared/train_labels.npz',
            'source_prepared': DISCOVERY/'local/prepared.npz',
            'source_receipt': DISCOVERY/'PREPARED.json',
            'source_result': DISCOVERY/'RESULTS.json',
            'source_seal': DISCOVERY/'RESULTS_SEAL.json',
            'source_freeze': DISCOVERY/'FREEZE.json',
            'old_execution_manifest': OLD/'execution_freeze_manifest.json',
            'old_data_manifest': OLD/'data_freeze_manifest.json',
            'partition_receipt': OLD/'data_partition_receipt.json',
            'inventory_metadata': RAW/'inventory_records.json'}


def input_hashes():
    return {name: sha(path) for name, path in input_paths().items()}


def versions():
    return {'python': platform.python_version(), 'numpy': np.__version__}


def verify_discovery_chain(paths, expected_results_sha):
    """Anchor cached sequence provenance to the previously sealed result hash."""
    result_sha = sha(paths['source_result'])
    seal = json.loads(paths['source_seal'].read_text())
    if result_sha != expected_results_sha or seal['results_sha256'] != result_sha:
        raise ValueError('previous source result/seal changed')
    result = json.loads(paths['source_result'].read_text())
    receipt = json.loads(paths['source_receipt'].read_text())
    if (sha(paths['source_receipt']) != result['prepared_receipt_sha256'] or
            sha(paths['source_freeze']) != result['freeze_sha256'] or
            receipt['freeze_sha256'] != result['freeze_sha256'] or
            sha(paths['source_prepared']) != result['prepared_sha256'] or
            receipt['prepared_sha256'] != result['prepared_sha256']):
        raise ValueError('previous source preparation provenance changed')


def freeze():
    if (HERE/'FREEZE.json').exists() or (HERE/'local/started.json').exists():
        raise ValueError('freeze/start exists; no replacement allowed')
    verify_old_sources()  # Checks code and original train-only inputs; no held truth.
    r = json.loads((HERE/'REVIEW.json').read_text())
    if r['verdict'] != 'APPROVE' or r['source_sha256'] != sources() or r['blockers']:
        raise ValueError('unreviewed source or unresolved blockers')
    tests = json.loads((HERE/'UNIT_TESTS.json').read_text())
    if tests['source_sha256'] != sources() or tests['failed'] != 0:
        raise ValueError('tests do not bind reviewed source')
    plan = json.loads((HERE/'PLAN.json').read_text())
    verify_discovery_chain(input_paths(), plan['expected_discovery_v1_results_sha256'])
    ledger = json.loads(LEDGER.read_text())
    if ledger['deadline_kst'] != plan['budget']['deadline_kst']:
        raise ValueError('deadline mismatch')
    allowed = input_hashes()
    old = json.loads((OLD/'execution_freeze_manifest.json').read_text())
    previous = json.loads((DISCOVERY/'PREPARED.json').read_text())
    partition = json.loads((OLD/'data_partition_receipt.json').read_text())
    expected = {'prefix_features': old['prepared_files']['features.npz'],
                'train_labels': old['prepared_files']['train_labels.npz'],
                'source_prepared': previous['prepared_sha256'],
                'inventory_metadata': partition['original_inventory_sha256']}
    if any(allowed[k] != v for k, v in expected.items()):
        raise ValueError('previously sealed source input changed')
    dump(HERE/'FREEZE.json', {'sources': sources(), 'inputs': allowed, 'versions': versions(),
          'review_sha256': sha(HERE/'REVIEW.json'), 'tests_sha256': sha(HERE/'UNIT_TESTS.json'),
          'scope': plan['scope'], 'deadline_kst': ledger['deadline_kst'],
          'raw_identity': 'each selected source WAV checked against frozen inventory PCM identity at read time'})
    print(json.dumps({'freeze_sha256': sha(HERE/'FREEZE.json')}))


def check():
    verify_old_sources()
    plan = json.loads((HERE/'PLAN.json').read_text())
    verify_discovery_chain(input_paths(), plan['expected_discovery_v1_results_sha256'])
    f = json.loads((HERE/'FREEZE.json').read_text())
    if (f['sources'] != sources() or f['inputs'] != input_hashes() or f['versions'] != versions()
            or f['review_sha256'] != sha(HERE/'REVIEW.json')
            or f['tests_sha256'] != sha(HERE/'UNIT_TESTS.json')
            or f['deadline_kst'] != json.loads(LEDGER.read_text())['deadline_kst']):
        raise ValueError('frozen contract, allowed input or environment changed')
    return f


def load_source():
    with np.load(OLD/'local/prepared/features.npz', allow_pickle=False) as z:
        sel = z['split'] == 'train'
        x, people, ids = z['X'][sel], z['participant_ids'][sel], z['source_ids'][sel]
    with np.load(DISCOVERY/'local/prepared.npz', allow_pickle=False) as z:
        for key, expected in [('X', x), ('people', people), ('source_ids', ids)]:
            if not np.array_equal(z[key], expected):
                raise ValueError('source cached prefix/order mismatch: ' + key)
        sequence, fold = z['sequence'].copy(), z['fold'].copy()
    if not np.array_equal(fold, source_folds(people)):
        raise ValueError('source person folds changed')
    with np.load(OLD/'local/prepared/train_labels.npz', allow_pickle=False) as z:
        targets, guard = z['targets'].copy(), z['target_guard'].copy()
    if len(set(map(str, ids))) != len(ids) or targets.shape != (len(ids), 2) or guard.shape != (len(ids),):
        raise ValueError('source target alignment or duplicate IDs')
    return x, sequence, people, ids, fold, targets, guard


def reserve_attempt(local, freeze_sha):
    # Exclusive creation survives exceptions; a partial preparation cannot be retried.
    with (local/'started.json').open('x') as f:
        json.dump({'started_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
                   'freeze_sha256': freeze_sha, 'max_attempts': 1, 'new_training_jobs': 0}, f)


def run():
    local = HERE/'local'; local.mkdir(exist_ok=True)
    with (local/'run.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (local/'started.json').exists() or (HERE/'RESULTS.json').exists():
            raise ValueError('preparation already started; preserve outputs/failure')
        plan = json.loads((HERE/'PLAN.json').read_text())
        deadline = datetime.datetime.fromisoformat(plan['budget']['deadline_kst'])
        if datetime.datetime.now(datetime.timezone.utc) >= deadline:
            raise TimeoutError('preparation deadline')
        check()
        reserve_attempt(local, sha(HERE/'FREEZE.json'))
        started = time.monotonic()
        try:
            result = prepare(plan, started, deadline)
            check()
            budget_check(started, plan['budget']['max_seconds'], clock=time.monotonic,
                         now=lambda: datetime.datetime.now(datetime.timezone.utc), deadline=deadline)
            result.update(seconds=time.monotonic()-started, freeze_sha256=sha(HERE/'FREEZE.json'))
            dump(HERE/'RESULTS.json', result)
            dump(HERE/'RESULTS_SEAL.json', {'results_sha256': sha(HERE/'RESULTS.json')})
            dump(local/'complete.json', {'verdict': result['verdict'], 'seconds': result['seconds']})
            print(json.dumps(result, indent=2))
        except Exception as error:
            dump(local/'failed.json', {'type': type(error).__name__, 'reason': str(error),
                                      'seconds': time.monotonic()-started, 'retry_allowed': False})
            raise


def prepare(plan, started, deadline):
    local = HERE/'local'
    x, sequence, people, ids, fold, old_targets, old_guard = load_source()
    if len(ids) != plan['expected_rows'] or len(np.unique(people)) != plan['expected_people']:
        raise ValueError('source count changed')
    fold_rows = [int((fold==i).sum()) for i in range(3)]
    fold_people = [len(np.unique(people[fold==i])) for i in range(3)]
    if fold_rows != plan['expected_fold_rows'] or fold_people != plan['expected_fold_people']:
        raise ValueError('fold contract changed')
    # Inference arrays and row bookkeeping are separate from all future labels.
    np.savez_compressed(local/'prefix_inputs.npz', X=x, sequence=sequence)
    np.savez_compressed(local/'source_metadata.npz', people=people, source_ids=ids, fold=fold)
    inventory = json.loads((RAW/'inventory_records.json').read_text())['records']
    records = {r['source_id']: r for r in inventory}
    audited = mutation_ids(ids, plan['real_future_mutation_checks'])
    labels, prefix_checks, reverse_checks = [], 0, 0
    for i, sid in enumerate(ids):
        budget_check(started, plan['budget']['max_seconds'], clock=time.monotonic,
                     now=lambda: datetime.datetime.now(datetime.timezone.utc), deadline=deadline)
        row = records[str(sid)]
        if row['participant_id'] != people[i]:
            raise ValueError('record person differs from source manifest')
        samples = read_supported_samples(row, RAW)
        label = future_labels(samples, row['sample_rate'], row['subtype'])
        labels.append(label)
        if sid in audited:
            base_x, base_seq = prefix_inputs(samples, row['sample_rate'], row['subtype'])
            if not np.array_equal(base_x, x[i]) or not np.array_equal(base_seq, sequence[i]):
                raise ValueError('real prefix does not reproduce sealed input')
            changed = samples.astype(np.float64) / 32768
            changed[row['sample_rate']:] = np.nan
            alt_x, alt_seq = prefix_inputs(changed, row['sample_rate'], 'DOUBLE')
            if not np.array_equal(base_x, alt_x) or not np.array_equal(base_seq, alt_seq):
                raise ValueError('future mutation changes inference input')
            prefix_checks += 1
            changed = samples.astype(np.float64) / 32768
            a, b = [round(t*row['sample_rate']) for t in (1.25,1.75)]
            changed[:a] = np.nan; changed[b:] = np.nan
            alt = future_labels(changed, row['sample_rate'], 'DOUBLE')
            if any(not np.array_equal(label[k], alt[k], equal_nan=True) for k in label):
                raise ValueError('future descriptor sees samples outside declared future')
            reverse_checks += 1
    stacked = {k: np.asarray([label[k] for label in labels]) for k in labels[0]}
    rebuilt, guard = reconstruct(stacked['pitch_cents'], stacked['power_dbfs'], stacked['voiced'])
    if (not np.array_equal(guard, old_guard) or
            not np.allclose(rebuilt, old_targets, atol=plan['continuous_atol'],
                            rtol=plan['continuous_rtol'], equal_nan=True)):
        raise ValueError('path target/guard reconstruction differs from sealed train labels')
    if prefix_checks != len(audited) or reverse_checks != len(audited):
        raise ValueError('missing preselected real leakage checks')
    np.savez_compressed(local/'future_labels.npz', **stacked)
    mask = stacked['voiced']
    counts = []
    crit = plan['catalogue_screen']
    for i in range(3):
        inner = fold != i; held = ~inner
        # No held masks are supplied to construction.
        catalogue = build_catalogue(mask[inner])
        detail = catalogue_counts(catalogue, mask[held])
        valid_fraction = detail['held_valid_unknown_fraction']
        counts.append({'fold': i, 'inner_train_rows': int(inner.sum()),
                       'inner_train_unique_masks': len(catalogue), **detail,
                       'catalogue_screen_pass': len(catalogue) <= crit['max_inner_train_unique_masks']
                          and valid_fraction is not None and valid_fraction <= crit['max_held_valid_unknown_fraction']})
    finite = np.isfinite(old_targets)
    error = float(np.abs(rebuilt[finite]-old_targets[finite]).max())
    unique, observed = np.unique(mask.sum(axis=1), return_counts=True)
    passed = all(r['catalogue_screen_pass'] for r in counts)
    return {'verdict': 'PREPARATION_PASS_CATALOGUE_SCREEN_PASS' if passed else 'PREPARATION_PASS_CATALOGUE_SCREEN_FAIL',
            'rows': len(ids), 'people': len(np.unique(people)), 'fold_rows': fold_rows,
            'fold_people': fold_people, 'future_shape': list(stacked['pitch_cents'].shape),
            'valid_rows': int(guard.sum()), 'invalid_rows_retained': int((~guard).sum()),
            'target_reconstruction_max_abs': error, 'target_guard_exact_match': True,
            'real_future_mutation_checks': prefix_checks, 'real_future_support_checks': reverse_checks,
            'voiced_count_histogram': {str(int(k)): int(v) for k,v in zip(unique,observed)},
            'total_unique_masks_descriptive_only': len(set(mask_keys(mask))), 'fold_catalogue': counts,
            'catalogue_screen': crit, 'prepared_files': {n: sha(local/n) for n in
                       ['prefix_inputs.npz','source_metadata.npz','future_labels.npz']},
            'new_training_jobs': 0, 'new_downloads': 0, 'old_dev_test_truth_accessed': False,
            'scope': 'historically exposed source-only preparation, no independent confirmation or forecasting performance',
            'observer_limit': 'no claim of Gaussian innovation independence, latent-factor adequacy or missing-at-random voicing'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('action', choices=['freeze','check','run'])
    args = parser.parse_args()
    if args.action == 'freeze': freeze()
    elif args.action == 'check': check(); print('source data contract intact')
    else: run()
