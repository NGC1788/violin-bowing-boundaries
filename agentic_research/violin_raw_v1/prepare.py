"""Build private partitioned data after a separate reviewed data-code freeze.

No model training or score computation. Test labels are written to a separate
file that the subsequent trainer must never open. Identity hashes are not X.
"""
from __future__ import annotations
import datetime
import fcntl
import hashlib
import json
from pathlib import Path
import numpy as np
from .data import assemble_rows, digest
from .features import extract_example, extract_prefix
from .inventory import file_sha, wav_metadata

HERE = Path(__file__).resolve().parent
REPOSITORY = HERE.parents[1]
DEFAULT_ROOT = Path.home() / 'violin-rne/data/external/blanco2021_raw'


def verify_freeze(freeze_path):
    freeze_path = Path(freeze_path)
    manifest = json.loads(freeze_path.read_text())
    if manifest.get('approved_scope') != 'data_preparation_only':
        raise ValueError('a reviewed data-preparation freeze is required')
    if datetime.datetime.now(datetime.timezone.utc) >= datetime.datetime.fromisoformat(manifest['end_utc']):
        raise ValueError('contract deadline reached')
    for name, expected in manifest['source_hashes'].items():
        if file_sha(REPOSITORY / name) != expected:
            raise ValueError('frozen source changed: ' + name)
    review = json.loads((HERE / 'DATA_REVIEW.json').read_text())
    if (review['verdict'] != 'approve' or review['snapshot_sha256'] != manifest['review_snapshot_sha256']
            or review['blockers']):
        raise ValueError('independent data review has unresolved blockers')
    return manifest


def read_supported_samples(row, root):
    path = Path(row['local_path']).resolve()
    if not path.is_relative_to((Path(root) / 'violin_wav').resolve()):
        raise ValueError('audio is outside the verified extraction directory')
    actual = wav_metadata(path)
    keys = ('sample_rate', 'channels', 'subtype', 'frames', 'pcm_sha256', 'prefix_pcm_sha256', 'data_offset')
    if any(actual[k] != row[k] for k in keys):
        raise ValueError('audio identity or layout changed')
    if (row['sample_rate'], row['channels'], row['subtype']) != (44100, 1, 'PCM_16'):
        raise ValueError('actual inventory encoding changed; review a new decoder')
    count = round(1.75 * row['sample_rate'])
    with path.open('rb') as f:
        f.seek(row['data_offset']); raw = f.read(count * 2)
    if len(raw) != count * 2:
        raise ValueError('incomplete requested PCM support')
    return np.frombuffer(raw, dtype='<i2').copy()


def write_partitioned(dataset, destination):
    """No held-out acoustic targets or labels are stored in the feature file."""
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    features = {k: dataset[k] for k in ('X', 'prefix_states', 'prefix_Y9', 'prefix_guard',
                                      'participant_ids', 'split', 'source_ids')}
    np.savez_compressed(destination / 'features.npz', **features)
    for split in ('train', 'dev', 'test'):
        mask = dataset['split'] == split
        np.savez_compressed(destination / f'{split}_labels.npz', Y9=dataset['Y9'][mask],
                            targets=dataset['targets'][mask], target_guard=dataset['target_guard'][mask])
    names = ['features.npz', 'train_labels.npz', 'dev_labels.npz', 'test_labels.npz']
    return {name: file_sha(destination / name) for name in names}


def build(root=DEFAULT_ROOT, destination=None, freeze_path=None):
    root = Path(root); destination = Path(destination or HERE / 'local/prepared')
    freeze_path = Path(freeze_path or HERE / 'data_freeze_manifest.json')
    manifest = verify_freeze(freeze_path)
    protocol = json.loads((HERE / 'protocol.json').read_text())
    partition = json.loads((HERE / 'data_partition_receipt.json').read_text())
    inventory_path = root / 'inventory_records.json'
    if file_sha(inventory_path) != partition['original_inventory_sha256']:
        raise ValueError('metadata inventory changed')
    split = json.loads((root / 'participant_split_v1.json').read_text())
    if digest(split) != protocol['data']['split_manifest_sha256']:
        raise ValueError('participant split changed')
    records = json.loads(inventory_path.read_text())['records']
    selected = sorted((r for r in records if r['eligible_duration']), key=lambda r: r['source_id'])
    if len(selected) != protocol['data']['rows'] or digest([r['source_id'] for r in selected]) != partition['selected_source_ids_sha256']:
        raise ValueError('eligible recording selection changed')
    destination.parent.mkdir(parents=True, exist_ok=True)
    with (destination.parent / 'prepare.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if destination.exists():
            raise ValueError('prepared or partial output exists; preserve it and inspect its state')
        destination.mkdir()
        (destination / 'started.json').write_text(json.dumps({'freeze_sha256': file_sha(freeze_path),
                'started_utc': datetime.datetime.now(datetime.timezone.utc).isoformat()}) + '\n')
        try:
            audited = set(sorted(selected, key=lambda r: hashlib.sha256(r['source_id'].encode()).hexdigest())[i]['source_id']
                          for i in range(min(8, len(selected))))
            rows = []
            for i, r in enumerate(selected, 1):
                samples = read_supported_samples(r, root)
                observation = extract_example(samples, r['sample_rate'], r['subtype'])
                if r['source_id'] in audited:
                    x = samples.astype(np.float64) / 32768
                    x[r['sample_rate']:] = np.nan
                    altered = extract_prefix(x, r['sample_rate'], 'DOUBLE')['X']
                    if not np.array_equal(observation['X'], altered):
                        raise ValueError('real prefix depends on future waveform values')
                rows.append({**r, **observation})
                if i % 250 == 0: print(f'Prepared recording {i}/{len(selected)}', flush=True)
            dataset = assemble_rows(rows, split)
            for key in ('rows', 'participant_counts', 'row_counts'):
                if key != 'rows' and dataset['metadata'][key] != protocol['data'][key]:
                    raise ValueError('prepared partition counts changed')
            if len(dataset['X']) != protocol['data']['rows']:
                raise ValueError('unexpected deduplication or row removal')
            hashes = write_partitioned(dataset, destination)
            # Report training threshold provenance; no dev/test target distribution.
            receipt = {'scope': 'private prepared data; no model scores or held-out target summaries',
                       'files': hashes, 'data_freeze_sha256': file_sha(freeze_path),
                       'rows': len(dataset['X']), 'feature_count': dataset['X'].shape[1],
                       'participant_counts': dataset['metadata']['participant_counts'],
                       'row_counts': dataset['metadata']['row_counts'],
                       'train_thresholds': dataset['thresholds'].tolist(),
                       'threshold_fit': dataset['metadata']['threshold_fit'],
                       'real_prefix_future_mutation_checks': len(audited),
                       'test_labels_separate': True, 'models_trained': False}
            verify_freeze(freeze_path)
            (destination / 'receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
            print(json.dumps({k: receipt[k] for k in ('rows', 'feature_count', 'participant_counts',
                                                     'row_counts', 'real_prefix_future_mutation_checks')}, indent=2), flush=True)
            return receipt
        except Exception as error:
            (destination / 'failed.json').write_text(json.dumps({'error_type': type(error).__name__,
                'error': str(error), 'models_trained': False}) + '\n')
            raise


if __name__ == '__main__': build()
