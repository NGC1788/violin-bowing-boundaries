#!/usr/bin/env python3
"""One companion to replicate.py: pin archives, run unchanged Claude extractors.

Never starts/stops/modifies replicate.py or deletes an original archive. A hardlink
is acquired before hashing/listing, so unlink by the original pipeline is harmless.
The watch loop keeps acquiring links while one extraction job runs. Failure retains
both the archive link and staging output. Final publication follows validation.

Run only after review: python archive_companion.py watch [--poll-seconds 15]
One scan: python archive_companion.py once
Adopt already produced A1/A2: python archive_companion.py adopt-legacy ARCHIVE.7z
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
import csv
from datetime import datetime, timezone
import fcntl
import gzip
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import subprocess
import sys
import time
import uuid

import numpy as np

HOME = Path.home()
ROOT = HOME / 'violin-rne'
REPO = HOME / 'violin-bowing-boundaries'
HERE = ROOT / 'astra_20261018'
DEFAULT_WORK = HERE / 'archive_companion'
PYTHON = REPO / '.venv-mac/bin/python'
SEVEN = REPO / '.local-tools/7zip-26.03/7zz'
SCRIPTS = {name: ROOT / 'claude_extra' / f'{name}.py' for name in ('ramp_meta', 'stroke_blocks')}
PROTECTED_A2 = REPO / 'data/raw/zenodo/17782542/2024-04-19_TypeA_sample2.7z'
BLOCK_FIELDS = ['speed', 'position', 'force_dc', 'force_ac', 'force_ac_lo', 'bridge_dc', 'nut_dc', 'bridge_p2p', 'bridge_edge', 'coh']
RESERVE_BYTES = 60_000_000_000
OUTPUT_ALLOWANCE = 1_000_000_000
LEGACY = {
    '2024-03-25_TypeA_sample1.7z': (
        ROOT / 'claude_extra/ramp/ramp_meta_2024-03-25_TypeA_sample1.csv',
        ROOT / 'claude_extra/blocks/blocks_A1.npz'),
    '2024-04-19_TypeA_sample2.7z': (
        ROOT / 'claude_extra/ramp_meta_A2.csv', ROOT / 'claude_extra/blocks/blocks_A2.npz'),
}


def now():
    return datetime.now(timezone.utc).isoformat()


def log(message):
    print(f'[{now()}] {message}', flush=True)


def digest(path, algorithm='sha256'):
    h = hashlib.new(algorithm)
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def fsync_dir(path):
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
    with tmp.open('x') as f:
        json.dump(value, f, indent=2, allow_nan=False)
        f.write('\n')
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)
    fsync_dir(path.parent)


@contextmanager
def singleton(work):
    work.mkdir(parents=True, exist_ok=True)
    with (work / 'companion.lock').open('a+') as f:
        try:
            fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as e:
            raise RuntimeError('another archive companion holds the lock') from e
        f.seek(0)
        f.truncate()
        f.write(f'{os.getpid()}\n')
        f.flush()
        try:
            yield f.fileno()
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


def valid_name(name):
    if Path(name).name != name or not name.endswith('.7z') or name.startswith('.'):
        raise ValueError('expected an exact .7z archive basename')
    return name


def pin_archive(source, work):
    source = Path(source)
    valid_name(source.name)
    pins = work / 'pins'
    pins.mkdir(parents=True, exist_ok=True)
    pin = pins / source.name
    if pin.exists():
        if source.exists() and not os.path.samefile(source, pin):
            raise RuntimeError('existing pin has different inode; preserve and inspect')
        return pin
    # link() atomically acquires a durable reference; cross-filesystem copies are forbidden.
    os.link(source, pin, follow_symlinks=False)
    if pin.is_symlink() or not pin.is_file():
        raise RuntimeError('archive must be a regular file, not a symlink')
    fsync_dir(pins)
    return pin


def metadata_spec(name):
    matches = []
    for directory in (ROOT / 'data/manifests', REPO / 'data/manifests'):
        for path in directory.glob('zenodo_*.json'):
            d = json.loads(path.read_text())
            record = d.get('record', d)
            for item in record.get('files', []):
                if item.get('key') == name:
                    checksum = item.get('checksum', '')
                    if not re.fullmatch(r'md5:[0-9a-fA-F]{32}', checksum):
                        raise ValueError('missing MD5 in local metadata')
                    matches.append((int(item['size']), checksum[4:].lower(), str(path)))
    if not matches:
        raise ValueError(f'no local Zenodo manifest for {name}; archive remains pinned')
    if len({m[:2] for m in matches}) != 1:
        raise ValueError('conflicting archive metadata')
    size, md5, source = matches[0]
    return {'bytes': size, 'md5': md5, 'metadata_path': source, 'metadata_sha256': digest(source)}


def parse_inventory(text):
    """Require one beta and whole item per full (folder, trial) identity."""
    entries, path = {}, None
    for line in text.splitlines():
        if line.startswith('Path = '):
            path = line[7:].strip().replace('\\', '/')
        elif line.startswith('Size = ') and path is not None:
            match = re.fullmatch(r'(.+)/(whole|beta)_(\d+)\.csv', path)
            if match:
                folder, kind, trial = match[1], match[2], int(match[3])
                pp = PurePosixPath(folder)
                if pp.is_absolute() or '..' in pp.parts:
                    raise ValueError('unsafe archive member')
                key = (folder, trial)
                if kind in entries.setdefault(key, {}):
                    raise ValueError('duplicate archive member identity')
                entries[key][kind] = int(line[7:])
            path = None
    if not entries or any(set(v) != {'whole', 'beta'} or min(v.values()) <= 0 for v in entries.values()):
        raise ValueError('incomplete whole/beta inventory')
    leaf_map(entries)  # Claude block extractor loses parent directory information.
    return entries


def leaf_map(keys):
    mapping = {}
    for folder, trial in keys:
        leaf = folder.rsplit('/', 1)[-1]
        if leaf in mapping and mapping[leaf] != folder:
            raise ValueError('ambiguous folder basename in unchanged stroke_blocks.py')
        mapping[leaf] = folder
    return mapping


def archive_inventory(pin):
    result = subprocess.run([str(SEVEN), 'l', '-slt', str(pin)], text=True, capture_output=True, check=True)
    return parse_inventory(result.stdout)


def validate_outputs(ramp_path, block_path, expected):
    """Exact trial identities, unique keys, beta agreement, valid full block spans.

NaN coherence is allowed for a constant signal. The remaining block fields must be
finite over every recorded full 50 ms block and NaN only after recording end.
The upstream fixed 8 s cap is rejected when it would truncate complete blocks.
"""
    expected = set(expected)
    mapping = leaf_map(expected)
    rows = {}
    with Path(ramp_path).open(newline='') as f:
        reader = csv.DictReader(f)
        required = {'folder', 'trial', 'beta', 'v_max', 'onset_s', 'plateau_on_s', 'plateau_off_s', 'total_s', 'ramp_s', 'accel', 'force_at_onset', 'error'}
        if not required <= set(reader.fieldnames or []):
            raise ValueError('ramp schema mismatch')
        for row in reader:
            key = (row['folder'], int(row['trial']))
            if key in rows or key not in expected or row['error']:
                raise ValueError('ramp duplicate/unexpected/error trial')
            nums = {k: float(row[k]) for k in required - {'folder', 'trial', 'error'}}
            if not all(math.isfinite(x) for x in nums.values()):
                raise ValueError('nonfinite ramp metadata')
            if not (0 < nums['beta'] < 1 and nums['v_max'] > 0 and nums['accel'] > 0 and
                    0 <= nums['onset_s'] <= nums['plateau_on_s'] < nums['plateau_off_s'] < nums['total_s']):
                raise ValueError('invalid ramp timing/beta')
            rows[key] = nums
    if set(rows) != expected:
        raise ValueError('ramp trial coverage mismatch')
    with np.load(block_path, allow_pickle=False) as z:
        if set(z.files) != {'data', 'fields', 'folder', 'trial', 'beta', 'block_s'}:
            raise ValueError('block schema mismatch')
        data = z['data']
        fields = z['fields'].tolist()
        folder, trials, beta = z['folder'], z['trial'], z['beta']
        if fields != BLOCK_FIELDS or float(z['block_s']) != .05:
            raise ValueError('block fields/time resolution mismatch')
        n = len(expected)
        if data.shape != (n, 10, 160) or any(a.shape != (n,) for a in (folder, trials, beta)):
            raise ValueError('block shape/coverage mismatch')
        seen, total_blocks = set(), 0
        for i in range(n):
            leaf = str(folder[i])
            key = (mapping.get(leaf, ''), int(trials[i]))
            if key not in expected or key in seen:
                raise ValueError('block duplicate/unexpected trial')
            seen.add(key)
            r = rows[key]
            if not math.isfinite(float(beta[i])) or not math.isclose(float(beta[i]), r['beta'], abs_tol=1e-10, rel_tol=1e-8):
                raise ValueError('ramp/block beta mismatch')
            samples = round(r['total_s'] * 50000)
            count = samples // 2500
            if not 1 <= count <= 160:
                raise ValueError('empty or truncated 8 s block recording')
            if not np.isfinite(data[i, :9, :count]).all() or not np.isnan(data[i, :, count:]).all():
                raise ValueError('block corruption/parse failure/length mismatch')
            coh = data[i, 9, :count]
            if np.isinf(coh).any() or np.any(np.abs(coh[np.isfinite(coh)]) > 1.00001):
                raise ValueError('invalid block coherence')
            if np.any(data[i, [3, 4, 7, 8], :count] < 0):
                raise ValueError('negative block magnitude')
            total_blocks += count
    if seen != expected:
        raise ValueError('block trial coverage mismatch')
    counts = {}
    for folder, _ in expected:
        counts[folder] = counts.get(folder, 0) + 1
    return {'trials': len(expected), 'blocks': total_blocks, 'folders': counts,
            'ramp_sha256': digest(ramp_path), 'blocks_sha256': digest(block_path),
            'coverage_sha256': hashlib.sha256(json.dumps(sorted(expected)).encode()).hexdigest()}


def publish(stage, destination):
    """Publish whole directory only after all files are closed and validated."""
    for path in stage.iterdir():
        if path.is_file():
            with path.open('rb') as f:
                os.fsync(f.fileno())
    fsync_dir(stage)
    if destination.exists():
        raise FileExistsError('output already published; refusing overwrite')
    os.rename(stage, destination)
    fsync_dir(destination.parent)


def run_archive(pin, work):
    # Child Python inherits this lock descriptor. If the watcher is killed, a
    # still-running extractor retains the lock, preventing duplicate extraction.
    with singleton(work / 'job_locks' / valid_name(pin.name)) as fd:
        return _run_archive(pin, work, fd)


def _run_archive(pin, work, lock_fd):
    name = valid_name(pin.name)
    final = work / name[:-3]
    receipt_path = work / 'receipts' / (name + '.json')
    stage = work / 'staging' / (name[:-3] + '-' + uuid.uuid4().hex)
    stage.mkdir(parents=True)
    receipt = {'archive': name, 'started_utc': now(), 'status': 'running', 'pin': str(pin),
               'stage': str(stage), 'commands': [], 'extractor_sha256': {k: digest(v) for k, v in SCRIPTS.items()},
               'companion_sha256': digest(__file__)}
    atomic_json(receipt_path, receipt)
    try:
        spec = metadata_spec(name)
        if pin.stat().st_size != spec['bytes'] or digest(pin, 'md5') != spec['md5']:
            raise ValueError('archive bytes/MD5 disagree with local metadata')
        receipt['archive_integrity'] = spec
        expected = archive_inventory(pin)
        if shutil.disk_usage(work).free < RESERVE_BYTES + OUTPUT_ALLOWANCE:
            raise RuntimeError('less than 60 GB reserve plus output allowance; pin retained')
        ramp, block = stage / ('ramp_meta_' + name[:-3] + '.csv'), stage / ('blocks_' + name[:-3] + '.npz')
        legacy = LEGACY.get(name)
        if legacy and all(p.exists() for p in legacy):
            try:
                validate_outputs(*legacy, expected)
            except (ValueError, OSError, KeyError) as e:
                receipt['legacy_rejected'] = str(e)
            else:
                shutil.copy2(legacy[0], ramp)
                shutil.copy2(legacy[1], block)
                receipt['legacy_reused'] = [str(p) for p in legacy]
                receipt['legacy_exit_codes'] = 'unknown; adopted only after current coverage/integrity validation'
        if not ramp.exists():
            for key, output in [('ramp_meta', ramp), ('stroke_blocks', block)]:
                command = [str(PYTHON), '-I', str(SCRIPTS[key]), str(pin), str(output)]
                record = {'extractor': key, 'command': command, 'started_utc': now()}
                receipt['commands'].append(record)
                atomic_json(receipt_path, receipt)
                with (stage / (key + '.log')).open('xb') as fh:
                    result = subprocess.run(command, stdout=fh, stderr=subprocess.STDOUT, pass_fds=(lock_fd,))
                record.update(exit_code=result.returncode, ended_utc=now())
                atomic_json(receipt_path, receipt)
                if result.returncode:
                    raise RuntimeError(f'{key} exited {result.returncode}')
        receipt['validation'] = validate_outputs(ramp, block, expected)
        receipt.update(status='validated', ended_utc=now(), published=str(final))
        atomic_json(stage / 'receipt.json', receipt)
        publish(stage, final)
        # External receipt is durable before pin release. A crash may retain an extra
        # link; that is harmless and recoverable, unlike premature release.
        receipt['status'] = 'complete'
        atomic_json(receipt_path, receipt)
        pin.unlink()
        fsync_dir(pin.parent)
        log(f'{name}: companion complete, archive link released')
        return receipt
    except Exception as e:
        receipt.update(status='failed', ended_utc=now(), error=f'{type(e).__name__}: {e}')
        atomic_json(receipt_path, receipt)
        log(f'{name}: failed, archive and staged outputs retained: {e}')
        return receipt


def legacy_expected(name):
    """For a deleted A1 original, independently cross-check feature IDs and state.

This validates legacy output coverage, not original archive MD5 or extractor exit
status. Only the two explicitly named already-exposed archives are admissible.
"""
    if name not in LEGACY:
        raise ValueError('legacy adoption allowed only for explicitly mapped A1/A2')
    state_path = ROOT / 'timeres/replicate_state.json'
    entry = json.loads(state_path.read_text())[name]
    if entry.get('status') != 'done':
        raise ValueError('legacy pipeline archive is not complete')
    feature = ROOT / 'timeres/features' / (name[:-3] + '.csv.gz')
    keys = set()
    with gzip.open(feature, 'rt', newline='') as f:
        for row in csv.DictReader(f):
            if row.get('error'):
                raise ValueError('legacy feature error row')
            keys.add((row['folder'], int(row['trial'])))
    expected = {(folder, n) for folder, count in entry['folders'].items() for n in range(1, count + 1)}
    if keys != expected or set(entry['folders_done']) != set(entry['folders']):
        raise ValueError('legacy features/state exact trial coverage disagreement')
    return expected, {'feature_sha256': digest(feature), 'state_sha256': digest(state_path),
                      'scope': 'feature IDs cross-checked against completed legacy state and contiguous 1..N trial identifiers; not raw archive verification'}


def adopt_legacy(name, work):
    name = valid_name(name)
    expected, reference = legacy_expected(name)
    validation = validate_outputs(*LEGACY[name], expected)
    final = work / name[:-3]
    stage = work / 'staging' / (name[:-3] + '-' + uuid.uuid4().hex)
    stage.mkdir(parents=True)
    for src, prefix, suffix in zip(LEGACY[name], ['ramp_meta_', 'blocks_'], ['.csv', '.npz']):
        shutil.copy2(src, stage / (prefix + name[:-3] + suffix))
    receipt = {'archive': name, 'status': 'legacy_outputs_validated', 'ended_utc': now(), 'validation': validation,
               'reference': reference, 'legacy_paths': [str(p) for p in LEGACY[name]],
               'extractor_exit_codes': None, 'original_archive_md5_verified': False}
    atomic_json(stage / 'receipt.json', receipt)
    publish(stage, final)
    atomic_json(work / 'receipts' / (name + '.json'), receipt)
    return receipt


def discover(work):
    originals = list((ROOT / 'data/raw/zenodo').glob('*/*.7z'))
    if PROTECTED_A2.exists():
        originals.append(PROTECTED_A2)
    for source in originals:
        receipt = work / 'receipts' / (source.name + '.json')
        if receipt.exists():
            status = json.loads(receipt.read_text())['status']
            if status in {'complete', 'legacy_outputs_validated', 'failed'}:
                continue
        try:
            pin_archive(source, work)
        except FileNotFoundError:
            log(f'{source.name}: source removed before pin; not marked complete')
    return sorted((work / 'pins').glob('*.7z'))


def loop(work, once=False, poll=15, retry_failed=False):
    """Only one extractor job; scanning continues while it runs."""
    attempted = set()
    current = None
    with singleton(work), ThreadPoolExecutor(max_workers=1) as pool:
        while True:
            pins = discover(work)
            if current is not None and current.done():
                current.result()
                current = None
            if current is None:
                for pin in pins:
                    if pin.name in attempted:
                        continue
                    rp = work / 'receipts' / (pin.name + '.json')
                    status = json.loads(rp.read_text())['status'] if rp.exists() else None
                    if status in {'complete', 'legacy_outputs_validated'}:
                        # A previously completed job can leave a pin after a crash.
                        # Preserve it pending explicit review, rather than deleting blindly.
                        continue
                    if status == 'failed' and not retry_failed:
                        continue
                    attempted.add(pin.name)
                    current = pool.submit(run_archive, pin, work)
                    break
            if once and current is None:
                return
            time.sleep(poll)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('command', choices=['watch', 'once', 'adopt-legacy'])
    p.add_argument('archive', nargs='?')
    p.add_argument('--work', type=Path, default=DEFAULT_WORK)
    p.add_argument('--poll-seconds', type=float, default=15)
    p.add_argument('--retry-failed', action='store_true')
    args = p.parse_args()
    if not 1 <= args.poll_seconds <= 60:
        p.error('--poll-seconds must be between 1 and 60')
    if args.command == 'adopt-legacy':
        if not args.archive:
            p.error('adopt-legacy requires an archive basename')
        with singleton(args.work):
            print(json.dumps(adopt_legacy(args.archive, args.work), indent=2))
    else:
        loop(args.work, args.command == 'once', args.poll_seconds, args.retry_failed)


if __name__ == '__main__':
    main()
