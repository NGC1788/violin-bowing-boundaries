"""One-pass archive/WAV identity audit. Does not compute any future targets.

Selected violin WAVs and per-record metadata remain outside Git. All extracted
paths are generated hashes, never archive-provided names. No source is deleted.
"""
from __future__ import annotations
import argparse
import collections
import ctypes as C
import ctypes.util
import datetime
import fcntl
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import stat
import struct


def file_sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(4 * 1024 ** 2), b''):
            h.update(block)
    return h.hexdigest()


def classify_member(name):
    name = name.replace('\\', '/')
    p = PurePosixPath(name)
    if p.is_absolute() or '..' in p.parts or any(ord(c) < 32 for c in name) or re.match(r'^[A-Za-z]:', name):
        raise ValueError('unsafe archive member path')
    if p.suffix.lower() != '.wav':
        return None, 'not_wav'
    lower = p.stem.lower()
    if 'sing' in lower:
        return None, 'voice'
    if lower.endswith('_ref'):
        return None, 'reference'
    positions = [(i, re.fullmatch(r'Participant(\d+)_(CG|FG|EG|AG)', part, re.I))
                 for i, part in enumerate(p.parts)]
    positions = [(i, m) for i, m in positions if m]
    if len(positions) != 1:
        return None, 'unclassified_wav'
    i, match = positions[0]
    cohort = match[2].upper()
    if cohort == 'AG':
        return None, 'different_AG_cohort'
    if len(p.parts) != i + 3 or p.parts[i + 1].lower() not in ('pre', 'feed', 'post'):
        return None, 'unclassified_wav'
    stage = p.parts[i + 1].lower()
    trial = re.fullmatch(r'Note\d+_Trial\d+_pitch', p.stem, re.I)
    feed = re.fullmatch(r'Note\d+_(GS|MC|PV_pitch)', p.stem, re.I)
    if not trial and not (stage == 'feed' and feed):
        return None, 'unclassified_wav'
    return {'source_id': name, 'participant_id': f'Participant{int(match[1])}_{cohort}',
            'cohort': cohort, 'stage': stage}, 'selected_violin'


def wav_metadata(path):
    """Parse RIFF headers and hash PCM bytes; never interpret acoustic values."""
    path = Path(path)
    size = path.stat().st_size
    with path.open('rb') as f:
        header = f.read(12)
        if len(header) != 12 or header[:4] != b'RIFF' or header[8:] != b'WAVE':
            raise ValueError('only little-endian RIFF/WAVE is supported')
        riff_end = struct.unpack('<I', header[4:8])[0] + 8
        if riff_end != size:
            raise ValueError('RIFF size does not match file size')
        fmt, payload = None, None
        while f.tell() + 8 <= riff_end:
            chunk = f.read(8)
            name, length = chunk[:4], struct.unpack('<I', chunk[4:])[0]
            start = f.tell()
            if start + length > riff_end:
                raise ValueError('truncated WAV chunk')
            if name == b'fmt ':
                if fmt is not None or length < 16 or length > 1024:
                    raise ValueError('unsupported WAV fmt chunk')
                raw = f.read(length)
                tag, channels, sr, byte_rate, align, bits = struct.unpack('<HHIIHH', raw[:16])
                if tag == 0xfffe:
                    if length < 40 or raw[28:40] != bytes.fromhex('00001000800000aa00389b71'):
                        raise ValueError('unsupported extensible WAV subformat')
                    valid_bits = struct.unpack('<H', raw[18:20])[0]
                    if valid_bits != bits:
                        raise ValueError('padded valid bits need an explicit decoding contract')
                    tag = struct.unpack('<I', raw[24:28])[0]
                subtype = {(1, 8): 'PCM_U8', (1, 16): 'PCM_16', (1, 24): 'PCM_24',
                           (1, 32): 'PCM_32', (3, 32): 'FLOAT', (3, 64): 'DOUBLE'}.get((tag, bits))
                if subtype is None or channels not in (1, 2) or not 8000 <= sr <= 192000:
                    raise ValueError('unsupported audio encoding')
                if align != channels * (bits // 8) or byte_rate != sr * align:
                    raise ValueError('inconsistent WAV sample layout')
                fmt = {'sample_rate': sr, 'channels': channels, 'bits': bits,
                       'block_align': align, 'subtype': subtype}
            elif name == b'data':
                if payload is not None:
                    raise ValueError('multiple data chunks are not supported')
                payload = (start, length)
            f.seek(start + length + length % 2)
        if fmt is None or payload is None or payload[1] % fmt['block_align']:
            raise ValueError('missing or partial audio samples')
        n = payload[1] // fmt['block_align']
        if n == 0:
            raise ValueError('empty audio')
        prefix_size = min(n, fmt['sample_rate']) * fmt['block_align']
        identity = json.dumps(fmt, sort_keys=True).encode() + b'\0'
        full = hashlib.sha256(identity)
        f.seek(payload[0]); left = payload[1]
        prefix = f.read(prefix_size); full.update(prefix); left -= len(prefix)
        while left:
            block = f.read(min(left, 1024 ** 2))
            if not block:
                raise ValueError('truncated PCM')
            full.update(block); left -= len(block)
        constant = prefix == prefix[:fmt['block_align']] * (len(prefix) // fmt['block_align'])
        return {**fmt, 'frames': n, 'duration_seconds': n / fmt['sample_rate'],
                'eligible_duration': n >= round(1.75 * fmt['sample_rate']),
                'pcm_sha256': full.hexdigest(),
                'prefix_pcm_sha256': hashlib.sha256(identity + prefix).hexdigest(),
                'prefix_constant_pcm': constant, 'file_bytes': size,
                'data_offset': payload[0], 'data_bytes': payload[1]}


def _library():
    name = ctypes.util.find_library('archive')
    if name is None:
        raise RuntimeError('system libarchive is unavailable')
    lib = C.CDLL(name)
    signatures = {
        'archive_read_new': (C.c_void_p, []),
        'archive_read_support_filter_all': (C.c_int, [C.c_void_p]),
        'archive_read_support_format_all': (C.c_int, [C.c_void_p]),
        'archive_read_open_filename': (C.c_int, [C.c_void_p, C.c_char_p, C.c_size_t]),
        'archive_read_next_header': (C.c_int, [C.c_void_p, C.POINTER(C.c_void_p)]),
        'archive_read_data': (C.c_ssize_t, [C.c_void_p, C.c_void_p, C.c_size_t]),
        'archive_read_data_skip': (C.c_int, [C.c_void_p]),
        'archive_read_free': (C.c_int, [C.c_void_p]),
        'archive_error_string': (C.c_char_p, [C.c_void_p]),
        'archive_entry_pathname': (C.c_char_p, [C.c_void_p]),
        'archive_entry_filetype': (C.c_uint, [C.c_void_p]),
        'archive_entry_size': (C.c_int64, [C.c_void_p]),
        'archive_entry_symlink': (C.c_char_p, [C.c_void_p]),
        'archive_entry_hardlink': (C.c_char_p, [C.c_void_p]),
        'archive_version_string': (C.c_char_p, [])}
    for function, (result, arguments) in signatures.items():
        f = getattr(lib, function); f.restype = result; f.argtypes = arguments
    return lib


def scan_archive(archive, destination):
    lib = _library(); a = lib.archive_read_new()
    destination = Path(destination); destination.mkdir(parents=True, exist_ok=True)
    records, exclusions, entries = [], collections.Counter(), 0
    extracted_bytes = 0
    def check(code):
        if code < 0:
            raise RuntimeError((lib.archive_error_string(a) or b'libarchive failure').decode(errors='replace'))
    try:
        check(lib.archive_read_support_filter_all(a)); check(lib.archive_read_support_format_all(a))
        check(lib.archive_read_open_filename(a, str(archive).encode(), 65536))
        entry = C.c_void_p(); buffer = C.create_string_buffer(1024 ** 2)
        while True:
            code = lib.archive_read_next_header(a, C.byref(entry))
            if code == 1: break
            check(code); entries += 1
            name = lib.archive_entry_pathname(entry).decode('utf-8')
            row, reason = classify_member(name)
            if row is None:
                exclusions[reason] += 1; check(lib.archive_read_data_skip(a)); continue
            if (lib.archive_entry_filetype(entry) != stat.S_IFREG or lib.archive_entry_symlink(entry)
                    or lib.archive_entry_hardlink(entry)):
                raise ValueError('selected WAV is not an ordinary file')
            expected = int(lib.archive_entry_size(entry))
            if not 44 <= expected <= 128 * 1024 ** 2 or extracted_bytes + expected > 8 * 1024 ** 3:
                raise ValueError('selected file exceeds declared extraction bounds')
            target = destination / (hashlib.sha256(name.encode()).hexdigest() + '.wav')
            temporary = target.with_suffix('.wav.tmp')
            with temporary.open('wb') as out:
                written = 0
                while True:
                    count = lib.archive_read_data(a, buffer, len(buffer))
                    check(count)
                    if count == 0: break
                    written += count
                    if written > expected: raise ValueError('archive member size changed')
                    out.write(buffer.raw[:count])
            if written != expected: raise ValueError('incomplete extracted member')
            # Never replace previously preserved raw material with differing bytes.
            if target.exists():
                if file_sha(temporary) != file_sha(target): raise ValueError('existing extraction differs')
                # Keep the duplicate temporary copy; no source or raw material deletion.
            else:
                temporary.rename(target)
            extracted_bytes += expected
            records.append({**row, 'local_path': str(target), **wav_metadata(target)})
            if len(records) % 250 == 0: print(f'WAV metadata checked: {len(records)}', flush=True)
    finally:
        lib.archive_read_free(a)
    if not records: raise ValueError('no eligible violin name patterns found; do not delete archive')
    return records, {'archive_entries': entries, 'exclusions': dict(exclusions),
                     'selected_wav_count': len(records), 'selected_wav_bytes': extracted_bytes,
                     'libarchive_version': lib.archive_version_string().decode()}


def summarize(records, base):
    eligible = [r for r in records if r['eligible_duration']]
    summary = dict(base)
    summary.update({'duration_eligible': len(eligible), 'too_short': len(records) - len(eligible),
                    'sample_rate_counts': dict(collections.Counter(r['sample_rate'] for r in records)),
                    'subtype_counts': dict(collections.Counter(r['subtype'] for r in records)),
                    'channel_counts': dict(collections.Counter(r['channels'] for r in records)),
                    'stage_counts': dict(collections.Counter(r['stage'] for r in eligible)),
                    'participant_counts': {g: len({r['participant_id'] for r in eligible if r['cohort'] == g})
                                           for g in ('CG', 'FG', 'EG')},
                    'duration_min': min(r['duration_seconds'] for r in records),
                    'duration_max': max(r['duration_seconds'] for r in records),
                    'constant_prefix_count': sum(r['prefix_constant_pcm'] for r in eligible)})
    collisions = {}
    for field in ('pcm_sha256', 'prefix_pcm_sha256'):
        groups = collections.defaultdict(list)
        for r in eligible: groups[r[field]].append(r)
        cross = [v for v in groups.values() if len({r['participant_id'] for r in v}) > 1]
        summary[field + '_extra_copies'] = sum(len(v) - 1 for v in groups.values())
        summary[field + '_cross_participant_groups'] = len(cross)
        collisions[field] = [[r['source_id'] for r in group] for group in cross]
    summary['metadata_gate_passed'] = (bool(eligible) and base['exclusions'].get('unclassified_wav', 0) == 0
                                        and all(summary[field + '_cross_participant_groups'] == 0
                                                for field in ('pcm_sha256', 'prefix_pcm_sha256')))
    summary['scope'] = 'Headers and byte identities only. No acoustic targets, scores, or questionnaire access.'
    summary['recording_origin_provenance'] = 'Unverified: file-relative support does not prove upstream segmentation was causal.'
    return summary, collisions


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--root', type=Path,
        default=Path.home() / 'violin-rne/data/external/blanco2021_raw')
    args = parser.parse_args(); here = Path(__file__).resolve().parent
    with (args.root / 'inventory.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        receipt = json.loads((here / 'download_receipt.json').read_text())
        archive = args.root / receipt['filename']
        if archive.stat().st_size != receipt['bytes'] or file_sha(archive) != receipt['sha256']:
            raise ValueError('archive identity differs from verified acquisition receipt')
        if (args.root / 'inventory_records.json').exists():
            raise ValueError('existing inventory preserved; review its receipt instead of overwriting')
        records, base = scan_archive(archive, args.root / 'violin_wav')
        summary, collisions = summarize(records, base)
        output = args.root / 'inventory_records.json'
        output.write_text(json.dumps({'records': records, 'collisions': collisions}, indent=2) + '\n')
        summary.update({'archive_sha256': receipt['sha256'], 'inventory_sha256': file_sha(output),
                        'code_sha256': file_sha(__file__), 'checked_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
                        'record_id': '4553940', 'license': 'CC-BY-4.0'})
        (here / 'inventory_summary.json').write_text(json.dumps(summary, indent=2) + '\n')
        print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__': main()
