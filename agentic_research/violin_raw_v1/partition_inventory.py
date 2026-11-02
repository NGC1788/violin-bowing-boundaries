"""Reconcile the observed AG2 folder and seal a target-blind participant split.

The original inventory and its failed unknown-file gate remain untouched.
Only CG/FG/EG were eligible before file inspection; AG2 is explicitly outside
that allowlist. No claim is made about the undocumented meaning of AG2.
"""
from __future__ import annotations
import argparse
import collections
import datetime
import fcntl
import json
from pathlib import Path, PurePosixPath
import re
import subprocess
from .data import participant_split, deduplicate_manifest, digest
from .inventory import classify_member, file_sha


def reconcile(records, member_names, original_summary):
    counts = collections.Counter()
    selected = []
    for name in member_names:
        row, reason = classify_member(name)
        if reason == 'unclassified_wav':
            parts = PurePosixPath(name.replace('\\', '/')).parts
            if sum(bool(re.fullmatch(r'Participant\d+_AG2', p, re.I)) for p in parts) == 1:
                reason = 'AG2_outside_declared_cohorts'
        if row is not None:
            selected.append(row['source_id'])
        else:
            counts[reason] += 1
    if len(member_names) != original_summary['archive_entries']:
        raise ValueError('archive listing entry count differs from original inventory')
    if sorted(selected) != sorted(r['source_id'] for r in records):
        raise ValueError('reconciliation would change selected waveforms')
    if counts['unclassified_wav']:
        raise ValueError('unresolved WAV names remain; no partition permitted')
    eligible = [r for r in records if r['eligible_duration']]
    unique, aliases = deduplicate_manifest(eligible)
    people = sorted({(r['participant_id'], r['cohort']) for r in unique})
    if len({pid for pid, _ in people}) != len(people):
        raise ValueError('participant assigned to multiple cohorts')
    manifest, split_info = participant_split([{'participant_id': p, 'cohort': c} for p, c in people])
    selected_rows = [r['source_id'] for r in unique]
    receipt = {'status': 'METADATA_GATE_PASSED_AND_TARGET_BLIND_SPLIT_FIXED',
               'eligible_cohorts': ['CG', 'FG', 'EG'], 'exclusions': dict(counts),
               'AG2_interpretation': 'Undocumented extra folder label; excluded by the pre-existing CG/FG/EG allowlist.',
               'selection_changed': False, 'original_unknown_count': original_summary['exclusions'].get('unclassified_wav', 0),
               'eligible_recordings': len(unique), 'aliases_removed': sum(len(v) - 1 for v in aliases.values()),
               'split': split_info, 'participant_counts': {s: list(manifest.values()).count(s) for s in ('train', 'dev', 'test')},
               'row_counts': {s: sum(manifest[r['participant_id']] == s for r in unique) for s in ('train', 'dev', 'test')},
               'selected_source_ids_sha256': digest(selected_rows),
               'selection_uses': ['cohort/folder and filename metadata', 'recorded sample count', 'exact full/prefix PCM identity hashes'],
               'selection_does_not_use': ['future acoustic descriptor', 'target guard', 'future rule label', 'model score'],
               'future_targets_computed': False, 'independent_review': 'pending', 'training_authorized_by_this_receipt': False,
               'scope': 'Recording-relative prediction only; upstream recording/segmentation provenance still unverified.'}
    return manifest, receipt


def write_once(path, value):
    path = Path(path)
    if path.exists():
        if json.loads(path.read_text()) != value:
            raise ValueError('existing metadata seal differs; preserve it and review changes')
        return
    with path.open('x') as f:
        f.write(json.dumps(value, indent=2, sort_keys=True) + '\n')


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--root', type=Path,
        default=Path.home() / 'violin-rne/data/external/blanco2021_raw')
    args = parser.parse_args(); here = Path(__file__).resolve().parent
    with (args.root / 'partition.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        original = json.loads((here / 'inventory_summary.json').read_text())
        raw = args.root / 'inventory_records.json'
        if file_sha(raw) != original['inventory_sha256']:
            raise ValueError('original inventory identity changed')
        records = json.loads(raw.read_text())['records']
        archive = args.root / 'Participants_Raw_data.rar'
        if file_sha(archive) != original['archive_sha256']:
            raise ValueError('archive identity changed')
        listing = subprocess.run(['tar', '-tf', str(archive)], check=True, capture_output=True, text=True, timeout=120)
        names = listing.stdout.splitlines()
        manifest, receipt = reconcile(records, names, original)
        write_once(args.root / 'participant_split_v1.json', manifest)
        receipt.update({'archive_sha256': original['archive_sha256'], 'original_inventory_sha256': original['inventory_sha256'],
                        'original_summary_sha256': file_sha(here / 'inventory_summary.json'),
                        'archive_listing_sha256': digest(names), 'partition_code_sha256': file_sha(__file__),
                        'data_contract_code_sha256': file_sha(here / 'data.py')})
        # No changing timestamp in a write-once receipt.
        write_once(here / 'data_partition_receipt.json', receipt)
        print(json.dumps(receipt, indent=2), flush=True)


if __name__ == '__main__': main()
