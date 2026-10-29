"""One bounded campaign tick: preflight -> exact headroom -> reviewed experiment.

No adaptive endpoint changes or new Python generation. Frozen v1 is preserved.
"""
from __future__ import annotations
import argparse
import fcntl
import json
from pathlib import Path
import sys
import control
import planner
from harness.headroom import make_report


def advance(campaign, llm_timeout=180):
    p = control.paths(campaign)
    manifest = control.integrity(p)
    status = control.status(campaign)
    if status['stopped'] or control.expired(manifest):
        return {'status': 'stopped', 'campaign': campaign, 'reason': 'STOP or expiry'}
    protocol = manifest['protocol']
    gate = control.headroom_gate(p, manifest)
    if not gate['ready']:
        # The five configurations are fixed before any full-development scores.
        directory = p['root'] / protocol['planner']['candidate_dir']
        required = protocol['preflight_candidate_ids']
        ids = []
        for name in required:
            candidate = json.loads((directory / (name + '.json')).read_text())
            ids.append(control.enqueue(campaign, candidate)['run_id'])
        worker = control.run(campaign, max_jobs=len(required), max_seconds=300)
        report = control.report(campaign)
        rows = {row['run_id']: row for row in report['runs']}
        bad = [rid for rid in ids if rows.get(rid, {}).get('status') != 'completed']
        if bad:
            control.set_stop(campaign, True)
            return {'status': 'stopped', 'reason': 'preflight incomplete or failed',
                    'run_ids': bad, 'worker': worker}
        result = make_report(Path(manifest['data_path']), protocol,
                             [rows[rid] for rid in ids], manifest['manifest_sha'])
        control.register_headroom(campaign, result)
        gate = control.headroom_gate(p, manifest)
    if not gate['passed']:
        control.set_stop(campaign, True)
        return {'status': 'stopped', 'reason': 'primary improvement mathematically unattainable',
                'headroom': gate}
    # Shared with direct planner CLI; a second caller cannot propose concurrently.
    with (p['state'] / 'planner.lock').open('a') as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return {'status': 'busy', 'reason': 'another planner owns this campaign'}
        result = planner.one_round(campaign, execute=True, llm_timeout=llm_timeout)
    if result.get('status') == 'stopped':
        control.set_stop(campaign, True)
    # Always materialize fresh sealed results; do not print all column data.
    report = control.report(campaign)
    return {'status': result.get('status'), 'campaign': campaign,
            'candidate_sha': result.get('candidate_sha'),
            'reason': result.get('reason'),
            'completed_runs': sum(r['status'] == 'completed' for r in report['runs']),
            'state': control.status(campaign)}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--campaign', default='development480_20261030_v2')
    ap.add_argument('--llm-timeout', type=int, default=180)
    a = ap.parse_args()
    if not 30 <= a.llm_timeout <= 600:
        ap.error('LLM timeout must be30..600seconds')
    p = control.paths(a.campaign)
    p['state'].mkdir(parents=True, exist_ok=True)
    try:
        with (p['state'] / 'stage.lock').open('a') as handle:
            try:
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                print(json.dumps({'status':'busy','reason':'another stage owns campaign'}))
                return 0
            result = advance(a.campaign, a.llm_timeout)
        print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
        return 0
    except Exception as exc:
        # Inspection needed; never silently refreeze, retry, or change endpoints.
        if p['settings'].exists():
            control.set_stop(a.campaign, True)
        print(f'ERROR: {type(exc).__name__}: {exc}', file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
