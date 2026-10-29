"""Bounded LLM proposer/reviewer. Only a registered JSON configuration can execute.

The CLI uses the user's saved Codex login. This is not a secure multiparty test
service: read-only CLI permissions and hashes detect ordinary mistakes; the OS
account remains trusted. The evaluator never executes LLM-generated Python.
"""
from __future__ import annotations
import argparse
import fcntl
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time
import uuid
import control

ROOT = Path(__file__).resolve().parent


def llm(prompt, schema, directory, name, timeout):
    executable = shutil.which('codex')
    if not executable:
        raise control.ControlError('Codex CLI missing; install/login before an agent round')
    schema_path = directory / (name + '_schema.json')
    output = directory / (name + '_response.json')
    control.atomic_json(schema_path, schema)
    (directory / (name + '_prompt.txt')).write_text(prompt)
    command = [executable, 'exec', '--sandbox', 'read-only', '--ephemeral',
               '--ignore-user-config', '--skip-git-repo-check', '--color', 'never',
               '--json', '--output-schema', str(schema_path),
               '--output-last-message', str(output), '-C', str(directory), '-']
    started = time.monotonic()
    with (directory / (name + '_events.jsonl')).open('w') as log:
        result = subprocess.run(command, input=prompt, text=True, stdout=log,
                                stderr=subprocess.STDOUT, timeout=timeout)
    if result.returncode != 0 or not output.exists():
        raise control.ControlError(f'{name} CLI failed; inspect {directory}')
    value = json.loads(output.read_text())
    control.atomic_json(directory / (name + '_call.json'),
                        {'elapsed_seconds': time.monotonic()-started,
                         'returncode': result.returncode, 'command': command,
                         'response_sha': control.digest(value)})
    return value


def schema(properties):
    return {'type':'object', 'properties': properties,
            'required':list(properties), 'additionalProperties':False}


def available(p, manifest):
    with control.database(p) as db:
        submitted = {row[0] for row in db.execute('SELECT run_id FROM jobs')}
    pool = {}
    directory=control.resolve_candidate_dir(ROOT,manifest['protocol'])
    for path in sorted(directory.glob('*.json')):
        raw = json.loads(path.read_text())
        normalized = control.normalized_candidate(raw, manifest['protocol'])
        key = control.identity(normalized, manifest)
        if not isinstance(raw.get('id'),str):raise control.ControlError('catalog candidate needs an id')
        if raw['id'] in pool:raise control.ControlError('duplicate candidate ID in frozen catalog')
        if key not in submitted:
            pool[raw['id']] = {'config':raw, 'semantic_sha':key}
    return pool


def one_round(campaign, execute=False, llm_timeout=180):
    p = control.paths(campaign)
    manifest = control.integrity(p)
    status = control.status(campaign)
    if status.get('safety_halt'):
        return {'status':'stopped','reason':'verifier/integrity safety halt'}
    if status['stopped'] or control.expired(manifest):
        return {'status':'stopped', 'reason':'STOP or campaign expiry'}
    if status['counts'].get('running', 0) or status['counts'].get('queued', 0):
        if execute:
            return {'status':'queue_processed', 'worker':control.run(campaign, 1, 600)}
        return {'status':'waiting_for_queue'}
    headroom=control.headroom_gate(p,manifest)
    if not headroom['passed']:
        return {'status':'stopped','reason':'headroom prerequisite not passed','headroom':headroom}
    prior_rounds = list((p['state']/'agent_rounds').glob('*/round.json'))
    max_rounds=manifest['protocol'].get('planner',{}).get('max_rounds',12)
    if len(prior_rounds) >= max_rounds:
        return {'status':'stopped', 'reason':f'{max_rounds}-round LLM budget exhausted'}
    pool = available(p, manifest)
    if not pool:
        return {'status':'stopped', 'reason':'registered hypothesis pool exhausted; review before expanding'}
    if status['consumed_seconds'] >= status['budget']['max_campaign_seconds']:
        return {'status':'stopped', 'reason':'compute budget exhausted'}
    name = time.strftime('%Y%m%dT%H%M%S') + '_' + uuid.uuid4().hex[:8]
    directory = p['state'] / 'agent_rounds' / name
    directory.mkdir(parents=True)
    # Record the round before any call; failed/time-out calls also spend a round.
    record = {'campaign':campaign, 'manifest_sha':manifest['manifest_sha'],
              'status':'started', 'created_at':time.time()}
    control.atomic_json(directory/'round.json', record)
    report = control.report(campaign)
    control.assert_safe(p)
    # The proposer needs all attempts but not full per-column probabilities.
    compact = []
    for row in report.get('runs', []):
        item = {k: row.get(k) for k in ('run_id','alias','status','config','elapsed','error')}
        if 'metrics' in row:
            item['metrics'] = {k:row['metrics'].get(k) for k in
                ('mean_queries','p90_queries','max_queries','certificate_error_count',
                 'all_certified','eval_columns','rules','groups','timing')}
        compact.append(item)
    report['runs'] = compact
    context = {'protocol':manifest['protocol'], 'available_candidates':pool,
               'status':status, 'report':report,'headroom_gate':headroom}
    prefix = (ROOT/'prompts/proposer.md').read_text()
    fields = {'candidate_id':{'type':'string','enum':['STOP', *pool]},
              'hypothesis':{'type':'string'}, 'falsifier':{'type':'string'},
              'reason':{'type':'string'}}
    try:
        proposal = llm(prefix+'\n\nCONTEXT JSON:\n'+control.canonical(context),
                       schema(fields), directory, 'proposer', llm_timeout)
        if proposal.get('candidate_id') == 'STOP':
            record.update(status='stopped', proposal=proposal)
            control.atomic_json(directory/'round.json',record)
            control.set_stop(campaign, True)
            return record
        selected = pool[proposal['candidate_id']]
        if any(not isinstance(proposal.get(k),str) or len(proposal[k].strip())<10
               for k in ('hypothesis','falsifier','reason')):
            raise control.ControlError('proposal must state a concrete hypothesis and falsifier')
        review_fields = {'verdict':{'type':'string','enum':['APPROVE','REJECT']},
                         'candidate_sha':{'type':'string','enum':[selected['semantic_sha']]},
                         'reason':{'type':'string'}, 'limitations':{'type':'string'}}
        review_context = {'selected':selected,'proposal':proposal,'context':context}
        review = llm((ROOT/'prompts/reviewer.md').read_text()+'\n\nCONTEXT JSON:\n'+
                     control.canonical(review_context), schema(review_fields),
                     directory,'reviewer',llm_timeout)
        if review.get('candidate_sha') != selected['semantic_sha']:
            raise control.ControlError('review does not bind to the selected configuration')
        record.update(proposal=proposal, review=review,
                      candidate_sha=selected['semantic_sha'])
        if review.get('verdict') != 'APPROVE':
            record['status']='rejected';control.atomic_json(directory/'round.json',record)
            return record
        # Final deterministic validation: an LLM verdict never bypasses these checks.
        control.integrity(p)
        if not control.headroom_gate(p,manifest)['passed']:raise control.ControlError('headroom prerequisite changed during proposal')
        submitted = control.enqueue(campaign,selected['config'])
        record.update(status='submitted',submission=submitted)
        control.atomic_json(directory/'round.json',record)
        if execute:
            record['worker']=control.run(campaign,1,600)
            record['status']='evaluated'
            control.atomic_json(directory/'round.json',record)
        return record
    except Exception as exc:
        record.update(status='failed',error=f'{type(exc).__name__}: {exc}')
        control.atomic_json(directory/'round.json',record)
        raise


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--campaign',required=True)
    ap.add_argument('--execute',action='store_true')
    ap.add_argument('--llm-timeout',type=int,default=180)
    args=ap.parse_args()
    if not 30 <= args.llm_timeout <= 600:
        ap.error('llm timeout must be between30and600seconds')
    try:
        p=control.paths(args.campaign)
        p['state'].mkdir(parents=True,exist_ok=True)
        with (p['state']/'planner.lock').open('a') as handle:
            try:fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
            except BlockingIOError:raise control.ControlError('another planner owns this campaign')
            result=one_round(args.campaign,args.execute,args.llm_timeout)
        print(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False))
    except (control.ControlError, subprocess.TimeoutExpired, ValueError, OSError, KeyError) as exc:
        print(f'ERROR: {exc}',file=sys.stderr);return 2
    return 0


if __name__=='__main__':raise SystemExit(main())
