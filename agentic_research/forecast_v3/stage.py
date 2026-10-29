"""Finite portable research loop; each tick runs fixed baselines or one review."""
from __future__ import annotations
import argparse
import fcntl
import json
from pathlib import Path
import control
import planner


def advance(campaign,proposer_backend='codex',reviewer_backend='codex',timeout=180):
    p=control.paths(campaign)
    manifest=control.integrity(p)
    status=control.status(campaign)
    if status['stopped'] or control.expired(manifest):
        return {'status':'stopped','campaign':campaign}
    protocol=manifest['protocol']
    directory=control.resolve_candidate_dir(p['root'],protocol)
    # Baseline identities are frozen and always precede adaptive run ordering.
    for name in protocol['preflight_candidate_ids']:
        control.enqueue(campaign,json.loads((directory/f'{name}.json').read_text()))
    report=control.report(campaign)
    required=set(protocol['preflight_candidate_ids'])
    outstanding=[r for r in report['runs'] if r['alias'] in required and r['status']!='completed']
    if outstanding:
        work=control.run(campaign,max_jobs=len(outstanding),max_seconds=600)
        report=control.report(campaign)
        if any(r['status']!='completed' for r in report['runs'] if r['alias'] in required):
            control.set_stop(campaign,True)
            return {'status':'stopped','reason':'baseline failure or interruption','worker':work}
    with (p['state']/'planner.lock').open('a') as handle:
        try:
            fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:
            return {'status':'busy','reason':'another planner owns campaign'}
        planner.PROPOSER_BACKEND=proposer_backend
        planner.REVIEWER_BACKEND=reviewer_backend
        result=planner.one_round(campaign,execute=True,llm_timeout=timeout)
    if result.get('status')=='stopped':
        control.set_stop(campaign,True)
    report=control.report(campaign)
    return {'status':result.get('status'),'campaign':campaign,'candidate_sha':result.get('candidate_sha'),
            'reason':result.get('reason'),'error':result.get('error'),
            'counts':control.status(campaign)['counts'],
            'latest':[{ 'method':r['config']['method'],'seed':r['config']['seed'],
                       'loss':r.get('metrics',{}).get('mean_log_loss'),'status':r['status']}
                      for r in report['runs'][-2:]]}


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--campaign',default='forecast_20261030_v3')
    ap.add_argument('--proposer-backend',choices=['codex','claude','manual'],default='codex')
    ap.add_argument('--reviewer-backend',choices=['codex','claude','manual'],default='codex')
    ap.add_argument('--llm-timeout',type=int,default=180)
    ap.add_argument('--max-rounds',type=int,default=1)
    args=ap.parse_args()
    if not 1<=args.max_rounds<=18 or not 30<=args.llm_timeout<=600:
        ap.error('max-rounds1..18, timeout30..600 required')
    p=control.paths(args.campaign)
    p['state'].mkdir(parents=True,exist_ok=True)
    with (p['state']/'stage.lock').open('a') as handle:
        try:
            fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:
            print(json.dumps({'status':'busy','reason':'another stage owns campaign'}));return 0
        try:
            for _ in range(args.max_rounds):
                result=advance(args.campaign,args.proposer_backend,args.reviewer_backend,args.llm_timeout)
                print(json.dumps(result,ensure_ascii=False,allow_nan=False),flush=True)
                if result['status'] in ('stopped','busy','awaiting_manual_input'):
                    break
        except Exception as exc:
            if p['settings'].exists():control.set_stop(args.campaign,True)
            print(json.dumps({'status':'error','type':type(exc).__name__,'error':str(exc)}));return 2
    return 0


if __name__=='__main__':
    raise SystemExit(main())
