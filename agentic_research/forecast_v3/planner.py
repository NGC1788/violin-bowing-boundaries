"""Finite JSON-only proposer/reviewer with resumable, request-bound manual turns.

Integrity hashes are mistake detection, not an OS/multiparty security boundary.
LLM-generated code never executes; only the frozen candidate catalog can run.
"""
from __future__ import annotations
import argparse
import fcntl
import json
from pathlib import Path
import re
import subprocess
import sys
import time
import uuid
try:
    from . import control
except ImportError:
    import control

ROOT=Path(__file__).resolve().parent
PROPOSER_BACKEND='codex'
REVIEWER_BACKEND='codex'
MANUAL_DIR=None


def llm(prompt,schema,directory,name,timeout):
    try:
        from .llm_backend import run_structured
    except ImportError:
        from llm_backend import run_structured
    backend=PROPOSER_BACKEND if name=='proposer' else REVIEWER_BACKEND
    return run_structured(prompt,schema,directory,name,backend,timeout,MANUAL_DIR)


def schema(properties):
    return {'type':'object','properties':properties,'required':list(properties),'additionalProperties':False}


def available(p,manifest):
    with control.database(p) as db:
        submitted={row[0] for row in db.execute('SELECT run_id FROM jobs')}
    pool={}
    directory=control.resolve_candidate_dir(p.get('root',ROOT),manifest['protocol'])
    for path in sorted(directory.glob('*.json')):
        raw=json.loads(path.read_text());normalized=control.normalized_candidate(raw,manifest['protocol'])
        key=control.identity(normalized,manifest)
        if not isinstance(raw.get('id'),str):raise control.ControlError('catalog candidate needs an id')
        if raw['id'] in pool:raise control.ControlError('duplicate candidate ID in frozen catalog')
        if key not in submitted:pool[raw['id']]={'config':raw,'semantic_sha':key}
    return pool


def _bound_llm(prompt,base_schema,directory,role,timeout):
    """A response must repeat the exact request hash, including manual resumes."""
    backend=PROPOSER_BACKEND if role=='proposer' else REVIEWER_BACKEND
    request_sha=control.digest({'prompt':prompt,'schema':base_schema,'backend':backend,'role':role})
    bound_schema=json.loads(json.dumps(base_schema))
    bound_schema['properties']['request_sha']={'type':'string','enum':[request_sha]}
    bound_schema['required'].append('request_sha')
    bound_prompt=prompt+'\n\nReturn JSON only under the supplied schema. Include request_sha exactly: '+request_sha
    binding={'request_sha':request_sha,'backend':backend,'role':role,
             'prompt_sha':control.digest(bound_prompt),'schema_sha':control.digest(bound_schema)}
    path=directory/f'{role}_binding.json'
    if path.exists():
        if json.loads(path.read_text())!=binding:raise control.ControlError('resumed request binding mismatch')
        saved_prompt=directory/f'{role}_prompt.txt';saved_schema=directory/f'{role}_schema.json'
        if saved_prompt.exists() and saved_prompt.read_text()!=bound_prompt:raise control.ControlError('saved manual prompt changed')
        if saved_schema.exists() and json.loads(saved_schema.read_text())!=bound_schema:raise control.ControlError('saved manual schema changed')
    else:control.atomic_json(path,binding)
    answer=llm(bound_prompt,bound_schema,directory,role,timeout)
    if not isinstance(answer,dict) or answer.get('request_sha')!=request_sha:
        raise control.ControlError('response does not bind to the exact request')
    try:
        from .llm_backend import validate
    except ImportError:
        from llm_backend import validate
    validate(answer,bound_schema)
    return answer


def _stop(campaign,reason):
    control.set_stop(campaign,True)
    return {'status':'stopped','reason':reason}


def one_round(campaign,execute=False,llm_timeout=180,resume_round=None):
    p=control.paths(campaign);manifest=control.integrity(p);status=control.status(campaign)
    if status.get('safety_halt'):return {'status':'stopped','reason':'verifier/integrity safety halt'}
    if status['stopped']:return {'status':'stopped','reason':'STOP requested'}
    if control.expired(manifest):return _stop(campaign,'campaign expiry')
    if status.get('exhaustion_reason'):return _stop(campaign,status['exhaustion_reason'])
    if status['consumed_seconds']>=status['budget']['max_campaign_seconds']:
        return _stop(campaign,'compute budget exhausted')
    rounds_dir=p['state']/'agent_rounds'
    prior_rounds=sorted(rounds_dir.glob('*/round.json'))
    pending=[path for path in prior_rounds if json.loads(path.read_text()).get('status')=='awaiting_manual_input']
    if resume_round is None and pending:
        if len(pending)>1:raise control.ControlError('multiple pending manual rounds need explicit review')
        old=json.loads(pending[0].read_text())
        return {'status':'awaiting_manual_input','round_id':old['round_id'],
                'pending_role':old.get('pending_role'),'reason':'resume this same round with --resume-round'}
    if status['counts'].get('running',0) or status['counts'].get('queued',0):
        if resume_round:raise control.ControlError('cannot resume a manual round while queue state changed')
        if execute:
            worker=control.run(campaign,1,600)
            if worker.get('stopped'):return {'status':'stopped','reason':worker.get('reason'),'worker':worker}
            if not worker.get('processed'):return _stop(campaign,'queued worker made no progress; review before resuming')
            return {'status':'queue_processed','worker':worker}
        return {'status':'waiting_for_queue'}
    if resume_round is not None:
        if not re.fullmatch(r'[A-Za-z0-9_-]{1,96}',resume_round):raise control.ControlError('invalid resume round ID')
        directory=rounds_dir/resume_round
        record=json.loads((directory/'round.json').read_text())
        if record.get('campaign')!=campaign or record.get('manifest_sha')!=manifest['manifest_sha']:
            raise control.ControlError('manual round campaign/manifest mismatch')
        if record.get('status')!='awaiting_manual_input':raise control.ControlError('only a pending manual round can resume')
        if record['backends']!={'proposer':PROPOSER_BACKEND,'reviewer':REVIEWER_BACKEND}:
            raise control.ControlError('resume requires the original proposer/reviewer backends')
        context=json.loads((directory/'context.json').read_text())
        if control.digest(context)!=record['context_sha']:raise control.ControlError('manual context changed')
        if record.get('proposal') and control.digest(record['proposal'])!=record.get('proposal_sha'):
            raise control.ControlError('saved proposal changed')
        pool=context['available_candidates']
        current=available(p,manifest)
        if any(k not in current or current[k]!=v for k,v in pool.items()):
            raise control.ControlError('candidate availability changed during manual wait')
    else:
        limit=manifest['protocol'].get('planner',{}).get('max_rounds',12)
        if len(prior_rounds)>=limit:return _stop(campaign,f'{limit}-round LLM budget exhausted')
        if sum(status['counts'].values())>=status['budget']['max_trials']:
            return _stop(campaign,'trial budget exhausted')
        pool=available(p,manifest)
        if not pool:return _stop(campaign,'registered hypothesis pool exhausted; review before expanding')
        report=control.report(campaign);control.assert_safe(p)
        compact=[]
        for row in report.get('runs',[]):
            item={k:row.get(k) for k in ('run_id','alias','status','config','elapsed','error')}
            if 'metrics' in row:
                item['metrics']={k:row['metrics'].get(k) for k in
                    ('mean_log_loss','mean_brier','verification_error_count','verification_ok',
                     'development_rows','rules','groups','sessions','timing','auxiliary')}
            compact.append(item)
        report['runs']=compact
        context={'protocol':manifest['protocol'],'available_candidates':pool,'status':status,'report':report}
        name=time.strftime('%Y%m%dT%H%M%S')+'_'+uuid.uuid4().hex[:8]
        directory=rounds_dir/name;directory.mkdir(parents=True)
        record={'campaign':campaign,'round_id':name,'manifest_sha':manifest['manifest_sha'],
                'status':'started','created_at':time.time(),'context_sha':control.digest(context),
                'backends':{'proposer':PROPOSER_BACKEND,'reviewer':REVIEWER_BACKEND}}
        control.atomic_json(directory/'context.json',context)
        control.atomic_json(directory/'round.json',record)  # failed calls spend one round
    try:
        if 'proposal' not in record:
            record['pending_role']='proposer';control.atomic_json(directory/'round.json',record)
            fields={'candidate_id':{'type':'string','enum':['STOP',*pool]},
                    'hypothesis':{'type':'string'},'falsifier':{'type':'string'},'reason':{'type':'string'}}
            proposal=_bound_llm((ROOT/'prompts/proposer.md').read_text()+'\n\nCONTEXT JSON:\n'+control.canonical(context),
                                schema(fields),directory,'proposer',llm_timeout)
            record.update(proposal=proposal,proposal_sha=control.digest(proposal))
            control.atomic_json(directory/'round.json',record)
        proposal=record['proposal']
        if proposal['candidate_id']=='STOP':
            record['status']='stopped';control.atomic_json(directory/'round.json',record)
            control.set_stop(campaign,True);return record
        if any(len(proposal[k].strip())<10 for k in ('hypothesis','falsifier','reason')):
            raise control.ControlError('proposal must state a concrete hypothesis and falsifier')
        selected=pool[proposal['candidate_id']]
        fields={'verdict':{'type':'string','enum':['APPROVE','REJECT']},
                'candidate_sha':{'type':'string','enum':[selected['semantic_sha']]},
                'reason':{'type':'string'},'limitations':{'type':'string'}}
        review_context={'selected':selected,'proposal':proposal,'context':context}
        record['pending_role']='reviewer';control.atomic_json(directory/'round.json',record)
        review=_bound_llm((ROOT/'prompts/reviewer.md').read_text()+'\n\nCONTEXT JSON:\n'+control.canonical(review_context),
                          schema(fields),directory,'reviewer',llm_timeout)
        if review['candidate_sha']!=selected['semantic_sha']:raise control.ControlError('review candidate hash mismatch')
        record.update(review=review,candidate_sha=selected['semantic_sha'])
        if review['verdict']!='APPROVE':
            record['status']='rejected';control.atomic_json(directory/'round.json',record);return record
        control.integrity(p)
        submitted=control.enqueue(campaign,selected['config'])
        record.update(status='submitted',submission=submitted)
        control.atomic_json(directory/'round.json',record)
        if execute:
            record['worker']=control.run(campaign,1,600)
            record['status']='stopped' if record['worker'].get('stopped') else 'evaluated'
            control.atomic_json(directory/'round.json',record)
        return record
    except Exception as exc:
        try:
            from .llm_backend import ManualInputRequired
        except ImportError:
            from llm_backend import ManualInputRequired
        if isinstance(exc,ManualInputRequired):
            record.update(status='awaiting_manual_input',error=str(exc))
            control.atomic_json(directory/'round.json',record);return record
        record.update(status='failed',error=f'{type(exc).__name__}: {exc}')
        control.atomic_json(directory/'round.json',record);raise


def main():
    global PROPOSER_BACKEND,REVIEWER_BACKEND,MANUAL_DIR
    ap=argparse.ArgumentParser()
    ap.add_argument('--campaign',required=True);ap.add_argument('--execute',action='store_true')
    ap.add_argument('--llm-timeout',type=int,default=180)
    ap.add_argument('--proposer-backend',choices=['codex','claude','manual'],default='codex')
    ap.add_argument('--reviewer-backend',choices=['codex','claude','manual'],default='codex')
    ap.add_argument('--manual-dir',type=Path);ap.add_argument('--resume-round')
    args=ap.parse_args()
    PROPOSER_BACKEND=args.proposer_backend;REVIEWER_BACKEND=args.reviewer_backend;MANUAL_DIR=args.manual_dir
    if not 30<=args.llm_timeout<=600:ap.error('llm timeout must be between 30 and 600 seconds')
    try:
        p=control.paths(args.campaign);p['state'].mkdir(parents=True,exist_ok=True)
        with (p['state']/'planner.lock').open('a') as handle:
            try:fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
            except BlockingIOError:raise control.ControlError('another planner owns this campaign')
            result=one_round(args.campaign,args.execute,args.llm_timeout,args.resume_round)
        print(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False))
    except (control.ControlError,subprocess.TimeoutExpired,ValueError,OSError,KeyError) as exc:
        print(f'ERROR: {exc}',file=sys.stderr);return 2
    return 0


if __name__=='__main__':raise SystemExit(main())
