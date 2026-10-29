"""Finite research campaign control. Integrity checks are not an OS security boundary."""
from __future__ import annotations
import contextlib
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import signal
import sqlite3
import subprocess
import sys
import time
from datetime import datetime
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parent
FIELDS = {'id','seed','method','epochs','hidden','batch_size','learning_rate','hgb_max_iter'}
METHODS = {'source_frequency','persistence','source_markov','hgb_history','mlp_ind','mlp_joint'}
TERMINAL = ('completed','failed','interrupted','invalid')

class ControlError(RuntimeError): pass


def canonical(x):
    return json.dumps(x, sort_keys=True, ensure_ascii=False, separators=(',', ':'), allow_nan=False)


def digest(x): return hashlib.sha256(canonical(x).encode()).hexdigest()


def file_sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024*1024), b''): h.update(chunk)
    return h.hexdigest()


def atomic_json(path, obj):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f'.tmp.{os.getpid()}')
    with tmp.open('x', encoding='utf-8') as f:
        f.write(json.dumps(obj, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)+'\n')
        f.flush(); os.fsync(f.fileno())
    os.replace(tmp, path)


def valid_campaign(name):
    if not re.fullmatch(r'[a-zA-Z0-9][a-zA-Z0-9_-]{0,63}', name):
        raise ControlError('campaign must be a short alphanumeric name, underscore or hyphen')
    return name


def paths(campaign, root=ROOT, lock_root=None):
    campaign = valid_campaign(campaign); root = Path(root).resolve()
    state = root/'state'/campaign
    locks = Path(lock_root) if lock_root is not None else Path.home()/'.violin_research_loop'/'locks'
    return {'root':root, 'state':state, 'db':state/'ledger.sqlite', 'settings':state/'campaign.json',
            'stop':state/'STOP', 'safety_halt':state/'SAFETY_HALT.json', 'headroom':state/'headroom.json', 'worker_lock':state/'worker.lock', 'manifest':locks/(campaign+'.json'),
            'runs':root/'runs'/campaign, 'reports':root/'reports'/campaign}


@contextlib.contextmanager
def database(p):
    c = sqlite3.connect(p['db'], timeout=30, isolation_level=None)
    c.row_factory = sqlite3.Row
    c.execute('PRAGMA journal_mode=WAL'); c.execute('PRAGMA busy_timeout=30000')
    try: yield c
    finally: c.close()


def source_files(root):
    files = list(root.glob('*.py'))
    files += [f for f in (root/'tests').rglob('*.py') if '__pycache__' not in f.parts]
    files += [f for folder in ('prompts','candidates') for f in (root/folder).rglob('*') if f.is_file()]
    for name in ('control.py','cli.py','worker.py','models.py','experiment.py','data.py','rule_patterns.py','llm_backend.py'):
        if not (root/name).is_file():raise ControlError('Missing required trusted source: '+name)
    return {str(f.relative_to(root)):file_sha(f) for f in sorted(set(files))}


def read_settings(p):
    if not p['settings'].exists(): raise ControlError('campaign not initialized')
    return json.loads(p['settings'].read_text())


def protocol_validate(protocol,root=None):
    b=protocol.get('budget',{})
    for key in ('max_trials','max_campaign_seconds','max_trial_seconds'):
        value=b.get(key)
        if isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value) or value<=0:
            raise ControlError('Invalid budget: '+key)
    datetime.strptime(b['end_date'],'%Y-%m-%d')
    if protocol.get('primary_metric') not in ('mean_log_loss','mean_future_rule_log_loss'):raise ControlError('Unexpected forecast endpoint')
    if protocol.get('data_scope')!='retrospective_landmark_forecast':raise ControlError('Unexpected causal scope')
    if protocol.get('headroom',{}).get('required',False):raise ControlError('Exact-query headroom gate is inapplicable to forecasting')
    rounds=protocol.get('planner',{}).get('max_rounds')
    if isinstance(rounds,bool) or not isinstance(rounds,int) or rounds<1:raise ControlError('Positive LLM budget required')
    for key in ('max_trials','max_consecutive_failures'):
        value=b.get(key,3)
        if isinstance(value,bool) or not isinstance(value,int) or value<1:raise ControlError('Positive integer budget required: '+key)
    if root is not None:resolve_candidate_dir(root,protocol)
    return protocol


def verify_dataset_receipt(path):
    """Bind the adjacent receipt and embedded metadata, without fitting a model."""
    import numpy as np
    path=Path(path);sidecar=path.with_suffix('.json')
    try:
        receipt=json.loads(sidecar.read_text())
        if receipt['npz_sha256']!=file_sha(path):raise ControlError('dataset receipt hash mismatch')
        with np.load(path,allow_pickle=False) as z: metadata=json.loads(str(z['metadata_json']))
        if metadata!=receipt['metadata']:raise ControlError('dataset embedded metadata/receipt mismatch')
    except (OSError,ValueError,KeyError) as exc:raise ControlError('invalid prepared dataset receipt: '+str(exc)) from exc
    return sidecar


def resolve_candidate_dir(root,protocol,require_exists=True):
    raw=protocol.get('planner',{}).get('candidate_dir','candidates')
    if not isinstance(raw,str) or Path(raw).is_absolute() or '..' in Path(raw).parts:
        raise ControlError('candidate_dir must be a safe relative path')
    root=Path(root).resolve(); target=(root/raw).resolve()
    if not target.is_relative_to(root/'candidates'):
        raise ControlError('candidate_dir must resolve inside this engine candidates directory')
    if require_exists and not target.is_dir(): raise ControlError('candidate directory missing')
    return target


def init(campaign, data, protocol, root=ROOT, lock_root=None):
    p=paths(campaign,root,lock_root)
    if p['settings'].exists(): raise ControlError('campaign exists; choose a new name')
    data=Path(data).resolve(); protocol=Path(protocol).resolve()
    if not data.is_file() or not protocol.is_file(): raise ControlError('data/protocol file missing')
    protocol_validate(json.loads(protocol.read_text()),p['root'])
    verify_dataset_receipt(data)
    p['state'].mkdir(parents=True,exist_ok=True)
    atomic_json(p['settings'],{'campaign':campaign,'root':str(p['root']),'data':str(data),'protocol':str(protocol),'created_at':time.time()})
    with database(p) as c:
        c.executescript('''CREATE TABLE IF NOT EXISTS jobs (
          run_id TEXT PRIMARY KEY, alias TEXT, config_json TEXT NOT NULL,
          status TEXT NOT NULL CHECK(status IN ('queued','running','completed','failed','interrupted','invalid')),
          submitted REAL NOT NULL, started REAL, ended REAL, elapsed REAL DEFAULT 0,
          pid INTEGER, result_sha TEXT, error TEXT);
          CREATE TABLE IF NOT EXISTS events (seq INTEGER PRIMARY KEY AUTOINCREMENT,
          at REAL NOT NULL, kind TEXT NOT NULL, run_id TEXT, detail TEXT NOT NULL);''')
        event(c,'initialized',None,{'data':str(data),'protocol':str(protocol)})
    return {'campaign':campaign,'initialized':True,'next':'freeze after the complete harness passes its tests'}


def event(c, kind, run_id, detail):
    c.execute('INSERT INTO events(at,kind,run_id,detail) VALUES(?,?,?,?)',(time.time(),kind,run_id,canonical(detail)))


def freeze(campaign,root=ROOT,lock_root=None):
    p=paths(campaign,root,lock_root); settings=read_settings(p)
    if p['manifest'].exists(): raise ControlError('already frozen; automatic relocking is forbidden; choose a new campaign')
    with database(p) as c:
        if c.execute('SELECT COUNT(*) FROM jobs').fetchone()[0]: raise ControlError('cannot freeze after submitting jobs')
    protocol=protocol_validate(json.loads(Path(settings['protocol']).read_text()),p['root'])
    sidecar=verify_dataset_receipt(settings['data'])
    manifest={'campaign':campaign,'root':str(p['root']),'frozen_at':time.time(),
              'data_path':settings['data'],'data_sha':file_sha(settings['data']),
              'data_receipt_path':str(sidecar),'data_receipt_sha':file_sha(sidecar),
              'protocol_path':settings['protocol'],'protocol_sha':digest(protocol),'protocol':protocol,
              'sources':source_files(p['root'])}
    manifest['source_sha']=digest(manifest['sources'])
    manifest['manifest_sha']=digest(manifest)
    p['manifest'].parent.mkdir(parents=True,exist_ok=True)
    # O_EXCL prevents a second initializer from replacing the trust baseline.
    with p['manifest'].open('x') as f:
        f.write(json.dumps(manifest,sort_keys=True,indent=2)+'\n');f.flush();os.fsync(f.fileno())
    with database(p) as c: event(c,'frozen',None,{'manifest_sha':manifest['manifest_sha']})
    return {'manifest':str(p['manifest']),'manifest_sha':manifest['manifest_sha']}


def integrity(p):
    read_settings(p)
    if not p['manifest'].exists(): raise ControlError('campaign is not frozen')
    m=json.loads(p['manifest'].read_text()); unsigned={k:v for k,v in m.items() if k!='manifest_sha'}
    if digest(unsigned)!=m['manifest_sha']: raise ControlError('manifest self-check failed')
    if m['root']!=str(p['root']): raise ControlError('manifest root mismatch')
    if source_files(p['root'])!=m['sources']: raise ControlError('frozen source integrity mismatch')
    try:
        if file_sha(m['data_path'])!=m['data_sha']: raise ControlError('data integrity mismatch')
        if file_sha(m['data_receipt_path'])!=m['data_receipt_sha']:raise ControlError('data receipt integrity mismatch')
        if digest(json.loads(Path(m['protocol_path']).read_text()))!=m['protocol_sha']: raise ControlError('protocol integrity mismatch')
    except OSError as e: raise ControlError(f'frozen input missing: {e}')
    return m


def normalized_candidate(config,protocol):
    if not isinstance(config,dict) or set(config)-FIELDS:raise ControlError('Unknown candidate fields')
    if 'id' in config and (not isinstance(config['id'],str) or not re.fullmatch(r'[a-zA-Z0-9][a-zA-Z0-9_.-]{0,95}',config['id'])):
        raise ControlError('Candidate id is metadata, not a path')
    cfg={**protocol['model_parameters'],'seed':0,'method':'source_frequency'}
    cfg.update({k:v for k,v in config.items() if k!='id'})
    if not isinstance(cfg['method'],str) or cfg['method'] not in METHODS.intersection(protocol['allowed']['methods']):raise ControlError('Unknown model method')
    if isinstance(cfg['seed'],bool) or not isinstance(cfg['seed'],int) or cfg['seed'] not in protocol['allowed']['seeds']:raise ControlError('Invalid seed')
    if cfg['method'] in {'source_frequency','persistence','source_markov'} and cfg['seed']!=0:raise ControlError('Deterministic baseline seed must be zero')
    for key,expected in protocol['model_parameters'].items():
        if isinstance(cfg[key],bool) or cfg[key]!=expected:raise ControlError('Hyperparameter is frozen: '+key)
    return cfg


def identity(config,m):
    return digest({'config':config,'data_sha':m['data_sha'],'source_sha':m['source_sha'],'protocol_sha':m['protocol_sha']})


def budget(m):
    return {'max_trials':36,'max_campaign_seconds':2700,'max_trial_seconds':600,'max_consecutive_failures':3,'end_date':'2026-11-06',**m['protocol'].get('budget',{})}


def expired(m):
    return datetime.now(ZoneInfo('Asia/Seoul')).date().isoformat()>=budget(m)['end_date']


def consumed(c):
    return float(c.execute('SELECT COALESCE(SUM(elapsed),0) FROM jobs').fetchone()[0])


def exhaustion_reason(c,m):
    if expired(m):return 'campaign end date reached'
    if consumed(c)>=budget(m)['max_campaign_seconds']:return 'campaign runtime budget exhausted'
    count=budget(m)['max_consecutive_failures']
    recent=c.execute("SELECT status FROM jobs WHERE status IN ('completed','failed','interrupted','invalid') ORDER BY ended DESC LIMIT ?",(count,)).fetchall()
    if len(recent)>=count and all(r['status']!='completed' for r in recent):return 'consecutive failure budget exhausted'
    pending=c.execute("SELECT COUNT(*) FROM jobs WHERE status IN ('queued','running')").fetchone()[0]
    total=c.execute('SELECT COUNT(*) FROM jobs').fetchone()[0]
    if not pending and total>=budget(m)['max_trials']:return 'campaign trial budget exhausted'
    return None


def budget_stop(p,c,reason):
    p['stop'].write_text(reason+'\n')
    event(c,'budget_stop',None,{'reason':reason})


def enqueue(campaign,config,root=ROOT,lock_root=None):
    p=paths(campaign,root,lock_root); m=integrity(p)
    assert_safe(p)
    if p['stop'].exists(): raise ControlError('campaign stopped; resume explicitly')
    if expired(m): raise ControlError('campaign end date reached')
    cfg=normalized_candidate(config,m['protocol']); rid=identity(cfg,m)
    with database(p) as c:
        reason=exhaustion_reason(c,m)
        if reason:budget_stop(p,c,reason);raise ControlError(reason)
        c.execute('BEGIN IMMEDIATE')
        old=c.execute('SELECT status FROM jobs WHERE run_id=?',(rid,)).fetchone()
        if old: c.execute('ROLLBACK'); return {'run_id':rid,'duplicate':True,'status':old['status']}
        if c.execute('SELECT COUNT(*) FROM jobs').fetchone()[0]>=budget(m)['max_trials']:
            c.execute('ROLLBACK'); raise ControlError('campaign trial budget exhausted')
        if consumed(c)>=budget(m)['max_campaign_seconds']:
            c.execute('ROLLBACK'); raise ControlError('campaign runtime budget exhausted')
        c.execute('INSERT INTO jobs(run_id,alias,config_json,status,submitted) VALUES(?,?,?,?,?)',
                  (rid,config.get('id',rid[:12]),canonical(cfg),'queued',time.time()))
        event(c,'queued',rid,{'candidate':cfg});c.execute('COMMIT')
    integrity(p)
    return {'run_id':rid,'duplicate':False,'status':'queued'}


@contextlib.contextmanager
def exclusive_worker(p):
    p['state'].mkdir(parents=True,exist_ok=True)
    with p['worker_lock'].open('a+') as f:
        try: fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError: raise ControlError('another worker already owns this campaign')
        try: yield
        finally: fcntl.flock(f,fcntl.LOCK_UN)


def terminate_child(pid):
    if not pid: return
    try: os.killpg(pid,signal.SIGTERM)
    except PermissionError:
        try: os.kill(pid,signal.SIGTERM)
        except ProcessLookupError: return
    except ProcessLookupError: return
    time.sleep(.1)
    try: os.killpg(pid,signal.SIGKILL)
    except PermissionError:
        try: os.kill(pid,signal.SIGKILL)
        except ProcessLookupError: pass
    except ProcessLookupError: pass


def recover(c,p,m):
    for row in c.execute("SELECT * FROM jobs WHERE status='running'").fetchall():
        pid=row['pid']
        if pid:
            try:
                command=subprocess.check_output(['ps','-p',str(pid),'-o','command='],text=True,stderr=subprocess.DEVNULL).strip()
            except subprocess.CalledProcessError: command=''
            if command:
                # Only terminate a process that contains this exact worker and run ID.
                if str(p['root']/'worker.py') not in command or row['run_id'] not in command:
                    raise ControlError('stale PID belongs to an unverified process; recovery requires review')
                terminate_child(pid)
        elapsed=min(max(0,time.time()-(row['started'] or time.time())),budget(m)['max_trial_seconds'])
        c.execute("UPDATE jobs SET status='interrupted',ended=?,elapsed=?,error=? WHERE run_id=?",
                  (time.time(),elapsed,'coordinator restarted; stale attempt preserved without automatic retry',row['run_id']))
        event(c,'interrupted',row['run_id'],{'recovered_after_exclusive_lock':True})


def run(campaign,max_jobs=1,max_seconds=2700,root=ROOT,lock_root=None):
    p=paths(campaign,root,lock_root); started=time.monotonic(); done=[]
    if isinstance(max_jobs,bool) or not isinstance(max_jobs,int) or max_jobs<1 or not math.isfinite(max_seconds) or max_seconds<=0: raise ControlError('finite positive run limits required')
    with exclusive_worker(p):
        m=integrity(p)
        assert_safe(p)
        with database(p) as c:
            recover(c,p,m)
            for _ in range(max_jobs):
                m=integrity(p)
                remaining=min(max_seconds-(time.monotonic()-started),budget(m)['max_campaign_seconds']-consumed(c))
                reason=exhaustion_reason(c,m)
                if reason:budget_stop(p,c,reason);break
                if p['stop'].exists() or remaining<=0: break
                c.execute('BEGIN IMMEDIATE')
                row=c.execute("SELECT * FROM jobs WHERE status='queued' ORDER BY submitted,run_id LIMIT 1").fetchone()
                if not row: c.execute('ROLLBACK'); break
                rid=row['run_id']; out=p['runs']/rid
                try:
                    candidate=json.loads(row['config_json'])
                    if normalized_candidate(candidate,m['protocol'])!=candidate or identity(candidate,m)!=rid:
                        raise ControlError('queued config identity mismatch')
                except Exception as e:
                    c.execute("UPDATE jobs SET status='invalid',error=? WHERE run_id=?",(str(e),rid))
                    event(c,'invalid',rid,{'reason':str(e)});c.execute('COMMIT');safety_halt(p,rid,str(e));break
                if out.exists():
                    c.execute("UPDATE jobs SET status='invalid',error='output directory already exists' WHERE run_id=?",(rid,))
                    event(c,'invalid',rid,{'reason':'output exists; never overwrite'});c.execute('COMMIT');safety_halt(p,rid,'output directory already exists');break
                out.mkdir(parents=True)
                cfg=json.loads(row['config_json'])
                atomic_json(out/'request.json',{'run_id':rid,'config':cfg,'data_path':m['data_path'],'output_dir':str(out),
                             'data_sha':m['data_sha'],'data_receipt_sha':m['data_receipt_sha'],
                             'protocol_sha':m['protocol_sha'],'source_sha':m['source_sha']})
                t0=time.monotonic(); wall=time.time()
                c.execute("UPDATE jobs SET status='running',started=? WHERE run_id=?",(wall,rid))
                event(c,'running',rid,{});c.execute('COMMIT')
                limit=min(budget(m)['max_trial_seconds'],remaining)
                state='failed';error=None;result_sha=None;proc=None
                try:
                    with (out/'stdout.log').open('x') as log:
                        env={**os.environ,'OMP_NUM_THREADS':'2','OPENBLAS_NUM_THREADS':'2','MKL_NUM_THREADS':'2',
                             'VECLIB_MAXIMUM_THREADS':'2','NUMEXPR_NUM_THREADS':'2','PYTHONPATH':str(p['root'].parent)}
                        proc=subprocess.Popen([sys.executable,str(p['root']/'worker.py'),'--request',str(out/'request.json'),'--run-id',rid],
                             cwd=p['root'],stdout=log,stderr=subprocess.STDOUT,env=env,start_new_session=True)
                        c.execute('UPDATE jobs SET pid=? WHERE run_id=?',(proc.pid,rid))
                        while proc.poll() is None:
                            if p['stop'].exists():
                                terminate_child(proc.pid);error='STOP requested';state='interrupted';break
                            if time.monotonic()-t0>=limit:
                                terminate_child(proc.pid);error='wall-clock budget exceeded';state='interrupted';break
                            time.sleep(.1)
                        proc.wait(timeout=5)
                    if error is None:
                        if proc.returncode!=0:
                            error=f'worker exited {proc.returncode}'
                            if (out/'worker_failure.json').exists():
                                failure=json.loads((out/'worker_failure.json').read_text())
                                if failure.get('verification_failed'):
                                    state='invalid';error='verification failure: '+failure.get('error','unknown')
                        else:
                            result=out/'worker_result.json'
                            metrics=json.loads(result.read_text())
                            if not isinstance(metrics,dict): raise ControlError('harness result must be an object')
                            canonical(metrics)  # rejects NaN/Infinity
                            if metrics.get('verification_error_count')!=0 or metrics.get('verification_ok') is not True:
                                state='invalid';raise ControlError('verification failure in returned forecast metrics')
                            result_sha=file_sha(result)
                            atomic_json(out/'seal.json',{'run_id':rid,'metrics_sha':result_sha,'candidate_sha':digest(cfg),
                                        'source_sha':m['source_sha'],'protocol_sha':m['protocol_sha'],'data_sha':m['data_sha']})
                            state='completed'
                except Exception as e:
                    if proc is not None and proc.poll() is None: terminate_child(proc.pid)
                    error=f'{type(e).__name__}: {e}';state='invalid' if state=='invalid' else 'failed'
                try: integrity(p)
                except Exception as e: state='invalid';error=f'post-run integrity failed: {e}'
                elapsed=time.monotonic()-t0
                c.execute('BEGIN IMMEDIATE')
                c.execute('UPDATE jobs SET status=?,ended=?,elapsed=?,result_sha=?,error=? WHERE run_id=?',
                          (state,time.time(),elapsed,result_sha,error,rid))
                event(c,state,rid,{'elapsed':elapsed,'error':error});c.execute('COMMIT')
                done.append({'run_id':rid,'status':state,'elapsed':elapsed,'error':error})
                if state=='invalid': safety_halt(p,rid,error);break
                reason=exhaustion_reason(c,m)
                if reason:budget_stop(p,c,reason);break
    return {'campaign':campaign,'processed':done,'stopped':p['stop'].exists(),
            'reason':p['stop'].read_text().strip() if p['stop'].exists() else None}


def set_stop(campaign,stopped=True,root=ROOT,lock_root=None):
    p=paths(campaign,root,lock_root);read_settings(p)
    if stopped: p['stop'].write_text('user requested stop\n')
    else: integrity(p);assert_safe(p);p['stop'].unlink(missing_ok=True)
    with database(p) as c:event(c,'stopped' if stopped else 'resumed',None,{})
    return {'campaign':campaign,'stopped':stopped}


def status(campaign,root=ROOT,lock_root=None):
    p=paths(campaign,root,lock_root);read_settings(p)
    try: m=integrity(p);valid=True;issue=None
    except Exception as e: m=None;valid=False;issue=str(e)
    with database(p) as c:
        counts={r['status']:r['n'] for r in c.execute('SELECT status,COUNT(*) n FROM jobs GROUP BY status')}
        used=consumed(c); jobs=[dict(r) for r in c.execute('SELECT run_id,alias,status,elapsed,error FROM jobs ORDER BY submitted')]
        exhausted=exhaustion_reason(c,m) if m else None
    return {'campaign':campaign,'integrity_ok':valid,'integrity_error':issue,'stopped':p['stop'].exists(),
            'counts':counts,'consumed_seconds':used,'budget':budget(m) if m else None,'jobs':jobs,
            'safety_halt':json.loads(p['safety_halt'].read_text()) if p['safety_halt'].exists() else None,
            'exhaustion_reason':exhausted,
            'data_scope':'retrospective_landmark_forecast'}


def report(campaign,root=ROOT,lock_root=None):
    p=paths(campaign,root,lock_root);m=integrity(p);records=[]
    with database(p) as c:
        for row in c.execute('SELECT * FROM jobs ORDER BY submitted'):
            d=dict(row);cfg=json.loads(d.pop('config_json'));d['config']=cfg
            if d['status']=='completed':
                try:
                    out=p['runs']/d['run_id'];seal=json.loads((out/'seal.json').read_text())
                    expected={'run_id':d['run_id'],'metrics_sha':d['result_sha'],'candidate_sha':digest(cfg),
                              'source_sha':m['source_sha'],'protocol_sha':m['protocol_sha'],'data_sha':m['data_sha']}
                    if identity(cfg,m)!=d['run_id'] or file_sha(out/'worker_result.json')!=d['result_sha'] or seal!=expected:
                        raise ControlError('result/identity seal mismatch')
                    d['metrics']=json.loads((out/'worker_result.json').read_text())
                except (OSError,ValueError,ControlError) as e:
                    d['status']='invalid';d['error']=f'result unavailable or invalid: {e}'
                    c.execute("UPDATE jobs SET status='invalid',error=? WHERE run_id=?",(d['error'],d['run_id']))
                    event(c,'invalid',d['run_id'],{'reason':d['error']})
                    safety_halt(p,d['run_id'],d['error'])
            records.append(d)
    payload={'campaign':campaign,'protocol_sha':m['protocol_sha'],'data_sha':m['data_sha'],
             'source_sha':m['source_sha'],'scope':'Exploratory campaign on previously exposed data; not untouched confirmation.',
             'runs':records}
    atomic_json(p['reports']/'results.json',payload)
    return payload


def safety_halt(p,run_id,reason):
    """Verifier/integrity failures cannot be resumed into the same scientific campaign."""
    if not p['safety_halt'].exists():
        atomic_json(p['safety_halt'],{'run_id':run_id,'reason':str(reason),'at':time.time()})
    p['stop'].write_text('Permanent verifier/integrity halt; preserve this campaign and review a new version.\n')


def assert_safe(p):
    if p['safety_halt'].exists(): raise ControlError('campaign has a verifier/integrity safety halt; new reviewed campaign required')


