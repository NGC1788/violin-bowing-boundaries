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
FIELDS = {'id','seed','prior','policy','rule_bank','max_train_columns','max_eval_columns','mlp_epochs','hidden','learning_rate'}
PRIORS = {'uniform','constant','logreg','hgb','mlp_bce','mlp_boundary'}
POLICIES = {'ascending','random','even_first','uncertainty','entropy'}
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
            'stop':state/'STOP', 'worker_lock':state/'worker.lock', 'manifest':locks/(campaign+'.json'),
            'runs':root/'runs'/campaign, 'reports':root/'reports'/campaign}


@contextlib.contextmanager
def database(p):
    c = sqlite3.connect(p['db'], timeout=30, isolation_level=None)
    c.row_factory = sqlite3.Row
    c.execute('PRAGMA journal_mode=WAL'); c.execute('PRAGMA busy_timeout=30000')
    try: yield c
    finally: c.close()


def source_files(root):
    files = [root/n for n in ('control.py','cli.py','worker.py')]
    files += [f for d in ('harness','tests') for f in (root/d).rglob('*.py') if '__pycache__' not in f.parts]
    if (root/'planner.py').is_file(): files.append(root/'planner.py')
    files += [f for f in (root/'prompts').rglob('*') if f.is_file()]
    if not all(f.is_file() for f in files[:3]): raise ControlError('control/cli/worker sources missing')
    return {str(f.relative_to(root)):file_sha(f) for f in sorted(set(files))}


def read_settings(p):
    if not p['settings'].exists(): raise ControlError('campaign not initialized')
    return json.loads(p['settings'].read_text())


def protocol_validate(protocol):
    b = protocol.get('budget', {})
    for key, default in [('max_trials',36),('max_campaign_seconds',2700),('max_trial_seconds',600)]:
        v=b.get(key,default)
        if isinstance(v,bool) or not isinstance(v,(int,float)) or not math.isfinite(v) or v<=0:
            raise ControlError(f'invalid budget {key}')
    end=b.get('end_date','2026-11-06')
    try: datetime.strptime(end,'%Y-%m-%d')
    except (ValueError,TypeError): raise ControlError('end_date must be YYYY-MM-DD (Asia/Seoul)')
    benchmark=protocol.get('benchmark',{})
    if benchmark.get('rule_bank','nine') not in ('nine','baseline'): raise ControlError('invalid rule bank')
    for key in ['max_eval_columns','max_train_columns']:
        v=benchmark.get(key,16 if key=='max_eval_columns' else None)
        if v is not None and (isinstance(v,bool) or not isinstance(v,int) or v<=0):
            raise ControlError(f'invalid benchmark {key}')
    if benchmark.get('eval_selection_seed',20261030)!=20261030:
        raise ControlError('current harness fixes evaluation selection seed at 20261030')
    return protocol


def init(campaign, data, protocol, root=ROOT, lock_root=None):
    p=paths(campaign,root,lock_root)
    if p['settings'].exists(): raise ControlError('campaign exists; choose a new name')
    data=Path(data).resolve(); protocol=Path(protocol).resolve()
    if not data.is_file() or not protocol.is_file(): raise ControlError('data/protocol file missing')
    protocol_validate(json.loads(protocol.read_text()))
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
    protocol=protocol_validate(json.loads(Path(settings['protocol']).read_text()))
    manifest={'campaign':campaign,'root':str(p['root']),'frozen_at':time.time(),
              'data_path':settings['data'],'data_sha':file_sha(settings['data']),
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
        if digest(json.loads(Path(m['protocol_path']).read_text()))!=m['protocol_sha']: raise ControlError('protocol integrity mismatch')
    except OSError as e: raise ControlError(f'frozen input missing: {e}')
    return m


def normalized_candidate(config, protocol):
    if not isinstance(config,dict): raise ControlError('candidate must be a JSON object')
    unknown=set(config)-FIELDS
    if unknown: raise ControlError('unknown candidate fields: '+', '.join(sorted(unknown)))
    benchmark=protocol.get('benchmark',{})
    cfg={'seed':0,'prior':'constant','policy':'ascending',
         'rule_bank':benchmark.get('rule_bank','nine'),
         'max_train_columns':benchmark.get('max_train_columns'),
         'max_eval_columns':benchmark.get('max_eval_columns',16),
         'mlp_epochs':8,'hidden':32,'learning_rate':0.003}
    cfg.update({k:v for k,v in config.items() if k!='id'})
    if 'id' in config and (not isinstance(config['id'],str) or not re.fullmatch(r'[a-zA-Z0-9][a-zA-Z0-9_.-]{0,95}',config['id'])):
        raise ControlError('id must be display metadata, not a path')
    allowed=protocol.get('allowed',{})
    if not isinstance(cfg['prior'],str) or cfg['prior'] not in PRIORS.intersection(allowed.get('priors',PRIORS)): raise ControlError('invalid prior')
    if not isinstance(cfg['policy'],str) or cfg['policy'] not in POLICIES.intersection(allowed.get('policies',POLICIES)): raise ControlError('invalid policy')
    if isinstance(cfg['seed'],bool) or not isinstance(cfg['seed'],int) or cfg['seed'] not in allowed.get('seeds',[0,1,2]): raise ControlError('invalid seed')
    for k in ('rule_bank','max_eval_columns','max_train_columns'):
        default={'rule_bank':'nine','max_eval_columns':16,'max_train_columns':None}[k]
        if cfg[k]!=benchmark.get(k,default): raise ControlError(f'{k} is frozen by the benchmark')
    for k,lo,hi in [('mlp_epochs',1,100),('hidden',4,128)]:
        v=cfg[k]
        if isinstance(v,bool) or not isinstance(v,int) or not lo<=v<=hi: raise ControlError(f'invalid {k}')
    v=cfg['learning_rate']
    if isinstance(v,bool) or not isinstance(v,(int,float)) or not math.isfinite(v) or not 1e-6<=v<=0.1:
        raise ControlError('invalid learning_rate')
    return cfg


def identity(config,m):
    return digest({'config':config,'data_sha':m['data_sha'],'source_sha':m['source_sha'],'protocol_sha':m['protocol_sha']})


def budget(m):
    return {'max_trials':36,'max_campaign_seconds':2700,'max_trial_seconds':600,'max_consecutive_failures':3,'end_date':'2026-11-06',**m['protocol'].get('budget',{})}


def expired(m):
    return datetime.now(ZoneInfo('Asia/Seoul')).date().isoformat()>=budget(m)['end_date']


def consumed(c):
    return float(c.execute('SELECT COALESCE(SUM(elapsed),0) FROM jobs').fetchone()[0])


def enqueue(campaign,config,root=ROOT,lock_root=None):
    p=paths(campaign,root,lock_root); m=integrity(p)
    if p['stop'].exists(): raise ControlError('campaign stopped; resume explicitly')
    if expired(m): raise ControlError('campaign end date reached')
    cfg=normalized_candidate(config,m['protocol']); rid=identity(cfg,m)
    with database(p) as c:
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
        with database(p) as c:
            recover(c,p,m)
            for _ in range(max_jobs):
                m=integrity(p)
                remaining=min(max_seconds-(time.monotonic()-started),budget(m)['max_campaign_seconds']-consumed(c))
                if p['stop'].exists() or expired(m) or remaining<=0: break
                recent=c.execute("SELECT status FROM jobs WHERE status IN ('completed','failed','interrupted','invalid') ORDER BY ended DESC LIMIT ?",
                                 (budget(m)['max_consecutive_failures'],)).fetchall()
                if len(recent)>=budget(m)['max_consecutive_failures'] and all(r['status']=='failed' for r in recent):
                    event(c,'failure_limit',None,{'count':len(recent)});break
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
                    event(c,'invalid',rid,{'reason':str(e)});c.execute('COMMIT');continue
                if out.exists():
                    c.execute("UPDATE jobs SET status='invalid',error='output directory already exists' WHERE run_id=?",(rid,))
                    event(c,'invalid',rid,{'reason':'output exists; never overwrite'});c.execute('COMMIT');continue
                out.mkdir(parents=True)
                cfg=json.loads(row['config_json'])
                atomic_json(out/'request.json',{'run_id':rid,'config':cfg,'data_path':m['data_path'],'output_dir':str(out),
                             'data_sha':m['data_sha'],'protocol_sha':m['protocol_sha'],'source_sha':m['source_sha']})
                t0=time.monotonic(); wall=time.time()
                c.execute("UPDATE jobs SET status='running',started=? WHERE run_id=?",(wall,rid))
                event(c,'running',rid,{});c.execute('COMMIT')
                limit=min(budget(m)['max_trial_seconds'],remaining)
                state='failed';error=None;result_sha=None;proc=None
                try:
                    with (out/'stdout.log').open('x') as log:
                        env={**os.environ,'OMP_NUM_THREADS':'2','OPENBLAS_NUM_THREADS':'2','MKL_NUM_THREADS':'2',
                             'VECLIB_MAXIMUM_THREADS':'2','NUMEXPR_NUM_THREADS':'2','PYTHONPATH':str(p['root'])}
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
                        if proc.returncode!=0: error=f'worker exited {proc.returncode}'
                        else:
                            result=out/'worker_result.json'
                            metrics=json.loads(result.read_text())
                            if not isinstance(metrics,dict): raise ControlError('harness result must be an object')
                            canonical(metrics)  # rejects NaN/Infinity
                            result_sha=file_sha(result)
                            atomic_json(out/'seal.json',{'run_id':rid,'metrics_sha':result_sha,'candidate_sha':digest(cfg),
                                        'source_sha':m['source_sha'],'protocol_sha':m['protocol_sha'],'data_sha':m['data_sha']})
                            state='completed'
                except Exception as e:
                    if proc is not None and proc.poll() is None: terminate_child(proc.pid)
                    error=f'{type(e).__name__}: {e}';state='failed'
                try: integrity(p)
                except Exception as e: state='invalid';error=f'post-run integrity failed: {e}'
                elapsed=time.monotonic()-t0
                c.execute('BEGIN IMMEDIATE')
                c.execute('UPDATE jobs SET status=?,ended=?,elapsed=?,result_sha=?,error=? WHERE run_id=?',
                          (state,time.time(),elapsed,result_sha,error,rid))
                event(c,state,rid,{'elapsed':elapsed,'error':error});c.execute('COMMIT')
                done.append({'run_id':rid,'status':state,'elapsed':elapsed,'error':error})
                if state=='invalid': break
    return {'campaign':campaign,'processed':done}


def set_stop(campaign,stopped=True,root=ROOT,lock_root=None):
    p=paths(campaign,root,lock_root);read_settings(p)
    if stopped: p['stop'].write_text('user requested stop\n')
    else: integrity(p);p['stop'].unlink(missing_ok=True)
    with database(p) as c:event(c,'stopped' if stopped else 'resumed',None,{})
    return {'campaign':campaign,'stopped':stopped}


def status(campaign,root=ROOT,lock_root=None):
    p=paths(campaign,root,lock_root);read_settings(p)
    try: m=integrity(p);valid=True;issue=None
    except Exception as e: m=None;valid=False;issue=str(e)
    with database(p) as c:
        counts={r['status']:r['n'] for r in c.execute('SELECT status,COUNT(*) n FROM jobs GROUP BY status')}
        used=consumed(c); jobs=[dict(r) for r in c.execute('SELECT run_id,alias,status,elapsed,error FROM jobs ORDER BY submitted')]
    return {'campaign':campaign,'integrity_ok':valid,'integrity_error':issue,'stopped':p['stop'].exists(),
            'counts':counts,'consumed_seconds':used,'budget':budget(m) if m else None,'jobs':jobs}


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
            records.append(d)
    payload={'campaign':campaign,'protocol_sha':m['protocol_sha'],'data_sha':m['data_sha'],
             'source_sha':m['source_sha'],'scope':'Exploratory campaign on previously exposed data; not untouched confirmation.',
             'runs':records}
    atomic_json(p['reports']/'results.json',payload)
    return payload
