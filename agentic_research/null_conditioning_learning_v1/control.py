"""Atomic finite-budget registration and a parent-process wall-clock watchdog."""
from pathlib import Path
import datetime,fcntl,json,os,signal,subprocess,time

CAMPAIGN='null_conditioning_learning_v1'


def dump(path,value):
    Path(path).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def register(ledger,status,*,freeze_sha=None,job=None,seconds=None):
    ledger=Path(ledger);lock=ledger.parent/'local/loop_budget.lock';lock.parent.mkdir(exist_ok=True)
    with lock.open('a') as f:
        fcntl.flock(f,fcntl.LOCK_EX)
        d=json.loads(ledger.read_text());entries=d['additional_batches_started']
        found=[e for e in entries if e['campaign']==CAMPAIGN]
        if status=='START':
            if found or len(entries)>=d['max_additional_training_batches']:raise ValueError('duplicate/exhausted training batch')
            if any(e.get('status')=='RUNNING' for e in entries):raise ValueError('another training batch is active')
            entries.append({'campaign':CAMPAIGN,'status':'RUNNING','freeze_sha256':freeze_sha,
                            'started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'jobs':[]})
        else:
            if len(found)!=1 or found[0]['status']!='RUNNING':raise ValueError('no active registered batch')
            entry=found[0]
            if status=='JOB_START':
                if any(j['status']=='RUNNING' for j in entry['jobs']) or len(entry['jobs'])>=d['max_jobs_per_additional_batch']:
                    raise ValueError('active job or job budget exhausted')
                if job['limit_seconds']<=0 or job['limit_seconds']>d['max_seconds_per_training_job']:raise ValueError('job limit invalid')
                if any(j['id']==job['id'] for j in entry['jobs']):raise ValueError('job already attempted')
                public={k:job[k] for k in ['id','fold','method','limit_seconds']}
                entry['jobs'].append({**public,'status':'RUNNING'})
            elif status=='JOB_END':
                matches=[j for j in entry['jobs'] if j['id']==job['id'] and j['status']=='RUNNING']
                if len(matches)!=1:raise ValueError('missing active job')
                matches[0].update(status=job['status'],seconds=seconds)
            else:entry.update(status=status,seconds=seconds)
        tmp=ledger.with_suffix('.tmp');dump(tmp,d);os.replace(tmp,ledger)


def job_limit(contract,elapsed,now):
    remaining=(datetime.datetime.fromisoformat(contract['deadline_kst'])-now).total_seconds()
    limit=min(contract['max_job_seconds'],contract['max_worker_seconds']-elapsed,remaining)
    if limit<=0:raise TimeoutError('worker budget or deadline reached')
    return limit


def stop_child(process,grace=2.):
    if process is None:return
    previous=signal.pthread_sigmask(signal.SIG_BLOCK,{signal.SIGALRM})
    try:
        if process.poll() is None:
            try:process.terminate()
            except ProcessLookupError:pass
            try:process.wait(timeout=grace)
            except subprocess.TimeoutExpired:
                try:process.kill()
                except ProcessLookupError:pass
                process.wait()
            except BaseException:
                try:process.kill()
                except ProcessLookupError:pass
                process.wait()
                raise
    finally:signal.pthread_sigmask(signal.SIG_SETMASK,previous)


def wait_child(process,limit,grace=2.):
    try:
        try:code=process.wait(timeout=limit)
        except subprocess.TimeoutExpired:raise TimeoutError('job parent watchdog reached')
        if code:raise ChildProcessError('job exited unsuccessfully')
    finally:stop_child(process,grace)


def run_child(spawn,limit,grace=2.):
    """Own process before pending whole-worker alarm can be delivered."""
    if limit<=0:raise TimeoutError('job expired before spawn')
    started=time.monotonic();previous=signal.pthread_sigmask(signal.SIG_BLOCK,{signal.SIGALRM});process=None
    try:
        process=spawn()
        signal.pthread_sigmask(signal.SIG_SETMASK,previous)
        remaining=limit-(time.monotonic()-started)
        if remaining<=0:raise TimeoutError('job expired during launch')
        return wait_child(process,remaining,grace)
    finally:
        try:stop_child(process,grace)
        finally:signal.pthread_sigmask(signal.SIG_SETMASK,previous)


def seal_private_output(directory,name,private_root=None):
    directory=Path(directory)
    root=Path(private_root) if private_root is not None else Path(__file__).resolve().parent/'local'
    if name not in {'predictions.npz','weights.npz','partial_weights.npz','trace.json','receipt.json','failure.json','request.json','started.json','stats.json','memory.npz','stderr.log','stdout.log'}:
        raise ValueError('undeclared private output')
    if (not directory.name.startswith('job_') or directory.is_symlink() or root.is_symlink()
            or directory.resolve().parent!=root.resolve()):raise ValueError('unsafe job directory')
    path=directory/name
    if path.is_symlink():raise ValueError('symlinked private output')
    return path
