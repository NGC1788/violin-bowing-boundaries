#!/usr/bin/env python3
"""Hold only the verified Part 6 pipeline parent; resume after exact preflight.

No download/extraction or deletion is performed. Root must review before launch.
A stopped parent leaves its existing downloader running. Cancellation or any failed
check keeps the parent stopped, with a durable explanatory receipt. An already
stopped parent is never resumed by this gate. This is a Part 6 boundary guard,
not a guarantee about later archives or unrelated concurrent disk writers.
"""
from __future__ import annotations
import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shlex
import shutil
import signal
import subprocess
import time
import uuid

ROOT = Path.home() / 'violin-rne'
REPO = Path.home() / 'violin-bowing-boundaries'
PYTHON = REPO / '.venv-mac/bin/python'
SEVEN = REPO / '.local-tools/7zip-26.03/7zz'
NAME = '2024-10-23_variability_setup.7z'
ARCHIVE = ROOT / 'data/raw/zenodo/17795326' / NAME
SIZE = 28218620427
MD5 = 'bddef0d9cf48a6dccbc27021116274a5'
RESERVE = 60_000_000_000
ALLOWANCE = 2_000_000_000
_CANCEL = None


def now(): return datetime.now(timezone.utc).isoformat()


def atomic_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name+'.'+uuid.uuid4().hex+'.tmp')
    with tmp.open('x') as f:
        json.dump(payload,f,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
    os.replace(tmp,path)
    fd=os.open(path.parent,os.O_RDONLY)
    try:os.fsync(fd)
    finally:os.close(fd)


@contextmanager
def singleton(path):
    with path.open('a+') as f:
        try:fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError as e:raise RuntimeError('another Part 6 disk gate is active') from e
        try:yield
        finally:fcntl.flock(f,fcntl.LOCK_UN)


def parse_process(line):
    parts=line.strip().split(None,8)
    if len(parts)!=9:raise ValueError('unexpected ps process format')
    return {'pid':int(parts[0]),'ppid':int(parts[1]),'state':parts[2],
            'started':' '.join(parts[3:8]),'command':parts[8]}


def process(pid):
    result=subprocess.run(['ps','-p',str(pid),'-o','pid=,ppid=,state=,lstart=,command='],text=True,capture_output=True)
    if not result.stdout.strip():return None
    if result.returncode:raise RuntimeError('ps failed')
    return parse_process(result.stdout)


def same_identity(a,b):
    return b is not None and all(a[k]==b[k] for k in ('pid','started','command'))


def stopped(p): return p is not None and 'T' in p['state']

def ended(p): return p is None or 'Z' in p['state']


def same_downloader(identity, current):
    # macOS can replace an exited child's command by '(python)' until the
    # stopped parent reaps it. PID plus birth time still prevents PID reuse.
    if current is None:
        return True
    if 'Z' in current['state']:
        return all(identity[k] == current[k] for k in ('pid', 'started'))
    return same_identity(identity, current)


def validate_pair(parent, child):
    if parent is None or child is None or ended(parent) or ended(child):
        raise ValueError('live parent and its active downloader are required before any signal')
    par=shlex.split(parent['command']);ch=shlex.split(child['command'])
    allowed_par=([str(PYTHON),'replicate.py'],[str(PYTHON),str(ROOT/'timeres/replicate.py')])
    if par not in allowed_par:raise ValueError('parent is not the exact expected Python pipeline')
    if child['ppid']!=parent['pid'] or ch[:3]!=[str(PYTHON),str(REPO/'scripts/zenodo_catalog.py'),'download']:
        raise ValueError('child is not the expected downloader belonging to this parent')
    for flag,value in [('--record','17795326'),('--file',NAME),('--data-dir',str(ROOT/'data'))]:
        if ch.count(flag)!=1 or ch.index(flag)+1>=len(ch) or ch[ch.index(flag)+1]!=value:
            raise ValueError('downloader does not match Part 6 destination/record/name')


def checked_signal(identity, sig):
    current=process(identity['pid'])
    if not same_identity(identity,current) or ended(current):
        raise RuntimeError('PID identity changed or ended; refusing signal')
    os.kill(identity['pid'],sig)


def direct_children(parent_pid):
    r=subprocess.run(['ps','-axo','pid=,ppid=,state=,lstart=,command='],text=True,capture_output=True,check=True)
    rows=[parse_process(line) for line in r.stdout.splitlines() if line.strip()]
    return [x for x in rows if x['ppid']==parent_pid]


def safe_children(parent, downloader):
    unexpected=[]
    for child in direct_children(parent['pid']):
        if child['pid']==downloader['pid'] and same_downloader(downloader,child):continue
        # The existing awake wrapper is allowed; no extractor or Python worker is.
        cmd=shlex.split(child['command'])
        if cmd and Path(cmd[0]).name=='caffeinate':continue
        unexpected.append(child)
    if unexpected:raise RuntimeError('unexpected child after parent hold; possible extraction race: '+json.dumps(unexpected))


def inventory(text):
    entries={};path=None
    for line in text.splitlines():
        if line.startswith('Path = '):path=line[7:].strip().replace('\\','/')
        elif line.startswith('Size = ') and path is not None:
            m=re.fullmatch(r'(.+)/(whole|beta)_(\d+)\.csv',path)
            if m:
                folder,kind,trial=m[1],m[2],int(m[3]);pp=PurePosixPath(folder)
                if pp.is_absolute() or '..' in pp.parts:raise ValueError('unsafe archive path')
                key=(folder,trial);row=entries.setdefault(key,{})
                if kind in row:raise ValueError('duplicate whole/beta member')
                row[kind]=int(line[7:])
            path=None
    if not entries or any(set(x)!={'whole','beta'} or min(x.values())<=0 for x in entries.values()):
        raise ValueError('unsupported layout or incomplete whole/beta pairs')
    leaves={};folders={};counts={}
    for (folder,_),sizes in entries.items():
        leaf=folder.rsplit('/',1)[-1]
        if leaf in leaves and leaves[leaf]!=folder:
            raise ValueError('leaf-folder collision unsupported by current stroke_blocks.py')
        leaves[leaf]=folder;folders[folder]=folders.get(folder,0)+sum(sizes.values());counts[folder]=counts.get(folder,0)+1
    return {'folder_bytes':folders,'folder_trials':counts,'trials':len(entries),'max_folder_bytes':max(folders.values()),
            'listing_sha256':hashlib.sha256(text.encode()).hexdigest()}


def budget(free, inv):
    need=RESERVE+inv['max_folder_bytes']+ALLOWANCE
    return {'free_bytes':free,'reserve_bytes':RESERVE,'output_allowance_bytes':ALLOWANCE,
            'required_free_bytes':need,'projected_free_after_largest_folder_and_allowance':free-inv['max_folder_bytes']-ALLOWANCE,
            'passes':free>=need}


def validate_archive(path):
    before=path.stat()
    if not path.is_file() or path.is_symlink() or before.st_size!=SIZE:raise ValueError('archive missing, symlink, or wrong published size')
    h=hashlib.md5()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(8<<20),b''):
            if _CANCEL:raise RuntimeError('cancelled during archive checksum')
            h.update(block)
    after=path.stat()
    if (before.st_ino,before.st_size,before.st_mtime_ns)!=(after.st_ino,after.st_size,after.st_mtime_ns):
        raise ValueError('archive changed during checksum')
    if h.hexdigest()!=MD5:raise ValueError('archive MD5 mismatch')
    r=subprocess.run([str(SEVEN),'l','-slt',str(path)],capture_output=True,text=True,check=True)
    inv=inventory(r.stdout)
    return inv,{'path':str(path),'bytes':SIZE,'md5':MD5,'inode':after.st_ino}


def cancellation(signum,frame):
    global _CANCEL
    _CANCEL=signum


def run(parent_pid,child_pid,work,poll=15,max_hours=12):
    work.mkdir(parents=True,exist_ok=True)
    receipt_path=work/'gate_receipt.json'
    with singleton(work/'gate.lock'):
        parent,child=process(parent_pid),process(child_pid)
        validate_pair(parent,child)  # no signal before exact initial verification
        was_stopped=stopped(parent)
        receipt={'started_utc':now(),'parent_identity':parent,'downloader_identity':child,
                 'parent_initially_stopped':was_stopped,'gate_sent_stop':False,'gate_sent_continue':False,
                 'status':'identity_verified','source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
        atomic_json(receipt_path,receipt)
        try:
            if not was_stopped:
                checked_signal(parent,signal.SIGSTOP)
                receipt['gate_sent_stop']=True
                # SIGSTOP delivery is asynchronous; wait briefly for the kernel state.
                for _ in range(20):
                    if stopped(process(parent_pid)):break
                    time.sleep(.05)
                else:raise RuntimeError('parent stop not observed')
            receipt.update(status='held_waiting_for_original_downloader',held_utc=now())
            atomic_json(receipt_path,receipt)
            safe_children(parent,child)
            deadline=time.monotonic()+max_hours*3600
            while True:
                if _CANCEL:raise RuntimeError(f'cancelled by signal {_CANCEL}; parent left stopped')
                if time.monotonic()>deadline:raise TimeoutError('download deadline exceeded; parent left stopped')
                current_parent=process(parent_pid)
                if not same_identity(parent,current_parent) or not stopped(current_parent):
                    raise RuntimeError('held parent changed/exited or resumed externally')
                current_child=process(child_pid)
                if not same_downloader(child,current_child):
                    raise RuntimeError('downloader PID reused')
                if ended(current_child):break
                time.sleep(poll)
            safe_children(parent,child)
            if not ARCHIVE.exists():raise FileNotFoundError('downloader ended without final archive; no resume')
            inv,verified=validate_archive(ARCHIVE)
            receipt.update(inventory=inv,archive_integrity=verified,budget=budget(shutil.disk_usage(ROOT).free,inv))
            atomic_json(receipt_path,receipt)
            if _CANCEL:raise RuntimeError('cancelled before resume; parent left stopped')
            if not receipt['budget']['passes']:raise RuntimeError('exact extraction budget fails; parent left stopped')
            if was_stopped:
                receipt.update(status='passed_parent_previously_stopped_no_resume',ended_utc=now())
            else:
                # Recheck current free space and parent identity at the final boundary.
                receipt['budget']=budget(shutil.disk_usage(ROOT).free,inv)
                if not receipt['budget']['passes']:raise RuntimeError('disk budget changed before resume')
                if not stopped(process(parent_pid)):raise RuntimeError('parent resumed externally; no signal')
                checked_signal(parent,signal.SIGCONT)
                receipt.update(status='passed_parent_resumed',gate_sent_continue=True,ended_utc=now())
            atomic_json(receipt_path,receipt)
            return receipt
        except BaseException as error:
            receipt.update(status='held_attention_required',ended_utc=now(),error=f'{type(error).__name__}: {error}',
                           parent_observed=process(parent_pid))
            atomic_json(receipt_path,receipt)
            raise


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--parent-pid',type=int,required=True);p.add_argument('--child-pid',type=int,required=True)
    p.add_argument('--work',type=Path,default=Path(__file__).parent/'gate_state')
    p.add_argument('--poll-seconds',type=float,default=15);p.add_argument('--max-hours',type=float,default=12)
    a=p.parse_args()
    if a.parent_pid<=1 or a.child_pid<=1 or not 2<=a.poll_seconds<=60 or not 0<a.max_hours<=24:p.error('invalid PID, polling interval or bounded duration')
    for sig in (signal.SIGINT,signal.SIGTERM,signal.SIGHUP):signal.signal(sig,cancellation)
    print(json.dumps(run(a.parent_pid,a.child_pid,a.work,a.poll_seconds,a.max_hours),indent=2),flush=True)


if __name__=='__main__':main()
