"""Bounded one-pass extraction for large solid 7z archives.

No classifier changes: each bounded temporary pair is passed to replicate.windows_of.
All archive identities retain their full paths. Metadata is computed with the existing
ramp_meta.stroke and stroke_blocks_full.blocks functions. Original/pinned archives
are NEVER removed here. Publication follows exact coverage, block, and stream checks.
"""
from __future__ import annotations
import csv, gzip, hashlib, json, math, os, re, shutil, subprocess, sys, tempfile
from collections import deque
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np
import disk_guard as DG

ROOT = Path(__file__).resolve().parent.parent
REPO = ROOT.parent / 'violin-bowing-boundaries'
SEVEN = REPO / '.local-tools/7zip-26.03/7zz'
MAX_MEMBER = 128 * 1024 * 1024
MAX_PENDING = 6
OUTPUT_ALLOWANCE = 2_000_000_000
# 2026-10-19 22h (Claude): some archives store whole_n immediately BEFORE beta_n (e.g. D2). At most MAX_HELD
# whole member may wait in memory (<= MAX_MEMBER bytes) for its beta; anything longer is still rejected.
MAX_HELD = 1


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8<<20), b''): h.update(b)
    return h.hexdigest()


def inventory(text):
    DG.parse_slt_listing(text)  # validates ALL member paths, pairing, size, uniqueness
    items=[]
    for block in text.split('----------',1)[1].split('\n\n'):
        fields=dict(line.split(' = ',1) for line in block.splitlines() if ' = ' in line)
        path=fields.get('Path','').replace('\\','/')
        m=re.fullmatch(r'(.+)/(whole|beta)_([1-9][0-9]*)\.csv',path)
        if m:
            size=int(fields['Size'])
            if size>MAX_MEMBER: raise ValueError('member exceeds bounded staging limit')
            items.append((path,size,m[1],m[2],int(m[3])))
    # x -so emits selected members in archive order; beta must precede its whole, or follow it with at most
    # MAX_HELD whole members waiting (bounded in-memory hold).
    betas=set();waiting=set()
    for _,_,folder,kind,trial in items:
        key=(folder,trial)
        if kind=='beta':betas.add(key);waiting.discard(key)
        elif key not in betas:
            waiting.add(key)
            if len(waiting)>MAX_HELD:raise ValueError('beta follows whole beyond bounded hold; unsupported streaming order')
    if waiting:raise ValueError('whole without beta in streaming order')
    return items


def read_exact(stream,size):
    chunks=[];left=size
    while left:
        b=stream.read(min(left,1<<20))
        if not b:raise EOFError(f'stream ended {left} bytes early')
        chunks.append(b);left-=len(b)
    return b''.join(chunks)


def worker(task):
    whole,beta_path,folder,trial=task
    sys.path.insert(0,str(ROOT/'timeres'))
    sys.path.insert(0,str(ROOT/'claude_extra'))
    sys.path.insert(0,str(REPO/'scripts'))
    import replicate, ramp_meta, stroke_blocks_full, trial_audit
    rows=replicate.windows_of(task)
    if not rows or any(r.get('error') for r in rows):
        raise ValueError(f'empty/error window features: {folder}/{trial}')
    raw=Path(whole).read_bytes()
    ramp=ramp_meta.stroke(raw)
    if ramp.get('error'):raise ValueError(f'ramp failed: {folder}/{trial}')
    sig=trial_audit.parse_signal(raw)
    if not 1<=len(sig)//2500<=stroke_blocks_full.MAXB:
        raise ValueError('blocks would be empty or truncated')
    blocks=stroke_blocks_full.blocks(sig)
    return rows,ramp,blocks


def validate_metadata(keys,betas,ramps,blocks,expected):
    if len(keys)!=len(set(keys)) or set(keys)!=set(expected):
        raise ValueError('stream trial identity coverage differs from inventory')
    if len(keys)!=len(ramps) or blocks.shape!=(len(keys),10,160):
        raise ValueError('metadata lengths differ')
    for i,key in enumerate(keys):
        r=ramps[i];beta=betas[key]
        required=['v_max','onset_s','plateau_on_s','plateau_off_s','total_s','ramp_s','accel','force_at_onset']
        if not (all(math.isfinite(r[x]) for x in required) and math.isfinite(beta) and 0<beta<1):
            raise ValueError('invalid protocol metadata')
        if not (r['v_max']>0 and r['accel']>0 and 0<=r['onset_s']<=r['plateau_on_s']<r['plateau_off_s']<r['total_s']):
            raise ValueError('invalid protocol times')
        nb=round(r['total_s']*50000)//2500
        if not 1<=nb<=160:raise ValueError('block truncation')
        b=blocks[i]
        if not np.isfinite(b[:9,:nb]).all() or not np.isnan(b[:,nb:]).all():raise ValueError('block finite span mismatch')
        coh=b[9,:nb]
        if np.isinf(coh).any() or np.any(np.abs(coh[np.isfinite(coh)])>1.00001):raise ValueError('invalid coherence')
    return {'trials':len(keys),'coverage_sha256':hashlib.sha256(json.dumps(sorted(keys)).encode()).hexdigest()}


def run(archive, output, expected_folders, log=print, integrity=None):
    archive=Path(archive);output=Path(output)
    sys.path.insert(0,str(ROOT/'timeres'));import replicate
    sys.path.insert(0,str(ROOT/'claude_extra'));import ramp_meta,stroke_blocks_full
    text=subprocess.run([str(SEVEN),'l','-slt',str(archive)],capture_output=True,text=True,check=True).stdout
    items=inventory(text)
    expected={(f,t) for _,_,f,k,t in items if k=='whole'}
    if expected!={(f,t) for f,ns in expected_folders.items() for t in ns}:raise ValueError('pipeline inventory differs')
    dest=ROOT/'astra_20261019/highforce_stream'/archive.stem
    # Atomic published directory is the restart checkpoint; do not recompute a finished pass.
    if dest.exists():
        receipt=json.loads((dest/'receipt.json').read_text())
        for name,sha in receipt['outputs'].items():
            if digest(dest/name)!=sha:raise ValueError('published stream hash mismatch')
        if receipt['validation']['coverage_sha256']!=hashlib.sha256(json.dumps(sorted(expected)).encode()).hexdigest():
            raise ValueError('published stream coverage mismatch')
        if not output.exists():
            os.link(dest/'features.csv.gz',output)
        elif digest(output)!=receipt['outputs']['features.csv.gz']:raise ValueError('pipeline output differs')
        return receipt
    required=DG.RESERVE_BYTES+DG.MARGIN_BYTES+OUTPUT_ALLOWANCE+MAX_PENDING*MAX_MEMBER
    replicate.wait_for_budget(required,'bounded archive stream')
    dest.parent.mkdir(parents=True,exist_ok=True)
    stage=Path(tempfile.mkdtemp(prefix=archive.stem+'.stage-',dir=dest.parent))
    pending=deque();keys=[];ramps=[];arrays=[];betas={};nrows=0
    command=[str(SEVEN),'x','-so',str(archive),'-ir!*/whole_*.csv','-ir!*/beta_*.csv']
    proc=None
    try:
        with (stage/'seven.log').open('wb') as err, gzip.open(stage/'features.csv.gz','wt',newline='') as fh, ProcessPoolExecutor(max_workers=4) as pool:
            writer=csv.DictWriter(fh,fieldnames=replicate.FIELDS,extrasaction='ignore');writer.writeheader()
            proc=subprocess.Popen(command,stdout=subprocess.PIPE,stderr=err,bufsize=1<<20)
            def finish_one():
                nonlocal nrows
                future,key,tmp=pending.popleft()
                rows,ramp,arr=future.result()
                writer.writerows(rows);nrows+=len(rows);keys.append(key);ramps.append(ramp);arrays.append(arr)
                shutil.rmtree(tmp)
                if len(keys)%500==0:log(f'{archive.name}: stream {len(keys)}/{len(expected)} strokes, free {shutil.disk_usage(ROOT).free/1e9:.1f} GB')
            def submit(key,raw,size):
                folder,trial=key
                if len(pending)>=MAX_PENDING:finish_one()
                if shutil.disk_usage(ROOT).free<DG.RESERVE_BYTES+DG.MARGIN_BYTES+size:
                    raise RuntimeError('reserve would be crossed; original archive retained')
                tmp=Path(tempfile.mkdtemp(prefix='stroke-',dir=stage))
                whole=tmp/'whole.csv';bp=tmp/'beta.csv'
                whole.write_bytes(raw);bp.write_text(repr(betas[key]))
                pending.append((pool.submit(worker,(str(whole),str(bp),folder,trial)),key,tmp))
            held={}
            for _,size,folder,kind,trial in items:
                raw=read_exact(proc.stdout,size);key=(folder,trial)
                if kind=='beta':
                    beta=float(re.split(r'[,\s;]+',raw.decode().strip())[0])
                    if not math.isfinite(beta) or not 0<beta<1:raise ValueError('invalid beta')
                    betas[key]=beta
                    if key in held:submit(key,*held.pop(key))
                    continue
                if key not in betas:
                    if len(held)>=MAX_HELD:raise ValueError('more wholes awaiting beta than MAX_HELD')
                    held[key]=(raw,size);continue
                submit(key,raw,size)
            if held:raise ValueError('whole member(s) without beta after stream end')
            if proc.stdout.read(1):raise ValueError('unexpected bytes beyond selected inventory')
            if proc.wait()!=0:raise RuntimeError('7zz extraction failed; see seven.log')
            while pending:finish_one()
        data=np.stack(arrays)
        validation=validate_metadata(keys,betas,ramps,data,expected)
        with (stage/'ramp_meta.csv').open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=ramp_meta.FIELDS);w.writeheader()
            for key,r in zip(keys,ramps):w.writerow({'folder':key[0],'trial':key[1],'beta':betas[key],**r})
        np.savez_compressed(stage/'blocks.npz',data=data,fields=np.array(stroke_blocks_full.FIELDS),
             folder=np.array([k[0] for k in keys]),trial=np.array([k[1] for k in keys]),
             beta=np.array([betas[k] for k in keys]),block_s=.05)
        receipt={'status':'validated','archive':str(archive),'archive_bytes':archive.stat().st_size,
            'archive_integrity':integrity,'validation':validation,'rows':nrows,'command':command,'full_folder_identity':True,
            'outputs':{n:digest(stage/n) for n in ['features.csv.gz','ramp_meta.csv','blocks.npz']},
            'sources':{str(p):digest(p) for p in [Path(__file__),Path(replicate.__file__),Path(ramp_meta.__file__),Path(stroke_blocks_full.__file__)]}}
        (stage/'receipt.json').write_text(json.dumps(receipt,indent=2))
        for p in stage.iterdir():
            if p.is_file():
                with p.open('rb') as f:os.fsync(f.fileno())
        os.rename(stage,dest)
        sync_directory(dest.parent)
        if output.exists():raise FileExistsError('unexpected old output; valid stream retained separately')
        os.link(dest/'features.csv.gz',output)
        sync_directory(output.parent)
        log(f'{archive.name}: validated {len(keys)} strokes / {nrows} windows; full-path ramp+blocks saved')
        return receipt
    finally:
        if proc is not None:
            if proc.poll() is None:proc.terminate();proc.wait()
            if proc.stdout:proc.stdout.close()


def sync_directory(path):
    fd=os.open(path,os.O_RDONLY)
    try:os.fsync(fd)
    finally:os.close(fd)


def verify_archive(archive):
    sys.path.insert(0,str(ROOT/'astra_20261018'))
    import archive_companion as ac
    spec=ac.metadata_spec(Path(archive).name)
    if Path(archive).stat().st_size!=spec['bytes'] or ac.digest(archive,'md5')!=spec['md5']:
        raise ValueError('archive MD5/size differ from exact local manifest')
    return spec


def release_redundant_pin(archive,receipt):
    """Release only a same-inode failed-companion link after validated replacement.

    The original remains present here. Do not touch arbitrary/raw/protected files.
    Existing failure receipt remains as history; replacement receipt is durable.
    """
    archive=Path(archive)
    base=ROOT/'astra_20261018/archive_companion'
    pin=base/'pins'/archive.name
    old=base/'receipts'/(archive.name+'.json')
    if not pin.exists():return False
    if not receipt.get('archive_integrity') or receipt.get('status')!='validated':
        raise ValueError('no integrity/validation receipt; pin retained')
    if not old.exists():raise ValueError('unknown companion state; pin retained')
    r=json.loads(old.read_text())
    if r.get('status')!='failed' or 'ambiguous folder basename' not in r.get('error',''):
        raise ValueError('companion may be active; pin retained')
    if not os.path.samefile(pin,archive):raise ValueError('pin differs from original; retained')
    dest=ROOT/'astra_20261019/highforce_stream'/archive.stem
    for n,sha in receipt['outputs'].items():
        if digest(dest/n)!=sha:raise ValueError('replacement outputs differ; pin retained')
    pin.unlink();sync_directory(pin.parent)
    (dest/'pin_release.json').write_text(json.dumps({'released_redundant_link':str(pin),'original_still_present':str(archive),'prior_receipt':str(old)},indent=2))
    return True
