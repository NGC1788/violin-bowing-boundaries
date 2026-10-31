"""Pinned raw-audio acquisition to resolve upstream feature lookahead uncertainty."""
from pathlib import Path
import datetime,fcntl,hashlib,json,shutil,subprocess,urllib.request,time,concurrent.futures
HERE=Path(__file__).resolve().parent
ROOT=Path.home()/'violin-rne/data/external/blanco2021_raw'
NAME='Participants_Raw_data.rar'
SIZE=2203971621
MD5='b7e8fef0b4a3e989319c5c5b38a819c2'

def digest(path,kind):
 h=hashlib.new(kind)
 with path.open('rb') as f:
  for block in iter(lambda:f.read(4*1024**2),b''):h.update(block)
 return h.hexdigest()

def main():
 ROOT.mkdir(parents=True,exist_ok=True)
 with (ROOT/'download.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  req=urllib.request.Request('https://zenodo.org/api/records/4553940',headers={'User-Agent':'ViolinForecastAudit/1.0'})
  with urllib.request.urlopen(req,timeout=40) as response:raw=response.read()
  metadata=json.loads(raw)
  entry=next(f for f in metadata['files'] if f['key']==NAME)
  assert metadata['metadata']['license']['id']=='cc-by-4.0'
  assert entry['size']==SIZE and entry['checksum']=='md5:'+MD5
  assert SIZE<3*1024**3 and shutil.disk_usage(ROOT).free>4*SIZE
  (ROOT/'record_4553940.json').write_bytes(raw)
  final=ROOT/NAME;partial=ROOT/(NAME+'.part')
  if not final.exists():
   print('Downloading pinned raw audio, 2.204 GB. Stored outside Git.',flush=True)
   parts=ROOT/'parts';parts.mkdir(exist_ok=True)
   chunk=8*1024**2
   # Retain the already downloaded prefix; completed chunks can be reused.
   if partial.exists() and not (ROOT/'single_stream_prefix.part').exists():
    with partial.open('rb') as f:
     for i in range(partial.stat().st_size//chunk):
      (parts/f'{i:05d}.part').write_bytes(f.read(chunk))
    partial.rename(ROOT/'single_stream_prefix.part')
   def fetch(i):
    lo=i*chunk;hi=min(SIZE,lo+chunk)-1;dest=parts/f'{i:05d}.part'
    if dest.exists() and dest.stat().st_size==hi-lo+1:return i
    for attempt in range(3):
     try:
      request=urllib.request.Request(entry['links']['self'],headers={
       'User-Agent':'ViolinForecastAudit/1.0','Range':f'bytes={lo}-{hi}'})
      with urllib.request.urlopen(request,timeout=120) as response:
       if response.status!=206 or response.headers.get('Content-Range')!=f'bytes {lo}-{hi}/{SIZE}':
        raise RuntimeError('Range response is not the pinned requested range')
       with dest.with_suffix('.tmp').open('wb') as f:
        shutil.copyfileobj(response,f,length=1024**2)
      if dest.with_suffix('.tmp').stat().st_size!=hi-lo+1:raise RuntimeError('Incomplete chunk')
      dest.with_suffix('.tmp').replace(dest)
      return i
     except Exception:
      if attempt==2:raise
      time.sleep(2**(attempt+1))
   total=(SIZE+chunk-1)//chunk
   with concurrent.futures.ThreadPoolExecutor(max_workers=16) as pool:
    futures=[pool.submit(fetch,i) for i in range(total)]
    for done,future in enumerate(concurrent.futures.as_completed(futures),1):
     future.result()
     if done%16==0 or done==total:print(f'Chunks {done}/{total}',flush=True)
   with partial.open('wb') as f:
    for i in range(total):
     with (parts/f'{i:05d}.part').open('rb') as inp:shutil.copyfileobj(inp,f,length=1024**2)
   assert partial.stat().st_size==SIZE and digest(partial,'md5')==MD5
   partial.rename(final)
  assert final.stat().st_size==SIZE and digest(final,'md5')==MD5
  receipt={'record_id':'4553940','url':'https://zenodo.org/records/4553940','filename':NAME,
   'bytes':SIZE,'md5':MD5,'sha256':digest(final,'sha256'),'license':'CC-BY-4.0',
   'creators':metadata['metadata']['creators'],'download_url':entry['links']['self'],
   'verified_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
   'budget_change':'Initial stored-descriptor pilot cap2GiB raised to3GiB before download, to investigate causal input provenance. No existing experiment changed.',
   'raw_storage':'Outside repository; audio files are not republished.'}
  (HERE/'download_receipt.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n')
  print('Complete, byte count and MD5/SHA256 verified.',flush=True)
if __name__=='__main__':main()
