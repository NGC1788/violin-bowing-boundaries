"""Recover missing B1 ramp/block data after the existing download pipeline exits.

Waiting avoids a second large archive competing with live folder extraction and
the 60 GB free-space reserve. Does not launch/stop/modify replicate.py. The running
archive companion independently pins and processes the completed download.
"""
from pathlib import Path
from datetime import datetime,timezone
import fcntl,json,os,shutil,subprocess,time

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
REPO=Path.home()/'violin-bowing-boundaries'
NAME='2024-04-10_TypeB_sample1.7z'

def log(s):print(datetime.now(timezone.utc).isoformat(),s,flush=True)

def active_download_pipeline():
    text=subprocess.check_output(['ps','-axo','pid=,command='],text=True)
    lines=[]
    for line in text.splitlines():
        if 'python' in line and (' replicate.py' in line or '/replicate.py' in line or 'zenodo_catalog.py download' in line):
            lines.append(line.strip())
    return lines

def main():
    with (HERE/'recover_B1.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        log('B1 recovery queued: waiting until existing replicate/download processes finish; no second pipeline.')
        while active_download_pipeline():time.sleep(30)
        while shutil.disk_usage(ROOT).free < 60_000_000_000+14_856_184_074+1_000_000_000:
            log('Waiting for 60 GB reserve plus full B1 archive and 1 GB output allowance.');time.sleep(60)
        receipt=HERE/'archive_companion/receipts'/f'{NAME}.json'
        if receipt.exists() and json.loads(receipt.read_text()).get('status')=='complete':
            log('B1 already recovered and validated; no download.');return
        cmd=[str(REPO/'.venv-mac/bin/python'),str(REPO/'scripts/zenodo_catalog.py'),'download',
             '--record','17749110','--file',NAME,'--max-gib','14','--connections','4','--retries','12',
             '--data-dir',str(ROOT/'data')]
        log('Starting B1 recovery download. The companion will retain it until metadata validation.')
        subprocess.run(cmd,cwd=REPO,check=True)
        log('B1 download complete. Waiting for archive-companion validation.')
        while True:
            if receipt.exists():
                state=json.loads(receipt.read_text())
                if state.get('status')=='complete':
                    log('B1 ramp and block outputs validated. Recovery original retained for review.');return
                if state.get('status')=='failed':
                    raise RuntimeError('Companion failed; archive preserved: '+state.get('error',''))
            time.sleep(30)

if __name__=='__main__':main()
