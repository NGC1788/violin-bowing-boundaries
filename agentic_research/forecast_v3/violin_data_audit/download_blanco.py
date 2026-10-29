"""Fetch only the pinned, CC-BY-4.0 Blanco 2021 archive; no automatic extraction.

The raw archive stays outside the repository. Metadata is rechecked before any
download. Re-running resumes an interrupted .part file and verifies every byte.
"""
from pathlib import Path
import argparse, datetime, hashlib, json, shutil, subprocess, urllib.request

RECORD = '4554201'
NAME = 'Sound_and_Motion_Data.rar'
SIZE = 468988201
MD5 = '352be95f849ebeeb76a3cc5e761208e7'
API = f'https://zenodo.org/api/records/{RECORD}'
HERE = Path(__file__).resolve().parent


def digest(path, method):
    h = hashlib.new(method)
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(4*1024**2), b''):
            h.update(block)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-root', type=Path, default=Path.home()/'violin-rne/data/external/blanco2021')
    args = parser.parse_args()
    root = args.data_root.expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(urllib.request.Request(API, headers={'User-Agent':'ViolinResearchDataAudit/1.0'}), timeout=40) as response:
        metadata_bytes = response.read()
    data = json.loads(metadata_bytes)
    item = next(f for f in data['files'] if f['key'] == NAME)
    assert data['metadata']['license']['id'] == 'cc-by-4.0', 'License changed; inspect manually'
    assert item['size'] == SIZE and item['checksum'] == f'md5:{MD5}', 'Pinned file changed'
    assert SIZE < 2*1024**3 and shutil.disk_usage(root).free > 2*1024**3
    (root/'zenodo_record_4554201.json').write_bytes(metadata_bytes)
    archive, partial = root/NAME, root/(NAME+'.part')
    if not archive.exists():
        print(f'Downloading {SIZE:,} bytes from pinned CC-BY-4.0 record', flush=True)
        subprocess.run(['curl','--fail','--location','--silent','--show-error','--retry','3',
                        '--retry-delay','2','--connect-timeout','20','--max-time','3600',
                        '--continue-at','-','--output',str(partial),item['links']['self']],check=True)
        assert partial.stat().st_size == SIZE, 'Incomplete download retained as .part'
        assert digest(partial, 'md5') == MD5, 'Checksum mismatch; keep file for inspection'
        partial.rename(archive)
    assert archive.stat().st_size == SIZE and digest(archive, 'md5') == MD5
    receipt = {'record_id':RECORD, 'record_url':f'https://zenodo.org/records/{RECORD}',
               'api_url':API, 'download_url':item['links']['self'], 'file':NAME,
               'size_bytes':SIZE, 'md5':MD5, 'sha256':digest(archive,'sha256'),
               'license':'CC-BY-4.0', 'title':data['metadata']['title'],
               'creators':data['metadata']['creators'], 'record_version':data['metadata'].get('version'),
               'api_metadata_sha256':hashlib.sha256(metadata_bytes).hexdigest(),
               'verified_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
               'raw_storage':'Outside repository; set --data-root to reproduce',
               'scope':'Source/archive integrity only; no feasibility or model accuracy implied'}
    (HERE/'download_provenance.json').write_text(json.dumps(receipt,indent=2,ensure_ascii=False)+'\n')
    print(json.dumps(receipt,indent=2,ensure_ascii=False),flush=True)


if __name__ == '__main__':
    main()
