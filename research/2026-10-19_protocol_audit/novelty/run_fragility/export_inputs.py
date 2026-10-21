"""Export compact derived inputs, without waveforms or original archives."""
from pathlib import Path
import sys,json,gzip,hashlib
import numpy as np
ROOT=Path.home()/'violin-rne';HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'novelty/tilt_principle'))
import tilt as T


def main():
    out={'description':'Derived t=.2s,window=.2s base-H labels, measured mean force and beta; original data CC BY4.0, Lampis/IWK. No waveform samples.',
         'source_records':['https://zenodo.org/records/17749111','https://zenodo.org/records/17782542','https://zenodo.org/records/17782552','https://zenodo.org/records/17782565','https://zenodo.org/records/17782538','https://zenodo.org/records/17795326'],
         'excluded_session':'D_T2: original clamp/zero-width problem; fixed protocol', 'columns':[], 'source_sha256':{}}
    for session,st in T.LS.STEM.items():
        if session=='D_T2':continue
        stem,rep=st if isinstance(st,tuple) else(st,None)
        raw,_=T.read2(stem,rep);d=T.LS.build(raw,T.BASE)
        for (folder,column),ix in T.LS.C.columns_of(d,.2).items():
            ix=sorted(ix,key=lambda i:d['force'][i])
            out['columns'].append({'session':session,'folder':str(d['folders'][folder]),'column':int(column),
                'beta':float(np.median(d['beta'][ix])),'speed':float(d['speed'][ix[0]]),
                'force':d['force'][ix].tolist(),'labels':d['H'][ix].astype(int).tolist(),
                'trial':d['trial'][ix].tolist()})
        p=T.LS.D.FEAT/f'{stem}.csv.gz';out['source_sha256'][str(p.relative_to(ROOT))]=hashlib.sha256(p.read_bytes()).hexdigest()
    for p in [Path(__file__),ROOT/'novelty/label_sensitivity/label_sweep.py',ROOT/'novelty/tilt_principle/tilt.py']:
        out['source_sha256'][str(p.relative_to(ROOT))]=hashlib.sha256(p.read_bytes()).hexdigest()
    raw=json.dumps(out,separators=(',',':'),allow_nan=False).encode()
    with (HERE/'input_columns.json.gz').open('wb') as file:
        with gzip.GzipFile(fileobj=file,mode='wb',mtime=0) as z:z.write(raw)
    print(json.dumps({'columns':len(out['columns']),'windows':sum(len(c['labels']) for c in out['columns']),
                      'compressed_bytes':(HERE/'input_columns.json.gz').stat().st_size}))

if __name__=='__main__':main()
