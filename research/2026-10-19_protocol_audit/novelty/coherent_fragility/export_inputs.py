"""Export a compact waveform-feature fixture aligned to the prior exact-edit input."""
from pathlib import Path
import sys,json,gzip,hashlib,math
import numpy as np
ROOT=Path.home()/'violin-rne';HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'novelty/tilt_principle'))
import tilt as T
FEATURES=('periodicity','second_flyback_ratio','amplitude','flyback_height')

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def serial(v):return float(v) if math.isfinite(v) else None

def main():
    old_path=ROOT/'novelty/run_fragility/input_columns.json.gz'
    with gzip.open(old_path,'rt') as fh:old=json.load(fh)
    out={'description':'Same measured forces, beta and original labels as run_fragility; added waveform features and fixed rule guards. See DATA.md. No raw waveform.',
         'source_records':old['source_records'],'source_sha256':{'prior_fixture':sha(old_path),'PLAN.md':sha(HERE/'PLAN.md'),'export_inputs.py':sha(Path(__file__))},
         'features':list(FEATURES),'columns':[]}
    by_session={}
    for c in old['columns']:by_session.setdefault(c['session'],[]).append(c)
    audit={'windows':0,'columns':0,'baseline_mismatches':{},'missing_values':{},'negative_values':{},'frozen_missing_H':{},'classifier_constants':T.LS.AP,'dependency_sha256':{}}
    for session,cols in by_session.items():
        st=T.LS.STEM[session];stem,rep=st if isinstance(st,tuple) else(st,None)
        raw,height=T.read2(stem,rep)
        records={r[0]:(r,h) for r,h in zip(raw[0],height)}
        assert len(records)==len(raw[0])
        featpath=T.LS.D.FEAT/f'{stem}.csv.gz';out['source_sha256'][str(featpath.relative_to(ROOT))]=sha(featpath)
        for c in cols:
            values={name:[] for name in FEATURES};guards={name:[] for name in FEATURES};fixed={name:[] for name in FEATURES}
            for i,tr in enumerate(c['trial']):
                row,h=records[c['folder'],tr];key,force,v,amp,p,f0,fpp,ratio=row
                assert force==c['force'][i]
                base=T.LS.helm(amp,p,f0,fpp,ratio,T.BASE)
                assert base==bool(c['labels'][i]),(session,key)
                physical=(not(math.isfinite(f0) and f0<T.LS.AP['subharmonic_ratio']*T.LS.REF_F0)) and math.isfinite(fpp) and fpp>=1
                scores={'periodicity':p,'second_flyback_ratio':ratio,'amplitude':amp,'flyback_height':h}
                gate={'periodicity':physical and math.isfinite(p) and p>=T.LS.AP['max_aperiodic'] and math.isfinite(ratio) and ratio<.5,
                      'second_flyback_ratio':physical and math.isfinite(p) and p>=max(.8,T.LS.AP['max_aperiodic']) and math.isfinite(ratio),
                      'amplitude':base and math.isfinite(amp),'flyback_height':base and math.isfinite(h)}
                for name,x in scores.items():
                    lock=bool(base and not math.isfinite(x));g=bool(gate[name] and math.isfinite(x) and not lock)
                    values[name].append(serial(x));guards[name].append(int(g));fixed[name].append(int(lock))
                    audit['missing_values'][name]=audit['missing_values'].get(name,0)+int(not math.isfinite(x))
                    audit['negative_values'][name]=audit['negative_values'].get(name,0)+int(math.isfinite(x) and x<0)
                    audit['frozen_missing_H'][name]=audit['frozen_missing_H'].get(name,0)+int(lock)
                    threshold={'periodicity':.8,'second_flyback_ratio':.5,'amplitude':0.,'flyback_height':0.}[name]
                    label=lock or (g and (x<threshold if name=='second_flyback_ratio' else x>=threshold))
                    audit['baseline_mismatches'][name]=audit['baseline_mismatches'].get(name,0)+int(label!=base)
            new=dict(c);new['id']=f'{session}|{c["folder"]}|{c["column"]}';new['group']=f'{session}|{c["folder"]}'
            new['features']={name:{'values':values[name],'guard':guards[name],'fixed':fixed[name]} for name in FEATURES}
            out['columns'].append(new);audit['columns']+=1;audit['windows']+=len(c['trial'])
        print(session,flush=True)
    assert audit['windows']==78000 and not any(audit['baseline_mismatches'].values()),audit
    for path in [Path(T.__file__),Path(T.LS.__file__),Path(T.LS.C.__file__),Path(T.LS.D.__file__),Path(T.LS.V.__file__)]:
        audit['dependency_sha256'][str(path)]=sha(path)
    for key,value in old['source_sha256'].items():
        if key.endswith('.csv.gz'):assert out['source_sha256'][key]==value
    payload=json.dumps(out,separators=(',',':'),allow_nan=False).encode()
    with (HERE/'input_features.json.gz').open('wb') as fh:
        with gzip.GzipFile(fileobj=fh,mode='wb',mtime=0) as z:z.write(payload)
    audit['input_sha256']=sha(HERE/'input_features.json.gz');audit['prior_fixture_sha256']=sha(old_path)
    (HERE/'input_audit.json').write_text(json.dumps(audit,indent=2,allow_nan=False)+'\n')
    print(json.dumps(audit,indent=2))
if __name__=='__main__':main()
