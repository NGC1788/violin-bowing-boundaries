"""Copy-only complete production reproduction in a new output directory."""
from pathlib import Path
import argparse,hashlib,json,os,shutil,subprocess,sys,time
import numpy as np
HERE=Path(__file__).resolve().parent

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);args=ap.parse_args();dest=args.out.expanduser().resolve()
    if dest==HERE or HERE in dest.parents or dest in HERE.parents:raise ValueError('output must not contain or be contained in source')
    if dest.exists() and any(dest.iterdir()):raise ValueError('output must be a new or empty directory')
    dest.mkdir(parents=True,exist_ok=True)
    source_files=[p for p in HERE.rglob('*') if p.is_file() and '__pycache__' not in str(p)]
    before={str(p.relative_to(HERE)):hashlib.sha256(p.read_bytes()).hexdigest() for p in source_files}
    for name in ('model.py','run.py','summarize.py','input_joint.json.gz','reference_baselines.json','cases.json','prior_obstructions.json'):shutil.copyfile(HERE/name,dest/name)
    env=os.environ|{'OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'1','VECLIB_MAXIMUM_THREADS':'1'};steps=[];start=time.monotonic()
    for label,cmd in [('prepare',[sys.executable,'run.py','--prepare']),('controls',[sys.executable,'run.py','--workers','2']),('summarize',[sys.executable,'summarize.py'])]:
        with (dest/(label+'.log')).open('w') as f:subprocess.run(cmd,cwd=dest,env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
        steps.append(label)
    compared=0;arrays=0;maxdiff=0.
    for path in sorted((HERE/'controls').glob('*.npz')):
        a=np.load(path);b=np.load(dest/'controls'/path.name);assert set(a.files)==set(b.files)
        for k in a.files:
            assert a[k].shape==b[k].shape
            if np.issubdtype(a[k].dtype,np.floating):
                assert np.allclose(a[k],b[k],rtol=1e-9,atol=1e-8,equal_nan=True),(path,k)
                finite=np.isfinite(a[k])&np.isfinite(b[k]);maxdiff=max(maxdiff,float(np.max(np.abs(a[k][finite]-b[k][finite]),initial=0)))
            else:assert np.array_equal(a[k],b[k]),(path,k)
            compared+=a[k].size;arrays+=1
    # Deterministic summaries include exact ranks, tie counts and all selected-case effects.
    for filename in ('summary.json','selected.json','baselines.json','rows.json'):
        assert json.load(open(HERE/filename))==json.load(open(dest/filename)),filename
    after={str(p.relative_to(HERE)):hashlib.sha256(p.read_bytes()).hexdigest() for p in source_files};assert before==after
    report={'status':'PASS','steps':steps,'seconds':time.monotonic()-start,'npz_arrays':arrays,'npz_elements':compared,'max_float_difference':maxdiff,'exact_json_files':4,'source_hashes_unchanged':True}
    (dest/'reproduction_report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
if __name__=='__main__':main()
