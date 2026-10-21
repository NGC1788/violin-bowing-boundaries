"""Copy-only full audit reproduction. No network, raw data, installation, or source overwrite."""
from pathlib import Path
import hashlib,json,os,shutil,subprocess,sys,tempfile,time
import numpy as np
HERE=Path(__file__).resolve().parent
SKIP={'seconds','duration_seconds','elapsed_seconds','source_sha256','output','source_hashes'}

def compare(a,b,path=''):
    if isinstance(a,dict):
        assert set(a)==set(b),(path,'keys')
        return sum(compare(a[k],b[k],path+'/'+k) for k in a if k not in SKIP)
    if isinstance(a,list):
        assert len(a)==len(b),(path,'length');return sum(compare(x,y,path+f'/{i}') for i,(x,y) in enumerate(zip(a,b)))
    if isinstance(a,(float,int)) and not isinstance(a,bool):
        assert np.isclose(a,b,atol=2e-7,rtol=5e-6,equal_nan=True),(path,a,b);return 1
    assert a==b,(path,a,b);return 0

def main():
    logdir=HERE/'reproduction';logdir.mkdir(exist_ok=True)
    sources=[p for p in HERE.rglob('*') if p.is_file() and 'reproduction' not in p.relative_to(HERE).parts and '__pycache__' not in p.parts]
    before={str(p.relative_to(HERE)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    start=time.monotonic();steps=[]
    env=os.environ|{'OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'1','VECLIB_MAXIMUM_THREADS':'1','MPLBACKEND':'Agg'}
    with tempfile.TemporaryDirectory(prefix='violin_mle_bridge_') as tmp:
        work=Path(tmp)/'mle_bridge';work.mkdir()
        def run(name,args,cwd=work):
            ts=time.monotonic()
            with (logdir/f'{len(steps)+1:02}_{name}.log').open('w') as stream:
                subprocess.run([sys.executable]+args,cwd=cwd,env=env,stdout=stream,stderr=subprocess.STDOUT,check=True)
            steps.append({'name':name,'status':'PASS','seconds':time.monotonic()-ts});print(name,'PASS',flush=True)
        run('full_production',[str(HERE/'production/reproduce.py'),'--out',str(work/'production')])
        for p in HERE.rglob('*.py'):
            rel=p.relative_to(HERE)
            if '__pycache__' in rel.parts or 'reproduction' in rel.parts:continue
            q=work/rel;q.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,q)
        for name in ('PLAN.md','FAMILY_EXTENSION.md','source_obstruction_certificates.json'):
            shutil.copy2(HERE/name,work/name)
        run('coherent_replay',['coherent_replay.py'])
        run('synthetic_theory',['theory/synthetic_bridge.py'])
        run('synthetic_oracle',['verification/synthetic.py'])
        run('independent_refits',['verification/independent_refits.py'])
        run('independent_comparison',['verification/compare_results.py'])
        run('all_control_checks',['verification/control_audit.py'])
        run('all_certificate_families',['theory/family.py'])
        run('independent_family_checks',['verification/family_audit.py'])
        run('figures',['plot_results.py'])
        files=['coherent_replay.json','theory/synthetic_results.json','verification/synthetic.json','verification/independent_refits.json','verification/compare_results.json','verification/control_audit.json','theory/family.json','verification/family_audit.json']
        numbers=sum(compare(json.loads((HERE/f).read_text()),json.loads((work/f).read_text()),f) for f in files)
        prod=json.loads((work/'production/reproduction_report.json').read_text())
    after={str(p.relative_to(HERE)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    assert before==after,'source mutation'
    result={'status':'PASS','seconds':time.monotonic()-start,'steps':steps,'compared_additional_json_files':len(files),'compared_additional_numbers':numbers,'production_reproduction':prod,'source_hashes_unchanged':True}
    (logdir/'summary.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
