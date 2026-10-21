"""Independent reconstruction and numerical fits for fixed selected cases.

No production fitter, parameter starts, rows, or baseline scores are imported.
The certificate IDs were chosen in the completed earlier geometric study.
"""
from pathlib import Path
import argparse, gzip, hashlib, json, time
import numpy as np
import oracle as O

HERE=Path(__file__).resolve().parent
BRIDGE=HERE.parent


def fit_rows(rows,p=None):
    return O.independent_fit([r['x'] for r in rows],[r['lower'] for r in rows],
                             [r['upper'] for r in rows],p)


def main(output):
    start=time.monotonic()
    sources=[BRIDGE/'production/input_joint.json.gz',BRIDGE/'source_obstruction_certificates.json',BRIDGE/'PLAN.md']
    columns=json.load(gzip.open(sources[0],'rt'))['columns']
    rows=O.original_rows_from_columns(columns)
    assert len(rows)==1560 and len(set(r['id'] for r in rows))==1560
    bysession={s:sorted([r for r in rows if r['session']==s],key=lambda r:r['id'])
               for s in sorted(set(r['session'] for r in rows))}
    baseline={}
    for session,rr in bysession.items():
        assert len(rr)==120
        free=fit_rows(rr);fixed={str(p):fit_rows(rr,p) for p in [-2.,-1.2]}
        assert free['success'] and all(f['success'] for f in fixed.values()),(session,free,fixed)
        for p,f in fixed.items(): assert f['nll']>=free['nll']-1e-7
        baseline[session]={'free':free,'fixed':fixed,'row_ids':[r['id'] for r in rr],
                           'censor_counts':{'finite':int(sum(np.isfinite(r['lower']) and np.isfinite(r['upper']) for r in rr)),
                                            'left':int(sum(np.isneginf(r['lower']) for r in rr)),
                                            'right':int(sum(np.isposinf(r['upper']) for r in rr))}}
        print('independent baseline',session,flush=True)
    certs=json.load(sources[1].open());cases=[]
    for cert in certs['certificates']+certs['triple_certificates']:
        chosen=cert.get('chosen_pair',cert.get('chosen_triple'));drop=set(chosen['ids'])
        session=cert['group'].split('|')[0];rr=bysession[session]
        assert len(drop) in (2,3) and all(r['eligible'] for r in rr if r['id'] in drop)
        left=[r for r in rr if r['id'] not in drop]
        assert len(left)==120-len(drop)
        free=fit_rows(left);fixed=fit_rows(left,cert['p'])
        assert free['success'] and fixed['success'],(cert['group'],cert['p'],free,fixed)
        assert fixed['nll']>=free['nll']-1e-7
        before=baseline[session]
        cases.append({'group':cert['group'],'p':cert['p'],'deleted_ids':sorted(drop),
                      'remaining_ids':[r['id'] for r in left],'free':free,'fixed':fixed,
                      'T':2*(fixed['nll']-free['nll']),
                      'delta_beta':free['params'][2]-before['free']['params'][2],
                      'delta_speed':free['params'][1]-before['free']['params'][1]})
        print('independent case',len(cases),cert['group'],cert['p'],flush=True)
    out={'passed':True,'plan_commit':'aed035e','method':'independent rows/logmass; canonical eta/tau numerical-gradient bounded optimizer; no production parameter starts',
         'n_rows':len(rows),'baseline':baseline,'cases':cases,'n_fits':39+2*len(cases),
         'source_sha256':{str(s):hashlib.sha256(s.read_bytes()).hexdigest() for s in sources},
         'seconds':time.monotonic()-start}
    output.write_text(json.dumps(out,indent=2)+'\n')
    print('PASS independent fits',out['n_fits'],'seconds',out['seconds'],flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,default=HERE/'independent_refits.json')
    main(parser.parse_args().output)
