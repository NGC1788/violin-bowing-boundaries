"""Post-hoc endpoint locality diagnostic; plan commit4172da3; no exponent rescoring."""
from pathlib import Path
import sys,json,hashlib
import numpy as np
ROOT=Path.home()/'violin-rne';HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'novelty/tilt_principle'))
import tilt as T
LS=T.LS

def ends(h):
    starts=np.flatnonzero(h[:-1]&h[1:])
    if not len(starts): return None
    return int(starts[0]),int(starts[-1]+1)

def main():
    out={'post_hoc':True,'sessions':{},'summary':{}}
    for sess,st in LS.STEM.items():
        if sess=='D_T2':continue
        stem,rep=st if isinstance(st,tuple) else(st,None)
        raw,fh=T.read2(stem,rep);d=LS.build(raw,T.BASE);h=d['H']
        feats={'amplitude':(np.array([r[3] for r in raw[0]]),T.A_LOW,False),
               'flyback_height':(fh,T.A_LOW,False),
               'periodicity':(np.array([r[4] for r in raw[0]]),[-.6,-.4,-.2,0.,.2,.4,.6],False),
               'second_flyback_ratio':(np.array([r[7] for r in raw[0]]),[-1.,-.5,0.,.5,1.],True)}
        cols=[]
        for key,idx in LS.C.columns_of(d,.2).items():
            ii=np.array(sorted(idx,key=lambda i:d['force'][i]));ee=ends(h[ii]);cols.append((ii,ee))
        rec={}
        for name,(x,grid,upper) in feats.items():
            rec[name]={}
            for a in grid:
                hv=T.tilted(x,d['beta'],d['folder'],h,a,.15,upper)
                n=lost=lh=uh=lc=uc=0
                for ii,ee in cols:
                    if ee is None:continue
                    n+=1;s,e=ee;now=ends(hv[ii]);lh+=int(not np.all(hv[ii[s:s+2]]));uh+=int(not np.all(hv[ii[e-1:e+1]]))
                    lost+=int(now is None);lc+=int(now is None or now[0]!=s);uc+=int(now is None or now[1]!=e)
                rec[name][str(a)]={'base_banded_columns':n,'lower_run_hit':lh,'upper_run_hit':uh,
                                  'lower_endpoint_changed':lc,'upper_endpoint_changed':uc,'band_lost':lost,
                                  'n_H':int(h.sum()),'removed_H':int((h&~hv).sum())}
        out['sessions'][sess]=rec;print(sess,flush=True)
    for feat in feats:
        rows=[r for d in out['sessions'].values() for r in d[feat].values()]
        den=sum(r['base_banded_columns'] for r in rows)
        out['summary'][feat]={'n_session_tilt_cells':len(rows),'denominator_repeated_columns':den,
           **{k:sum(r[k] for r in rows)/den for k in ('lower_run_hit','upper_run_hit','lower_endpoint_changed','upper_endpoint_changed','band_lost')}}
    out['source_sha256']={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),ROOT/'novelty/tilt_principle/tilt.py']}
    (HERE/'results.json').write_text(json.dumps(out,indent=2));print(json.dumps(out['summary'],indent=2),flush=True)
if __name__=='__main__':main()
