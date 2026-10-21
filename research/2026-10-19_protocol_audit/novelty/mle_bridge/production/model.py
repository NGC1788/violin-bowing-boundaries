"""Portable original-window interval Gaussian MLE; all optimization stages bounded."""
import numpy as np
from scipy.special import log_ndtr
from scipy.optimize import minimize

BOUNDS=(-9.,3.5)

def extract_rows(columns, labels=None):
    out=[]
    for c in columns:
        f=np.log(np.maximum(np.asarray(c['force'],float),.001))
        h=np.asarray(c['baseline'] if labels is None else labels[c['id']],bool)
        starts=np.flatnonzero(h[:-1]&h[1:])
        if len(starts):
            s=int(starts[0]);lo,hi=(-np.inf,f[0]) if s==0 else (f[s-1],f[s])
        else: lo,hi=f[-1],np.inf
        out.append({k:c[k] for k in ('id','session','group','column','eligible')}|{'lv':float(np.log(c['speed'])),'lb':float(np.log(c['beta'])),'lower':[float(lo),float(hi)]})
    return out

def arrays(rows):
    X=np.asarray([[1.,r['lv'],r['lb']] for r in rows]);lo,hi=np.asarray([r['lower'] for r in rows]).T
    return X,lo,hi

def logmass(a,b):
    a,b=np.asarray(a),np.asarray(b);out=np.empty_like(a,dtype=float)
    left=np.isneginf(a);right=np.isposinf(b);both=left&right
    out[left]=log_ndtr(b[left]);out[right]=log_ndtr(-a[right]);out[both]=0.
    finite=~(left|right);pos=finite&(a>0);neg=finite&~pos
    for mask,top,bottom in ((neg,b,a),(pos,-a,-b)):
        u=log_ndtr(top[mask]);v=log_ndtr(bottom[mask]);out[mask]=u+np.log(-np.expm1(v-u))
    return out

def row_nll(params,rows):
    X,lo,hi=arrays(rows);p=np.asarray(params);s=np.exp(p[3]);return -logmass((lo-X@p[:3])/s,(hi-X@p[:3])/s)

def objective(q,X,lo,hi,off):
    sig=np.exp(q[-1]);mu=X@q[:-1]+off;a=(lo-mu)/sig;b=(hi-mu)/sig;lp=logmass(a,b)
    if not np.isfinite(lp).all():return 1e100,np.zeros_like(q)
    ra=np.exp(-.5*a*a-.5*np.log(2*np.pi)-lp);rb=np.exp(-.5*b*b-.5*np.log(2*np.pi)-lp)
    aa=np.where(np.isfinite(a),a,0.);bb=np.where(np.isfinite(b),b,0.)
    return float(-lp.sum()),np.r_[X.T@((rb-ra)/sig),np.sum(bb*rb-aa*ra)]

def fit(rows,fixed_beta=None,init=None):
    X,lo,hi=arrays(rows)
    return fit_arrays(X,lo,hi,p=fixed_beta,init=init)

def fit_arrays(X,lo,hi,p=None,init=None):
    X,lo,hi=np.asarray(X,float),np.asarray(lo,float),np.asarray(hi,float)
    if X.ndim!=2 or X.shape!=(len(lo),3) or len(hi)!=len(lo):raise ValueError('shape')
    if not np.all(lo<hi):raise ValueError('zero/reversed interval')
    J=[0,1,2] if p is None else [0,1];D=X[:,J]
    if len(lo)<6 or np.linalg.matrix_rank(D)<len(J):raise ValueError('insufficient design')
    center=D[:,1:].mean(0);scale=D[:,1:].std(0)
    XX=np.c_[np.ones(len(lo)),(D[:,1:]-center)/scale];off=np.zeros(len(lo)) if p is None else p*X[:,2]
    if init is None:
        fin=np.isfinite(lo)&np.isfinite(hi)
        yy=np.where(fin,(np.where(np.isfinite(lo),lo,0)+np.where(np.isfinite(hi),hi,0))/2,np.where(np.isfinite(hi),hi-.2,lo+.2))-off
        q=np.linalg.lstsq(XX,yy,rcond=None)[0];q=np.r_[q,np.log(max(.05,np.std(yy-XX@q)))]
    else:
        pp=np.asarray(init,float);coef=pp[J];q=np.r_[coef[0]+center@coef[1:],coef[1:]*scale,pp[3]]
    q[-1]=np.clip(q[-1],*BOUNDS)
    bounds=[(None,None)]*len(J)+[BOUNDS];attempts=1
    opt=minimize(objective,q,args=(XX,lo,hi,off),jac=True,method='L-BFGS-B',bounds=bounds,options={'maxiter':500,'ftol':1e-13,'gtol':1e-7,'maxls':30})
    def accepted(o):return bool(o.success and np.isfinite(o.fun) and np.max(np.abs(objective(o.x,XX,lo,hi,off)[1]))<=2e-3)
    if not accepted(opt):
        attempts+=1
        alt=minimize(lambda z:objective(z,XX,lo,hi,off)[0],opt.x,method='Nelder-Mead',bounds=bounds,options={'maxiter':2000,'xatol':1e-9,'fatol':1e-10})
        # Finish every fallback with bounded analytic-gradient optimization.
        alt=minimize(objective,alt.x,args=(XX,lo,hi,off),jac=True,method='L-BFGS-B',bounds=bounds,options={'maxiter':1000,'ftol':2e-15,'gtol':1e-8,'maxls':50})
        if (accepted(alt) and not accepted(opt)) or alt.fun<opt.fun:opt=alt
    if not accepted(opt):
        attempts+=1
        alt=minimize(objective,opt.x,args=(XX,lo,hi,off),jac=True,method='SLSQP',bounds=bounds,options={'maxiter':1000,'ftol':1e-11})
        if (accepted(alt) and not accepted(opt)) or alt.fun<opt.fun:opt=alt
    q=opt.x;coef=np.r_[q[0]-center@(q[1:-1]/scale),q[1:-1]/scale]
    pp=np.r_[coef,q[-1]] if p is None else np.array([coef[0],coef[1],p,q[-1]])
    grad=float(np.max(np.abs(objective(q,XX,lo,hi,off)[1])));hit=bool(q[-1]<=BOUNDS[0]+1e-6 or q[-1]>=BOUNDS[1]-1e-6)
    return {'params':pp.tolist(),'nll':float(opt.fun),'success':bool(accepted(opt) and not hit),'optimizer_success':bool(opt.success),'grad_max':grad,'scale_hit':hit,'sigma':float(np.exp(q[-1])),'iterations':int(opt.nit),'attempts':attempts,'n':len(lo),'rank':int(np.linalg.matrix_rank(D)),'fixed_beta':p,'message':str(opt.message)}
