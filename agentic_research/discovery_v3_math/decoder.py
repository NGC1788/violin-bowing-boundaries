"""Restricted shared-factor, mask-mixture variance decoder; classical formulas.

No data access or model fitting. CPU float64, first-order gradients only.
"""
from functools import lru_cache
import numpy as np
import torch
from scipy.special import ndtr,ndtri
from scipy.stats import ncx2,qmc

N=46
DTYPE=torch.float64


class _NCCDF(torch.autograd.Function):
    @staticmethod
    def forward(ctx,x,nc,df):
        if x.device.type!='cpu' or nc.device.type!='cpu' or x.dtype!=DTYPE or nc.dtype!=DTYPE:
            raise ValueError('CPUfloat64 only')
        if (not bool(torch.isfinite(x).all()) or not bool(torch.isfinite(nc).all()) or
            bool(torch.any(x<0)) or bool(torch.any(nc<0)) or df<=0):raise ValueError('invalid ncchi-square parameters')
        value=torch.from_numpy(np.asarray(ncx2.cdf(x.detach().numpy(),df,nc.detach().numpy())))
        ctx.save_for_backward(x,nc,value);ctx.df=df
        return value
    @staticmethod
    def backward(ctx,grad):
        x,nc,value=ctx.saved_tensors
        xx,nn=x.detach().numpy(),nc.detach().numpy()
        dx=torch.from_numpy(np.asarray(ncx2.pdf(xx,ctx.df,nn)))
        dn=.5*(torch.from_numpy(np.asarray(ncx2.cdf(xx,ctx.df+2,nn)))-value)
        return grad*dx,grad*dn,None


def nc_cdf(x,nc,df):
    x,nc=torch.broadcast_tensors(x,nc)
    return _NCCDF.apply(x,nc,int(df))


@lru_cache(maxsize=8)
def quadrature(order):
    x,w=np.polynomial.legendre.leggauss(order);z=8*x
    weights=8*w*np.exp(-.5*z*z)/np.sqrt(2*np.pi)
    total=float(weights.sum());scale=min(1.,1/total)
    return z,weights*scale,{'raw_mass':total,'roundoff_scale':scale,'forward_normal_tail_bound':float(2*ndtr(-8.))}


def tensor(a):return torch.as_tensor(a,dtype=DTYPE)


def fixture(name):
    t=np.linspace(0,1,N);h=(t-t.mean())/t.std(ddof=1)
    full=np.ones(N,bool)
    c={'mean':np.zeros((2,N)),'loading':np.zeros((2,N)),'sigma':np.array([.7,1.1]),
       'masks':full[None],'logits':np.zeros(1),'slopes':np.zeros(1),
       'thresholds':np.array([[.4,.7,1.],[.6,1.1,1.6]])}
    if name=='drift':
        c.update(mean=np.array([1.5*t,.8*np.sin(2*np.pi*t)]),sigma=np.array([.3,.4]),thresholds=np.tile([.4,.8,1.2],(2,1)))
    elif name in ['shared','invariance']:
        c.update(loading=np.array([.8*h,.8*h]),sigma=np.array([.5,.5]),thresholds=np.tile([.4,.8,1.2],(2,1)))
        if name=='invariance':
            c['mean']+=np.array([13.,-7.])[:,None];c['loading']+=np.array([3.,-2.])[:,None]
    elif name=='mask':
        c.update(mean=np.array([4*t,np.zeros(N)]),loading=np.array([.15*h,.3*h]),sigma=np.array([.35,.4]),
                 masks=np.array([full,np.arange(N)<23,np.arange(N)<22]),
                 logits=np.log([.4,.4,.2]),slopes=np.array([.3,-.3,0.]),
                 thresholds=np.array([[.6,.9,1.3],[.2,.3,.5]]))
    elif name not in ['central','guard22','guard23']:raise ValueError(name)
    if name in ['guard22','guard23']:c['masks']=(np.arange(N)<int(name[-2:]))[None]
    return c


def tensor_case(c):
    return {k:np.asarray(v,bool) if k=='masks' else tensor(v) for k,v in c.items()}


def decode(c,order=256):
    """Integrate a declared mask distribution, never an observed future mask."""
    z0,w0,_=quadrature(order);z,w=tensor(z0),tensor(w0)
    pi=torch.softmax(c['logits'][None]+z[:,None]*c['slopes'][None],1)
    mu=c['mean'][None]+z[:,None,None]*c['loading'][None]
    noise=c['sigma'];thresholds=c['thresholds']
    if bool(torch.any(noise<=0)) or bool(torch.any(thresholds<0)):raise ValueError('invalid path parameters')
    centered=mu[:,1]-mu[:,1].mean(1,keepdim=True)
    lam=centered.square().sum(1,keepdim=True)/noise[1].square()
    x=(N-1)*thresholds[1:2].square()/noise[1].square()
    fb=nc_cdf(x,lam,N-1)
    joint=torch.zeros((3,3),dtype=DTYPE);marg=torch.zeros((2,3),dtype=DTYPE);valid=torch.zeros((),dtype=DTYPE)
    for j,mask in enumerate(c['masks']):
        k=int(mask.sum())
        if k<23:continue
        selected=mu[:,0,mask];centered=selected-selected.mean(1,keepdim=True)
        lam=centered.square().sum(1,keepdim=True)/noise[0].square()
        x=(k-1)*thresholds[:1].square()/noise[0].square()
        fa=nc_cdf(x,lam,k-1);weight=w*pi[:,j]
        joint=joint+(weight[:,None,None]*fa[:,:,None]*fb[:,None,:]).sum(0)
        marg=marg+torch.stack([(weight[:,None]*fa).sum(0),(weight[:,None]*fb).sum(0)])
        valid=valid+weight.sum()
    return {'joint':joint,'marginals_with_guard':marg,'valid_probability':valid}


def independent_analytic(c):
    if np.any(c['loading']) or len(c['masks'])!=1 or not c['masks'][0].all():raise ValueError('not independent fixture')
    probabilities=[]
    for j in range(2):
        lam=np.square(c['mean'][j]-c['mean'][j].mean()).sum()/c['sigma'][j]**2
        x=(N-1)*np.square(c['thresholds'][j])/c['sigma'][j]**2
        probabilities.append(ncx2.cdf(x,N-1,lam))
    return np.outer(*probabilities)


def path_reference(c,power,seed):
    """Independent direct-path QMC implementation; no analytic variance CDF."""
    u=qmc.Sobol(1+2*N,scramble=True,seed=seed).random_base2(power)
    normal=ndtri(np.clip(u,1e-12,1-1e-12));z=normal[:,0]
    path=c['mean'][None]+z[:,None,None]*c['loading'][None]+c['sigma'][None,:,None]*normal[:,1:].reshape(-1,2,N)
    db=np.std(path[:,1],axis=1,ddof=1);pb=db[:,None]<=c['thresholds'][1]
    logits=c['logits'][None]+z[:,None]*c['slopes'][None];pi=np.exp(logits-logits.max(1,keepdims=True));pi/=pi.sum(1,keepdims=True)
    joint=np.zeros((3,3));valid=0.
    for j,mask in enumerate(c['masks']):
        if mask.sum()<23:continue
        pitch=np.std(path[:,0,mask],axis=1,ddof=1);pa=pitch[:,None]<=c['thresholds'][0]
        joint+=(pa.T*pi[:,j])@pb/len(path);valid+=pi[:,j].mean()
    return {'joint':joint,'valid_probability':float(valid),'paths':len(path),'seed':seed,'power':power}


def cell_masses(joint):
    cells=np.diff(np.diff(np.pad(np.asarray(joint),((1,0),(1,0))),axis=0),axis=1)
    return np.r_[1-np.asarray(joint)[-1,-1],cells.ravel()]


def gradient_case(theta):
    base=tensor_case(fixture('mask'));t=tensor(np.linspace(0,1,N));h=(t-t.mean())/t.std(correction=1)
    base['mean']=torch.stack([3*theta[0]*t,.2*torch.sin(2*torch.pi*t)])
    base['loading']=torch.stack([.25*h,.35*h]);base['sigma']=torch.stack([torch.exp(theta[1]),tensor(.4)])
    base['logits']=torch.stack([theta[2],tensor(0.),tensor(-1.)])
    return base


def gradient_objective(theta,order=256):
    q=decode(gradient_case(theta),order)['joint']
    return (q*torch.arange(1,10,dtype=DTYPE).reshape(3,3)/45).sum()
