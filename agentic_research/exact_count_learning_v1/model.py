"""Prefix-only neural forecasts with common generative supervision.

CPU float64. Pure model functions; no data loading, optimizer or experiment run.
Classical nc-chi-square derivatives are imported from the frozen prototype.
"""
import math
import numpy as np
import torch
from torch import nn
from ..discovery_v3_math.decoder import nc_cdf

DTYPE=torch.float64


def normal_cdf(x):return .5*torch.erfc(-x/math.sqrt(2))


def bank():
    rows=[np.zeros(9)]
    for a in range(3):
        for b in range(3):rows.append(np.array([i>=a and j>=b for i in range(3) for j in range(3)],float))
    return torch.tensor(np.unique(rows,axis=0),dtype=DTYPE)


def count_distribution(initial,transitions,blocks):
    """Exact batched count/end-bit DP; no observed mask or pruning."""
    if len(blocks)!=2 or min(blocks)<1:raise ValueError('two positive future blocks required')
    if initial.ndim!=2 or transitions.shape!=(*initial.shape,2,2):raise ValueError('bad mask law shape')
    d=initial.new_zeros((*initial.shape,blocks[0]+1,blocks[1]+1,2));d[...,0,0,0]=1.
    for t in range(sum(blocks)):
        b=int(t>=blocks[0])
        p0=initial if t==0 else transitions[:,:,b,0]
        p1=initial if t==0 else transitions[:,:,b,1]
        p0,p1=p0[:,:,None,None],p1[:,:,None,None]
        zero=d[...,0]*(1-p0)+d[...,1]*(1-p1);one=d[...,0]*p0+d[...,1]*p1
        if b==0:one=torch.cat((torch.zeros_like(one[:,:,:1,:]),one[:,:,:-1,:]),2)
        else:one=torch.cat((torch.zeros_like(one[:,:,:,:1]),one[:,:,:,:-1]),3)
        d=torch.stack((zero,one),-1)
    return d.sum(-1).flatten(2)


def guard_probability(initial,transitions,blocks,minimum):
    """Cheaper total-count DP for a direct-MDN's predicted validity."""
    tmax=sum(blocks);d=initial.new_zeros((*initial.shape,tmax+1,2));d[...,0,0]=1.
    for t in range(tmax):
        b=int(t>=blocks[0]);p0=initial if t==0 else transitions[:,:,b,0];p1=initial if t==0 else transitions[:,:,b,1]
        p0,p1=p0[:,:,None],p1[:,:,None]
        zero=d[...,0]*(1-p0)+d[...,1]*(1-p1);one=d[...,0]*p0+d[...,1]*p1
        one=torch.cat((torch.zeros_like(one[:,:,:1]),one[:,:,:-1]),2)
        d=torch.stack((zero,one),-1)
    return d[:,:,minimum:,:].sum((2,3))


def positive_nc_cdf(x,nc,df):
    # Avoid the df1 PDF singularity at exactly0 in generic tiny functional tests.
    safe=torch.where(x>0,x,torch.full_like(x,1e-100))
    value=nc_cdf(safe,nc,df)
    return torch.where(x>0,value,torch.zeros_like(value))


def approximate_cdf(x,nc,df):
    """Truncated normal approximation to chi-square using its first2moments."""
    mean=df+nc;sd=torch.sqrt(2*(df+2*nc));bottom=normal_cdf(-mean/sd)
    return ((normal_cdf((x-mean)/sd)-bottom)/(1-bottom)).clamp(0,1)


def parameterize(raw,config):
    h=config['regimes'];blocks=config['future_blocks'];r=raw[:,h:].reshape(-1,h,9)
    low,high=config['scale_range_normalized_units'];eps=config['mask_probability_epsilon']
    delta=config['contrast_bound_normalized_units']*torch.tanh(r[:,:,5:7])
    means=torch.stack((delta,-delta*blocks[0]/blocks[1]),-1)
    return {'mixture':torch.softmax(raw[:,:h],1),'log_mixture':torch.log_softmax(raw[:,:h],1),
            'initial':eps+(1-2*eps)*torch.sigmoid(r[:,:,0]),
            'transitions':eps+(1-2*eps)*torch.sigmoid(r[:,:,1:5].reshape(-1,h,2,2)),
            'means':means,'sigma':low+(high-low)*torch.sigmoid(r[:,:,7:9])}


def count_queries(p,thresholds,blocks,minimum,*,moment=False):
    if thresholds.shape!=(2,3) or not bool(torch.isfinite(thresholds).all()) or bool((thresholds<0).any()):
        raise ValueError('finite nonnegative2x3 thresholds required')
    w=count_distribution(p['initial'],p['transitions'],blocks)
    a,b=torch.meshgrid(torch.arange(blocks[0]+1),torch.arange(blocks[1]+1),indexing='ij')
    a,b=a.flatten(),b.flatten();n=a+b;valid=n>=minimum;a,b,n=a[valid],b[valid],n[valid]
    w=w[:,:,valid];mu=p['means'];sigma=p['sigma'];sp=sigma[:,:,0,None];sd=sigma[:,:,1,None]
    factor=a.to(DTYPE)*b.to(DTYPE)/n.to(DTYPE)
    lam=factor[None,None,:]*(mu[:,:,0,0]-mu[:,:,0,1])[:,:,None].square()/sp.square()
    xx=(n-1)[None,None,:,None]*thresholds[0].square()[None,None,None,:]/sp[:,:,:,None].square()
    lam=lam[:,:,:,None].expand_as(xx)
    if moment:fp=approximate_cdf(xx,lam,(n-1)[None,None,:,None])
    else:
        pieces=[];order=[]
        for count in torch.unique(n,sorted=True).tolist():
            index=torch.nonzero(n==count,as_tuple=True)[0];order.append(index)
            pieces.append(positive_nc_cdf(xx[:,:,index,:],lam[:,:,index,:],count-1))
        restore=torch.argsort(torch.cat(order));fp=torch.cat(pieces,2)[:,:,restore,:]
    total=sum(blocks);ld=(blocks[0]*blocks[1]/total)*(mu[:,:,1,0]-mu[:,:,1,1]).square()/sigma[:,:,1].square()
    xd=(total-1)*thresholds[1].square()[None,None,:]/sd.square()
    fd=approximate_cdf(xd,ld[:,:,None],total-1) if moment else positive_nc_cdf(xd,ld[:,:,None],total-1)
    return torch.einsum('bh,bhc,bhci,bhj->bij',p['mixture'],w,fp,fd).flatten(1)


def mdn_queries(raw,p,thresholds,blocks,minimum,config):
    h=config['regimes'];r=raw.reshape(-1,h,5)
    weights=torch.softmax(r[:,:,0],1);means=config['mdn_log_std_location_bound']*torch.tanh(r[:,:,1:3])
    lo,hi=config['mdn_log_std_scale_range'];scale=lo+(hi-lo)*torch.sigmoid(r[:,:,3:5])
    logt=torch.log(thresholds.clamp_min(1e-100))
    f=normal_cdf((logt[None,None]-means[:,:,:,None])/scale[:,:,:,None])
    f=torch.where(thresholds[None,None]>0,f,torch.zeros_like(f))
    cdf=torch.einsum('bh,bhi,bhj->bij',weights,f[:,:,0],f[:,:,1]).flatten(1)
    g=(guard_probability(p['initial'],p['transitions'],blocks,minimum)*p['mixture']).sum(1)
    return g[:,None]*cdf


def auxiliary_nll(p,pitch,power,voiced,blocks):
    """Observed future labels only; no future information is sent to forward()."""
    total=sum(blocks);batch=len(p['mixture'])
    if pitch.shape!=(batch,total) or power.shape!=pitch.shape or voiced.shape!=pitch.shape or voiced.dtype!=torch.bool:
        raise ValueError('future label shape/dtype mismatch')
    if not bool(torch.isfinite(power).all()) or not bool(torch.isfinite(pitch[voiced]).all()):
        raise ValueError('observed future values must be finite')
    pitch=torch.where(voiced,pitch,torch.zeros_like(pitch))
    block=torch.tensor([0]*blocks[0]+[1]*blocks[1],dtype=torch.long)
    means=p['means'][:,:,:,block];scale=p['sigma'][:,:,:,None]
    y=torch.stack((pitch,power),1)[:,None]
    # Normalized density in orthonormal contrast coordinates, not an iid
    # product density for already centered frames. Degrees of freedom matter.
    residual=y-means
    weights=torch.stack((voiced,torch.ones_like(voiced)),1)[:,None].to(DTYPE)
    n=weights.sum(-1)
    centered=residual-(residual*weights).sum(-1,keepdim=True)/n.clamp_min(1)[:,:,:,None]
    energy=(centered.square()*weights).sum(-1)
    degrees=(n-1).clamp_min(0)
    channel=-.5*energy/p['sigma'].square()-degrees*(torch.log(p['sigma'])+.5*math.log(2*math.pi))
    path=channel.sum(2)
    init=p['initial'];m=voiced[:,None,0]
    mask=torch.where(m,torch.log(init),torch.log1p(-init))
    prob=p['transitions'][:,:,block[1:],:]
    prev=voiced[:,None,:-1,None].long().expand(-1,len(init[0]),-1,1)
    prob=torch.gather(prob,3,prev).squeeze(3)
    mask=mask+torch.where(voiced[:,None,1:],torch.log(prob),torch.log1p(-prob)).sum(2)
    return -torch.logsumexp(p['log_mixture']+mask+path,1)/total


class Forecaster(nn.Module):
    def __init__(self,method,config):
        super().__init__()
        if method not in config['methods']:raise ValueError('undeclared method')
        self.method,self.config=method,config
        width=config['hidden_width'];inputs=config['input_dimension'];h=config['regimes']
        self.encoder=nn.Sequential(nn.Linear(inputs,width),nn.GELU(),nn.Linear(width,width),nn.GELU())
        self.generator=nn.Linear(width,h+h*9)
        self.free_head=nn.Linear(width,10 if method=='joint_aux' else h*5) if method in ['joint_aux','mdn_aux'] else None
        self.register_buffer('pattern_bank',bank());self.double()

    def forward(self,prefix,thresholds):
        if prefix.ndim!=2 or prefix.shape[1]!=self.config['input_dimension'] or not bool(torch.isfinite(prefix).all()):
            raise ValueError('finite prefix-only feature matrix required')
        hidden=self.encoder(prefix);p=parameterize(self.generator(hidden),self.config)
        blocks=self.config['future_blocks'];minimum=self.config['minimum_voiced']
        if self.method=='joint_aux':q=torch.softmax(self.free_head(hidden),1)@self.pattern_bank
        elif self.method=='mdn_aux':q=mdn_queries(self.free_head(hidden),p,thresholds,blocks,minimum,self.config)
        else:q=count_queries(p,thresholds,blocks,minimum,moment=self.method=='moment_count')
        return q,p

    def loss_rows(self,prefix,thresholds,event_labels,pitch,power,voiced):
        q,p=self(prefix,thresholds);eps=self.config['probability_clipping']
        bce=nn.functional.binary_cross_entropy(q.clamp(eps,1-eps),event_labels,reduction='none').mean(1)
        aux=auxiliary_nll(p,pitch,power,voiced,self.config['future_blocks'])
        return bce+self.config['auxiliary_weight']*aux,{'bce':bce,'auxiliary':aux,'q':q}
