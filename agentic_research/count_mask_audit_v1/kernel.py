"""Toy-only two-block count marginalization. No data loader or training code."""
import itertools
import numpy as np
import torch
from scipy.stats import ncx2, chi2

DTYPE=torch.float64


def tensor(x):return torch.as_tensor(x,dtype=DTYPE)


def logits_for(plan,large=None):
    if large is None:
        t=plan['small_frames'];result=np.zeros((2,t,2))
        for h in range(2):
            result[h,0,:]=plan['small_initial_logits'][h]
            for i in range(1,t):
                for prev in range(2):result[h,i,prev]=.35*np.sin((h+1)*(i+1))+(.9 if prev else -.6)
        return result
    result=np.zeros((2,plan['large_frames'],2))
    for h in range(2):
        p=np.asarray(large['transition_probabilities'][h]);result[h,:,:]=np.log(p/(1-p))
        first=large['initial_probabilities'][h];result[h,0,:]=np.log(first/(1-first))
    return result


def dp(logits,blocks,beam=None,fixed_trace=None):
    """State probability is surviving-path mass. Returns counts and trace."""
    if len(blocks)!=2 or sum(blocks)!=len(logits) or min(blocks)<1:
        raise ValueError('two positive blocks matching time are required')
    if logits.shape!=(sum(blocks),2) or not torch.isfinite(logits).all():raise ValueError('finite [time,2] logits required')
    if beam is not None and (isinstance(beam,bool) or not isinstance(beam,int) or beam<1):raise ValueError('positive beam required')
    shape=(blocks[0]+1,blocks[1]+1,2)
    d=torch.zeros(shape,dtype=logits.dtype);d[0,0,0]=1.
    trace=[]
    for t in range(sum(blocks)):
        p=torch.sigmoid(logits[t]);zero=d[:,:,0]*(1-p[0])+d[:,:,1]*(1-p[1])
        one=d[:,:,0]*p[0]+d[:,:,1]*p[1]
        if t<blocks[0]:one=torch.cat((torch.zeros_like(one[:1]),one[:-1]),dim=0)
        else:one=torch.cat((torch.zeros_like(one[:,:1]),one[:,:-1]),dim=1)
        d=torch.stack((zero,one),dim=-1)
        selected=None
        if fixed_trace is not None:selected=torch.tensor(fixed_trace[t],dtype=torch.long)
        elif beam is not None and beam<d.numel():
            selected=torch.argsort(d.detach().reshape(-1),descending=True,stable=True)[:beam]
        if selected is not None:
            mask=torch.zeros(d.numel(),dtype=d.dtype).scatter(0,selected,1.).reshape(shape)
            d=d*mask;trace.append(selected.tolist())
        else:trace.append(list(range(d.numel())))
    return d.sum(-1),trace


def guard_dp(logits,minimum):
    tmax=len(logits);d=torch.zeros((tmax+1,2),dtype=logits.dtype);d[0,0]=1.
    for t in range(tmax):
        p=torch.sigmoid(logits[t]);zero=d[:,0]*(1-p[0])+d[:,1]*(1-p[1])
        one=d[:,0]*p[0]+d[:,1]*p[1]
        one=torch.cat((torch.zeros_like(one[:1]),one[:-1]))
        d=torch.stack((zero,one),-1)
    return d[minimum:].sum()


def numpy_counts(logits,blocks):
    """Independent dictionary recurrence rather than Torch grid/padding."""
    state={(0,0,0):1.}
    p=1/(1+np.exp(-np.asarray(logits)))
    for t in range(sum(blocks)):
        new={}
        for (a,b,old),weight in state.items():
            for s in [0,1]:
                key=(a+(s if t<blocks[0] else 0),b+(s if t>=blocks[0] else 0),s)
                new[key]=new.get(key,0.)+weight*(p[t,old] if s else 1-p[t,old])
        state=new
    out=np.zeros((blocks[0]+1,blocks[1]+1))
    for (a,b,_),weight in state.items():out[a,b]+=weight
    return out


def enumerate_paths(logits,blocks):
    p=1/(1+np.exp(-np.asarray(logits)));out=[]
    for values in itertools.product([0,1],repeat=len(logits)):
        m=np.asarray(values,bool);weight=1.;prev=0
        for t,s in enumerate(m):
            weight*=p[t,prev] if s else 1-p[t,prev];prev=int(s)
        n=(int(m[:blocks[0]].sum()),int(m[blocks[0]:].sum()))
        out.append((m,weight,n))
    return out


def tables(plan,blocks,minimum,*,coarsening=False,count_conditioned=False):
    """Known nc-chi² law, not a learned model; no mask is an inference input."""
    t=sum(blocks);pitch=np.asarray(plan['small_pitch_block_means']);power=np.asarray(plan['small_power_block_means'])
    up=np.asarray(plan['thresholds_pitch']);vp=np.asarray(plan['thresholds_power'])
    result=np.zeros((2,blocks[0]+1,blocks[1]+1,3,3));low=result.copy();high=result.copy()
    a,b=np.indices((blocks[0]+1,blocks[1]+1));n=a+b;valid=n>=minimum;dfn=np.maximum(n-1,1)
    for h in range(2):
        mu=np.repeat(pitch[h],blocks);dm=np.repeat(power[h],blocks)
        original=mu+.12*np.sin(1.7*(np.arange(t)+1)+h)
        errors=np.array([np.abs(original[:blocks[0]]-mu[:blocks[0]]).max(),
                         np.abs(original[blocks[0]:]-mu[blocks[0]:]).max()])
        ps,ds=plan['scales'][h];difference=np.full_like(n,pitch[h,0]-pitch[h,1],dtype=float)
        ps=np.full_like(n,ps,dtype=float);ds=np.full_like(n,ds,dtype=float);dscale=np.ones_like(n,dtype=float)
        if count_conditioned:
            difference+=.5*(a-b)/t;ps*=1+.3*n/t;ds*=1+.15*n/t;dscale+=.2*(a-b)/t
        lam=a*b/np.maximum(n,1)*difference**2/ps**2
        dl=np.sum((dm-dm.mean())**2)*dscale**2/ds**2
        pf=ncx2.cdf(dfn[:,:,None]*up**2/ps[:,:,None]**2,dfn[:,:,None],lam[:,:,None])
        df=ncx2.cdf((t-1)*vp**2/ds[:,:,None]**2,t-1,dl[:,:,None])
        result[h]=pf[:,:,:,None]*df[:,:,None,:]*valid[:,:,None,None]
        if coarsening:
            delta=np.sqrt((a*errors[0]**2+b*errors[1]**2)/dfn)
            lo=up-delta[:,:,None];hi=up+delta[:,:,None]
            lf=ncx2.cdf(dfn[:,:,None]*np.maximum(lo,0)**2/ps[:,:,None]**2,dfn[:,:,None],lam[:,:,None]);lf[lo<0]=0.
            hf=ncx2.cdf(dfn[:,:,None]*hi**2/ps[:,:,None]**2,dfn[:,:,None],lam[:,:,None])
            low[h]=lf[:,:,:,None]*df[:,:,None,:]*valid[:,:,None,None]
            high[h]=hf[:,:,:,None]*df[:,:,None,:]*valid[:,:,None,None]
        else:low[h]=result[h];high[h]=result[h]
    return result,low,high


def direct_path_query(plan,logits,blocks,minimum,*,perturbed=False,count_conditioned=False):
    """Enumerate binary masks, compute noncentrality from selected full means."""
    total=np.zeros((3,3));up=np.asarray(plan['thresholds_pitch']);vp=np.asarray(plan['thresholds_power']);t=sum(blocks)
    counts=np.zeros((2,blocks[0]+1,blocks[1]+1))
    for h in range(2):
        mu=np.repeat(plan['small_pitch_block_means'][h],blocks)
        dm=np.repeat(plan['small_power_block_means'][h],blocks)
        if perturbed:mu=mu+.12*np.sin(1.7*(np.arange(t)+1)+h)
        for mask,w,(a,b) in enumerate_paths(logits[h],blocks):
            counts[h,a,b]+=w;n=a+b
            if n<minimum:continue
            observed=mu[mask].copy();dd=dm.copy();ps,ds=plan['scales'][h]
            if count_conditioned:
                adjustment=np.repeat(np.array([.3,-.2])*(a-b)/t,blocks)
                observed+=adjustment[mask];ps*=1+.3*n/t
                dd*=1+.2*(a-b)/t;ds*=1+.15*n/t
            pl=np.sum((observed-observed.mean())**2)/ps**2
            dl=np.sum((dd-dd.mean())**2)/ds**2
            pf=ncx2.cdf((n-1)*up**2/ps**2,n-1,pl)
            df=ncx2.cdf((t-1)*vp**2/ds**2,t-1,dl)
            total+=plan['mixture'][h]*w*pf[:,None]*df[None,:]
    return total,counts


def inference(logits,blocks,minimum,table,mixture,beam=None,fixed_traces=None):
    q=torch.zeros((3,3),dtype=DTYPE);guard=tensor(0.);mass=tensor(0.);traces=[];weights=[]
    a,b=np.indices((blocks[0]+1,blocks[1]+1));valid=tensor(a+b>=minimum)
    for h,pi in enumerate(mixture):
        w,trace=dp(logits[h],blocks,beam,None if fixed_traces is None else fixed_traces[h])
        q=q+pi*(w[:,:,None,None]*tensor(table[h])).sum((0,1))
        guard=guard+pi*guard_dp(logits[h],minimum);mass=mass+pi*(w*valid).sum()
        traces.append(trace);weights.append(w)
    return {'q':q,'guard':guard,'retained_valid':mass,'residual':guard-mass,'traces':traces,'weights':weights}


def cells(q):
    q=np.asarray(q);padded=np.pad(q,((1,0),(1,0)))
    return np.diff(np.diff(padded,axis=0),axis=1)


def correlation_counterexample(plan):
    c=plan['correlation_negative_control'];r=c['rho'];s=c['sigma'];u=c['std_threshold']
    cdfs=[float(chi2.cdf(u*u/(s*s*(1-r**d)),1)) for d in c['pair_distances']]
    return {'cdf':cdfs,'abs_gap':abs(cdfs[0]-cdfs[1]),'law':'S²=sigma²(1-rho^distance) chi²_1; equal count2 does not imply equal law'}
