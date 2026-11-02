"""Conventional baselines and representation/supervision probes, not novelty."""
import time
import numpy as np
import torch
from torch import nn
from sklearn.ensemble import HistGradientBoostingClassifier
from threadpoolctl import threadpool_limits
from ..violin_raw_v1.models import patterns,participant_weights


class TCN(nn.Module):
    def __init__(self,out,width=16):
        super().__init__();layers=[];channels=4
        for dilation in [1,2,4]:
            layers.extend([nn.Conv1d(channels,width,3,padding=dilation,dilation=dilation),nn.GELU()]);channels=width
        self.encoder=nn.Sequential(*layers);self.head=nn.Linear(width*2,out)
    def forward(self,x):
        z=self.encoder(x.transpose(1,2))
        return self.head(torch.cat([z.mean(2),z[:,:,-1]],1))


def mdn_distribution(raw,thresholds,lower):
    weights=torch.softmax(raw[:,:2],1);mu=raw[:,2:6].reshape(-1,2,2)
    sigma=torch.nn.functional.softplus(raw[:,6:10].reshape(-1,2,2))+.1
    guard=torch.sigmoid(raw[:,10])
    log_normalizer=torch.special.log_ndtr(-(lower[None,None]-mu)/sigma)
    z=(thresholds[None,None]-mu[:,:,:,None])/sigma[:,:,:,None]
    log_survival_ratio=torch.special.log_ndtr(-z)-log_normalizer[:,:,:,None]
    cdf=(-torch.expm1(log_survival_ratio)).clamp(0,1)
    joint=(cdf[:,:,0,:,None]*cdf[:,:,1,None,:]).reshape(-1,2,9)
    probability=guard[:,None]*(weights[:,:,None]*joint).sum(1)
    return probability,weights,mu,sigma,log_normalizer


def predict(method,train_seq,train_x,y,continuous,guard,people,eval_seq,eval_x,fold,config,clock_start,time_limit):
    if method=='summary_hgb':
        out=[];weights=participant_weights(people)
        for j in range(9):
            if len(np.unique(y[:,j]))==1:out.append(np.full(len(eval_x),y[0,j]));continue
            m=HistGradientBoostingClassifier(max_iter=config['hgb_iterations'],early_stopping=False,
                max_leaf_nodes=15,min_samples_leaf=20,l2_regularization=.5,learning_rate=.07,random_state=0)
            with threadpool_limits(limits=1):m.fit(train_x,y[:,j],sample_weight=weights);out.append(m.predict_proba(eval_x)[:,1])
            if time.monotonic()-clock_start>time_limit:raise TimeoutError('job time limit')
        return np.column_stack(out)
    torch.set_num_threads(1);torch.manual_seed(config['seed']);torch.use_deterministic_algorithms(True)
    continuous_head=method=='ordered_tcn_mdn';model=TCN(11 if continuous_head else 10,config['tcn_width'])
    optimizer=torch.optim.AdamW(model.parameters(),lr=config['learning_rate'],weight_decay=config['weight_decay'])
    x=torch.tensor(train_seq,dtype=torch.float32);yt=torch.tensor(y,dtype=torch.float32)
    g=torch.tensor(guard,dtype=torch.float32);w=torch.tensor(participant_weights(people),dtype=torch.float32)
    # Invalid pitch observations are never used in continuous likelihood.
    c=np.zeros_like(continuous);c[guard]=(np.log1p(continuous[guard])-fold['continuous_mean'])/fold['continuous_scale']
    ct=torch.tensor(c,dtype=torch.float32);bank=torch.tensor(patterns(),dtype=torch.float32)
    thresholds=torch.tensor((np.log1p(fold['thresholds'])-fold['continuous_mean'][:,None])/fold['continuous_scale'][:,None],dtype=torch.float32)
    lower=torch.tensor(-fold['continuous_mean']/fold['continuous_scale'],dtype=torch.float32)
    generator=torch.Generator().manual_seed(1831)
    for epoch in range(config['epochs']):
        for batch in torch.randperm(len(x),generator=generator).split(config['batch_size']):
            raw=model(x[batch])
            if continuous_head:
                p,mixture,mu,sigma,log_normalizer=mdn_distribution(raw,thresholds,lower)
                loss=nn.functional.binary_cross_entropy(torch.clamp(p,1e-6,1-1e-6),yt[batch],reduction='none').mean(1)
                logn=-.5*(((ct[batch,None]-mu)/sigma)**2+2*torch.log(sigma)+np.log(2*np.pi)).sum(2)-log_normalizer.sum(2)
                nll=-torch.logsumexp(torch.log(mixture.clamp_min(1e-12))+logn,1)
                guard_loss=nn.functional.binary_cross_entropy_with_logits(raw[:,10],g[batch],reduction='none')
                loss=loss+config['mdn_nll_weight']*(guard_loss+g[batch]*nll)
            else:
                p=torch.softmax(raw,1)@bank
                loss=nn.functional.binary_cross_entropy(p.clamp(1e-6,1-1e-6),yt[batch],reduction='none').mean(1)
            objective=(w[batch]*loss).mean()
            if not torch.isfinite(objective):raise ArithmeticError('nonfinite training loss')
            optimizer.zero_grad();objective.backward();nn.utils.clip_grad_norm_(model.parameters(),5.);optimizer.step()
            if time.monotonic()-clock_start>time_limit:raise TimeoutError('job time limit')
    model.eval()
    with torch.inference_mode():
        raw=model(torch.tensor(eval_seq,dtype=torch.float32))
        if continuous_head:
            result=mdn_distribution(raw.double(),thresholds.double(),lower.double())[0].numpy()
        else:
            mass=torch.softmax(raw,1).numpy().astype(float);mass/=mass.sum(1,keepdims=True)
            result=mass@patterns()
    if result.shape!=(len(eval_seq),9) or not np.isfinite(result).all():raise ArithmeticError('invalid forecast')
    if np.any(result < -1e-10) or np.any(result > 1+1e-10):raise ArithmeticError('probability outside numerical tolerance')
    return np.clip(result,0,1)
