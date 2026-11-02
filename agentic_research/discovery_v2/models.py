import time
import numpy as np
import torch
from torch import nn
from ..discovery_v1.models import mdn_distribution
from ..violin_raw_v1.models import patterns,participant_weights


class MomentModel(nn.Module):
    def __init__(self,attention=False,delta=False):
        super().__init__();self.attention=attention;self.delta=delta
        self.query=nn.Sequential(nn.Linear(73,32),nn.GELU(),nn.Linear(32,32),nn.GELU())
        if attention:
            self.token=nn.Linear(13,32);self.cross=nn.MultiheadAttention(32,2,batch_first=True)
            t=torch.linspace(0,1,96);pos=[t]
            for f in [1,2,4,8]:pos.extend([torch.sin(2*torch.pi*f*t),torch.cos(2*torch.pi*f*t)])
            self.register_buffer('position',torch.stack(pos,1))
        self.norm=nn.LayerNorm(32);self.head=nn.Linear(32,11 if delta else 10)
    def forward(self,query,seq,anchor):
        h=self.query(query)
        if self.attention:
            position=self.position[None].expand(len(seq),-1,-1)
            context=self.token(torch.cat([seq,position],2))
            attended,_=self.cross(h[:,None],context,context,need_weights=False)
            h=h+attended[:,0]
        raw=self.head(self.norm(h))
        if self.delta:raw=torch.cat([raw[:,:2],raw[:,2:6]+anchor.repeat(1,2),raw[:,6:]],1)
        return raw


def train_predict(method,q,s,a,y,target,guard,people,qe,se,ae,params,config,start,limit):
    torch.set_num_threads(1);torch.manual_seed(config['seed']);torch.use_deterministic_algorithms(True)
    delta=method.endswith('_delta');model=MomentModel('attention' in method,delta)
    optimizer=torch.optim.AdamW(model.parameters(),lr=config['learning_rate'],weight_decay=config['weight_decay'])
    qt,st,at=[torch.tensor(v,dtype=torch.float32) for v in [q,s,a]]
    yt=torch.tensor(y,dtype=torch.float32);g=torch.tensor(guard,dtype=torch.float32)
    w=torch.tensor(participant_weights(people),dtype=torch.float32)
    c=np.zeros_like(target);c[guard]=(np.log1p(target[guard])-params['continuous_mean'])/params['continuous_scale']
    ct=torch.tensor(c,dtype=torch.float32)
    thresholds=torch.tensor((np.log1p(params['thresholds'])-params['continuous_mean'][:,None])/params['continuous_scale'][:,None],dtype=torch.float32)
    lower=torch.tensor(-params['continuous_mean']/params['continuous_scale'],dtype=torch.float32)
    bank=torch.tensor(patterns(),dtype=torch.float32);gen=torch.Generator().manual_seed(1831);trace=[]
    def probability(raw):return mdn_distribution(raw,thresholds,lower)[0] if delta else torch.softmax(raw,1)@bank
    for epoch in range(config['epochs']):
        for batch in torch.randperm(len(qt),generator=gen).split(config['batch_size']):
            raw=model(qt[batch],st[batch],at[batch]);p=probability(raw)
            loss=nn.functional.binary_cross_entropy(p.clamp(1e-6,1-1e-6),yt[batch],reduction='none').mean(1)
            if delta:
                _,mix,mu,sigma,logz=mdn_distribution(raw,thresholds,lower)
                logn=-.5*(((ct[batch,None]-mu)/sigma)**2+2*torch.log(sigma)+np.log(2*np.pi)).sum(2)-logz.sum(2)
                nll=-torch.logsumexp(torch.log(mix.clamp_min(1e-12))+logn,1)
                gl=nn.functional.binary_cross_entropy_with_logits(raw[:,10],g[batch],reduction='none')
                loss=loss+config['mdn_nll_weight']*(gl+g[batch]*nll)
            objective=(loss*w[batch]).mean()
            if not torch.isfinite(objective):raise ArithmeticError('nonfinite objective')
            optimizer.zero_grad();objective.backward();nn.utils.clip_grad_norm_(model.parameters(),config['gradient_norm_clip']);optimizer.step()
            if time.monotonic()-start>limit:raise TimeoutError('job budget')
        if (epoch+1)%config['log_train_bce_every_epochs']==0:
            with torch.inference_mode():
                p=probability(model(qt,st,at)).clamp(1e-6,1-1e-6)
                bce=(w*nn.functional.binary_cross_entropy(p,yt,reduction='none').mean(1)).mean()
            trace.append({'epoch':epoch+1,'train_person_macro_bce':float(bce)})
    model.eval()
    with torch.inference_mode():
        raw=model(*[torch.tensor(v,dtype=torch.float32) for v in [qe,se,ae]])
        if delta:p=mdn_distribution(raw.double(),thresholds.double(),lower.double())[0].numpy()
        else:
            mass=torch.softmax(raw,1).numpy().astype(float);mass/=mass.sum(1,keepdims=True);p=mass@patterns()
    if not np.isfinite(p).all() or np.any(p< -1e-10) or np.any(p>1+1e-10):raise ArithmeticError('invalid forecast')
    weights={k:v.detach().numpy() for k,v in model.state_dict().items()}
    return np.clip(p,0,1),{'trace':trace,'parameter_count':sum(p.numel() for p in model.parameters())},weights
