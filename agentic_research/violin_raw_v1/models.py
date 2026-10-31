"""Fixed conventional baselines and controlled neural ablations; no novelty claim."""
from __future__ import annotations
import random,time
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits
import torch
from torch import nn


def patterns():
    bank={tuple(np.zeros(9))}
    for row in range(3):
        for col in range(3):
            bank.add(tuple(float(i>=row and j>=col) for i in range(3) for j in range(3)))
    return np.asarray(sorted(bank),dtype=np.float64)


def pattern_ids(y):
    y=np.asarray(y)
    if y.ndim!=2 or y.shape[1]!=9 or not np.isin(y,[0,1]).all():raise ValueError('Invalid labels')
    match=(y[:,None,:]==patterns()[None,:,:]).all(2)
    if not match.any(1).all():raise ValueError('Invalid outcome pattern')
    return match.argmax(1)


def participant_weights(ids):
    ids=np.asarray(ids);unique,counts=np.unique(ids,return_counts=True)
    lookup=dict(zip(unique,counts))
    w=np.array([1/lookup[x] for x in ids],float)
    return w/w.mean()


def transition_prior(y,states,weights):
    ids=pattern_ids(y);states=np.asarray(states)
    if not np.isin(states,np.arange(11)).all():raise ValueError('Invalid prefix states')
    global_counts=np.bincount(ids,weights=weights,minlength=10)+.1
    global_p=global_counts/global_counts.sum()
    counts=np.zeros((11,10));np.add.at(counts,(states,ids),weights)
    rows=np.where(counts.sum(1,keepdims=True)>0,(counts+.1)/(counts.sum(1,keepdims=True)+1),global_p)
    return global_p,rows


def fit_predict(method,seed,train_x,train_y,train_states,train_participants,eval_x,eval_states,config):
    """Training labels only; no dev/test labels or early stopping enter the API."""
    started=time.perf_counter()
    x=np.asarray(train_x,float);y=np.asarray(train_y,float);xe=np.asarray(eval_x,float)
    if not np.isfinite(x).all() or not np.isfinite(xe).all():raise ValueError('Nonfinite features')
    if x.ndim!=2 or xe.shape[1]!=x.shape[1] or len(y)!=len(x):raise ValueError('Shape mismatch')
    if not np.isin(train_states,np.arange(11)).all() or not np.isin(eval_states,np.arange(11)).all():raise ValueError('Invalid states')
    weights=participant_weights(train_participants)
    global_p,prior=transition_prior(y,train_states,weights)
    bank=patterns();meta={'method':method,'seed':int(seed),'train_rows':len(x),'train_participants':len(set(train_participants))}
    if method=='frequency':p=np.tile(global_p@bank,(len(xe),1))
    elif method=='markov':p=prior[eval_states]@bank
    elif method=='hgb':
        columns=[]
        for j in range(9):
            if len(np.unique(y[:,j]))==1:columns.append(np.full(len(xe),y[0,j]));continue
            m=HistGradientBoostingClassifier(max_iter=config['hgb_iterations'],learning_rate=.07,
                max_leaf_nodes=15,min_samples_leaf=20,l2_regularization=.5,early_stopping=False,random_state=seed)
            with threadpool_limits(limits=1):m.fit(x,y[:,j],sample_weight=weights);columns.append(m.predict_proba(xe)[:,1])
        p=np.column_stack(columns)
    elif method in ('mlp_ind','mlp_joint','mlp_offset'):
        torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
        random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
        scaler=StandardScaler().fit(x)
        tx=torch.tensor(np.clip(scaler.transform(x),-8,8),dtype=torch.float32)
        ex=torch.tensor(np.clip(scaler.transform(xe),-8,8),dtype=torch.float32)
        ty=torch.tensor(y,dtype=torch.float32);tw=torch.tensor(weights,dtype=torch.float32)
        b=torch.tensor(bank,dtype=torch.float32)
        model=nn.Sequential(nn.Linear(x.shape[1],config['hidden']),nn.GELU(),
                            nn.Linear(config['hidden'],9 if method=='mlp_ind' else 10))
        opt=torch.optim.AdamW(model.parameters(),lr=config['learning_rate'],weight_decay=config['weight_decay'])
        train_offset=torch.tensor(np.log(prior[train_states]),dtype=torch.float32)
        eval_offset=torch.tensor(np.log(prior[eval_states]),dtype=torch.float32)
        generator=torch.Generator().manual_seed(seed+1831)
        for epoch in range(config['epochs']):
            permutation=torch.randperm(len(tx),generator=generator)
            for batch in permutation.split(config['batch_size']):
                logits=model(tx[batch])
                if method=='mlp_ind':loss=nn.functional.binary_cross_entropy_with_logits(logits,ty[batch],reduction='none').mean(1)
                else:
                    if method=='mlp_offset':logits=logits+train_offset[batch]
                    logmass=torch.log_softmax(logits,1)
                    log_positive=torch.stack([torch.logsumexp(logmass[:,b[:,j]==1],1) for j in range(9)],1)
                    log_negative=torch.stack([torch.logsumexp(logmass[:,b[:,j]==0],1) for j in range(9)],1)
                    loss=-(ty[batch]*log_positive+(1-ty[batch])*log_negative).mean(1)
                objective=(loss*tw[batch]).mean()
                opt.zero_grad();objective.backward();opt.step()
        with torch.no_grad():
            logits=model(ex)
            if method=='mlp_ind':p=torch.sigmoid(logits).numpy().astype(float)
            else:
                if method=='mlp_offset':logits=logits+eval_offset
                mass=torch.softmax(logits,1).numpy().astype(float);mass/=mass.sum(1,keepdims=True);p=mass@bank
        meta.update({'parameter_count':sum(a.numel() for a in model.parameters()),'epochs':config['epochs'],
                     'scaler_mean':scaler.mean_.tolist(),'scaler_scale':scaler.scale_.tolist(),
                     'state_dict':{name:value.detach().numpy() for name,value in model.state_dict().items()}})
    else:raise ValueError('Unknown method')
    if p.shape!=(len(xe),9) or not np.isfinite(p).all() or np.any((p<0)|(p>1)):
        raise ArithmeticError('Invalid model output')
    meta['seconds']=time.perf_counter()-started
    return p,meta
