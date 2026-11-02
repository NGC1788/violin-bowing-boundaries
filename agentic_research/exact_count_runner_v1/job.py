"""One registered child job. No direct unregistered fitting entry point."""
from pathlib import Path
import hashlib,json,os,signal,tempfile,time
import numpy as np
import torch
from threadpoolctl import threadpool_limits
from .data import fit_inner,prediction_inputs,verify_predictions
from .control import dump,seal_private_output
from ..exact_count_learning_v1.model import Forecaster


def state_hash(model):
    h=hashlib.sha256()
    for name,value in model.state_dict().items():
        if not (name.startswith('encoder.') or name.startswith('generator.')):continue
        a=value.detach().cpu().numpy();h.update(name.encode());h.update(str(a.shape).encode());h.update(a.tobytes())
    return h.hexdigest()


def save_weights(model,path):
    path=Path(path)
    with tempfile.NamedTemporaryFile(mode='wb',dir=path.parent,prefix='.'+path.name+'.',suffix='.tmp',delete=False) as f:
        np.savez_compressed(f,**{k:v.detach().numpy() for k,v in model.state_dict().items()});temp=f.name
    os.replace(temp,path)


def train_model(train,eval_prefix,method,config,deadline,checkpoint):
    """Only inner-train LABELS and held PREFIX are accepted."""
    torch.set_num_threads(1);torch.manual_seed(config['seed']);torch.use_deterministic_algorithms(True)
    model=Forecaster(method,config);initial_sha=state_hash(model);trace=[]
    optimizer=torch.optim.AdamW(model.parameters(),lr=config['optimizer']['learning_rate'],weight_decay=config['optimizer']['weight_decay'])
    tensors={k:torch.tensor(v,dtype=torch.bool if k=='voiced' else torch.float64) for k,v in train.items()}
    generator=torch.Generator().manual_seed(config['seed']+1831);order=hashlib.sha256()
    def budget():
        if time.monotonic()>=deadline:raise TimeoutError('job deadline during training/evaluation')
    try:
        for epoch in range(config['epochs']):
            budget();sums=np.zeros(3);permutation=torch.randperm(len(tensors['prefix']),generator=generator)
            order.update(permutation.numpy().tobytes())
            for batch in permutation.split(config['batch_size']):
                budget();rows,info=model.loss_rows(tensors['prefix'][batch],tensors['thresholds'],tensors['labels'][batch],
                                                   tensors['pitch'][batch],tensors['power'][batch],tensors['voiced'][batch])
                weights=tensors['weights'][batch];objective=(weights*rows).mean()
                if not torch.isfinite(objective):raise ArithmeticError('nonfinite training objective')
                optimizer.zero_grad();objective.backward()
                if any(p.grad is None or not torch.isfinite(p.grad).all() for p in model.parameters()):raise ArithmeticError('nonfinite/missing gradient')
                torch.nn.utils.clip_grad_norm_(model.parameters(),config['optimizer']['gradient_norm_clip']);optimizer.step()
                sums+=np.array([float((weights*rows).detach().sum()),float((weights*info['bce']).detach().sum()),
                                float((weights*info['auxiliary']).detach().sum())])
                budget()
            trace.append({'epoch':epoch+1,'online_weighted_loss':float(sums[0]/len(tensors['prefix'])),
                          'online_weighted_bce':float(sums[1]/len(tensors['prefix'])),
                          'online_weighted_auxiliary':float(sums[2]/len(tensors['prefix'])),
                          'kind':'online within-epoch changing-model averages, not epoch-end fitted loss'})
            checkpoint(model,trace)
        model.eval();out=[]
        with torch.inference_mode():
            for begin in range(0,len(eval_prefix),config['batch_size']):
                budget();x=torch.tensor(eval_prefix[begin:begin+config['batch_size']],dtype=torch.float64)
                q,_=model(x,tensors['thresholds']);out.append(q.numpy());budget()
        p=np.concatenate(out)
        if not np.isfinite(p).all() or (p< -1e-10).any() or (p>1+1e-10).any():raise ArithmeticError('invalid probability')
        return np.clip(p,0,1),model,{'trace':trace,'initial_shared_sha256':initial_sha,'minibatch_order_sha256':order.hexdigest(),
                  'parameters':sum(v.numel() for v in model.parameters()),'epochs_completed':len(trace)}
    finally:checkpoint(model,trace)


def execute(job_id):
    from .runner import HERE,check,load_frozen,sha
    freeze=check();directory=HERE/'local'/job_id
    request=json.loads(seal_private_output(directory,'request.json').read_text())
    state=json.loads((HERE/'local/state.json').read_text());active=state.get('active_job')
    if (not freeze['execution_authorized'] or state['status']!='RUNNING' or not active or active['id']!=job_id
            or request['freeze_sha256']!=sha(HERE/'FREEZE.json') or active['nonce']!=request['nonce']):
        raise ValueError('job is not authorized by active registered parent')
    os.kill(request['parent_pid'],0)
    with seal_private_output(directory,'started.json').open('x') as f:json.dump({'job':job_id,'freeze_sha256':request['freeze_sha256']},f)
    def terminated(_signum,_frame):raise TimeoutError('parent terminated timed-out child')
    signal.signal(signal.SIGTERM,terminated)
    model=None
    try:
        data=load_frozen(freeze);train,stats,_=fit_inner(data,request['fold'])
        eval_prefix,indices=prediction_inputs(data,request['fold'],stats)
        serialized={k:v.tolist() if isinstance(v,np.ndarray) else v for k,v in stats.items()}
        dump(seal_private_output(directory,'stats.json'),serialized)
        def checkpoint(m,trace):
            save_weights(m,seal_private_output(directory,'partial_weights.npz'));dump(seal_private_output(directory,'trace.json'),trace)
        with threadpool_limits(limits=1):
            p,model,receipt=train_model(train,eval_prefix,request['method'],request['model_config'],request['absolute_monotonic_deadline'],checkpoint)
        verify_predictions(p,indices,indices)
        np.savez_compressed(seal_private_output(directory,'predictions.npz'),probability=p,row_indices=indices)
        save_weights(model,seal_private_output(directory,'weights.npz'))
        receipt.update(job=job_id,method=request['method'],fold=request['fold'],preprocessing_sha256=sha(seal_private_output(directory,'stats.json')),
                       predictions_sha256=sha(seal_private_output(directory,'predictions.npz')),weights_sha256=sha(seal_private_output(directory,'weights.npz')),
                       trace_sha256=sha(seal_private_output(directory,'trace.json')))
        dump(seal_private_output(directory,'receipt.json'),receipt)
    except Exception as error:
        dump(seal_private_output(directory,'failure.json'),{'type':type(error).__name__,'reason':str(error)})
        raise


if __name__=='__main__':
    import sys
    if len(sys.argv)!=2:raise SystemExit('registered job id required')
    execute(sys.argv[1])
