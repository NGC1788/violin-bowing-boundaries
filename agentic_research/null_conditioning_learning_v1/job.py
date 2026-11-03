"""One registered bounded fit; trainer receives train labels and held prefix only."""
import hashlib,json,os,signal,time
import numpy as np
import torch
from threadpoolctl import threadpool_limits
from .data import fit_inner,prediction_inputs
from .control import dump,seal_private_output
from .model import ConditioningForecaster
from ..exact_count_runner_v1.job import state_hash,save_weights


def full_state_hash(model):
    h=hashlib.sha256()
    for k,v in model.state_dict().items():
        a=v.detach().numpy();h.update(k.encode());h.update(str(a.shape).encode());h.update(a.tobytes())
    return h.hexdigest()


def train_model(train,eval_prefix,method,config,deadline,checkpoint):
    torch.set_num_threads(1);torch.manual_seed(config['seed']);torch.use_deterministic_algorithms(True)
    model=ConditioningForecaster(method,config)
    initial_sha=state_hash(model);initial_model_sha=full_state_hash(model);trace=[]
    main_params=model.main_parameters()
    optimizer=torch.optim.AdamW(main_params,lr=config['optimizer']['learning_rate'],weight_decay=config['optimizer']['weight_decay'])
    tensors={k:torch.tensor(v,dtype=torch.bool if k=='voiced' else torch.long if k in {'fine_labels','coarse_labels'} else torch.float64) for k,v in train.items()}
    generator=torch.Generator().manual_seed(config['seed']+1831);order=hashlib.sha256()
    main_seconds=fine_seconds=0.
    def budget():
        if time.monotonic()>=deadline:raise TimeoutError('job deadline during training/evaluation')
    try:
        for epoch in range(config['epochs']):
            budget();sums=np.zeros(3);step_info=[]
            permutation=torch.randperm(len(tensors['prefix']),generator=generator);order.update(permutation.numpy().tobytes())
            for batch in permutation.split(config['batch_size']):
                budget();started=time.monotonic();weights=tensors['weights'][batch]
                rows,primary,aux=model.loss_rows(tensors['prefix'][batch],tensors['labels'][batch],
                    tensors['pitch'][batch],tensors['power'][batch],tensors['voiced'][batch])
                objective=(weights*rows).mean()
                if not torch.isfinite(objective):raise ArithmeticError('nonfinite main objective')
                optimizer.zero_grad(set_to_none=True);objective.backward()
                if any(v.grad is None or not torch.isfinite(v.grad).all() for v in main_params):raise ArithmeticError('nonfinite/missing main gradient')
                torch.nn.utils.clip_grad_norm_(main_params,config['optimizer']['gradient_norm_clip']);optimizer.step()
                main_seconds+=time.monotonic()-started
                sums+=np.array([float((weights*rows).detach().sum()),float((weights*primary).detach().sum()),float((weights*aux).detach().sum())])
                budget();started=time.monotonic()
                step_info.append(model.fine_step(tensors['prefix'][batch],tensors['fine_labels'][batch],tensors['coarse_labels'][batch],weights))
                fine_seconds+=time.monotonic()-started;budget()
            diagnostics={}
            for key in step_info[0]:
                values=[x[key] for x in step_info]
                diagnostics[key]={'min':float(min(values)),'mean':float(np.mean(values)),'max':float(max(values))}
            trace.append({'epoch':epoch+1,'online_weighted_loss':float(sums[0]/len(tensors['prefix'])),
                'online_weighted_bce':float(sums[1]/len(tensors['prefix'])),'online_weighted_auxiliary':float(sums[2]/len(tensors['prefix'])),
                'fine_steps':len(step_info),'fine_step_diagnostics':diagnostics,
                'kind':'online changing-model main averages; separate fine step diagnostics, not endpoint risk/convergence evidence'})
            checkpoint(model,trace)
        model.eval();parts=[];inference_started=time.monotonic()
        with torch.inference_mode():
            for start in range(0,len(eval_prefix),config['batch_size']):
                budget();q,_=model(torch.tensor(eval_prefix[start:start+config['batch_size']],dtype=torch.float64))
                parts.append(q.numpy());budget()
        probability=np.concatenate(parts)
        if not np.isfinite(probability).all() or (probability<0).any() or (probability>1).any():raise ArithmeticError('invalid forecast')
        return probability,model,{'initial_shared_sha256':initial_sha,'initial_model_sha256':initial_model_sha,
            'minibatch_order_sha256':order.hexdigest(),'parameters':sum(v.numel() for v in model.parameters()),
            'epochs_completed':len(trace),'main_step_seconds':main_seconds,'fine_step_seconds':fine_seconds,
            'held_inference_seconds':time.monotonic()-inference_started,'inference_memory_array_bytes':0,
            'fine_steps_completed':sum(x['fine_steps'] for x in trace)}
    finally:checkpoint(model,trace)


def execute(job_id):
    from .runner import HERE,check,load_frozen,sha,contract,execution_review
    execution_review();freeze=check();directory=HERE/'local'/job_id
    request=json.loads(seal_private_output(directory,'request.json').read_text())
    state=json.loads((HERE/'local/state.json').read_text());active=state.get('active_job')
    if (not freeze['execution_authorized'] or state['status']!='RUNNING' or not active or active['id']!=job_id
        or request['freeze_sha256']!=sha(HERE/'FREEZE.json') or active['nonce']!=request['nonce']):raise ValueError('unregistered job')
    if (request['model_config']!=contract()['model_config'] or any(request[k]!=active[k] for k in ['fold','method','limit_seconds'])
        or job_id!=f"job_{request['fold']}_{request['method']}"
        or {'fold':request['fold'],'method':request['method']} not in contract()['job_order']):raise ValueError('job request differs from contract')
    os.kill(request['parent_pid'],0)
    with seal_private_output(directory,'started.json').open('x') as f:json.dump({'job':job_id,'freeze_sha256':request['freeze_sha256']},f)
    def terminated(_signum,_frame):raise TimeoutError('parent terminated child')
    signal.signal(signal.SIGTERM,terminated)
    try:
        data=load_frozen(freeze);train,stats,fine_stats=fit_inner(data,request['fold'])
        eval_prefix,indices=prediction_inputs(data,request['fold'],stats)
        dump(seal_private_output(directory,'stats.json'),{k:v.tolist() if isinstance(v,np.ndarray) else v for k,v in stats.items()})
        parent=json.loads((HERE.parent/'exact_count_runner_v1/RESULTS.json').read_text())
        historical=parent['common_control_hashes'][str(request['fold'])]
        if sha(directory/'stats.json')!=historical['preprocessing_sha256']:raise ValueError('historical preprocessing mismatch')
        binding=json.loads((HERE/'INPUT_RECEIPT.json').read_text())['fine_training_bindings'][str(request['fold'])]
        if hashlib.sha256(json.dumps(fine_stats,sort_keys=True).encode()).hexdigest()!=binding['fine_statistics_sha256']:
            raise ValueError('fine statistics binding mismatch')
        def checkpoint(model,trace):
            save_weights(model,seal_private_output(directory,'partial_weights.npz'));dump(seal_private_output(directory,'trace.json'),trace)
        with threadpool_limits(limits=1):
            probability,model,receipt=train_model(train,eval_prefix,request['method'],request['model_config'],request['absolute_monotonic_deadline'],checkpoint)
        if any(receipt[k]!=historical[k] for k in ['initial_shared_sha256','minibatch_order_sha256']):raise ValueError('historical initialization/order mismatch')
        np.savez_compressed(seal_private_output(directory,'predictions.npz'),probability=probability,row_indices=indices)
        save_weights(model,seal_private_output(directory,'weights.npz'))
        receipt.update(job=job_id,method=request['method'],fold=request['fold'],fine_stats=fine_stats,
            weights_artifact_bytes=(directory/'weights.npz').stat().st_size,predictions_artifact_bytes=(directory/'predictions.npz').stat().st_size,
            preprocessing_sha256=sha(directory/'stats.json'),predictions_sha256=sha(directory/'predictions.npz'),
            weights_sha256=sha(directory/'weights.npz'),trace_sha256=sha(directory/'trace.json'))
        dump(seal_private_output(directory,'receipt.json'),receipt)
    except Exception as error:
        dump(seal_private_output(directory,'failure.json'),{'type':type(error).__name__,'reason':str(error)});raise


if __name__=='__main__':
    import sys
    if len(sys.argv)!=2:raise SystemExit('registered job id required')
    execute(sys.argv[1])
