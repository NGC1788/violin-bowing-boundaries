"""One bounded registered fit; only train labels and held prefix reach trainer."""
from pathlib import Path
import hashlib,json,os,signal,time
import numpy as np
import torch
from threadpoolctl import threadpool_limits
from .data import fit_inner,prediction_inputs,training_memory,training_support_summary
from .control import dump,seal_private_output
from .model import RetrievalForecaster,QueryDRO
from ..exact_count_runner_v1.job import state_hash,save_weights


def train_model(train,memory,eval_prefix,method,config,deadline,checkpoint):
    torch.set_num_threads(1);torch.manual_seed(config['seed']);torch.use_deterministic_algorithms(True)
    model=RetrievalForecaster(method,config);initial_sha=state_hash(model);trace=[]
    optimizer=torch.optim.AdamW(model.parameters(),lr=config['optimizer']['learning_rate'],weight_decay=config['optimizer']['weight_decay'])
    tensors={k:torch.tensor(v,dtype=torch.bool if k=='voiced' else torch.float64) for k,v in train.items()}
    mem={k:torch.tensor(memory[k],dtype=torch.long if k=='groups' else torch.float64) for k in ['prefix','labels','groups']}
    dro=QueryDRO(mem['groups'],config['query_group_dro_eta'])
    generator=torch.Generator().manual_seed(config['seed']+1831)
    random_generator=torch.Generator().manual_seed(config['seed']+77129)
    order=hashlib.sha256()
    def budget():
        if time.monotonic()>=deadline:raise TimeoutError('job deadline during training/evaluation')
    try:
        for epoch in range(config['epochs']):
            budget();sums=np.zeros(4);permutation=torch.randperm(len(tensors['prefix']),generator=generator)
            order.update(permutation.numpy().tobytes())
            for batch in permutation.split(config['batch_size']):
                budget();groups=mem['groups'][batch]
                primary,aux,info=model.loss_rows(tensors['prefix'][batch],mem,tensors['labels'][batch],
                    tensors['pitch'][batch],tensors['power'][batch],tensors['voiced'][batch],groups,random_generator)
                weights=tensors['weights'][batch]
                multiplier=dro.multipliers(groups) if method=='retrieval_query_group_dro' else torch.ones_like(weights)
                rows=multiplier*primary+config['auxiliary_weight']*aux
                objective=(weights*rows).mean()
                if not torch.isfinite(objective):raise ArithmeticError('nonfinite training objective')
                optimizer.zero_grad();objective.backward()
                if any(v.grad is None or not bool(torch.isfinite(v.grad).all()) for v in model.parameters()):
                    raise ArithmeticError('nonfinite/missing gradient')
                torch.nn.utils.clip_grad_norm_(model.parameters(),config['optimizer']['gradient_norm_clip']);optimizer.step()
                if method=='retrieval_query_group_dro':dro.update(primary,groups)
                sums+=np.array([float((weights*rows).detach().sum()),float((weights*primary).detach().sum()),
                                float((weights*aux).detach().sum()),float((weights*multiplier*primary).detach().sum())])
                budget()
            trace.append({'epoch':epoch+1,'online_weighted_loss':float(sums[0]/len(tensors['prefix'])),
                          'online_person_macro_primary_diagnostic':float(sums[1]/len(tensors['prefix'])),
                          'online_optimized_weighted_primary':float(sums[3]/len(tensors['prefix'])),
                          'online_weighted_auxiliary':float(sums[2]/len(tensors['prefix'])),
                          'kind':'online changing-model averages; primary objectives differ between arms, not endpoint training scores'})
            checkpoint(model,trace)
        inference_started=time.monotonic();model.eval();parts=[]
        with torch.inference_mode():
            for begin in range(0,len(eval_prefix),config['batch_size']):
                budget();x=torch.tensor(eval_prefix[begin:begin+config['batch_size']],dtype=torch.float64)
                held_groups=torch.full((len(x),),-1,dtype=torch.long)
                info,_=model(x,mem['prefix'],mem['labels'],mem['groups'],held_groups)
                parts.append({k:info[k].numpy() for k in ['probability','group_mass','positive_support','negative_support']})
                budget()
        out={k:np.concatenate([part[k] for part in parts]) for k in parts[0]}
        out['donor_codes']=np.arange(-1,len(dro.counts),dtype=np.int64)
        out['dro_logweights']=dro.logweights.numpy()
        receipt={'trace':trace,'initial_shared_sha256':initial_sha,'minibatch_order_sha256':order.hexdigest(),
                 'parameters':sum(v.numel() for v in model.parameters()),'epochs_completed':len(trace),
                 'held_full_plus_all_counterfactual_inference_seconds':time.monotonic()-inference_started,
                 'inference_memory_array_bytes':sum(np.asarray(memory[k]).nbytes for k in ['prefix','labels','groups','source_rows']),
                 'training_memory_sha256':hashlib.sha256(b''.join(np.asarray(memory[k]).tobytes() for k in ['prefix','labels','groups','source_rows'])).hexdigest()}
        return out,model,receipt
    finally:checkpoint(model,trace)


def execute(job_id):
    from .runner import HERE,check,load_frozen,sha,contract
    freeze=check();directory=HERE/'local'/job_id
    request=json.loads(seal_private_output(directory,'request.json').read_text())
    state=json.loads((HERE/'local/state.json').read_text());active=state.get('active_job')
    if (not freeze['execution_authorized'] or state['status']!='RUNNING' or not active or active['id']!=job_id
        or request['freeze_sha256']!=sha(HERE/'FREEZE.json') or active['nonce']!=request['nonce']):
        raise ValueError('job is not authorized by active registered parent')
    if (request['model_config']!=contract()['model_config'] or
        any(request[k]!=active[k] for k in ['fold','method','limit_seconds']) or
        job_id!=f"job_{request['fold']}_{request['method']}" or
        {'fold':request['fold'],'method':request['method']} not in contract()['job_order']):
        raise ValueError('registered request differs from frozen contract')
    os.kill(request['parent_pid'],0)
    with seal_private_output(directory,'started.json').open('x') as f:json.dump({'job':job_id,'freeze_sha256':request['freeze_sha256']},f)
    def terminated(_signum,_frame):raise TimeoutError('parent terminated timed-out child')
    signal.signal(signal.SIGTERM,terminated)
    try:
        data=load_frozen(freeze);train,stats,_=fit_inner(data,request['fold'])
        eval_prefix,indices=prediction_inputs(data,request['fold'],stats);memory=training_memory(data,request['fold'],train)
        serialized={k:v.tolist() if isinstance(v,np.ndarray) else v for k,v in stats.items()}
        dump(seal_private_output(directory,'stats.json'),serialized)
        parent=json.loads((HERE.parent/'exact_count_runner_v1/RESULTS.json').read_text())
        if sha(directory/'stats.json')!=parent['common_control_hashes'][str(request['fold'])]['preprocessing_sha256']:
            raise ValueError('historical preprocessing mismatch')
        support=training_support_summary(memory)
        np.savez_compressed(seal_private_output(directory,'memory.npz'),**memory)
        def checkpoint(model,trace):
            save_weights(model,seal_private_output(directory,'partial_weights.npz'));dump(seal_private_output(directory,'trace.json'),trace)
        with threadpool_limits(limits=1):
            out,model,receipt=train_model(train,memory,eval_prefix,request['method'],request['model_config'],request['absolute_monotonic_deadline'],checkpoint)
        if any(receipt[k]!=parent['common_control_hashes'][str(request['fold'])][k]
               for k in ['initial_shared_sha256','minibatch_order_sha256']):raise ValueError('historical shared initialization/order mismatch')
        np.savez_compressed(seal_private_output(directory,'predictions.npz'),
            probability=out['probability'][:,0],counterfactual_probability=out['probability'],row_indices=indices,
            donor_codes=out['donor_codes'],group_mass=out['group_mass'],positive_support=out['positive_support'],
            negative_support=out['negative_support'],dro_logweights=out['dro_logweights'])
        save_weights(model,seal_private_output(directory,'weights.npz'))
        receipt.update(job=job_id,method=request['method'],fold=request['fold'],training_support_summary=support,
            memory_artifact_bytes=(directory/'memory.npz').stat().st_size,
            weights_artifact_bytes=(directory/'weights.npz').stat().st_size,
            predictions_artifact_bytes=(directory/'predictions.npz').stat().st_size,
            preprocessing_sha256=sha(directory/'stats.json'),predictions_sha256=sha(directory/'predictions.npz'),
            weights_sha256=sha(directory/'weights.npz'),trace_sha256=sha(directory/'trace.json'),memory_artifact_sha256=sha(directory/'memory.npz'))
        dump(seal_private_output(directory,'receipt.json'),receipt)
    except Exception as error:
        dump(seal_private_output(directory,'failure.json'),{'type':type(error).__name__,'reason':str(error)})
        raise


if __name__=='__main__':
    import sys
    if len(sys.argv)!=2:raise SystemExit('registered job id required')
    execute(sys.argv[1])
