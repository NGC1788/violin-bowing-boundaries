import itertools,json,math
from pathlib import Path
import numpy as np
from scipy.stats import ncx2
import torch
import pytest
from .model import (Forecaster,parameterize,count_queries,auxiliary_nll,count_distribution,
                    guard_probability)
from .preprocess import fit_training_stats,transform_prefix,transform_labels,normalized_thresholds

torch.set_num_threads(1)


def config(tiny=True):
    c=json.loads((Path(__file__).parent/'CONTRACT_PROPOSAL.json').read_text())
    if tiny:c.update(future_blocks=[2,2],future_frames=4,minimum_voiced=2,hidden_width=8)
    return c


def thresholds():return torch.tensor([[.3,.8,1.5],[.4,.9,1.7]],dtype=torch.float64)


def independent_query(p,thresholds,blocks,minimum):
    result=np.zeros((len(p['mixture']),9));t=sum(blocks)
    p={k:v.detach().numpy() for k,v in p.items()}
    for row in range(len(result)):
        for h,mixture in enumerate(p['mixture'][row]):
            for bits in itertools.product([0,1],repeat=t):
                m=np.array(bits,bool);n=int(m.sum())
                if n<minimum:continue
                w=mixture*(p['initial'][row,h] if bits[0] else 1-p['initial'][row,h])
                for i in range(1,t):
                    pp=p['transitions'][row,h,int(i>=blocks[0]),bits[i-1]]
                    w*=pp if bits[i] else 1-pp
                values=[]
                for ch in [0,1]:
                    mu=np.repeat(p['means'][row,h,ch],blocks)
                    if ch==0:mu=mu[m]
                    k=len(mu);sd=p['sigma'][row,h,ch]
                    lam=np.sum((mu-mu.mean())**2)/sd**2
                    values.append(ncx2.cdf((k-1)*thresholds[ch]**2/sd**2,k-1,lam))
                result[row]+=w*np.outer(*values).flatten()
    return result


def test_batched_gaussian_head_matches_selected_path_enumeration():
    torch.manual_seed(2);c=config();m=Forecaster('exact_count',c)
    x=torch.randn(2,73,dtype=torch.float64)*.2;q,p=m(x,thresholds())
    reference=independent_query(p,thresholds().numpy(),c['future_blocks'],c['minimum_voiced'])
    np.testing.assert_allclose(q.detach(),reference,atol=2e-13,rtol=2e-12)
    serial=torch.cat([m(x[i:i+1],thresholds())[0] for i in range(2)])
    torch.testing.assert_close(q,serial,atol=2e-13,rtol=2e-12)


def test_gaussian_contrast_scale_partials_match_finite_difference():
    c=config();raw=torch.linspace(-.2,.3,20,dtype=torch.float64)[None].requires_grad_()
    def objective(r):return count_queries(parameterize(r,c),thresholds(),[2,2],2).mean()
    q=objective(raw);g=torch.autograd.grad(q,raw)[0]
    for index in [7,8,9,16,18,19]:
        a=raw.detach().clone();b=a.clone();step=1e-5;a[0,index]+=step;b[0,index]-=step
        fd=float((objective(a)-objective(b))/(2*step))
        assert float(g[0,index])==pytest.approx(fd,abs=2e-8,rel=2e-5)
    assert abs(float(g[0,9]))>1e-8  # scale parameter, not just mask-logit gradients
    assert abs(float(g[0,7]))>1e-8  # pitch contrast parameter


@pytest.mark.parametrize('method',['joint_aux','mdn_aux','exact_count','moment_count'])
def test_training_objective_backpropagates_without_optimizer_or_data(method):
    torch.manual_seed(3);c=config();model=Forecaster(method,c)
    x=torch.randn(2,73,dtype=torch.float64)*.1
    pitch=torch.tensor([[.1,.4,.7,float('nan')],[.2,float('nan'),float('nan'),float('nan')]],dtype=torch.float64)
    power=torch.tensor([[.2,-.1,.1,.4],[.3,.2,.1,0]],dtype=torch.float64)
    mask=torch.tensor([[True,True,True,False],[True,False,False,False]])
    y=torch.zeros((2,9),dtype=torch.float64);y[0]=1
    rows,info=model.loss_rows(x,thresholds(),y,pitch,power,mask)
    assert rows.shape==(2,) and torch.isfinite(rows).all()
    rows.mean().backward()
    for name,p in model.named_parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all(),name
    assert float(model.encoder[0].weight.grad.abs().sum())>0
    # Actual future labels never enter the inference API or mutate its state.
    before=info['q'].detach().clone()
    with torch.no_grad():after=model(x,thresholds())[0]
    torch.testing.assert_close(before,after,atol=0,rtol=0)


def test_full46frame_code_shape_and_gaussian_backward_are_functional_only():
    c=config(False);torch.manual_seed(4);m=Forecaster('exact_count',c)
    q,p=m(torch.zeros((1,73),dtype=torch.float64),thresholds())
    assert q.shape==(1,9) and torch.isfinite(q).all()
    q.mean().backward();assert torch.isfinite(m.generator.bias.grad).all()
    w=count_distribution(p['initial'],p['transitions'],c['future_blocks'])
    torch.testing.assert_close(w.sum(2),torch.ones((1,2),dtype=torch.float64),atol=1e-12,rtol=0)


def test_same_encoder_generator_and_auxiliary_information_for_all_methods():
    models=[]
    for method in config()['methods']:
        torch.manual_seed(5);models.append(Forecaster(method,config()))
    x=torch.ones((2,73),dtype=torch.float64)*.1
    base=models[0](x,thresholds())[1]
    for m in models[1:]:
        p=m(x,thresholds())[1]
        for k in base:torch.testing.assert_close(base[k],p[k],atol=0,rtol=0)
    counts=[sum(p.numel() for p in m.parameters()) for m in models]
    assert counts[0]==counts[1]>counts[2]==counts[3]


def test_unvoiced_nan_ignored_and_invalid_row_still_has_mask_power_likelihood():
    c=config();p=parameterize(torch.zeros((2,20),dtype=torch.float64),c)
    p['means']=torch.zeros((2,2,2,2),dtype=torch.float64);p['sigma']=torch.ones((2,2,2),dtype=torch.float64)
    pitch=torch.full((2,4),float('nan'),dtype=torch.float64);mask=torch.zeros((2,4),dtype=torch.bool)
    power=torch.zeros((2,4),dtype=torch.float64)
    nll=auxiliary_nll(p,pitch,power,mask,[2,2])
    expected=math.log(2)+(3/4)*.5*math.log(2*math.pi)  # power contrast hasT-1 dimensions
    torch.testing.assert_close(nll,torch.full((2,),expected,dtype=torch.float64),atol=1e-13,rtol=0)
    mask[0,0]=True
    with pytest.raises(ValueError):auxiliary_nll(p,pitch,power,mask,[2,2])


def test_zero_threshold_is_zero_and_gradients_remain_finite():
    raw=torch.zeros((1,20),dtype=torch.float64,requires_grad=True);c=config()
    t=thresholds();t[:,0]=0
    q=count_queries(parameterize(raw,c),t,[2,2],2).reshape(1,3,3)
    assert torch.count_nonzero(q[:,0,:])==0 and torch.count_nonzero(q[:,:,0])==0
    q.sum().backward();assert torch.isfinite(raw.grad).all()


def test_exact_and_moment_outputs_have_nonnegative_common_grid_cells():
    for method in ['exact_count','moment_count','mdn_aux','joint_aux']:
        torch.manual_seed(6);m=Forecaster(method,config());q,_=m(torch.randn(3,73,dtype=torch.float64),thresholds())
        grid=q.detach().numpy().reshape(-1,3,3)
        for row in grid:
            cells=np.diff(np.diff(np.pad(row,((1,0),(1,0))),axis=0),axis=1)
            assert cells.min()>-1e-12 and row[-1,-1]<=1+1e-12


def synthetic_arrays():
    rng=np.random.default_rng(7);x=rng.normal(size=(12,60));x[:,57:59]=np.abs(x[:,57:59]);x[:,59]=1
    pitch=rng.normal(size=(12,4));power=rng.normal(size=(12,4));m=np.ones((12,4),bool)
    m[1,1:]=False;g=m.sum(1)>=2
    targets=np.column_stack([np.array([np.std(v[mask],ddof=1) if guard else np.nan for v,mask,guard in zip(pitch,m,g)]),power.std(1,ddof=1)])
    return x,pitch,power,m,targets,g


def test_preprocessing_fit_is_inner_train_only_and_inference_prefix_only():
    arrays=synthetic_arrays();stats=fit_training_stats(*[v[:8] for v in arrays])
    before=transform_prefix(arrays[0],stats)
    mutated=[v.copy() for v in arrays]
    for v in mutated:
        if v.dtype==bool:v[8:]=False
        else:v[8:]=np.nan
    again=fit_training_stats(*[v[:8] for v in mutated])
    for k in stats:np.testing.assert_equal(stats[k],again[k])
    np.testing.assert_array_equal(before,transform_prefix(arrays[0],again))
    assert before.shape==(12,73)
    assert normalized_thresholds(stats).shape==(2,3)
    pp,pw=transform_labels(arrays[1],arrays[2],arrays[3],stats)
    assert np.isfinite(pp).all() and np.isfinite(pw).all() and np.all(pp[~arrays[3]]==0)


def test_unclipped_moment_anchors_preserve_prefix_information():
    arrays=synthetic_arrays();stats=fit_training_stats(*[v[:8] for v in arrays])
    x=arrays[0][:2].copy();x[0,57:59]=1e3;x[1,57:59]=1e4
    features=transform_prefix(x,stats)
    np.testing.assert_array_equal(features[0,57:59],features[1,57:59])
    assert np.all(features[0,-2:]!=features[1,-2:])


def test_projected_auxiliary_density_matches_known_contrast_gaussian():
    c=config();p=parameterize(torch.zeros((1,20),dtype=torch.float64),c)
    p['means']=torch.zeros((1,2,2,2),dtype=torch.float64);p['sigma']=torch.ones((1,2,2),dtype=torch.float64)
    pitch=torch.tensor([[1.,3.,float('nan'),float('nan')]],dtype=torch.float64)
    power=torch.tensor([[1.,2.,3.,4.]],dtype=torch.float64);mask=torch.tensor([[True,True,False,False]])
    expected=(4*math.log(2)+.5*(2+5)+.5*(1+3)*math.log(2*math.pi))/4
    assert float(auxiliary_nll(p,pitch,power,mask,[2,2]))==pytest.approx(expected,abs=1e-13)


def test_future_level_shifts_do_not_change_projected_auxiliary_or_scaling():
    arrays=synthetic_arrays();before=fit_training_stats(*[v[:8] for v in arrays])
    shifted=[v.copy() for v in arrays];shifted[1]+=np.arange(12)[:,None]*123.;shifted[2]-=np.arange(12)[:,None]*45.
    after=fit_training_stats(*[v[:8] for v in shifted])
    np.testing.assert_allclose(before['path_scale'],after['path_scale'],atol=2e-13,rtol=1e-12)
    for a,b in zip(transform_labels(*arrays[1:4],before),transform_labels(*shifted[1:4],after)):
        np.testing.assert_allclose(a,b,atol=2e-12,rtol=1e-12)
    c=config();p=parameterize(torch.zeros((2,20),dtype=torch.float64),c)
    pitch=torch.tensor(arrays[1][:2],dtype=torch.float64);power=torch.tensor(arrays[2][:2],dtype=torch.float64)
    mask=torch.tensor(arrays[3][:2]);a=auxiliary_nll(p,pitch,power,mask,[2,2])
    b=auxiliary_nll(p,pitch+torch.tensor([[17.],[-23.]]),power+torch.tensor([[33.],[-11.]]),mask,[2,2])
    torch.testing.assert_close(a,b,atol=2e-12,rtol=1e-12)
