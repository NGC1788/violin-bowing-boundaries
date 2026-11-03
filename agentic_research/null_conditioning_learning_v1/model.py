"""Linear finite-step conditioning; no fitting or IO at import."""
import torch
from torch import nn
from .data import fine_to_coarse
from ..exact_count_learning_v1.model import bank,parameterize,auxiliary_nll


def null_projector(a,rtol):
    ac=a-a.mean(0,keepdim=True)
    with torch.no_grad():
        singular=torch.linalg.svdvals(ac)
        rank=int((singular>rtol*singular.max()).sum())
        if rank>9:raise ArithmeticError('centered readout rank exceeds9')
        p=torch.eye(a.shape[1],dtype=a.dtype)-torch.linalg.pinv(ac,rtol=rtol)@ac
        residual=float(torch.linalg.vector_norm(ac@p)/(1+torch.linalg.vector_norm(ac)))
    return p,rank,residual


def match_norm(value,target):
    norm=torch.linalg.vector_norm(value)
    if float(target)==0 or float(norm)==0:return torch.zeros_like(value)
    return value*(target/norm)


def applied_displacement(proposal,p,random_basis,rank,method):
    null=p@proposal;target=torch.linalg.vector_norm(null)
    if method=='main_only':return torch.zeros_like(proposal),target
    if method=='ordinary_matched':return match_norm(proposal,target),target
    if method=='random_matched':
        basis=random_basis[:,:proposal.shape[0]-rank]
        return match_norm(basis@(basis.T@proposal),target),target
    if method=='null_conditioned':return null,target
    raise ValueError('unknown arm')


class ConditioningForecaster(nn.Module):
    def __init__(self,method,config):
        super().__init__()
        if method not in config['methods']:raise ValueError('undeclared method')
        self.method,self.config=method,config
        w=config['hidden_width'];h=config['regimes']
        # These first three modules match historical joint_aux initialization order.
        self.encoder=nn.Sequential(nn.Linear(config['input_dimension'],w),nn.GELU(),nn.Linear(w,w),nn.GELU())
        self.generator=nn.Linear(w,h+h*9)
        self.free_head=nn.Linear(w,10)
        self.adapter=nn.Linear(w,w,bias=False)
        with torch.no_grad():self.adapter.weight.copy_(torch.eye(w))
        self.fine_head=nn.Linear(w,65)
        self.register_buffer('pattern_bank',bank())
        self.register_buffer('fine_map',torch.tensor(fine_to_coarse(),dtype=torch.long))
        rng=torch.Generator().manual_seed(config['seed']+config['random_basis_seed_offset'])
        basis=torch.linalg.qr(torch.randn((w,w),generator=rng,dtype=torch.float64)).Q
        self.register_buffer('random_basis',basis)
        self.double()

    def main_parameters(self):
        return [p for n,p in self.named_parameters() if not n.startswith('fine_head.')]

    def forward(self,prefix):
        if prefix.ndim!=2 or prefix.shape[1]!=self.config['input_dimension'] or not torch.isfinite(prefix).all():
            raise ValueError('finite prefix-only feature matrix required')
        u=self.encoder(prefix)
        q=torch.softmax(self.free_head(self.adapter(u)),1)@self.pattern_bank
        return q,parameterize(self.generator(u),self.config)

    def loss_rows(self,prefix,labels,pitch,power,voiced):
        q,p=self(prefix);eps=self.config['probability_clipping']
        primary=nn.functional.binary_cross_entropy(q.clamp(eps,1-eps),labels,reduction='none').mean(1)
        aux=auxiliary_nll(p,pitch,power,voiced,self.config['future_blocks'])
        return primary+self.config['auxiliary_weight']*aux,primary,aux

    def fine_rows(self,u,fine,coarse):
        if fine.dtype!=torch.long or coarse.dtype!=torch.long or fine.shape!=(len(u),) or coarse.shape!=fine.shape:
            raise ValueError('fine/coarse label schema')
        if (fine<0).any() or (fine>=65).any() or not torch.equal(self.fine_map[fine],coarse):
            raise ValueError('fine/coarse fibre mismatch')
        logits=self.fine_head(self.adapter(u))
        allowed=self.fine_map[None]==coarse[:,None]
        return torch.logsumexp(logits.masked_fill(~allowed,-torch.inf),1)-logits.gather(1,fine[:,None]).squeeze(1)

    def fine_step(self,prefix,fine,coarse,weights):
        """SGD proposal then actual displacement projection; no shared optimizer state."""
        with torch.no_grad():
            u=self.encoder(prefix).detach()
            before=torch.softmax(self.free_head(self.adapter(u)),1)
            p,rank,residual=null_projector(self.free_head.weight.detach(),self.config['projection_rtol'])
        if residual>self.config['projection_relative_residual_limit']:raise ArithmeticError('projector residual')
        objective=self.config['fine_auxiliary_weight']*(weights*self.fine_rows(u,fine,coarse)).mean()
        params=[self.adapter.weight,*self.fine_head.parameters()]
        gradients=torch.autograd.grad(objective,params)
        if not torch.isfinite(objective) or any(not torch.isfinite(g).all() for g in gradients):
            raise ArithmeticError('nonfinite fine objective/gradient')
        proposal=-self.config['fine_learning_rate']*gradients[0]
        with torch.no_grad():
            delta,target=applied_displacement(proposal,p,self.random_basis,rank,self.method)
            self.adapter.weight.add_(delta)
            for v,g in zip(params[1:],gradients[1:]):v.add_(g,alpha=-self.config['fine_learning_rate'])
            after=torch.softmax(self.free_head(self.adapter(u)),1)
            drift=float(torch.max(torch.abs(after-before)))
            qdrift=float(torch.max(torch.abs((after-before)@self.pattern_bank)))
            norm=float(torch.linalg.vector_norm(self.adapter.weight))
            condition=float(torch.linalg.cond(self.adapter.weight))
        if self.method=='null_conditioned' and max(drift,qdrift)>self.config['preservation_probability_atol']:
            raise ArithmeticError('null step probability preservation failed')
        if (not all(torch.isfinite(v).all() for v in params) or not condition<=self.config['maximum_adapter_condition']
                or norm>self.config['maximum_adapter_frobenius_norm']):
            raise ArithmeticError('adapter conditioning/norm failure')
        return {'fine_objective':float(objective.detach()),'proposal_norm':float(torch.linalg.vector_norm(proposal)),
                'target_norm':float(target),'applied_norm':float(torch.linalg.vector_norm(delta)),
                'readout_rank':rank,'projector_relative_residual':residual,'softmax_drift':drift,'query_drift':qdrift,
                'adapter_norm':norm,'adapter_condition':condition}
