"""Conventional calibration and risk-test controls; no new-method claims."""
import math
import numpy as np
from scipy.stats import binom
from ..violin_raw_v1.metrics import EPSILON
from ..violin_raw_v1.models import patterns


def row_loss(y,p):
    p=np.clip(p,EPSILON,1-EPSILON)
    return -(y[None]*np.log(p)+(1-y[None])*np.log1p(-p)).mean((0,2))


def person_loss(y,p,people):
    loss=row_loss(y,p)
    return np.asarray([loss[people==pid].mean() for pid in np.unique(people)])


def fit_alpha(y,b,p,weights,upper=.95):
    """Certified 1-D convex minimization on an explicitly unclipped interval.

    Shape p=seeds x rows x rules, b=rows x rules. Fixed row weights preserve
    person-macro weighting even when optimizing one prefix-state contribution.
    """
    if len(y)==0:return {'alpha':0.,'status':'empty_fallback','iterations':0,'bracket_width':0.}
    weights=np.asarray(weights,float);weights=weights/weights.sum()
    if (not 0<upper<1 or not np.isfinite(p).all() or
        not np.all((p>=0)&(p<=1)) or not np.all((b>EPSILON)&(b<1-EPSILON))):
        raise ValueError('invalid mixture or baseline support')
    end=b[None]+upper*(p-b[None])
    if np.any(end<=EPSILON) or np.any(end>=1-EPSILON):
        raise ValueError('clipping would invalidate the global convexity certificate')
    delta=p-b[None]
    def derivative(alpha):
        q=b[None]+alpha*delta
        return float(np.sum(weights*(((q-y[None])/(q*(1-q))*delta).mean((0,2)))))
    d0,d1=derivative(0.),derivative(upper)
    if d0>=0:return {'alpha':0.,'status':'lower_endpoint','iterations':0,'bracket_width':0.,'derivative':d0}
    if d1<=0:return {'alpha':upper,'status':'upper_endpoint','iterations':0,'bracket_width':0.,'derivative':d1}
    lo,hi=0.,upper
    for iteration in range(64):
        mid=(lo+hi)/2
        if derivative(mid)<0:lo=mid
        else:hi=mid
        if hi-lo<1e-11:break
    return {'alpha':(lo+hi)/2,'status':'interior','iterations':iteration+1,
            'bracket_width':hi-lo,'derivative_left':derivative(lo),'derivative_right':derivative(hi)}


def calibrate(y,b,p,people,states,training_mask,statewise=False):
    """Training mask slices truth before passing it to any fitting function."""
    mask=np.asarray(training_mask,bool)
    if mask.shape!=(len(y),) or not mask.any():raise ValueError('empty calibration split')
    yt,bt,pt=y[mask],b[mask],p[:,mask]
    ids,st=people[mask],states[mask]
    counts={pid:int((ids==pid).sum()) for pid in np.unique(ids)}
    weights=np.asarray([1/counts[pid] for pid in ids])
    fits=[];alpha=np.zeros(len(y))
    groups=range(11) if statewise else [None]
    for state in groups:
        take=np.ones(len(yt),bool) if state is None else st==state
        fit=fit_alpha(yt[take],bt[take],pt[:,take],weights[take])
        fits.append(fit)
        alpha[np.ones(len(y),bool) if state is None else states==state]=fit['alpha']
    return b[None]+alpha[None,:,None]*(p-b[None]),fits


def cap_predictions(prior,residual,cap):
    """Known bounded logit shrinkage, no parameter estimation."""
    residual=residual-residual.mean(-1,keepdims=True)
    span=np.ptp(residual,axis=-1,keepdims=True)
    limited=residual*np.minimum(1.,cap/np.maximum(span,1e-300))
    logits=np.log(prior)[None]+limited
    mass=np.exp(logits-logits.max(-1,keepdims=True));mass/=mass.sum(-1,keepdims=True)
    return mass


def betting_evalue(gain):
    """Uniform mixture over fixed nonnegative bets in[0,1].

    Valid only if each next gain has conditional expectation<=0 under the null.
    Gauss-Legendre integrates this degree-n polynomial with positive weights.
    """
    u=np.asarray(gain,float)
    if np.any(np.abs(u)>1+1e-9):raise ValueError('gain is not bounded by1')
    u=np.clip(u,-1,1)
    nodes,weights=np.polynomial.legendre.leggauss(max(1,(len(u)+2)//2))
    lam=(nodes+1)/2
    return float(np.sum(weights/2*np.prod(1+lam[:,None]*u[None],axis=1)))


def hb_pvalue(difference,cap):
    """LTT Prop1 with correct e-factor and ceil; assumptions NOT verified here."""
    d=np.asarray(difference,float)
    if np.any(np.abs(d)>cap+1e-9):raise ValueError('difference outside declared cap')
    mean=float(np.clip(np.mean((d+cap)/(2*cap)),0,1));n=len(d)
    a=min(mean,.5)
    kl=(a*math.log(2*a) if a else 0)+((1-a)*math.log(2*(1-a)) if a<1 else 0)
    return min(1.,math.exp(-n*kl),math.e*float(binom.cdf(math.ceil(n*mean),n,.5)))


def math_controls():
    n=8
    same_positive=betting_evalue(np.ones(n));same_negative=betting_evalue(-np.ones(n))
    return {'scope':'exact synthetic controls, not a certificate on existing dev people',
        'independent_symmetric_null_mean_mixture_evalue':sum(math.comb(n,k)*betting_evalue(np.r_[np.ones(k),-np.ones(n-k)]) for k in range(n+1))/2**n,
        'all_equal_symmetric_dependent_null_mean_evalue':(same_positive+same_negative)/2,
        'all_equal_dependent_null_rejection_probability_at_20':float(.5*(same_positive>=20)+.5*(same_negative>=20)),
        'iid_best_case_all_gains_one_evalue':same_positive,
        'iid_best_case_all_gains_one_hb_pvalue':hb_pvalue(-np.ones(n),1.),
        'note':'Small n can support a nonvacuous test when gains are strong and consistent. Marginal zero-mean dependent gains alone do not suffice. This is a counterexample to a general independence substitution, not evidence that actual fold dependence takes this form.',
        'conventional_online_alternative':online_control()}


def online_control():
    """Known exponential-weights control, synthetic only; no novelty claim.

    Geometric mean of the correct-event probabilities is concave. Thus the
    average marginal log loss is1-exp-concave on the unclipped probability
    domain. Uniform joint smoothing keeps ALL outcome marginals in that domain.
    """
    rng=np.random.default_rng(20261102);weights=np.array([.5,.5])
    cumulative=np.zeros(2);learner=0.;smallest_jensen_gap=1.
    for t in range(8):
        rows=2+t%3
        mass=rng.dirichlet(np.ones(10)*2,size=(2,rows))
        mass=(1-1e-4)*mass+1e-4/10
        p=mass@patterns()
        if not np.all((p>EPSILON)&(p<1-EPSILON)):raise ValueError('online score clipping active')
        mixed=np.sum(weights[:,None,None]*p,axis=0)
        # Current labels are not used to choose this round's weights.
        y=patterns()[rng.integers(0,10,rows)]
        expert=np.array([row_loss(y,pi[None]).mean() for pi in p])
        actual=float(row_loss(y,mixed[None]).mean())
        gap=float(np.exp(-actual)-weights@np.exp(-expert))
        smallest_jensen_gap=min(smallest_jensen_gap,gap)
        if gap < -1e-12:raise ValueError('exp-concavity control failed')
        learner+=actual;cumulative+=expert
        weights*=np.exp(-expert);weights/=weights.sum()
    bound=math.log(2)
    if np.any(learner-cumulative>bound+1e-12):raise ValueError('known online regret bound failed')
    return {'scope':'8synthetic participant rounds, no real predictions or training',
        'expconcavity_min_gap':smallest_jensen_gap,
        'max_cumulative_regret_vs_two_experts':float(np.max(learner-cumulative)),
        'known_cumulative_bound_log2':bound,'mean_bound_at8people':bound/8,
        'mean_bound_at34people':bound/34,
        'online_protocol_requires':'predict each person before that person labels, then update weights; individual next-person risk and offline posthoc generalization are not guaranteed',
        'novelty':False}
