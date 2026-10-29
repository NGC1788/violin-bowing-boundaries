"""Independent exhaustive checks; only this audit directory is written.

No harness implementation is reused to produce reference outcomes/distributions.
"""
from pathlib import Path
import hashlib, itertools, json, sys, time
import numpy as np

ROOT = Path('/Users/jeongjihoon/violin-bowing-boundaries/agentic_research')
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from harness import run_distribution as impl


def reference_first(y, k):
    # Direct Boolean definition, without automaton or a running-count recurrence.
    hits = [i for i in range(len(y)-k+1) if tuple(y[i:i+k]) == (1,)*k]
    return min(hits) if hits else len(y)


def completion_outcomes(obs, k):
    missing = [i for i, v in enumerate(obs) if v == -1]
    result = set()
    for fill in itertools.product((0,1), repeat=len(missing)):
        y = list(obs)
        for i, v in zip(missing, fill):
            y[i] = v
        result.add(reference_first(y,k))
    return result


def reference_distribution(p, k):
    p = np.asarray(p, float)
    n = len(p)
    out = np.zeros(n+1)
    for y in itertools.product((0,1), repeat=n):
        mass = float(np.prod([p[i] if bit else 1-p[i] for i, bit in enumerate(y)]))
        out[reference_first(y,k)] += mass
    return out


def reference_entropy(p):
    return -sum(float(x)*np.log(float(x)) for x in p if x > 0)


def minimum_certificate(y, known_outcomes):
    n = len(y)
    for size in range(n+1):
        for subset in itertools.combinations(range(n), size):
            obs = tuple(y[i] if i in subset else -1 for i in range(n))
            if len(known_outcomes[obs]) == 1:
                return size
    raise AssertionError('Full observation must certify')


def main():
    start = time.perf_counter()
    counts = dict(certificates=0, first_runs=0, minimum_certificates=0,
                  distributions=0, information_gains=0, joint_lower_bounds=0)
    max_prob_error, max_gain_error = 0., 0.
    rng = np.random.default_rng(20261030)
    for n in range(1,8):
        full = list(itertools.product((0,1), repeat=n))
        for k in range(1,n+1):
            cases = {}
            for obs in itertools.product((-1,0,1), repeat=n):
                outcomes = completion_outcomes(obs,k)
                cases[obs] = outcomes
                done, lo, hi = impl.certificate(obs,k)
                assert bool(done) == (len(outcomes)==1), (n,k,obs,outcomes,done)
                assert int(lo)==min(outcomes) and int(hi)==max(outcomes)
                counts['certificates'] += 1
            for y in full:
                assert int(impl.first_run(y,k)) == reference_first(y,k)
                counts['first_runs'] += 1
                expected = minimum_certificate(y,cases)
                actual, per_rule = impl.certificate_lower_bound(y,k)
                assert actual == expected and per_rule == [expected], (n,k,y,actual,expected)
                counts['minimum_certificates'] += 1
            # Include deterministic/near-deterministic priors and partial observations.
            probs = [np.zeros(n), np.ones(n), np.full(n,.5),
                     np.linspace(1e-8,1-1e-8,n), *rng.uniform(.02,.98,(5,n))]
            for p in probs:
                ref = reference_distribution(p,k)
                got = impl.run_distribution(p,k)
                err = float(np.max(np.abs(ref-got)))
                max_prob_error = max(max_prob_error,err)
                assert err<1e-12 and abs(got.sum()-1)<1e-12
                counts['distributions'] += 1
            for _ in range(3):
                p = rng.uniform(.02,.98,(3,n))
                known = rng.random(n)<.3
                for i in np.flatnonzero(known):
                    p[:,i] = rng.integers(0,2,3)
                unknown = ~known
                if not unknown.any():
                    unknown[0] = True
                got = impl.information_gain(p,unknown,k)
                base = sum(reference_entropy(reference_distribution(row,k)) for row in p)
                for i in np.flatnonzero(unknown):
                    expected_h=0.
                    for row in p:
                        for bit in (0,1):
                            conditioned=row.copy(); conditioned[i]=bit
                            mass=row[i] if bit else 1-row[i]
                            expected_h += mass*reference_entropy(reference_distribution(conditioned,k))
                    ref = base-expected_h
                    err=abs(float(got[i])-ref)
                    max_gain_error=max(max_gain_error,err)
                    assert err<2e-12,(n,k,i,got[i],ref)
                    counts['information_gains']+=1
                assert np.isneginf(got[known & ~unknown]).all()
            if n<=5:
                # Nine jointly revealed rules: max of individual minima is lower bound,
                # NOT necessarily the joint minimum.
                for _ in range(8):
                    ys=rng.integers(0,2,(9,n))
                    joint_min=None
                    for size in range(n+1):
                        for subset in itertools.combinations(range(n),size):
                            if all(len(cases[tuple(int(row[i]) if i in subset else -1 for i in range(n))])==1 for row in ys):
                                joint_min=size;break
                        if joint_min is not None:break
                    lower, per=impl.certificate_lower_bound(ys,k)
                    assert lower==max(per) and lower<=joint_min
                    counts['joint_lower_bounds']+=1
        print(f'n={n} exhaustive done',flush=True)
    model_report={}
    try:
        import torch
        from harness.models import torch_run_log_probs
        torch.manual_seed(123)
        max_torch_error=0.
        for n in range(2,8):
            logits=torch.randn((4,3,n),dtype=torch.float64,requires_grad=True)*3
            logs=torch_run_log_probs(logits)
            p=torch.sigmoid(logits).detach().numpy()
            ref=np.asarray([reference_distribution(row,2) for row in p.reshape(-1,n)]).reshape(4,3,n+1)
            err=float(np.max(np.abs(logs.exp().detach().numpy()-ref)))
            max_torch_error=max(max_torch_error,err)
            assert err<1e-12
            # Random finite valid category, including no-run.
            target=torch.tensor(rng.choice([*range(n-1),n],size=(4,3)))
            loss=-logs.gather(-1,target[...,None]).mean()
            grad=torch.autograd.grad(loss,logits)[0]
            assert torch.isfinite(grad).all()
        model_report={'status':'PASS','max_distribution_error':max_torch_error,'finite_gradient':True}
    except ImportError as exc:
        model_report={'status':'NOT_RUN','reason':str(exc)}
    report={'status':'PASS','scope':'Independent enumeration n<=7, all K; no fits',
            'counts':counts,'max_probability_error':max_prob_error,'max_information_gain_error':max_gain_error,
            'torch_differentiable_layer':model_report,'seconds':time.perf_counter()-start,
            'source_hashes':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
                             for p in (ROOT/'harness').glob('*.py')}}
    (HERE/'brute_force_results.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
