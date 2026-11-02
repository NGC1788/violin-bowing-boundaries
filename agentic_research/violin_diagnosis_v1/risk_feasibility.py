"""Algebraic feasibility checks, not model fitting or an empirical guarantee.

These elementary bounds are controls/obstructions, not new theorems.
"""
from pathlib import Path
import json
import math
import numpy as np
from ..violin_raw_v1.models import patterns
from .diagnose import EPS,sha


def outcome_scores(q):
    """Actual nine marginal BCE scores for each of the ten possible outcomes."""
    a = patterns(); p = np.clip(np.asarray(q) @ a,EPS,1-EPS)
    return -(a*np.log(p)+(1-a)*np.log1p(-p)).mean(1)


def logit_correct(b, residual):
    logq=np.log(b)+residual
    q=np.exp(logq-logq.max());return q/q.sum()


def independent_person_hoeffding_width(span, people, delta=.05):
    # A logit-span cap gives each person's paired mean score difference in [-c,c].
    return span*math.sqrt(2*math.log(1/delta)/people)


def main():
    here=Path(__file__).resolve().parent
    summary=json.loads((here/'SUMMARY.json').read_text())
    # Illustrative target from existing DEV only; not a newly registered criterion.
    if sha(here/'local/diagnostics.json') != summary['diagnostics_sha256']:
        raise ValueError('diagnosis identity changed')
    diagnostic=json.loads((here/'local/diagnostics.json').read_text())
    baseline=diagnostic['models']['markov']['dev']['log_loss']; desired=.02*baseline
    rng=np.random.default_rng(20261102); max_violation=0.; min_regret=1.
    for _ in range(32):
        b=rng.dirichlet(np.ones(10)*2); r=rng.normal(size=10)
        q=logit_correct(b,r); c=float(np.ptp(r))
        d=outcome_scores(q)-outcome_scores(b)
        max_violation=max(max_violation,float(np.max(np.abs(d))-c))
        # E_b excess equals the mean Bernoulli KL of its nine marginals.
        p0=b@patterns();p1=np.clip(q@patterns(),EPS,1-EPS)
        if not np.all((p0>EPS)&(p0<1-EPS)):
            raise ValueError('proper-score no-go check needs interior baseline marginals')
        kl=np.mean(p0*np.log(p0/p1)+(1-p0)*np.log((1-p0)/(1-p1)))
        if not np.isclose(b@d,kl,rtol=1e-10,atol=1e-12):
            raise ValueError('proper-score no-go identity failed')
        min_regret=min(min_regret,float(b@d))
        for alpha in (.01,.1,.5):
            mixed=(1-alpha)*b+alpha*q
            if np.max(outcome_scores(mixed)-outcome_scores(b)) > -math.log1p(-alpha)+1e-12:
                raise ValueError('mixture harm cap failed')
    result={
        'scope':'finite algebra checks and hypothetical sample-size arithmetic; no models, data rows or test labels',
        'new_models_fitted':0,'new_method_claim':False,'code_sha256':sha(Path(__file__)),
        'score':'mean of nine marginal BCE terms; not categorical joint NLL',
        'toy_checks':32,'max_logit_span_bound_violation':max_violation,
        'min_expected_excess_under_baseline_law_for_nontrivial_toys':min_regret,
        'pattern_map_augmented_rank':int(np.linalg.matrix_rank(np.column_stack([np.ones(10),patterns()]))),
        'no_go':'If uncertainty includes the baseline law and its marginals are strictly inside the clipping interval, exact robust noninferiority forces equal clipped marginals by proper scoring. Equality to interior baseline marginals then forces equal raw marginals. Here the augmented pattern map has full rank, hence the same pattern distribution. Do not extend the argument to baselines at/outside the clip boundaries.',
        'illustrative_dev_baseline_loss':baseline,'illustrative_two_percent_absolute_gain':desired,
        'hypothetical_independent_calibration_only':[
            {'logit_span_cap':c,'people':m,'delta':.05,
             'hoeffding_width':independent_person_hoeffding_width(c,m),
             'independent_people_needed_for_width_below_illustrative_gain':math.floor(2*c*c*math.log(20)/desired**2)+1}
            for c in (.025,.1) for m in (8,34)],
        'warning':'The actual 34 people trained the models, eight dev people informed this diagnosis, and nine test people were already evaluated. None is a fresh independent calibration set. Overlapping cross-fit losses are not iid people for this bound.',
        'diagnostic_source_sha256':summary['diagnostic_code_sha256']}
    (here/'RISK_FEASIBILITY.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
