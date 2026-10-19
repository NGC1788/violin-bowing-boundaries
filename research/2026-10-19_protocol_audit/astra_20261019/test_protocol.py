import numpy as np
import protocol_falsification as p
import descriptor_only as d


def test_nominal_ranks_reverse_only_A1():
    assert np.isclose(p.nominal_force('A1',[1])[0],4.)
    assert np.isclose(p.nominal_force('A2',[1])[0],.1)
    np.testing.assert_allclose(p.nominal_force('A2',[1,51,101]),[.1,.1,.1])


def test_crossvalidation_never_uses_held_group_response():
    x=np.c_[np.ones(6),np.arange(6.)];y=np.array([0.,1.,2.,4.,5.,6.]);g=np.array([0,0,1,1,2,2])
    a=p.cv_predict(x,y,g);y[:2]+=1000;b=p.cv_predict(x,y,g)
    np.testing.assert_allclose(a[:2],b[:2])


def test_interval_fit_recovers_known_location_without_cdf_cancellation():
    # Ten narrow intervals symmetrically around 10; wrong initial sigma creates far tails.
    center=10.+np.linspace(-.5,.5,10)
    rows=[{'lo':float(c-.02),'hi':float(c+.02)} for c in center]
    fit=p.censored(rows,np.ones((10,1)))
    assert abs(fit['coef'][0]-10)<1e-5
    assert fit['converged_starts']>=1


def test_descriptor_reads_no_bridge_or_label_columns():
    assert not any(s in d.FIELDS for s in ['H','periodicity','f0_hz','amplitude_std','flyback_height'])


def test_descriptor_is_order_invariant_and_uses_force_rank():
    fields={k:[] for k in d.FIELDS}
    for v,folder in zip((.05,.1,.2),('v1','v2','v3')):
        for trial in range(1,2001):
            for t in (0.,.05/v,.05/v+.1):
                row=dict(folder=folder,trial=trial,window_s=.1,t=t,force=1.-v*t,speed=v,error='')
                for k in fields:fields[k].append(row[k])
    fields={k:np.array(v) for k,v in fields.items()}
    out=d.from_columns(fields)
    assert abs(out['dec5_pl_N']+.05)<1e-12
    order=np.random.default_rng(3).permutation(len(fields['trial']))
    assert d.from_columns({k:v[order] for k,v in fields.items()})==out
