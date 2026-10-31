import numpy as np
from agentic_research.violin_raw_v1.models import patterns,pattern_ids,participant_weights,transition_prior,fit_predict

CONF={'hgb_iterations':2,'hidden':8,'learning_rate':.003,'weight_decay':.0001,'epochs':2,'batch_size':16}


def fixture():
 rng=np.random.default_rng(17);x=rng.normal(size=(60,7));y=patterns()[rng.integers(0,10,60)]
 states=rng.integers(0,11,60);people=np.array(['p'+str(i//10) for i in range(60)])
 return x,y,states,people


def test_patterns_and_weights():
 b=patterns();assert b.shape==(10,9);assert np.linalg.matrix_rank(np.vstack((np.ones(10),b.T)))==10
 np.testing.assert_array_equal(pattern_ids(b),np.arange(10))
 ids=np.array(['a','a','b']);w=participant_weights(ids)
 assert w[:2].sum()==w[2]


def test_prior_unseen_and_probability():
 x,y,states,people=fixture();states[:]=0
 gp,table=transition_prior(y,states,participant_weights(people))
 np.testing.assert_allclose(table[10],gp);np.testing.assert_allclose(table.sum(1),1)


def test_fit_only_train_and_eval_prefix_invariance():
 x,y,states,people=fixture()
 for method in ('frequency','markov','hgb','mlp_ind','mlp_joint','mlp_offset'):
  p,meta=fit_predict(method,0,x,y,states,people,x[:5],states[:5],CONF)
  # Extra future examples cannot change a fixed prefix's prediction or preprocessing.
  p2,meta2=fit_predict(method,0,x,y,states,people,np.r_[x[:5],x[5:]*100],states,CONF)
  np.testing.assert_allclose(p,p2[:5],atol=1e-7,rtol=0)
  if 'scaler_mean' in meta:np.testing.assert_array_equal(meta['scaler_mean'],x.mean(0))
  assert p.shape==(5,9)
  if method!='mlp_ind' and method!='hgb':
   masses=np.column_stack((np.ones(5),p))@np.linalg.inv(np.vstack((np.ones(10),patterns().T))).T
   assert masses.min()>-1e-12
