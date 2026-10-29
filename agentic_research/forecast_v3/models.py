"""Source-only probabilistic models for a frozen future-window prediction task.

Independent and joint neural heads share architecture, initialization of their
backbone, minibatch permutations, update count, and marginal binary loss. The
joint head restricts outputs to mixtures of the ten observer rule patterns;
this is a constraint ablation, not a new general learning algorithm.
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import hashlib
import math
import time
import warnings
import numpy as np
from . import rule_patterns

METHODS = {'source_frequency', 'source_markov', 'persistence', 'hgb_history', 'mlp_ind', 'mlp_joint'}
PATTERNS = rule_patterns.admissible_patterns().astype(np.float64)
_AFFINE = np.vstack((np.ones(10), PATTERNS.T))
_AFFINE_INVERSE = np.linalg.inv(_AFFINE)


def normalize_config(config):
    fields = {'id','method','seed','epochs','hidden','learning_rate','batch_size','hgb_max_iter'}
    if not isinstance(config, dict) or set(config)-fields:
        raise ValueError('Unexpected model configuration fields')
    c = {'method':'mlp_joint','seed':0,'epochs':8,'hidden':32,
         'learning_rate':.003,'batch_size':512,'hgb_max_iter':80}
    c.update({k:v for k,v in config.items() if k != 'id'})
    if c['method'] not in METHODS: raise ValueError('Unknown forecast method')
    for key, lo, hi in [('seed',0,2**31-1),('epochs',1,100),('hidden',4,128),
                        ('batch_size',16,4096),('hgb_max_iter',1,500)]:
        if isinstance(c[key], bool) or not isinstance(c[key], int) or not lo <= c[key] <= hi:
            raise ValueError(f'Invalid {key}')
    if isinstance(c['learning_rate'], bool) or not isinstance(c['learning_rate'], (int,float)) or not math.isfinite(c['learning_rate']) or not 1e-6 <= c['learning_rate'] <= .1:
        raise ValueError('Invalid learning_rate')
    return c


class SourcePreprocessor:
    """Median imputation and standardization fitted only on provided source rows.

    All-missing source features have median zero. Nonfinite prefix values are
    missing. No target values, development distribution, or future missingness
    enters these statistics. A transform call never updates fitted parameters.
    """
    def fit(self, x):
        x = np.asarray(x, dtype=np.float64)
        if x.ndim != 2 or not x.shape[0] or not x.shape[1]:
            raise ValueError('Source features must be a nonempty matrix')
        z = np.where(np.isfinite(x), x, np.nan)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            median = np.nanmedian(z, axis=0)
        self.median_ = np.where(np.isfinite(median), median, 0.)
        z = np.where(np.isfinite(z), z, self.median_)
        self.mean_ = z.mean(axis=0)
        self.scale_ = z.std(axis=0)
        self.scale_ = np.where(self.scale_ > 1e-12, self.scale_, 1.)
        if not np.isfinite(self.mean_).all() or not np.isfinite(self.scale_).all():
            raise ValueError('Source preprocessing statistics are not finite')
        self.n_source_rows_ = x.shape[0]
        return self

    def transform(self, x):
        if not hasattr(self, 'median_'): raise ValueError('Preprocessor has not been fitted')
        x = np.asarray(x, dtype=np.float64)
        if x.ndim != 2 or x.shape[1] != len(self.median_):
            raise ValueError('Feature dimension differs from fitted source data')
        z = (np.where(np.isfinite(x), x, self.median_)-self.mean_)/self.scale_
        z = z.astype(np.float32)
        if not np.isfinite(z).all(): raise ValueError('Transformed features exceed finite float32 range')
        return z

    def fit_transform(self, x): return self.fit(x).transform(x)

    def save(self, path):
        with Path(path).open('xb') as f:
            np.savez_compressed(f, median=self.median_, mean=self.mean_, scale=self.scale_,
                                n_source_rows=np.asarray(self.n_source_rows_))


def validate_probabilities(probabilities):
    p = np.asarray(probabilities, dtype=np.float64)
    if p.ndim != 2 or p.shape[1] != 9 or not len(p) or not np.isfinite(p).all() or np.any((p < -1e-12) | (p > 1+1e-12)):
        raise ValueError('Expected a nonempty, finite N-by-9 probability matrix in [0,1]')
    # Permit only roundoff from summing normalized pattern mixtures.
    return np.clip(p,0.,1.)


def signed_pattern_masses(probabilities):
    """Unique affine pattern coordinates; negative entries indicate incoherence."""
    p = validate_probabilities(probabilities)
    return np.column_stack((np.ones(len(p)), p)) @ _AFFINE_INVERSE.T


def simplex_projection(values):
    """Euclidean projection onto the probability simplex, independently per row."""
    v = np.asarray(values, dtype=np.float64)
    if v.ndim != 2 or not v.shape[0] or not v.shape[1] or not np.isfinite(v).all():
        raise ValueError('Expected a nonempty finite matrix')
    u = np.sort(v, axis=1)[:, ::-1]
    cssv = np.cumsum(u, axis=1)-1
    j = np.arange(1, v.shape[1]+1)
    rho = (u-cssv/j > 0).sum(axis=1)-1
    theta = cssv[np.arange(len(v)),rho]/(rho+1)
    projected = np.maximum(v-theta[:,None], 0.)
    return projected/projected.sum(axis=1,keepdims=True)


def project_marginals(probabilities):
    """Source-free coherence comparator; project *latent masses*, not marginals.

    This is NOT the nearest coherent vector under marginal probability error.
    No labels or evaluation statistics are used. Valid pattern mixtures remain
    unchanged up to floating point error.
    """
    masses = simplex_projection(signed_pattern_masses(probabilities))
    return masses @ PATTERNS, masses


def coherence(probabilities, tolerance=1e-6):
    p = validate_probabilities(probabilities)
    masses = signed_pattern_masses(p)
    q = p.reshape(-1,3,3)
    order_bad = ((q[:,1:,:]-q[:,:-1,:] > tolerance).any(axis=(1,2)) |
                 (q[:,:,:-1]-q[:,:,1:] > tolerance).any(axis=(1,2)))
    bad = (masses < -tolerance).any(axis=1)
    return {'coherent_fraction':float(np.mean(~bad)),
            'incoherent_rows':int(bad.sum()), 'order_violation_rows':int(order_bad.sum()),
            'minimum_signed_pattern_mass':float(masses.min()),
            'mean_negative_pattern_mass':float(np.maximum(-masses,0).sum(axis=1).mean()),
            'tolerance':tolerance}


def make_network(input_dim, hidden, output_dim, seed):
    import torch
    torch.manual_seed(seed)
    # Head is initialized after the backbone; its size cannot alter backbone initialization.
    return torch.nn.Sequential(torch.nn.Linear(input_dim,hidden),torch.nn.ReLU(),
                               torch.nn.Linear(hidden,hidden),torch.nn.ReLU(),
                               torch.nn.Linear(hidden,output_dim))


@dataclass
class Prediction:
    probabilities: np.ndarray
    pattern_probabilities: np.ndarray | None
    info: dict


def _training_sha(x,y):
    h = hashlib.sha256()
    for a in (x,y):
        a = np.ascontiguousarray(a)
        h.update(str(a.shape).encode());h.update(a.dtype.str.encode());h.update(a.tobytes())
    return h.hexdigest()


def fit_predict(train_x, train_y, eval_x, config, *, train_prefix_labels=None, eval_prefix_labels=None, output_dir=None):
    """Fit exclusively on train_x/train_y; evaluation labels are not accepted.

    eval_prefix_labels is the already observed latest prefix observer output,
    used by persistence/Markov. train_prefix_labels supplies source-only transition
    pairs to Markov. Identity/session information is not accepted.
    """
    c = normalize_config(config)
    x = np.asarray(train_x,dtype=np.float32); y = np.asarray(train_y)
    target = np.asarray(eval_x,dtype=np.float32)
    if x.ndim != 2 or target.ndim != 2 or x.shape[1] != target.shape[1] or not len(x) or not len(target):
        raise ValueError('Incompatible or empty train/evaluation feature matrices')
    if y.shape != (len(x),9): raise ValueError('Invalid source label dimensions')
    y_ids = rule_patterns.pattern_ids(y)
    out = Path(output_dir) if output_dir is not None else None
    if out is not None: out.mkdir(parents=True,exist_ok=True)
    wall_start=time.perf_counter();cpu_start=time.process_time()
    info={'config':c,'source_rows':len(x),'feature_count':x.shape[1],
          'source_training_sha256':_training_sha(x,y),'preprocessing_fit_split':'train',
          'prediction_kind':'future observer-rule probabilities; not physical truth',
          'optimizer_updates':0}
    pattern_p=None;preprocessor=None
    method=c['method']
    if method == 'persistence':
        if eval_prefix_labels is None: raise ValueError('Persistence requires observed prefix labels')
        ids=rule_patterns.pattern_ids(eval_prefix_labels)
        if len(ids)!=len(target): raise ValueError('Persistence prefix rows do not align')
        pattern_p=np.eye(10)[ids];prob=pattern_p@PATTERNS
        info['loss']='none; hard persistence';fit_end=time.perf_counter()
    elif method == 'source_frequency':
        counts=np.bincount(y_ids,minlength=10).astype(float)
        # Jeffreys smoothing of each marginal via half mass on empty and full patterns.
        counts[np.flatnonzero(PATTERNS.sum(1)==0)[0]]+=.5
        counts[np.flatnonzero(PATTERNS.sum(1)==9)[0]]+=.5
        frequency=counts/counts.sum()
        pattern_p=np.tile(frequency,(len(target),1));prob=pattern_p@PATTERNS
        if out is not None:
            with (out/'model.npz').open('xb') as f:np.savez_compressed(f,pattern_frequencies=frequency)
        info['loss']='source empirical frequency with marginal Jeffreys smoothing';fit_end=time.perf_counter()
    elif method == 'source_markov':
        if train_prefix_labels is None or eval_prefix_labels is None:
            raise ValueError('Source Markov requires source and observed evaluation prefix labels')
        source_ids=rule_patterns.pattern_ids(train_prefix_labels)
        observed_ids=rule_patterns.pattern_ids(eval_prefix_labels)
        if source_ids.shape!=(len(x),) or observed_ids.shape!=(len(target),):
            raise ValueError('Markov prefix labels do not align')
        counts=np.zeros((10,10),dtype=float)
        np.add.at(counts,(source_ids,y_ids),1.)
        prior=(np.bincount(y_ids,minlength=10)+.1)/(len(y_ids)+1.)
        conditional=(counts+.1)/(counts.sum(axis=1,keepdims=True)+1.)
        absent=counts.sum(axis=1)==0
        conditional[absent]=prior
        pattern_p=conditional[observed_ids];prob=pattern_p@PATTERNS
        if out is not None:
            with (out/'model.npz').open('xb') as f:
                np.savez_compressed(f,conditional_pattern_probabilities=conditional,
                    source_global_pattern_prior=prior,source_transition_counts=counts)
        info.update(loss='source-only empirical conditional pattern frequency',
                    learned_from_source=True,dirichlet_total_mass=1.,
                    dirichlet_mass_per_pattern=.1,empty_prefix_fallback='source global pattern prior',
                    source_unseen_prefix_patterns=int(absent.sum()))
        fit_end=time.perf_counter()
    else:
        preprocessor=SourcePreprocessor()
        source=preprocessor.fit_transform(x); destination=preprocessor.transform(target)
        if out is not None:preprocessor.save(out/'preprocessor.npz')
        if method == 'hgb_history':
            from sklearn.ensemble import HistGradientBoostingClassifier
            from threadpoolctl import threadpool_limits
            models=[]
            with threadpool_limits(limits=2):
                for j in range(9):
                    if np.unique(y[:,j]).size == 1:
                        models.append(float((y[:,j].sum()+.5)/(len(y)+1)))
                    else:
                        model=HistGradientBoostingClassifier(max_iter=c['hgb_max_iter'],
                            max_leaf_nodes=15,learning_rate=.08,l2_regularization=1.,
                            min_samples_leaf=20,early_stopping=False,random_state=c['seed'])
                        model.fit(source,y[:,j]);models.append(model)
                fit_end=time.perf_counter()
                prob=np.column_stack([np.full(len(target),m) if isinstance(m,float)
                                      else m.predict_proba(destination)[:,1] for m in models])
            if out is not None:
                import joblib
                with (out/'model.joblib').open('xb') as f:joblib.dump(models,f)
            info.update(loss='independent binary log loss',hgb_early_stopping=False,
                        hgb_max_iter=c['hgb_max_iter'],hgb_max_leaf_nodes=15,
                        hgb_learning_rate=.08,hgb_l2_regularization=1.)
        else:
            import torch
            torch.set_num_threads(2)
            torch.use_deterministic_algorithms(True)
            network=make_network(source.shape[1],c['hidden'],10 if method=='mlp_joint' else 9,c['seed'])
            optimizer=torch.optim.AdamW(network.parameters(),lr=c['learning_rate'],weight_decay=1e-4)
            tx=torch.from_numpy(source);ty=torch.from_numpy(y.astype(np.float32))
            generator=torch.Generator().manual_seed(c['seed']+10003)
            order_hash=hashlib.sha256();losses=[];updates=0
            network.train()
            for epoch in range(c['epochs']):
                order=torch.randperm(len(tx),generator=generator)
                order_hash.update(order.numpy().tobytes());total=0.
                for start in range(0,len(tx),c['batch_size']):
                    idx=order[start:start+c['batch_size']];logits=network(tx[idx])
                    loss=(rule_patterns.per_rule_bce(logits,ty[idx]) if method=='mlp_joint'
                          else torch.nn.functional.binary_cross_entropy_with_logits(logits,ty[idx]))
                    if not torch.isfinite(loss): raise FloatingPointError('Source training loss is nonfinite')
                    optimizer.zero_grad();loss.backward();optimizer.step()
                    total+=float(loss.detach())*len(idx);updates+=1
                losses.append(total/len(tx))
            fit_end=time.perf_counter();network.eval();predictions=[]
            with torch.no_grad():
                for start in range(0,len(destination),2048):
                    logits=network(torch.from_numpy(destination[start:start+2048]))
                    probabilities=torch.softmax(logits,-1) if method=='mlp_joint' else torch.sigmoid(logits)
                    predictions.append(probabilities.numpy().astype(np.float64))
            predicted=np.concatenate(predictions)
            if method=='mlp_joint':
                pattern_p=predicted/predicted.sum(axis=1,keepdims=True);prob=pattern_p@PATTERNS
            else:prob=predicted
            if out is not None:
                with (out/'model.pt').open('xb') as f:
                    torch.save({'state_dict':network.state_dict(),'config':c,'input_dim':source.shape[1]},f)
            info.update(loss='mean marginal binary cross-entropy across nine rules',
                        optimizer='AdamW',weight_decay=1e-4,
                        optimizer_updates=updates,minibatch_order_sha256=order_hash.hexdigest(),
                        train_loss_by_epoch=losses,parameter_count=sum(p.numel() for p in network.parameters()),
                        deterministic_algorithms=True,early_stopping=False)
    prob=validate_probabilities(prob)
    info.update(fit_seconds=fit_end-wall_start,total_seconds=time.perf_counter()-wall_start,
                cpu_seconds=time.process_time()-cpu_start)
    info['prediction_and_artifact_seconds']=info['total_seconds']-info['fit_seconds']
    return Prediction(prob,pattern_p,info)
