"""Closed trusted prior registry; learned priors never consume development labels."""
from __future__ import annotations
import time
import numpy as np
from .data import conditional_features
from .run_distribution import first_run


def torch_run_log_probs(logits):
    """Differentiable exact independent first-2-run log probabilities, stable log space."""
    import torch
    import torch.nn.functional as F
    n = logits.shape[-1]
    if n < 2:
        raise ValueError('K=2 requires n>=2')
    a = torch.zeros_like(logits[..., 0])
    b = torch.full_like(a, -torch.inf)
    terms = []
    for t in range(n):
        lp, lq = F.logsigmoid(logits[..., t]), F.logsigmoid(-logits[..., t])
        if t >= 1:
            terms.append(b + lp)
        a, b = torch.logaddexp(a, b) + lq, a + lp
    terms.extend((torch.full_like(a, -torch.inf), torch.logaddexp(a, b)))
    return torch.stack(terms, dim=-1)


class Prior:
    def __init__(self, kind, thresholds, estimator=None, mean=None, scale=None, constant=None):
        self.kind, self.thresholds, self.estimator = kind, thresholds, estimator
        self.mean, self.scale, self.constant = mean, scale, constant

    def predict(self, public):
        x = conditional_features(np.asarray(public)[None, ...], self.thresholds)[0]
        if self.kind in ('constant', 'uniform'):
            return np.broadcast_to(self.constant[:, None], x.shape[:2]).copy()
        if self.kind.startswith('mlp'):
            import torch
            with torch.no_grad():
                a = torch.as_tensor((x-self.mean)/self.scale, dtype=torch.float32)
                p = torch.sigmoid(self.estimator(a).squeeze(-1)).numpy()
        else:
            p = self.estimator.predict_proba(x.reshape(-1, 5))[:, 1].reshape(x.shape[:2])
        return np.clip(p, 1e-6, 1-1e-6)

    def save(self, path):
        """Trusted generated model artifact; not loaded from arbitrary candidate paths."""
        import joblib
        if self.kind.startswith('mlp'):
            import torch
            torch.save({'state_dict': self.estimator.state_dict(), 'mean': self.mean,
                        'scale': self.scale, 'thresholds': self.thresholds}, path/'prior.pt')
        else:
            joblib.dump(self, path/'prior.joblib')


def fit_prior(config, train, thresholds):
    start = time.perf_counter()
    kind, seed = config.get('prior', 'constant'), int(config.get('seed', 0))
    public = np.stack([c.public for c in train])
    y = np.stack([c.labels for c in train]).astype(float)
    x = conditional_features(public, thresholds)
    info = {'prior': kind, 'train_columns': len(train), 'train_stroke_rule_labels': int(y.size),
            'inputs': ['log_measured_force', 'log_beta', 'log_measured_speed', 'tau_P', 'tau_R'],
            'no_development_labels_used': True}
    if kind in ('constant', 'uniform'):
        mean = (y.sum(axis=(0, 2))+.5)/(len(train)*y.shape[-1]+1.) if kind == 'constant' else np.full(len(thresholds), .5)
        model = Prior(kind, thresholds, constant=mean)
        info['training_objective'] = 'Per-rule source prevalence with Jeffreys pseudocount' if kind == 'constant' else 'Fixed Bernoulli probability 0.5; no labels fitted'
    elif kind in ('logreg', 'hgb'):
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler, PolynomialFeatures
        from sklearn.linear_model import LogisticRegression
        from sklearn.ensemble import HistGradientBoostingClassifier
        if kind == 'logreg':
            estimator = make_pipeline(PolynomialFeatures(2, include_bias=False), StandardScaler(),
                                      LogisticRegression(C=1., max_iter=600, random_state=seed))
        else:
            estimator = HistGradientBoostingClassifier(max_iter=60, max_leaf_nodes=15,
                min_samples_leaf=30, learning_rate=.08, l2_regularization=1.,
                early_stopping=False, random_state=seed)
        estimator.fit(x.reshape(-1, 5), y.reshape(-1).astype(int))
        model = Prior(kind, thresholds, estimator=estimator)
        info['training_objective'] = 'Pointwise binary classification'
    elif kind in ('mlp_bce', 'mlp_boundary'):
        import torch
        import torch.nn.functional as F
        torch.set_num_threads(2)
        torch.manual_seed(seed)
        hidden, epochs = int(config.get('hidden', 32)), int(config.get('mlp_epochs', 8))
        rate = float(config.get('learning_rate', .003))
        if not 4 <= hidden <= 128 or not 1 <= epochs <= 100 or not 1e-5 <= rate <= .1:
            raise ValueError('MLP configuration out of bounded registry')
        mean = x.mean(axis=(0, 1, 2))
        scale = np.maximum(x.std(axis=(0, 1, 2)), .01)
        xt = torch.as_tensor((x-mean)/scale, dtype=torch.float32)
        yt = torch.as_tensor(y, dtype=torch.float32)
        target = torch.as_tensor(first_run(y.astype(bool), 2), dtype=torch.long)
        net = torch.nn.Sequential(torch.nn.Linear(5, hidden), torch.nn.SiLU(),
                                  torch.nn.Linear(hidden, hidden), torch.nn.SiLU(), torch.nn.Linear(hidden, 1))
        opt = torch.optim.AdamW(net.parameters(), lr=rate, weight_decay=1e-4)
        generator = torch.Generator().manual_seed(seed)
        losses = []
        # Both neural ablations use identical initialization, column batches,
        # ordering and optimizer-step count. Extra DP compute is timed separately
        # through the complete fit duration, not hidden as equal-FLOPs training.
        for _ in range(epochs):
            order = torch.randperm(len(xt), generator=generator)
            total = 0.
            for ix in order.split(32):
                opt.zero_grad()
                logits = net(xt[ix]).squeeze(-1)
                bce = F.binary_cross_entropy_with_logits(logits, yt[ix])
                if kind == 'mlp_boundary':
                    logp = torch_run_log_probs(logits)
                    nll = -logp.gather(-1, target[ix][..., None]).mean()
                    loss = bce + nll
                else:
                    loss = bce
                if not torch.isfinite(loss):
                    raise FloatingPointError('Nonfinite training loss')
                loss.backward(); torch.nn.utils.clip_grad_norm_(net.parameters(), 5.); opt.step()
                total += float(loss.detach())*len(ix)
            losses.append(total/len(xt))
        objective = ('Pointwise BCE + weight 1 exact first-2-run categorical NLL (joint objective)'
                     if kind == 'mlp_boundary' else 'Pointwise BCE')
        net.eval()
        model = Prior(kind, thresholds, estimator=net, mean=mean, scale=scale)
        info.update(training_objective=objective, epoch_losses=losses, hidden=hidden,
                    epochs=epochs, learning_rate=rate, device='cpu', torch_threads=2,
                    batch_columns=32, optimizer_steps=epochs*((len(train)+31)//32))
    else:
        raise ValueError('Unknown prior')
    info['fit_seconds'] = time.perf_counter()-start
    return model, info
