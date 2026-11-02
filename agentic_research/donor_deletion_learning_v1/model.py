"""Grouped empirical-pattern retrieval. No IO, fitting, or label access hooks."""
import copy
import torch
from torch import nn
from ..exact_count_learning_v1.model import Forecaster, parameterize, auxiliary_nll, bank

DTYPE = torch.float64


def event_losses(probability, labels):
    # Fixed prior puts every event strictly inside the score clip interval.
    if not bool(torch.isfinite(probability).all()) or bool(((probability <= 0) | (probability >= 1)).any()):
        raise ValueError('event probabilities must lie strictly inside (0,1)')
    return -(labels[:, None] * probability.log() +
             (1-labels[:, None]) * torch.log1p(-probability)).mean(-1)


def deletion_probabilities(scores, memory_labels, memory_groups, query_groups, rho):
    """All null/group-deleted distributions using stable positive remainder sums.

    query_group=-1 means a new held query with no same-person training memory.
    Memory group codes must be contiguous 0..G-1. No query future labels here.
    """
    if scores.ndim != 2 or memory_labels.shape != (scores.shape[1], 9):
        raise ValueError('memory score/label shape')
    if memory_groups.shape != (scores.shape[1],) or query_groups.shape != (scores.shape[0],):
        raise ValueError('group shape')
    if memory_groups.dtype != torch.long or query_groups.dtype != torch.long:
        raise ValueError('integer groups required')
    if not bool(torch.isfinite(scores).all()) or not (0 < rho < 1):
        raise ValueError('finite scores and positive fixed prior required')
    patterns = bank()
    if not bool((memory_labels[:, None] == patterns[None]).all(-1).any(-1).all()):
        raise ValueError('memory labels are not attainable joint patterns')
    codes = torch.unique(memory_groups, sorted=True)
    if len(codes) < 2 or not torch.equal(codes, torch.arange(len(codes))):
        raise ValueError('at least two contiguous memory donor groups required')
    if bool(((query_groups < -1) | (query_groups >= len(codes))).any()):
        raise ValueError('unknown query donor code')
    group_logmass, conditional, positives, negatives = [], [], [], []
    for g in codes:
        mask = memory_groups == g
        current = scores[:, mask]
        group_logmass.append(torch.logsumexp(current, -1))
        conditional.append(torch.softmax(current, -1) @ memory_labels[mask])
        positives.append(memory_labels[mask].sum(0))
        negatives.append((1-memory_labels[mask]).sum(0))
    group_logmass = torch.stack(group_logmass, 1)
    conditional = torch.stack(conditional, 1)
    allowed_group = codes[None] != query_groups[:, None]
    # Intervention axis: null followed by exactly one group removed.
    removed = torch.arange(-1, len(codes))
    remaining = allowed_group[:, None, :] & (codes[None, None] != removed[None, :, None])
    if not bool(remaining.any(-1).all()):
        raise ValueError('empty donor remainder under an allowed intervention')
    logits = group_logmass[:, None, :].expand(-1, len(removed), -1)
    mixing = torch.softmax(logits.masked_fill(~remaining, -torch.inf), -1)
    unsmoothed = torch.einsum('bdg,bgr->bdr', mixing, conditional)
    probability = (1-rho)*unsmoothed + rho*patterns.mean(0)
    # Deleting an already-excluded query person is a duplicate null, excluded
    # from the random/max intervention menu though its forecast is stored.
    eligible = (removed[None] == -1) | (removed[None] != query_groups[:, None])
    pos = torch.einsum('bdg,gr->bdr', remaining.to(DTYPE), torch.stack(positives))
    neg = torch.einsum('bdg,gr->bdr', remaining.to(DTYPE), torch.stack(negatives))
    return {'probability': probability, 'eligible': eligible, 'donor_codes': removed,
            'group_mass': mixing[:, 0], 'positive_support': pos, 'negative_support': neg}


class RetrievalForecaster(Forecaster):
    def __init__(self, method, config):
        if method not in config['methods']:
            raise ValueError('undeclared method')
        # Same initialization draw order and exact shared architecture as the
        # old count head (no free head), without ever invoking its decoder.
        base = copy.deepcopy(config)
        base['methods'] = ['exact_count']
        super().__init__('exact_count', base)
        self.method, self.config = method, config

    def forward(self, prefix, memory_prefix, memory_labels, memory_groups, query_groups):
        if (prefix.ndim != 2 or memory_prefix.ndim != 2 or
            prefix.shape[1] != self.config['input_dimension'] or
            memory_prefix.shape[1] != prefix.shape[1] or
            not bool(torch.isfinite(prefix).all()) or not bool(torch.isfinite(memory_prefix).all())):
            raise ValueError('finite prefix-only features required')
        hidden = self.encoder(prefix)
        keys = self.encoder(memory_prefix)
        query = nn.functional.normalize(hidden, dim=-1, eps=1e-12)
        key = nn.functional.normalize(keys, dim=-1, eps=1e-12)
        scores = (query @ key.T) / self.config['retrieval_temperature']
        if self.config['retrieval_group_count_correction']:
            counts = torch.bincount(memory_groups)
            scores = scores - counts[memory_groups].to(DTYPE).log()[None]
        info = deletion_probabilities(scores, memory_labels, memory_groups, query_groups,
                                      self.config['retrieval_prior_mass'])
        return info, parameterize(self.generator(hidden), self.config)

    def loss_rows(self, prefix, memory, labels, pitch, power, voiced, query_groups, random_generator):
        info, parameters = self(prefix, memory['prefix'], memory['labels'], memory['groups'], query_groups)
        losses = event_losses(info['probability'], labels)
        if self.method == 'retrieval_adversarial_delete':
            primary = losses.masked_fill(~info['eligible'], -torch.inf).max(1).values
        elif self.method == 'retrieval_random_delete':
            chosen = torch.multinomial(info['eligible'].to(DTYPE), 1, generator=random_generator).squeeze(1)
            primary = losses.gather(1, chosen[:, None]).squeeze(1)
        else:
            primary = losses[:, 0]
        auxiliary = auxiliary_nll(parameters, pitch, power, voiced, self.config['future_blocks'])
        return primary, auxiliary, info


class QueryDRO:
    """Fixed exp-weights control; not an exact solved max-group oracle."""
    def __init__(self, groups, eta):
        self.counts = torch.bincount(groups).to(DTYPE)
        if not bool((self.counts > 0).all()):
            raise ValueError('all training groups must be nonempty')
        self.logweights = torch.zeros(len(self.counts), dtype=DTYPE)
        self.eta = eta

    def multipliers(self, groups):
        return len(self.counts)*torch.softmax(self.logweights, 0)[groups]

    def update(self, losses, groups):
        # Shuffled-minibatch importance-scaled group-risk estimate. Adaptive
        # randperm iterates are not conditionally unbiased full-data oracles.
        # Update after current objective; auxiliary stays person-macro.
        sums = torch.zeros_like(self.counts).index_add_(0, groups, losses.detach())
        estimate = sums / len(groups) * self.counts.sum() / self.counts
        self.logweights += self.eta*estimate
        self.logweights -= torch.logsumexp(self.logweights, 0)
