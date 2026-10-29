"""Joint head for the ten admissible outcomes of one shared-guard 3×3 rule bank.

Rule order is row-major: tau_P=(.7,.8,.9), tau_R=(.3,.5,.7), and
H(P,R; tau_P,tau_R) = fixed OR (guard AND P>=tau_P AND R<tau_R).

Proof of the pattern count: increasing tau_P can only remove positives, so
active rows form a prefix of length a in {0,1,2,3}. Increasing tau_R can only
add positives, so active columns form a suffix of length b in {0,1,2,3}.
With one SHARED guard and fixed override the positive set is their Cartesian
product: one empty pattern plus 3*3 nonempty rectangles = ten. A false guard
gives the empty pattern; a true fixed override gives the full rectangle, both
already counted. Cross-rule monotonicity alone is weaker than this rectangle
property; not every monotone binary matrix belongs to this bank.

This is a structural parameterization using a categorical distribution, not a
new general method, trained result, or physical ground-truth claim. It assumes
exactly the shared guard/fixed convention above. An observer export audit must
precede use on future-window CSV targets; this module does not relabel data.
"""
from __future__ import annotations
import itertools
import numpy as np

PERIODICITY_THRESHOLDS = (.7, .8, .9)
RATIO_THRESHOLDS = (.3, .5, .7)
RULES = tuple(itertools.product(PERIODICITY_THRESHOLDS, RATIO_THRESHOLDS))


def boundary_representatives(thresholds):
    """Both adjacent floating values and equality at every threshold, sorted."""
    values = [v for t in thresholds for v in (np.nextafter(float(t), -np.inf),
                                               float(t), np.nextafter(float(t), np.inf))]
    return np.asarray(sorted(set(values)), dtype=np.float64)


def labels_from_features(periodicity, ratio, guard=True, fixed=False):
    """Broadcast scalar/array features; strict ratio '<' and periodicity '>='.

    Nonfinite values follow NumPy comparison semantics (NaN fails comparisons).
    This helper has no source-data I/O and does not infer observer guards.
    """
    p, r, g, f = np.broadcast_arrays(np.asarray(periodicity, dtype=float),
        np.asarray(ratio, dtype=float), np.asarray(guard, dtype=bool), np.asarray(fixed, dtype=bool))
    thresholds = np.asarray(RULES)
    return f[..., None] | (g[..., None] & (p[..., None] >= thresholds[:, 0])
                           & (r[..., None] < thresholds[:, 1]))


def _enumerate_patterns():
    rows = []
    for p, r, g, f in itertools.product(boundary_representatives(PERIODICITY_THRESHOLDS),
                                       boundary_representatives(RATIO_THRESHOLDS), (False, True), (False, True)):
        rows.append(labels_from_features(p, r, g, f))
    patterns = np.unique(np.asarray(rows, dtype=np.int8), axis=0)
    if patterns.shape != (10, 9):
        raise AssertionError('Shared-guard rule bank must contain exactly ten patterns')
    patterns.flags.writeable = False
    return patterns


_PATTERNS = _enumerate_patterns()
_LOOKUP = np.full(512, -1, dtype=np.int64)
_LOOKUP[_PATTERNS @ (2**np.arange(9))] = np.arange(10)
_LOOKUP.flags.writeable = False


def admissible_patterns():
    """Fresh 10×9 array; pattern IDs follow lexicographic binary-row order."""
    return _PATTERNS.copy()


def pattern_ids(labels):
    """Map ...×9 binary rule vectors to IDs; reject nonbinary/inadmissible vectors."""
    y = np.asarray(labels)
    if y.ndim < 1 or y.shape[-1] != 9 or not np.isin(y, [0, 1]).all():
        raise ValueError('Expected binary labels with last dimension nine')
    ids = _LOOKUP[y.astype(np.int64) @ (2**np.arange(9))]
    if np.any(ids < 0):
        raise ValueError('Labels violate the shared-guard rectangle rule structure')
    return ids


def _validate_logits(logits):
    import torch
    if not isinstance(logits, torch.Tensor) or logits.ndim < 1 or logits.shape[-1] != 10:
        raise ValueError('Expected a torch tensor with last dimension ten')
    if not logits.is_floating_point() or logits.numel() == 0 or not torch.isfinite(logits).all():
        raise ValueError('Logits must be nonempty, finite and floating-point')


def _torch_patterns(device, dtype):
    import torch
    return torch.tensor(_PATTERNS.tolist(), device=device, dtype=dtype)


def torch_pattern_ids(labels):
    import torch
    if not isinstance(labels, torch.Tensor) or labels.ndim < 1 or labels.shape[-1] != 9:
        raise ValueError('Expected a torch label tensor with last dimension nine')
    if not ((labels == 0) | (labels == 1)).all():
        raise ValueError('Rule labels must be binary')
    patterns = _torch_patterns(labels.device, labels.dtype)
    match = (labels[..., None, :] == patterns).all(dim=-1)
    if not match.any(dim=-1).all():
        raise ValueError('Labels violate the shared-guard rectangle rule structure')
    return match.to(torch.int64).argmax(dim=-1)


def joint_log_probabilities(logits):
    """Normalized log probabilities over admissible patterns, preserving gradients."""
    import torch
    _validate_logits(logits)
    logp = torch.log_softmax(logits, dim=-1)
    if not torch.isfinite(logp).all():
        raise FloatingPointError('Logits exceeded the representable log-softmax range')
    return logp


def marginal_log_probabilities(logits):
    """Return (log P(H=1), log P(H=0)), each ...×9, via stable log-sum-exp."""
    import torch
    logp = joint_log_probabilities(logits)
    mask = _torch_patterns(logits.device, torch.bool)
    expanded = logp[..., :, None]
    log_positive = torch.logsumexp(expanded.masked_fill(~mask, -torch.inf), dim=-2)
    log_negative = torch.logsumexp(expanded.masked_fill(mask, -torch.inf), dim=-2)
    return log_positive, log_negative


def marginal_probabilities(logits):
    """Marginal rule probabilities induced by the normalized ten-pattern head."""
    return marginal_log_probabilities(logits)[0].exp()


def _reduce(loss, reduction):
    if reduction == 'none':
        return loss
    if reduction == 'mean':
        return loss.mean()
    if reduction == 'sum':
        return loss.sum()
    raise ValueError('reduction must be none, mean or sum')


def per_rule_bce(logits, labels, reduction='mean'):
    """Stable binary loss on nine marginals; validates admissible binary targets.

    'none' returns ...×9; 'mean' averages both samples and rules. This is a
    marginal loss, not the categorical likelihood of the complete rule vector.
    """
    import torch
    _validate_logits(logits)
    if not isinstance(labels, torch.Tensor) or labels.shape != logits.shape[:-1]+(9,):
        raise ValueError('Target shape must match the logits batch dimensions plus nine')
    if labels.device != logits.device:
        raise ValueError('Logits and labels must be on the same device')
    torch_pattern_ids(labels)
    log_yes, log_no = marginal_log_probabilities(logits)
    loss = torch.where(labels.bool(), -log_yes, -log_no)
    return _reduce(loss, reduction)


def categorical_nll(logits, ids, reduction='mean'):
    """Categorical likelihood of one complete admissible nine-rule outcome."""
    import torch
    _validate_logits(logits)
    if not isinstance(ids, torch.Tensor) or ids.shape != logits.shape[:-1]:
        raise ValueError('Pattern-ID shape must match the logits batch dimensions')
    if ids.device != logits.device or ids.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8):
        raise ValueError('Pattern IDs must be same-device integer tensors')
    if not ((ids >= 0) & (ids < 10)).all():
        raise ValueError('Pattern IDs must be between zero and nine')
    logp = joint_log_probabilities(logits)
    loss = -logp.gather(-1, ids.to(torch.int64)[..., None]).squeeze(-1)
    return _reduce(loss, reduction)
