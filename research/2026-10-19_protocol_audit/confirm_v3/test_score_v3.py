"""Synthetic checks for score_v3 (no real data)."""
import numpy as np
import score_v3 as S


def synth(curv, g, seed=0):
    rng = np.random.default_rng(seed)
    data = {}
    lvs = np.log([0.05, 0.1, 0.2])
    for k, t in enumerate(S.STEMS):
        lv, lb, lo, hi = [], [], [], []
        for j, v in enumerate(lvs):
            for b in np.log(0.02 * 10 ** (np.arange(40) / 39)):
                e = 0.8 + (curv if j == 2 else 0) * (v - lvs[1]) / (lvs[2] - lvs[1]) * 0
                mu = 0.3 * k + 0.8 * v + (curv * (v - lvs[1]) ** 2 / (2 * (lvs[1] - lvs[0]) ** 2) * (lvs[1] - lvs[0]) if False else 0) \
                     + (0.5 * curv / (lvs[2] - lvs[1]) * (v - lvs[1]) ** 2 if True else 0) - 1.0 * b + g * v * b + rng.normal(0, .05)
                f = np.log(np.exp(np.linspace(np.log(0.1), np.log(40), 50)))
                i = np.searchsorted(f, mu)
                lv.append(v); lb.append(b); lo.append(f[i - 1] if i > 0 else -np.inf); hi.append(f[i] if i < 50 else np.inf)
        data[t] = tuple(np.array(x) for x in (lv, lb, lo, hi))
    return data


def test_flat_vs_curved_sign():
    r0 = S.pooled(synth(0.0, 0.0), nb=5)
    r1 = S.pooled(synth(-0.6, 0.0), nb=5)
    assert abs(r0["estimate"]["curvature"]) < 0.1
    assert r1["estimate"]["curvature"] < -0.2


def test_interaction_sign():
    r = S.pooled(synth(0.0, 0.4), nb=5)
    assert r["estimate"]["g"] > 0.25
