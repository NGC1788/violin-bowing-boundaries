"""Layer-0 headroom for task B (cello future forecasting, forecast_v3 benchmark).

Evolution may only use the train families A/B: fitness = cross-family transfer
(fit A -> score B, fit B -> score A). Families C/D stay sealed. Here we compare
  - Markov (prefix pattern -> future pattern), the frozen strong baseline;
  - HGB on inputs + prefix labels (the earlier best learner);
  - an in-distribution ceiling: HGB fit by grouped 4-fold CV *inside* the scored
    family (what a model could reach without family shift).
The gap HGB(transfer) - ceiling is the room a better transfer method could take.
"""
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import GroupKFold

from agentic_research.forecast_v3 import rule_patterns
from agentic_research.forecast_v3.data import load_prepared_dataset

REPO = Path(__file__).resolve().parents[2]
PATTERNS = rule_patterns.admissible_patterns().astype(float)


def load():
    d = load_prepared_dataset(REPO / "agentic_research/benchmarks/forecast_v3/dataset.npz")
    keep = d["split"] == "train"
    x = np.hstack([d["X"], d["prefix_Y9"]]).astype(float)[keep]
    return dict(x=x, y=d["Y9"][keep].astype(int), y_ids=d["pattern_ids"][keep],
                prefix_ids=d["prefix_pattern_ids"][keep], family=d["groups"][keep],
                session=d["measurement_sessions"][keep])


def log_loss(y, p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())


def markov(train_prefix, train_y_ids, eval_prefix):
    counts = np.zeros((10, 10))
    np.add.at(counts, (train_prefix, train_y_ids), 1.)
    prior = (np.bincount(train_y_ids, minlength=10) + .1) / (len(train_y_ids) + 1.)
    cond = (counts + .1) / (counts.sum(1, keepdims=True) + 1.)
    cond[counts.sum(1) == 0] = prior
    return cond[eval_prefix] @ PATTERNS


def hgb(x_train, y_train, x_eval, iters=200):
    out = np.zeros((len(x_eval), y_train.shape[1]))
    for r in range(y_train.shape[1]):
        if y_train[:, r].min() == y_train[:, r].max():
            out[:, r] = y_train[0, r]
            continue
        model = HistGradientBoostingClassifier(max_iter=iters, learning_rate=.05, random_state=0)
        out[:, r] = model.fit(x_train, y_train[:, r]).predict_proba(x_eval)[:, 1]
    return out


if __name__ == "__main__":
    d = load()
    result = {}
    for source, target in (("A", "B"), ("B", "A")):
        tr, ev = d["family"] == source, d["family"] == target
        row = dict(markov=log_loss(d["y"][ev], markov(d["prefix_ids"][tr], d["y_ids"][tr], d["prefix_ids"][ev])),
                   hgb_transfer=log_loss(d["y"][ev], hgb(d["x"][tr], d["y"][tr], d["x"][ev])))
        inside = np.zeros(d["y"][ev].shape)
        for fit, score in GroupKFold(4).split(d["x"][ev], groups=d["session"][ev]):
            inside[score] = hgb(d["x"][ev][fit], d["y"][ev][fit], d["x"][ev][score])
        row["hgb_in_family_cv"] = log_loss(d["y"][ev], inside)
        both = np.zeros(d["y"][ev].shape)  # source family plus in-family folds
        for fit, score in GroupKFold(4).split(d["x"][ev], groups=d["session"][ev]):
            both[score] = hgb(np.vstack([d["x"][tr], d["x"][ev][fit]]),
                              np.vstack([d["y"][tr], d["y"][ev][fit]]), d["x"][ev][score])
        row["hgb_source_plus_in_family"] = log_loss(d["y"][ev], both)
        result[f"{source}->{target}"] = row
        print(source, "->", target, {k: round(v, 5) for k, v in row.items()}, flush=True)
    print(json.dumps(result))
    (Path(__file__).parent / "headroom_b.json").write_text(json.dumps(result, indent=1))
    sys.exit(0)
