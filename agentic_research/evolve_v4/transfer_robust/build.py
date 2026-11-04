"""Transfer benchmark: an AI-robustness audit with the same certificate as the violin law test.

  .venv-mac/bin/python -m agentic_research.evolve_v4.transfer_robust.build

Question per case: for a group of correctly classified MNIST test images, is there a failure
criterion (one of 9 nested rules) under which every image's Gaussian-noise break point agrees with
the local-linear robustness law  sigma* = C * margin / ||grad margin||  within a factor e^epsilon?
A "yes" means a one-pass gradient score predicts noise robustness for the whole group; a "no" flags
images whose robustness the cheap score gets wrong (the situation gradient masking creates).

Mapping onto the frozen violin interface (nothing in core.py changes):
  column            = one test image                  rank = one of 50 noise levels (shared grid)
  force[j, r]       = noise std sigma_r               one read = N noisy forward passes at (j, r)
  log_beta[j]       = -0.5 * log(margin_j / grad_j)   so  log sigma + 2 log_beta = log(sigma / s_j)
  rules (9, 2)      = (fail-rate floor u, true-class-probability ceiling v), 3 x 3, nested
  label[j, k, r]    = fail_rate >= u_k  and  mean true-class probability < v_k      ("broken")
The violin law  log F* = -2 log beta + c  becomes  log sigma* = log s_j + c: same slope, same
intervals, same certificate. Design fixed before any policy was scored on it: epsilon (.2, .4) and
the 3 x 3 rule grid follow the violin task; groups 0-1 of each digit = search, 2-3 = sealed holdout.
"""
import gzip
import hashlib
import io
import json
import pickle
from pathlib import Path

import numpy as np
import torch
from torch import nn

from agentic_research.evolve_v4.task_a import EPSILONS
from agentic_research.law_decision_pilot_v1.pilot import certificates, possible_starts

HERE = Path(__file__).resolve().parent
OUT = HERE / "cases"
SIGMAS = np.geomspace(0.03, 3.0, 50)
RULES = np.array([(u, v) for u in (0.1, 0.3, 0.5) for v in (0.5, 0.7, 0.9)], dtype=float)
PER_GROUP, GROUPS, DRAWS = 24, 4, 64


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.f = nn.Sequential(nn.Conv2d(1, 16, 5), nn.ReLU(), nn.MaxPool2d(2), nn.Conv2d(16, 32, 5), nn.ReLU(),
                               nn.MaxPool2d(2), nn.Flatten(), nn.Linear(512, 64), nn.ReLU(), nn.Linear(64, 10))

    def forward(self, x):
        return self.f(x)


def mnist():
    from sklearn.datasets import fetch_openml
    x, y = fetch_openml("mnist_784", version=1, return_X_y=True, as_frame=False, parser="liac-arff")
    x = (x.astype(np.float32) / 255).reshape(-1, 1, 28, 28)
    y = y.astype(np.int64)
    return x[:60000], y[:60000], x[60000:], y[60000:]  # the standard train / test split


def train(xtr, ytr):
    torch.manual_seed(0)
    net = Net()
    opt = torch.optim.Adam(net.parameters(), 1e-3)
    xt, yt = torch.from_numpy(xtr), torch.from_numpy(ytr)
    for epoch in range(2):
        order = torch.randperm(len(xt), generator=torch.Generator().manual_seed(epoch))
        for i in range(0, len(xt), 128):
            idx = order[i:i + 128]
            loss = nn.functional.cross_entropy(net(xt[idx]), yt[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
    return net.eval()


def margin_and_grad(net, x, y):
    x = torch.from_numpy(x).requires_grad_(True)
    logits = net(x)
    true = logits.gather(1, torch.from_numpy(y)[:, None])[:, 0]
    other = logits.masked_fill(nn.functional.one_hot(torch.from_numpy(y), 10).bool(), -1e9).max(1).values
    margin = true - other
    (grad,) = torch.autograd.grad(margin.sum(), x)
    return margin.detach().numpy().astype(float), grad.flatten(1).norm(dim=1).numpy().astype(float)


@torch.no_grad()
def noisy_stats(net, x, y, image_id):
    """fail rate and mean true-class probability at every sigma, DRAWS fixed noise draws per cell."""
    g = torch.Generator().manual_seed(1_000_003 * image_id + 17)
    noise = torch.randn(DRAWS, 1, 28, 28, generator=g)  # common random numbers across sigma
    xs = torch.from_numpy(x)[None] + torch.from_numpy(SIGMAS.astype(np.float32))[:, None, None, None, None] * noise
    probs = net(xs.reshape(-1, 1, 28, 28)).softmax(1).reshape(len(SIGMAS), DRAWS, 10)
    fail = (probs.argmax(2) != y).float().mean(1).numpy()
    true_prob = probs[:, :, y].mean(1).numpy()
    return fail.astype(float), true_prob.astype(float)


def intervals(force, log_beta, epsilon):  # task_a.intervals, taking arrays instead of columns
    lower = np.full(force.shape, np.nan)
    upper = lower.copy()
    valid = (force[:, :-2] > 0) & (force[:, 1:-1] > force[:, :-2])
    rows, previous = np.where(valid)
    start = previous + 1
    lower[rows, start] = np.log(force[rows, start - 1]) + 2 * log_beta[rows] - epsilon
    upper[rows, start] = np.log(force[rows, start]) + 2 * log_beta[rows] + epsilon
    return lower[:, None, :], upper[:, None, :]


def main():
    OUT.mkdir(exist_ok=True)
    torch.set_num_threads(8)
    xtr, ytr, xte, yte = mnist()
    net = train(xtr, ytr)
    with torch.no_grad():
        pred = torch.cat([net(torch.from_numpy(xte[i:i + 2000])).argmax(1) for i in range(0, 10000, 2000)]).numpy()
    accuracy = float((pred == yte).mean())
    buf = io.BytesIO()
    torch.save(net.state_dict(), buf)
    (OUT / "mnist_cnn.pt").write_bytes(buf.getvalue())
    rng = np.random.default_rng(0)
    splits = {"search": [], "holdout": []}
    for digit in range(10):
        pool = np.flatnonzero((yte == digit) & (pred == yte))
        chosen = rng.choice(pool, GROUPS * PER_GROUP, replace=False)
        for group in range(GROUPS):
            ids = np.sort(chosen[group * PER_GROUP:(group + 1) * PER_GROUP])
            x, y = xte[ids], yte[ids]
            margin, grad = margin_and_grad(net, x, y)
            stats = [noisy_stats(net, x[i], int(y[i]), int(ids[i])) for i in range(len(ids))]
            fail = np.array([s[0] for s in stats])
            true_prob = np.array([s[1] for s in stats])
            labels = ((fail[:, None, :] >= RULES[None, :, 0, None]) &
                      (true_prob[:, None, :] < RULES[None, :, 1, None])).astype(np.int8)
            force = np.broadcast_to(SIGMAS, (len(ids), len(SIGMAS))).copy()
            log_beta = -0.5 * np.log(margin / grad)
            public = dict(force=force, log_beta=log_beta, rules=RULES.copy())
            hidden = dict(fail_rate=fail, true_prob=true_prob)
            for epsilon in EPSILONS:
                lower, upper = intervals(force, log_beta, epsilon)
                expected, _, _ = certificates(possible_starts(labels), lower, upper)
                assert expected is not None, "full observation must decide"
                case = dict(group=f"digit{digit}_g{group}", epsilon=epsilon, labels=labels, lower=lower,
                            upper=upper, expected=expected, public=public, hidden=hidden,
                            images=ids.tolist())
                splits["search" if group < 2 else "holdout"].append(case)
    manifest = dict(model_sha256=hashlib.sha256(buf.getvalue()).hexdigest(), test_accuracy=accuracy,
                    sigmas=[float(s) for s in SIGMAS], rules=RULES.tolist(), draws=DRAWS, per_group=PER_GROUP)
    for split, cases in splits.items():
        blob = gzip.compress(pickle.dumps(cases, protocol=4), mtime=0)
        (OUT / f"cases_{split}.pkl.gz").write_bytes(blob)
        manifest[split] = dict(cases=len(cases), sha256=hashlib.sha256(blob).hexdigest(),
                               true=sum(c["expected"] for c in cases),
                               read_everything=int(sum(c["labels"].shape[0] * c["labels"].shape[2] for c in cases)))
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=1))
    print(json.dumps({k: v for k, v in manifest.items() if k != "sigmas"}, indent=1))


def load_cases(split):
    if split == "holdout":
        import os
        if os.environ.get("EVOLVE_FINAL") != "1":
            raise SystemExit("robust holdout is sealed: scored once at the end")
    return pickle.loads(gzip.decompress((OUT / f"cases_{split}.pkl.gz").read_bytes()))


if __name__ == "__main__":
    main()
