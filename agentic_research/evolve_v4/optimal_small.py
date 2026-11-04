"""Exact Bayes-optimal adaptive querying on tiny synthetic certification instances.

Run from the repo root:  .venv-mac/bin/python -m agentic_research.evolve_v4.optimal_small [n_instances]

Instance: C columns x R ranks x K nested rules.  A read of (column, rank) reveals the
cell's "level" l in {0..K}; rule k (1-based) is positive iff l >= k, so rule 2 positive
implies rule 1 positive.  The frozen certificate (core.possible_starts / core.certificates)
is used unchanged for the stopping test.  Prior over labelings = product over columns of a
latent-onset model (onset rank, positive level) with per-cell noise, enumerated exactly.

Outputs optimal_small.json: exact optimal expected reads (memoised DP over partial label
tensors), exact minimax reads, hindsight lower bound, and expected reads of existing
policies (evaluated exactly on their decision tree under the same prior).
"""
import importlib.util
import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
sys.setrecursionlimit(10000)

_spec = importlib.util.spec_from_file_location(
    "frozen_core", HERE / "colab/bundle/agentic_research/evolve_v4/core.py")
_core = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_core)
View, certificates, possible_starts, replay = (
    _core.View, _core.certificates, _core.possible_starts, _core.replay)

C, R, K = 2, 5, 2
SEED = 20261104
POLICIES = {
    "seed": HERE / "seed_a.py",
    "t2": HERE / "runs/a1/programs/t2.py",
    "t6": HERE / "runs/a1/inbox/t6.py",
    "t9": HERE / "runs/a1/inbox/t9.py",
}
BASE = K + 2  # per-cell digit: 0 unknown, 1+level


# ----------------------------------------------------------------------------- instances
def make_rules():
    # (periodicity floor, ratio ceiling); rule 2 stricter than rule 1 => nested
    return np.array([[2.0, 4.0], [3.0, 2.0]])[:K]


def hidden_for_levels(levels, rules):
    """Hidden features as a deterministic function of the cell level (consistent with labels)."""
    lv = np.asarray(levels)
    per = np.where(lv == 0, rules[0, 0] - 1, np.where(lv == 1, (rules[0, 0] + rules[1, 0]) / 2,
                                                       rules[1, 0] + 1)).astype(float)
    rat = np.where(lv == 0, rules[0, 1] + 1, np.where(lv == 1, (rules[0, 1] + rules[1, 1]) / 2,
                                                       rules[1, 1] - 1)).astype(float)
    return dict(periodicity=per, ratio=rat, fixed=np.zeros(lv.shape), guard=np.ones(lv.shape))


def labels_from_levels(levels):
    lv = np.asarray(levels)  # (C, R)
    return np.stack([(lv >= k + 1) for k in range(K)], axis=-2).astype(np.int8)  # (C, K, R)


def intervals(force, log_beta, epsilon):
    """Mirror of law_decision_pilot_v1.pilot.intervals / task_a.intervals (slope -2, width eps)."""
    lower = np.full(force.shape, np.nan)
    upper = lower.copy()
    valid = (force[:, :-2] > 0) & (force[:, 1:-1] > force[:, :-2])
    rows, previous = np.where(valid)
    start = previous + 1
    lower[rows, start] = np.log(force[rows, start - 1]) + 2 * log_beta[rows] - epsilon
    upper[rows, start] = np.log(force[rows, start]) + 2 * log_beta[rows] + epsilon
    return lower[:, None, :], upper[:, None, :]


def synth_geometry(rng):
    """Synthetic: sorted forces, per-column log_beta so a slope -2 law is near-consistent."""
    logf = np.cumsum(rng.uniform(0.1, 0.6, (C, R)), axis=1)
    c0 = logf[:, 2].mean()
    s = rng.integers(1, R - 1, C)
    mid = np.array([(logf[i, s[i] - 1] + logf[i, s[i]]) / 2 for i in range(C)])
    log_beta = (c0 - mid) / 2 + rng.normal(0, rng.choice([0.05, 0.15, 0.3]), C)
    return np.exp(logf), log_beta, "synthetic"


def real_geometry(rng, pool):
    """Sub-sample C columns and R sorted ranks from a real SEARCH-split case."""
    case = pool[rng.integers(len(pool))]
    force, lb = case["public"]["force"], case["public"]["log_beta"]
    cols = rng.choice(len(force), C, replace=False)
    ranks = np.sort(rng.choice(np.arange(10, 50), R, replace=False))
    f = np.maximum.accumulate(force[cols][:, ranks], axis=1) * (1 + 1e-6 * np.arange(R))
    return f, lb[cols], "real_subsample"


def column_distribution(rng):
    """Latent onset s in 0..R-1 (Dirichlet), positive level in 1..K, cell noise q."""
    onset = rng.dirichlet(np.ones(R) * 2.0)
    lvl_p = rng.dirichlet(np.ones(K) * 2.0)
    q = float(rng.choice([0.02, 0.05, 0.1]))
    lv = np.array(list(itertools.product(range(K + 1), repeat=R)), dtype=np.int8)
    prob = np.zeros(len(lv))
    for s in range(R):
        for l in range(1, K + 1):
            base = np.where(np.arange(R) >= s, l, 0)
            pc = np.where(lv == base[None], 1 - q + q / (K + 1), q / (K + 1)).prod(axis=1)
            prob += onset[s] * lvl_p[l - 1] * pc
    return lv, prob / prob.sum(), dict(onset=onset.round(3).tolist(), level_p=lvl_p.round(3).tolist(), q=q)


def make_instance(i, pool):
    rng = np.random.default_rng(SEED + i)
    force, log_beta, kind = (real_geometry(rng, pool) if (pool and i % 2 == 1) else synth_geometry(rng))
    eps = float(rng.choice([0.2, 0.4]))
    lower, upper = intervals(force, log_beta, eps)
    (lv0, p0, info0), (lv1, p1, info1) = column_distribution(rng), column_distribution(rng)
    levels = np.concatenate([np.repeat(lv0, len(lv1), 0), np.tile(lv1, (len(lv0), 1))], axis=1)
    return dict(i=i, kind=kind, eps=eps, force=force, log_beta=log_beta, lower=lower, upper=upper,
                rules=make_rules(), levels=levels.astype(np.int8), prob=np.outer(p0, p1).ravel(),
                prior=[info0, info1])


# ----------------------------------------------------------------------------- stop table
def stop_table(lower, upper):
    """stop[code] for all (K+2)^(C*R) partial states; batched copy of core.certificates."""
    n = BASE ** (C * R)
    codes = np.arange(n, dtype=np.int64)
    out = np.zeros(n, bool)
    valid = np.isfinite(lower)[None]
    for a in range(0, n, 200000):
        d = ((codes[a:a + 200000, None] // BASE ** np.arange(C * R)[None]) % BASE).reshape(-1, C, R)
        part = np.full((len(d), C, K, R), -1, np.int8)
        for k in range(K):
            part[:, :, k, :] = np.where(d == 0, -1, (d - 1 >= k + 1)).astype(np.int8)
        poss = possible_starts(part)
        cand = poss & valid
        lo = np.where(cand, lower[None], np.inf).min(-1).max(1)
        hi = np.where(cand, upper[None], -np.inf).max(-1).min(1)
        could = cand.any(-1).all(1) & (lo <= hi)
        allv = ~(poss & ~valid).any(axis=(1, 3))
        il = np.where(cand, lower[None], -np.inf).max(-1).max(1)
        ih = np.where(cand, upper[None], np.inf).min(-1).min(1)
        out[a:a + 200000] = (allv & (il <= ih)).any(-1) | ~could.any(-1)
    return out


def partial_from_code(code):
    d = np.array([(code // BASE ** i) % BASE for i in range(C * R)]).reshape(C, R)
    part = np.full((C, K, R), -1, np.int8)
    for k in range(K):
        part[:, k, :] = np.where(d == 0, -1, (d - 1 >= k + 1))
    return part, d


# ----------------------------------------------------------------------------- solvers
def solve(inst, stop):
    levels, prob = inst["levels"], inst["prob"]
    pw = [BASE ** i for i in range(C * R)]
    memo = {}  # code -> (optimal expected reads, minimax reads, best cell)

    def rec(code, idx):
        if stop[code]:
            return 0.0, 0
        if code in memo:
            return memo[code][:2]
        w = prob[idx]
        tot = w.sum()
        best, bcell, worst = np.inf, None, 10 ** 9
        for cell in range(C * R):
            if (code // pw[cell]) % BASE:
                continue
            vals = levels[idx, cell]
            exp, wc = 1.0, 1
            for v in range(K + 1):
                m = vals == v
                if m.any():
                    e, wst = rec(code + (v + 1) * pw[cell], idx[m])
                    exp += w[m].sum() / tot * e
                    wc = max(wc, 1 + wst)
            if exp < best - 1e-12:
                best, bcell = exp, cell
            worst = min(worst, wc)
        memo[code] = (best, worst, bcell)
        return best, worst

    opt, minimax = rec(0, np.arange(len(prob)))
    return opt, minimax, memo


def hindsight(inst, stop):
    """E over labelings of the fewest cells whose reveal makes the certificate stop."""
    levels, prob = inst["levels"], inst["prob"]
    contrib = [(levels[:, i].astype(np.int64) + 1) * BASE ** i for i in range(C * R)]
    best = np.full(len(prob), C * R + 1)
    for s in sorted(range(1 << (C * R)), key=lambda s: bin(s).count("1")):
        pc = bin(s).count("1")
        code = sum((contrib[i] for i in range(C * R) if s >> i & 1), np.zeros(len(prob), np.int64))
        ok = stop[code] & (best > pc)
        best[ok] = pc
    return float((best * prob).sum()), int(best.max())


# ----------------------------------------------------------------------------- policies
def load_policy(path, name, shim, k0=False):
    spec = importlib.util.spec_from_file_location(f"pol_{name}_{int(shim)}{int(k0)}", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    if shim:
        # RUNTIME shim only (files untouched): the policies hard-code UNIFORM10 = 10 response
        # patterns (9 rules); with K=2 nested rules response_patterns() yields fewer patterns.
        from agentic_research.evolve_v4 import helpers
        n = len(helpers.response_patterns(make_rules()))
        mod.UNIFORM10 = np.full(n, 1.0 / n)
    if k0:
        # KSEED=8 means "defer to the seed for the first 8 reads"; with only C*R=10 cells the
        # evolved logic would almost never run, so +k0 sets KSEED=0 (runtime only).
        mod.KSEED = 0
    return mod.choose


def case_dict(inst):
    zero = np.zeros((C, R), int)
    return dict(lower=inst["lower"], upper=inst["upper"], epsilon=inst["eps"],
                public=dict(force=inst["force"], log_beta=inst["log_beta"], rules=inst["rules"]),
                hidden=hidden_for_levels(zero, inst["rules"]), labels=labels_from_levels(zero),
                expected=False)


def make_view(inst, code):
    part, d = partial_from_code(code)
    revealed = {k: np.where(d == 0, np.nan, v)
                for k, v in hidden_for_levels(np.maximum(d - 1, 0), inst["rules"]).items()}
    return View(part, possible_starts(part), case_dict(inst), revealed, int((d > 0).sum())), d


def eval_policy(inst, stop, choose):
    """Exact expected reads of a deterministic policy under the prior (decision-tree walk)."""
    levels, prob = inst["levels"], inst["prob"]
    memo = {}

    def pick(code):
        if code not in memo:
            view, d = make_view(inst, code)
            c, r = choose(view)
            c, r = int(c), int(r)
            if not (0 <= c < C and 0 <= r < R) or d[c, r] != 0:
                raise ValueError(f"invalid/repeated query {(c, r)}")
            memo[code] = c * R + r
        return memo[code]

    def rec(code, idx):
        if stop[code]:
            return 0.0
        cell = pick(code)
        w = prob[idx]
        vals = levels[idx, cell]
        exp = 1.0
        for v in range(K + 1):
            m = vals == v
            if m.any():
                exp += w[m].sum() / w.sum() * rec(code + (v + 1) * BASE ** cell, idx[m])
        return exp

    return rec(0, np.arange(len(prob)))


def replay_check(inst, stop, choose, n=40):
    """Cross-check the tree evaluator against frozen core.replay on random labelings."""
    rng = np.random.default_rng(1)
    got, want = [], []
    for j in rng.choice(len(inst["prob"]), n, p=inst["prob"]):
        lv = inst["levels"][j].reshape(C, R)
        lab = labels_from_levels(lv)
        exp = certificates(possible_starts(lab), inst["lower"], inst["upper"])[0]
        case = dict(case_dict(inst), labels=lab, hidden=hidden_for_levels(lv, inst["rules"]), expected=exp)
        got.append(replay(case, choose))
        code, reads = 0, 0
        while not stop[code]:
            view, _ = make_view(inst, code)
            c, r = choose(view)
            code += (int(lv[c, r]) + 1) * BASE ** (int(c) * R + int(r))
            reads += 1
        want.append(reads)
    return got == want


def brute_check(inst, memo):
    """Independent DP check: recompute V at states with <=6 unknown cells straight from the
    definition (no memo, frozen certificates() called directly on each partial tensor)."""
    levels, prob = inst["levels"], inst["prob"]
    rng = np.random.default_rng(7)
    small = [c for c in memo if sum(((c // BASE ** i) % BASE) == 0 for i in range(C * R)) <= 6]
    picks = [small[j] for j in rng.choice(len(small), min(5, len(small)), replace=False)]

    def V(d, idx):
        part = np.full((C, K, R), -1, np.int8)
        dd = d.reshape(C, R)
        for k in range(K):
            part[:, k, :] = np.where(dd == 0, -1, (dd - 1 >= k + 1))
        if certificates(possible_starts(part), inst["lower"], inst["upper"])[0] is not None:
            return 0.0
        best = np.inf
        tot = prob[idx].sum()
        for cell in np.flatnonzero(d == 0):
            e = 1.0
            for v in range(K + 1):
                sub = idx[levels[idx, cell] == v]
                if len(sub):
                    d2 = d.copy()
                    d2[cell] = v + 1
                    e += prob[sub].sum() / tot * V(d2, sub)
            best = min(best, e)
        return best

    out = []
    for code in picks:
        d = np.array([(code // BASE ** i) % BASE for i in range(C * R)])
        idx = np.flatnonzero(np.all((d[None] == 0) | (levels + 1 == d[None]), axis=1))
        out.append(dict(code=int(code), dp=memo[code][0], brute=V(d, idx)))
    return out


# ----------------------------------------------------------------------------- main
def main():
    t0 = time.time()
    n_inst = int(sys.argv[1]) if len(sys.argv) > 1 else 24
    pool = None
    try:
        from agentic_research.evolve_v4.task_a import load_cases
        pool = load_cases("search")  # SEARCH split only, only for geometry sub-sampling
    except Exception as e:
        print("real geometry unavailable:", repr(e))
    names = [n + s for n in POLICIES for s in (("", "+shim") if n == "seed" else ("", "+shim", "+shim+k0"))]
    rows = []
    skipped, i = [], -1
    while len(rows) < n_inst:
        i += 1
        inst = make_instance(i, pool)
        stop = stop_table(inst["lower"], inst["upper"])
        if stop[0]:  # certificate decides with zero reads (no valid start geometry): trivial, skip
            skipped.append(i)
            continue
        if not rows:
            for code in np.random.default_rng(3).integers(0, BASE ** (C * R), 3000):
                part, _ = partial_from_code(int(code))
                ref = certificates(possible_starts(part), inst["lower"], inst["upper"])[0] is not None
                assert ref == bool(stop[code]), "stop table mismatch"
        opt, minimax, memo = solve(inst, stop)
        hind, hmax = hindsight(inst, stop)
        row = dict(i=i, kind=inst["kind"], eps=inst["eps"], states=len(memo), optimal=opt,
                   minimax=minimax, hindsight=hind, hindsight_worst=hmax, policies={}, prior=inst["prior"])
        if not rows:
            row["brute_check"] = brute_check(inst, memo)
            print("brute-force DP check:", row["brute_check"])
        for name in names:
            base_name, shim, k0 = name.split("+")[0], "+shim" in name, name.endswith("+k0")
            try:
                ch = load_policy(POLICIES[base_name], base_name, shim, k0)
                val = eval_policy(inst, stop, ch)
                if not rows and shim and base_name == "seed":
                    row["replay_matches_tree_eval"] = replay_check(inst, stop, ch)
                row["policies"][name] = dict(reads=val, ratio=val / opt)
            except Exception as e:
                row["policies"][name] = dict(error=f"{type(e).__name__}: {str(e)[:120]}")
        rows.append(row)
        print(f"[{i:2d}] {inst['kind']:14s} eps={inst['eps']} states={len(memo):7d} opt={opt:.3f} "
              f"mm={minimax} hind={hind:.3f} " +
              " ".join(f"{k}={v.get('reads', float('nan')):.3f}" for k, v in row["policies"].items()) +
              f"  t={time.time() - t0:.0f}s", flush=True)

    avg = dict(optimal=float(np.mean([r["optimal"] for r in rows])),
               minimax=float(np.mean([r["minimax"] for r in rows])),
               hindsight=float(np.mean([r["hindsight"] for r in rows])), policies={})
    for name in names:
        ok = [r for r in rows if "reads" in r["policies"][name]]
        errs = [r["policies"][name]["error"] for r in rows if "error" in r["policies"][name]]
        e = dict(n_ok=len(ok), n_crash=len(rows) - len(ok), first_error=errs[0] if errs else None)
        if ok:
            e["reads"] = float(np.mean([r["policies"][name]["reads"] for r in ok]))
            e["optimal_same_instances"] = float(np.mean([r["optimal"] for r in ok]))
            e["ratio_to_optimal"] = e["reads"] / e["optimal_same_instances"]
            e["mean_of_ratios"] = float(np.mean([r["policies"][name]["ratio"] for r in ok]))
        avg["policies"][name] = e
    out = dict(settings=dict(C=C, R=R, K=K, seed=SEED, n_instances=len(rows), skipped_trivial_seeds_offsets=skipped,
                             geometry="odd i: sub-sampled real SEARCH-split columns/ranks (forces, log_beta); "
                                      "even i: synthetic; intervals mirror pilot.intervals (slope -2, eps)",
                             prior="per column latent onset + positive level + per-cell noise q; independent columns"),
               average=avg, instances=rows, seconds=time.time() - t0)
    (HERE / "optimal_small.json").write_text(json.dumps(out, indent=1, default=float))
    print("\nAVERAGE over", len(rows), "instances")
    print(f"optimal {avg['optimal']:.3f}  minimax {avg['minimax']:.3f}  hindsight {avg['hindsight']:.3f}")
    for name, e in avg["policies"].items():
        print(f"{name:10s} ok={e['n_ok']:2d} reads={e.get('reads', float('nan')):.3f} "
              f"ratio={e.get('ratio_to_optimal', float('nan')):.3f}  {e['first_error'] or ''}")


if __name__ == "__main__":
    main()
