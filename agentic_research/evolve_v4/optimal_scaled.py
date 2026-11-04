"""Exact Bayes-optimal adaptive querying, scaled past optimal_small.py (C=2, R up to 12, K=2..3).

Run from the repo root (results go to part files under $OPTIMAL_SCALED_PARTS, then `merge`):
  python -m agentic_research.evolve_v4.optimal_scaled check            # R=5 K=2 vs optimal_small.json
  python -m agentic_research.evolve_v4.optimal_scaled run R K n [key=json ...]
        e.g. run 8 2 24 part=0/2                      (exact branch and bound, exact policy walks)
             run 10 2 12 exact=false 'taus=[1e-4,1e-5]' policy_tau=1e-7 hindsight_sample=8000
  python -m agentic_research.evolve_v4.optimal_scaled merge            # -> optimal_scaled.json
  python -m agentic_research.evolve_v4.optimal_scaled report           # markdown table

Same instance generator, prior and frozen certificate as optimal_small.py (generalised to R, K;
identical at R=5, K=2, which `check` reproduces to 1e-15).  What makes larger sizes possible:
  * belief-state factorisation: the prior is a product over columns, so every predictive
    probability is a ratio of one column's marginal likelihood table m_c (size B^R, B = K+2)
    instead of a filter over all (K+1)^(C*R) joint labelings;
  * the stop test depends on a column only through a small label (per rule: candidate flags and
    the min/max interval ends); stopT is precomputed over label pairs with the frozen formula;
  * "dense": full Bellman recursion over all B^(2R) joint codes in C++ (validation, R<=7);
  * "bnb": depth-first branch and bound on the AND-OR tree with a memo, using an admissible
    column-decomposed bound  V(o0,o1) >= E[W0(o0 | col 1 fully known)] + E[W1(o1 | col 0 known)]
    (each term a single-column DP; a frozen column makes the state's value exactly that DP).
    The result is the exact optimum whenever the memo did not overflow;
  * certified bounds when exact search does not fit: states with marginal m0*m1 < tau become
    terminal with value LB, which by monotonicity of the Bellman operator gives V_lo <= V*; the
    policy that search chose, evaluated exactly with the greedy policy below tau, is a real
    policy, so its value V_hi >= V*.  Policy expected reads are bounded the same way.
Hindsight lower bound, minimax (dense only) and expected reads of seed_a, distilled_a and the
Colab population programs p25_t21, p12_t9 (decision-tree walk under the same prior) are reported.
Only the SEARCH split is touched (for real geometry sub-sampling); the holdout is never loaded.
"""
import ctypes
import hashlib
import importlib.util
import itertools
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
sys.setrecursionlimit(20000)

_spec = importlib.util.spec_from_file_location(
    "frozen_core", HERE / "colab/bundle/agentic_research/evolve_v4/core.py")
_core = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_core)
View, certificates, possible_starts, replay = (
    _core.View, _core.certificates, _core.possible_starts, _core.replay)

C = 2
SEED = 20261104  # same as optimal_small
POLICIES = {
    "seed": HERE / "seed_a.py",
    "distilled": HERE / "distilled_a.py",
    "p25_t21": HERE / "colab/bundle/population/p25_t21.py",
    "p12_t9": HERE / "colab/bundle/population/p12_t9.py",
}
EXTRA_SMALL = {  # only for the R=5 reproduction of optimal_small's t2 / t6 rows
    "p5_t2": HERE / "colab/bundle/population/p5_t2.py",
    "p10_t6": HERE / "colab/bundle/population/p10_t6.py",
}
OUT = HERE / "optimal_scaled.json"


# ----------------------------------------------------------------------------- instances
def make_rules(K):
    # (periodicity floor, ratio ceiling); each rule stricter than the previous => nested.
    # K=2 is exactly optimal_small's pair.
    return np.array([[2.0, 4.0], [3.0, 2.0], [4.0, 1.0]])[:K]


def hidden_for_levels(levels, rules):
    """Hidden features as a deterministic function of the cell level (consistent with labels).
    Identical to optimal_small for K=2; level l in 1..K-1 sits between rule l and rule l+1."""
    lv = np.asarray(levels)
    K = len(rules)
    f, c = rules[:, 0], rules[:, 1]
    per_of = [f[0] - 1] + [(f[l - 1] + f[l]) / 2 for l in range(1, K)] + [f[K - 1] + 1]
    rat_of = [c[0] + 1] + [(c[l - 1] + c[l]) / 2 for l in range(1, K)] + [c[K - 1] - 1]
    per = np.asarray(per_of, float)[lv]
    rat = np.asarray(rat_of, float)[lv]
    return dict(periodicity=per, ratio=rat, fixed=np.zeros(lv.shape), guard=np.ones(lv.shape))


def labels_from_levels(levels, K):
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


def synth_geometry(rng, R):
    """optimal_small.synth_geometry; the alignment rank 2 becomes R // 2 (== 2 at R=5)."""
    logf = np.cumsum(rng.uniform(0.1, 0.6, (C, R)), axis=1)
    c0 = logf[:, R // 2].mean()
    s = rng.integers(1, R - 1, C)
    mid = np.array([(logf[i, s[i] - 1] + logf[i, s[i]]) / 2 for i in range(C)])
    log_beta = (c0 - mid) / 2 + rng.normal(0, rng.choice([0.05, 0.15, 0.3]), C)
    return np.exp(logf), log_beta, "synthetic"


def real_geometry(rng, pool, R):
    """Sub-sample C columns and R sorted ranks from a real SEARCH-split case."""
    case = pool[rng.integers(len(pool))]
    force, lb = case["public"]["force"], case["public"]["log_beta"]
    cols = rng.choice(len(force), C, replace=False)
    ranks = np.sort(rng.choice(np.arange(10, 50), R, replace=False))
    f = np.maximum.accumulate(force[cols][:, ranks], axis=1) * (1 + 1e-6 * np.arange(R))
    return f, lb[cols], "real_subsample"


def column_prior(rng, R, K):
    """Same draws as optimal_small.column_distribution (onset Dirichlet, level Dirichlet, q)."""
    onset = rng.dirichlet(np.ones(R) * 2.0)
    lvl_p = rng.dirichlet(np.ones(K) * 2.0)
    q = float(rng.choice([0.02, 0.05, 0.1]))
    return onset, lvl_p, q


def make_instance(i, pool, R, K):
    rng = np.random.default_rng(SEED + i)
    force, log_beta, kind = (real_geometry(rng, pool, R) if (pool and i % 2 == 1)
                             else synth_geometry(rng, R))
    eps = float(rng.choice([0.2, 0.4]))
    lower, upper = intervals(force, log_beta, eps)
    priors = [column_prior(rng, R, K) for _ in range(C)]
    return dict(i=i, kind=kind, eps=eps, force=force, log_beta=log_beta, lower=lower, upper=upper,
                rules=make_rules(K), priors=priors, R=R, K=K,
                prior=[dict(onset=o.round(3).tolist(), level_p=l.round(3).tolist(), q=q)
                       for o, l, q in priors])


def load_pool():
    try:
        from agentic_research.evolve_v4.task_a import load_cases
        return load_cases("search")  # SEARCH split only, only for geometry sub-sampling
    except Exception as e:  # pragma: no cover
        print("real geometry unavailable:", repr(e))
        return None


# ----------------------------------------------------------------------------- column tables
class Tables:
    """Per-column tables over all B^R partial observations of one column."""

    def __init__(self, inst):
        R, K = inst["R"], inst["K"]
        self.R, self.K, self.B = R, K, K + 2
        B = self.B
        N = self.N = B ** R
        self.pw = B ** np.arange(R, dtype=np.int64)
        dig = np.zeros((N, R), np.int8)
        codes = np.arange(N, dtype=np.int64)
        for r in range(R):
            dig[:, r] = (codes // self.pw[r]) % B
        self.dig = dig
        # possible-start bitmasks per rule, via the frozen possible_starts (chunked)
        masks = np.zeros((N, K), np.int64)
        bits = (1 << np.arange(R)).astype(np.int64)
        for a in range(0, N, 1 << 18):
            d = dig[a:a + (1 << 18)]
            part = np.full((len(d), K, R), -1, np.int8)
            for k in range(K):
                part[:, k, :] = np.where(d == 0, -1, (d - 1 >= k + 1))
            masks[a:a + len(d)] = (possible_starts(part).astype(np.int64) * bits).sum(-1)
        self.lab, self.tup = [], []
        self.m = []
        for c in range(C):
            lab, tup = self._labels(masks, inst["lower"][c, 0], inst["upper"][c, 0])
            self.lab.append(lab)
            self.tup.append(tup)
            self.m.append(self._marginal(*inst["priors"][c]))
        self.stopT = self._stop_table()

    def _labels(self, masks, lower, upper):
        """Label of a column state = per rule (any candidate, any invalid possible start,
        min lower, max upper, max lower, min upper) over candidates = possible & valid."""
        R, K = self.R, self.K
        allm = np.arange(1 << R, dtype=np.int64)
        poss = ((allm[:, None] >> np.arange(R)) & 1).astype(bool)
        valid = np.isfinite(lower)
        cand = poss & valid
        t = np.stack([cand.any(-1), (poss & ~valid).any(-1),
                      np.where(cand, lower, np.inf).min(-1), np.where(cand, upper, -np.inf).max(-1),
                      np.where(cand, lower, -np.inf).max(-1), np.where(cand, upper, np.inf).min(-1)],
                     -1).astype(float)
        ut, rid = np.unique(t, axis=0, return_inverse=True)
        rid = rid.ravel()
        nr = len(ut)
        comb = np.zeros(self.N, np.int64)
        for k in range(K):
            comb = comb * nr + rid[masks[:, k]]
        uc, lab = np.unique(comb, return_inverse=True)
        per_rule = np.zeros((len(uc), K), np.int64)
        x = uc.copy()
        for k in reversed(range(K)):
            per_rule[:, k] = x % nr
            x //= nr
        return lab.ravel().astype(np.int32), ut[per_rule]  # tup: (nl, K, 6)

    def _stop_table(self):
        """stopT[l0, l1] = frozen certificates() verdict is not None (computed in row blocks)."""
        t0, t1 = self.tup
        out = np.zeros((len(t0), len(t1)), np.uint8)
        step = max(1, int(2e7 // max(1, len(t1) * self.K)))
        b = t1[None]
        for i in range(0, len(t0), step):
            a = t0[i:i + step, None]
            ol = np.maximum(a[..., 2], b[..., 2])
            oh = np.minimum(a[..., 3], b[..., 3])
            could = (a[..., 0] > 0) & (b[..., 0] > 0) & (ol <= oh)
            allv = (a[..., 1] == 0) & (b[..., 1] == 0)
            il = np.maximum(a[..., 4], b[..., 4])
            ih = np.minimum(a[..., 5], b[..., 5])
            guar = allv & (il <= ih)
            out[i:i + step] = guar.any(-1) | ~could.any(-1)
        return out

    def _marginal(self, onset, lvl_p, q):
        """m[code] = P(observed cells of the code) under the column's mixture prior."""
        R, K = self.R, self.K
        pm, pq = 1 - q + q / (K + 1), q / (K + 1)
        m = np.zeros(self.N)
        for s in range(R):
            for l in range(1, K + 1):
                f = np.ones(1)
                for r in reversed(range(R)):
                    base = l if r >= s else 0
                    fac = np.array([1.0] + [pm if v == base else pq for v in range(K + 1)])
                    f = np.kron(f, fac)
                m += onset[s] * lvl_p[l - 1] * f
        return m

    def frozen(self, c):
        """frozen[o]: no refinement of column state o can change its label."""
        if not hasattr(self, "_frozen"):
            self._frozen = {}
        if c not in self._frozen:
            R, K, N = self.R, self.K, self.N
            lab = self.lab[c]
            nobs = (self.dig > 0).sum(1)
            fz = np.ones(N, bool)
            for n in range(R - 1, -1, -1):
                idx = np.flatnonzero(nobs == n)
                ok = np.ones(len(idx), bool)
                for r in range(R):
                    un = self.dig[idx, r] == 0
                    for v in range(K + 1):
                        ch = idx[un] + (v + 1) * self.pw[r]
                        ok[un] &= fz[ch] & (lab[ch] == lab[idx[un]])
                fz[idx] = ok
            self._frozen[c] = fz
        return self._frozen[c]

    def canon(self, max_codes=1 << 21):
        """Per-column probabilistic bisimulation (labels + transition probabilities, useless
        self-loop reads dropped): canonical representative code per state, or None if too big."""
        if self.N > max_codes:
            return None
        if not hasattr(self, "_canon"):
            self._canon = [self._quotient(c) for c in range(C)]
        return self._canon

    def _quotient(self, c):
        R, K, N = self.R, self.K, self.N
        m, lab = self.m[c], self.lab[c]
        unob = self.dig == 0
        child = np.zeros((N, R, K + 1), np.int64)
        qp = np.zeros((N, R, K + 1), np.int64)
        codes = np.arange(N, dtype=np.int64)
        for r in range(R):
            for v in range(K + 1):
                ch = np.where(unob[:, r], codes + (v + 1) * self.pw[r], 0)
                child[:, r, v] = ch
                qp[:, r, v] = np.where(unob[:, r], np.round(m[ch] / m * 1e12), 0).astype(np.int64)
        block = lab.astype(np.int64)
        nb = block.max() + 1
        while True:
            cb = block[child]
            selfloop = (cb == block[:, None, None]).all(-1)
            act = np.concatenate([qp, cb], -1).reshape(N * R, -1)
            _, aid = np.unique(act, axis=0, return_inverse=True)
            aid = np.where(unob & ~selfloop, aid.reshape(N, R), -1)
            aid = np.sort(aid, axis=1)
            dup = np.zeros_like(aid, bool)
            dup[:, 1:] = aid[:, 1:] == aid[:, :-1]
            aid = np.sort(np.where(dup, -1, aid), axis=1)
            _, nbk = np.unique(np.concatenate([block[:, None], aid], 1), axis=0, return_inverse=True)
            nbk = nbk.ravel()
            if nbk.max() + 1 == nb:
                break
            block, nb = nbk, nbk.max() + 1
        rep = np.full(nb, N, np.int64)
        np.minimum.at(rep, block, codes)
        out = np.ascontiguousarray(rep[block])
        self.quotient_sizes = getattr(self, "quotient_sizes", []) + [int(nb)]
        return out

    def stop(self, code):
        N = self.N
        return bool(self.stopT[self.lab[0][code % N], self.lab[1][code // N]])

    def digits(self, code):
        N = self.N
        return np.stack([self.dig[code % N], self.dig[code // N]])  # (C, R)

    def full_codes(self):
        return np.flatnonzero((self.dig > 0).all(1)).astype(np.int64)


# ----------------------------------------------------------------------------- C++ kernel
_LIB = None


def lib():
    global _LIB
    if _LIB is None:
        src = (HERE / "optimal_scaled_kernel.cpp").read_text()
        tag = hashlib.sha256(src.encode()).hexdigest()[:12]
        so = Path(tempfile.gettempdir()) / f"optimal_scaled_{tag}.so"
        if not so.exists():
            tmp = so.with_suffix(f".{os.getpid()}.so")
            subprocess.run(["g++", "-O3", "-march=native", "-std=c++17", "-shared", "-fPIC",
                            str(HERE / "optimal_scaled_kernel.cpp"), "-o", str(tmp)], check=True)
            os.replace(tmp, so)
        L = ctypes.CDLL(str(so))
        P = ctypes.c_void_p
        L.dense_solve.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int64, P, P, ctypes.c_int, P,
                                  P, P, P, P]
        L.column_tables.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int64, P, P, P,
                                    ctypes.c_int, P, P, ctypes.c_int, P, P]
        L.bnb_solve.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int64, P, P, ctypes.c_int, P,
                                P, P, P, P, P, ctypes.c_int, P, P, ctypes.c_int, P, P, P, P,
                                ctypes.c_double, ctypes.c_int64, ctypes.c_double, ctypes.c_int,
                                ctypes.c_double, P, P]
        L.greedy_eval.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int64, P, P, ctypes.c_int, P,
                                  P, P, P, P, P, ctypes.c_int, P, P, ctypes.c_int, P, P,
                                  ctypes.c_int64, P]
        L.bnb_split.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int64, P, P, ctypes.c_int, P,
                                P, P, P, P, P, ctypes.c_int, P, P, ctypes.c_int, P, P, P,
                                ctypes.c_int64, P]
        L.hindsight_exact.argtypes = [ctypes.c_int, ctypes.c_int, P, P, ctypes.c_int, ctypes.c_int,
                                      P, ctypes.c_int, P, P, ctypes.c_int, P, P, P]
        _LIB = L
    return _LIB


def _p(a):
    return a.ctypes.data_as(ctypes.c_void_p)


def _common(T):
    keep = [np.ascontiguousarray(T.lab[0]), np.ascontiguousarray(T.lab[1]),
            np.ascontiguousarray(T.stopT), np.ascontiguousarray(T.m[0]),
            np.ascontiguousarray(T.m[1]), np.ascontiguousarray(T.dig)]
    return keep, [T.R, T.K, T.N, _p(keep[0]), _p(keep[1]), int(T.stopT.shape[1]), _p(keep[2]),
                  _p(keep[3]), _p(keep[4]), _p(keep[5])]


def dense_solve(T):
    keep, args = _common(T)
    out = np.zeros(4)
    lib().dense_solve(*args, _p(out))
    return dict(optimal=float(out[0]), minimax=int(out[1]), nonstopped_states=int(out[2]),
                optimal_policy_states=int(out[3]))


def _merge_classes(rows, weight, k):
    """Agglomerate stop rows (bool, n x L) into <= k groups by OR-ing; a merged group's stop row
    is the OR of its members, so its single-column value is <= each member's (bound stays valid)."""
    groups = [[i] for i in range(len(rows))]
    R_ = [rows[i].copy() for i in range(len(rows))]
    w = list(weight.astype(float))
    while len(groups) > k:
        M = np.array(R_)
        W_ = np.array(w)
        # cost of merging i, j: weight of i times new stop entries it gains, and vice versa
        gain = (~M[:, None, :] & M[None, :, :]).sum(-1)  # gain[i, j] = |row_j \ row_i|
        cost = W_[:, None] * gain + W_[None, :] * gain.T
        np.fill_diagonal(cost, np.inf)
        i, j = np.unravel_index(np.argmin(cost), cost.shape)
        i, j = min(i, j), max(i, j)
        groups[i] += groups.pop(j)
        R_[i] = R_[i] | R_.pop(j)
        w[i] += w.pop(j)
    gid = np.zeros(len(rows), np.int64)
    for g, mem in enumerate(groups):
        gid[mem] = g
    return np.array(R_), gid


def lb_tables(T, max_classes=None):
    """Single-column relaxation tables for the decomposed lower bound.  With max_classes the
    other column's full-label classes are merged (weaker but still admissible bound, less
    memory); frozen-state exact values are then switched off."""
    full = T.full_codes()
    st = T.stopT
    l1f = T.lab[1][full]
    u1, i1 = np.unique(l1f, return_inverse=True)
    cols1, c1 = np.unique(st[:, u1].T, axis=0, return_inverse=True)  # nA x nl0
    cls1 = c1.ravel()[i1.ravel()]
    l0f = T.lab[0][full]
    u0, i0 = np.unique(l0f, return_inverse=True)
    rows0, c0 = np.unique(st[u0, :], axis=0, return_inverse=True)    # nB x nl1
    cls0 = c0.ravel()[i0.ravel()]
    merged = False
    if max_classes and (len(cols1) > max_classes or len(rows0) > max_classes):
        merged = True
        w1 = np.bincount(cls1, weights=T.m[1][full], minlength=len(cols1))
        cols1, g1 = _merge_classes(cols1.astype(bool), w1, max_classes)
        cls1 = g1[cls1]
        w0 = np.bincount(cls0, weights=T.m[0][full], minlength=len(rows0))
        rows0, g0 = _merge_classes(rows0.astype(bool), w0, max_classes)
        cls0 = g0[cls0]
    nA, nB = len(cols1), len(rows0)
    fc0 = np.full(T.N, -1, np.int32)
    fc0[full] = cls0
    fc1 = np.full(T.N, -1, np.int32)
    fc1[full] = cls1
    stop0 = np.ascontiguousarray(cols1.T.astype(np.uint8))  # [l0, a]
    stop1 = np.ascontiguousarray(rows0.T.astype(np.uint8))  # [l1, b]
    W0 = np.zeros((T.N, nA), np.float64)
    F0 = np.zeros((T.N, nB), np.float32)
    W1 = np.zeros((T.N, nB), np.float64)
    F1 = np.zeros((T.N, nA), np.float32)
    # frozen column states (label can never change) -> class of that (final) label, else -1
    fz = []
    for c, (lf, cl) in enumerate(((l0f, cls0), (l1f, cls1))):
        if merged:
            fz.append(np.full(T.N, -1, np.int32))
            continue
        lab2cls = np.full(int(T.stopT.shape[c]), -1, np.int32)
        lab2cls[lf] = cl
        z = np.where(T.frozen(c), lab2cls[T.lab[c]], -1).astype(np.int32)
        assert (z[T.frozen(c)] >= 0).all()
        fz.append(np.ascontiguousarray(z))
    L = lib()
    for c, (stp, nF, fc, nG, W, F) in enumerate(((stop0, nA, fc0, nB, W0, F0),
                                                 (stop1, nB, fc1, nA, W1, F1))):
        lab = np.ascontiguousarray(T.lab[c])
        m = np.ascontiguousarray(T.m[c])
        L.column_tables(T.R, T.K, T.N, _p(lab), _p(m), _p(T.dig), nF, _p(stp), _p(fc), nG,
                        _p(W), _p(F))
    return dict(W0=W0, F1=F1, nA=nA, W1=W1, F0=F0, nB=nB, fz0=fz[0], fz1=fz[1], merged=merged)


def lb_value(T, tabs, code):
    """Python copy of the kernel's decomposed lower bound (for truncated policy evaluation)."""
    o0, o1 = code % T.N, code // T.N
    if T.stop(code):
        return 0.0
    s = float(tabs["F1"][o1].astype(np.float64) @ tabs["W0"][o0]
              + tabs["F0"][o0].astype(np.float64) @ tabs["W1"][o1])
    return max(1.0, s * (1 - 1e-6) - 1e-9)


def _tab_args(tabs):
    return [_p(tabs["W0"]), _p(tabs["F1"]), tabs["nA"], _p(tabs["W1"]), _p(tabs["F0"]), tabs["nB"],
            _p(tabs["fz0"]), _p(tabs["fz1"])]


def bnb_solve(T, tabs=None, ub=np.inf, max_entries=int(1.8e8), canon=False, tau=0.0, mode=0,
              gcut=1e-3):
    """Exact optimum if V(root) < ub and the memo stays under max_entries; if V(root) >= ub the
    returned value is a certified lower bound >= ub (fail-high).  With tau > 0 every state whose
    marginal probability m0*m1 is below tau is made terminal with value LB (mode 0) or #unread
    cells (mode 1); by monotonicity of the Bellman operator the result is then a certified lower
    (mode 0) or upper (mode 1) bound on the true optimum."""
    tabs = tabs or lb_tables(T)
    keep, args = _common(T)
    cn = T.canon() if canon else None
    cp = [_p(cn[0]), _p(cn[1])] if cn is not None else [None, None]
    out = np.zeros(12)
    hist = np.zeros(40)
    lib().bnb_solve(*args, *_tab_args(tabs), *cp, float(ub), int(max_entries), float(tau), int(mode),
                    float(gcut), _p(out), _p(hist))
    exact = bool(out[5]) and not out[3]
    return dict(value=float(out[0]), exact=exact and tau == 0.0, solved=exact, overflow=bool(out[3]),
                fail_high=(not out[3]) and not exact and out[0] >= ub,
                calls=int(out[1]), expanded=int(out[2]), memo=int(out[4]), root_lb=float(out[6]),
                frozen_hits=int(out[7]), truncated=int(out[8]), greedy_memo=int(out[9]),
                policy_ub=float(out[10]), policy_states=int(out[11]),
                tau=float(tau), mode=int(mode),
                lb_classes=[tabs["nA"], tabs["nB"]], ub=float(ub), canonical_keys=cn is not None,
                expanded_by_neglog10_marginal={int(b): int(h) for b, h in enumerate(hist) if h})


def greedy_eval(T, tabs=None, max_entries=int(9e7)):
    tabs = tabs or lb_tables(T)
    keep, args = _common(T)
    out = np.zeros(3)
    lib().greedy_eval(*args, *_tab_args(tabs), int(max_entries), _p(out))
    return dict(reads=float(out[0]), states=int(out[1]), overflow=bool(out[2]))


def hindsight(T, n_sample=None, seed=11):
    """Hindsight (clairvoyant) lower bound: E over full labelings of the fewest cells whose reveal
    makes the certificate stop.  Exact (all full labelings of both columns, product prior) unless
    n_sample is given; then n_sample full codes per column are drawn from the prior and all
    n_sample^2 pairs are averaged (unbiased), in 4 independent batches for a standard error."""
    full = T.full_codes()
    lab0, lab1 = np.ascontiguousarray(T.lab[0]), np.ascontiguousarray(T.lab[1])

    def call(f0, p0, f1, p1):
        out = np.zeros(4)
        lib().hindsight_exact(T.R, T.K, _p(lab0), _p(lab1), int(T.stopT.shape[0]),
                              int(T.stopT.shape[1]), _p(T.stopT), len(f0), _p(f0), _p(p0),
                              len(f1), _p(f1), _p(p1), _p(out))
        return float(out[0]), int(out[1])

    if not n_sample:
        pF = [T.m[c][full] / T.m[c][full].sum() for c in range(C)]
        v, w = call(full, pF[0], full, pF[1])
        return dict(hindsight=v, hindsight_worst=w, method="exact")
    rng = np.random.default_rng(seed)
    vals, worst = [], 0
    pF = [T.m[c][full] / T.m[c][full].sum() for c in range(C)]
    nb = max(1, n_sample // 4)
    for _ in range(4):
        f = [np.ascontiguousarray(rng.choice(full, nb, p=pF[c])) for c in range(C)]
        u = np.full(nb, 1.0 / nb)
        v, w = call(f[0], u, f[1], u)
        vals.append(v)
        worst = max(worst, w)
    return dict(hindsight=float(np.mean(vals)), hindsight_se=float(np.std(vals, ddof=1) / 2),
                hindsight_worst_sampled=worst, method=f"sampled {4 * nb}x{4 * nb} pairs in 4 batches")


# ----------------------------------------------------------------------------- policies
def load_policy(path, name, shim, k0, rules):
    spec = importlib.util.spec_from_file_location(f"pol_{name}_{int(shim)}{int(k0)}", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    if shim:
        # RUNTIME shim only (files untouched): the policies hard-code UNIFORM10 = 10 response
        # patterns (the real data's 9 rules); response_patterns() of our K rules gives n patterns.
        from agentic_research.evolve_v4 import helpers
        n = len(helpers.response_patterns(rules))
        mod.UNIFORM10 = np.full(n, 1.0 / n)
    if k0:
        # KSEED=8 = "defer to the seed for the first 8 reads"; +k0 sets KSEED=0 (runtime only).
        mod.KSEED = 0
    return mod.choose


def policy_variants(names):
    out = []
    for n in names:
        out += [n, n + "+shim"]
        src = (POLICIES.get(n) or EXTRA_SMALL[n]).read_text()
        if "KSEED" in src:
            out.append(n + "+shim+k0")
    return out


def case_dict(inst):
    K, R = inst["K"], inst["R"]
    zero = np.zeros((C, R), int)
    return dict(lower=inst["lower"], upper=inst["upper"], epsilon=inst["eps"],
                public=dict(force=inst["force"], log_beta=inst["log_beta"], rules=inst["rules"]),
                hidden=hidden_for_levels(zero, inst["rules"]), labels=labels_from_levels(zero, K),
                expected=False)


def make_view(inst, T, code, cd):
    d = T.digits(code)
    K = inst["K"]
    part = np.full((C, K, inst["R"]), -1, np.int8)
    for k in range(K):
        part[:, k, :] = np.where(d == 0, -1, (d - 1 >= k + 1))
    revealed = {k: np.where(d == 0, np.nan, v)
                for k, v in hidden_for_levels(np.maximum(d - 1, 0), inst["rules"]).items()}
    return View(part, possible_starts(part), cd, revealed, int((d > 0).sum())), d


def eval_policy(inst, T, choose, max_states=None, tau=0.0, tabs=None):
    """Expected reads of a deterministic policy under the prior (decision-tree walk).
    tau = 0: exact.  tau > 0: states with marginal probability m0*m1 < tau are not expanded; the
    policy's remaining reads there lie in [decomposed LB, #unread cells], so the walk returns a
    certified interval (lo, hi).  Returns (lo, hi, distinct states queried)."""
    R, K, N = inst["R"], inst["K"], T.N
    cd = case_dict(inst)
    picks, vals = {}, {}

    def pick(code):
        if code not in picks:
            if max_states and len(picks) >= max_states:
                raise RuntimeError(f"policy tree exceeds {max_states} states")
            view, d = make_view(inst, T, code, cd)
            c, r = choose(view)
            c, r = int(c), int(r)
            if not (0 <= c < C and 0 <= r < R) or d[c, r] != 0:
                raise ValueError(f"invalid/repeated query {(c, r)}")
            picks[code] = (c, r)
        return picks[code]

    def rec(code):
        if T.stop(code):
            return 0.0, 0.0
        if code in vals:
            return vals[code]
        o0, o1 = code % N, code // N
        if tau > 0 and T.m[0][o0] * T.m[1][o1] < tau:
            unread = int((T.dig[o0] == 0).sum() + (T.dig[o1] == 0).sum())
            lo = lb_value(T, tabs, code) if tabs is not None else 1.0
            vals[code] = (lo, float(unread))
            return vals[code]
        c, r = pick(code)
        o = o0 if c == 0 else o1
        m = T.m[c]
        lo = hi = 1.0
        for v in range(K + 1):
            step = (v + 1) * int(T.pw[r])
            p = m[o + step] / m[o]
            a, b = rec(code + (step if c == 0 else step * N))
            lo += p * a
            hi += p * b
        vals[code] = (lo, hi)
        return lo, hi

    lo, hi = rec(0)
    return lo, hi, len(picks)


def sample_levels(inst, rng, n):
    R, K = inst["R"], inst["K"]
    out = np.zeros((n, C, R), int)
    for c, (onset, lvl_p, q) in enumerate(inst["priors"]):
        s = rng.choice(R, n, p=onset)
        l = rng.choice(np.arange(1, K + 1), n, p=lvl_p)
        base = np.where(np.arange(R)[None] >= s[:, None], l[:, None], 0)
        noise = rng.random((n, R)) < q
        out[:, c] = np.where(noise, rng.integers(0, K + 1, (n, R)), base)
    return out


def replay_check(inst, T, choose, n=40):
    """Cross-check the tree walk's stop/transition logic against frozen core.replay."""
    rng = np.random.default_rng(1)
    K, R = inst["K"], inst["R"]
    cd = case_dict(inst)
    got, want = [], []
    for lv in sample_levels(inst, rng, n):
        lab = labels_from_levels(lv, K)
        exp = certificates(possible_starts(lab), inst["lower"], inst["upper"])[0]
        case = dict(cd, labels=lab, hidden=hidden_for_levels(lv, inst["rules"]), expected=exp)
        got.append(replay(case, choose))
        code, reads = 0, 0
        while not T.stop(code):
            view, _ = make_view(inst, T, code, cd)
            c, r = choose(view)
            step = (int(lv[c, r]) + 1) * int(T.pw[r])
            code += step if c == 0 else step * T.N
            reads += 1
        want.append(reads)
    return got == want


def stop_check(inst, T, n=3000):
    rng = np.random.default_rng(3)
    K, R = inst["K"], inst["R"]
    for code in rng.integers(0, T.N * T.N, n):
        d = T.digits(int(code))
        part = np.full((C, K, R), -1, np.int8)
        for k in range(K):
            part[:, k, :] = np.where(d == 0, -1, (d - 1 >= k + 1))
        ref = certificates(possible_starts(part), inst["lower"], inst["upper"])[0] is not None
        if ref != T.stop(int(code)):
            return False
    return True


# ----------------------------------------------------------------------------- driver
def optimum(T, tabs, cfg, greedy=None):
    """Exact branch and bound if it fits in the memo budget, else certified truncated bounds.
    greedy (exact value of a real policy) seeds the search bound: V <= greedy < ub."""
    out = {}
    ub = greedy + 1e-7 if greedy is not None else np.inf
    if cfg.get("exact", True):
        t = time.time()
        b = bnb_solve(T, tabs, ub=ub, max_entries=cfg.get("cap", int(9e7)))
        b["seconds"] = time.time() - t
        out["bnb_exact"] = b
        if b["exact"]:
            out.update(optimal=b["value"], optimal_lo=b["value"], optimal_hi=b["value"],
                       optimal_exact=True)
            return out
    out["optimal_exact"] = False
    cap = cfg.get("cap", int(9e7))
    for tau in cfg.get("taus", (1e-6, 1e-7)):
        t = time.time()
        lo = bnb_solve(T, tabs, ub=ub, tau=tau, mode=0, max_entries=cap, gcut=cfg.get("gcut", 1e-3))
        if lo["overflow"]:
            out.setdefault("bounds_failed", []).append(dict(tau=tau, run="lo", expanded=lo["expanded"],
                                                            seconds=time.time() - t))
            break
        rec = dict(tau=tau, lo=lo["value"], expanded_lo=lo["expanded"], lo_solved=lo["solved"],
                   policy_ub=lo["policy_ub"], policy_states=lo["policy_states"],
                   greedy_memo_lo=lo["greedy_memo"], seconds_lo=time.time() - t)
        out["optimal_lo"] = max(out.get("optimal_lo", 0.0), lo["value"])
        if greedy is not None:
            out["optimal_hi"] = min(out.get("optimal_hi", np.inf), greedy)
        if np.isfinite(lo["policy_ub"]):
            out["optimal_hi"] = min(out.get("optimal_hi", np.inf), lo["policy_ub"])
        if not cfg.get("hi_run", False) or out.get("optimal_hi", np.inf) - out["optimal_lo"] < cfg.get("width", 1e-3):
            out.setdefault("bounds_runs", []).append(rec)
            out["bounds"] = rec
            if out.get("optimal_hi", np.inf) - out["optimal_lo"] < cfg.get("width", 1e-3):
                break
            continue
        t = time.time()
        hi = bnb_solve(T, tabs, ub=ub, tau=tau, mode=cfg.get("hi_mode", 2), max_entries=cap,
                       gcut=cfg.get("gcut", 1e-3))
        rec.update(expanded_hi=hi["expanded"], greedy_memo_hi=hi["greedy_memo"],
                   hi_mode=cfg.get("hi_mode", 2), gcut=cfg.get("gcut", 1e-3), seconds_hi=time.time() - t,
                   hi_overflow=hi["overflow"], hi_solved=hi["solved"])
        # hi run: V <= hi when solved; a fail-high (hi >= ub) or overflow leaves the greedy policy
        # value (a real policy) as the upper bound.
        if hi["solved"] and not hi["overflow"]:
            rec["hi"] = hi["value"]
            out["optimal_hi"] = min(out.get("optimal_hi", np.inf), hi["value"])
        out.setdefault("bounds_runs", []).append(rec)
        out["bounds"] = rec
        if out.get("optimal_hi", np.inf) - out["optimal_lo"] < cfg.get("width", 1e-3):
            break
    if "optimal_lo" not in out:  # nothing fitted: only the root bound
        out["optimal_lo"] = bnb_solve(T, tabs, ub=0.0)["root_lb"]
    if greedy is not None:
        out["optimal_hi"] = min(out.get("optimal_hi", np.inf), greedy)
    if np.isfinite(out.get("optimal_hi", np.inf)):
        out["optimal"] = 0.5 * (out["optimal_lo"] + out["optimal_hi"])
    return out


def run_instance(inst, T, cfg, checks=False):
    t0 = time.time()
    row = dict(i=inst["i"], kind=inst["kind"], eps=inst["eps"], prior=inst["prior"],
               labels_per_column=[int(T.stopT.shape[0]), int(T.stopT.shape[1])])
    if checks:
        row["stop_table_matches_certificates"] = stop_check(inst, T)
    t = time.time()
    tabs = lb_tables(T, cfg.get("max_classes"))
    row["lb_tables"] = dict(classes=[tabs["nA"], tabs["nB"]], merged=tabs["merged"],
                            seconds=time.time() - t)
    if cfg.get("dense"):
        t = time.time()
        row["dense"] = dense_solve(T)
        row["dense"]["seconds"] = time.time() - t
        row["minimax"] = row["dense"]["minimax"]
    t = time.time()
    g = greedy_eval(T, tabs) if cfg.get("greedy", True) else dict(reads=None, overflow=True,
                                                                  skipped=True)
    row["lb_greedy"] = dict(g, seconds=time.time() - t)
    row.update(optimum(T, tabs, cfg, None if g["overflow"] else g["reads"]))
    if "dense" in row:
        assert abs(row["dense"]["optimal"] - row["optimal"]) < 1e-9, (row["dense"], row["optimal"])
    t = time.time()
    hs = cfg.get("hindsight_sample", "auto")
    if hs == "auto":  # exact when (K+1)^(2R) labeling pairs x 2^R subsets stays small
        hs = None if (T.K + 1) ** (2 * T.R) * 2 ** T.R <= 5e10 else 20000
    row.update(hindsight(T, hs))
    row["hindsight_seconds"] = time.time() - t
    row["policies"] = {}
    n_pat = len(_core.response_patterns(inst["rules"]))
    for name in policy_variants(cfg.get("policies", list(POLICIES))):
        base, shim, k0 = name.split("+")[0], "+shim" in name, name.endswith("+k0")
        if shim and not k0 and n_pat == 10:
            row["policies"][name] = dict(same_as=base, note="10 response patterns: shim is a no-op")
            continue
        t = time.time()
        try:
            ch = load_policy(POLICIES.get(base) or EXTRA_SMALL[base], base, shim, k0, inst["rules"])
            lo, hi, nst = eval_policy(inst, T, ch, cfg.get("policy_max_states"),
                                      cfg.get("policy_tau", 0.0), tabs)
            e = dict(reads=0.5 * (lo + hi), reads_lo=lo, reads_hi=hi, states=nst,
                     exact=bool(lo == hi), seconds=time.time() - t)
            if row.get("optimal_exact"):
                e["ratio"] = e["reads"] / row["optimal"]
            if checks and base == "seed" and (shim or n_pat == 10):
                row["replay_matches_tree_eval"] = replay_check(inst, T, ch)
        except Exception as ex:
            e = dict(error=f"{type(ex).__name__}: {str(ex)[:120]}", seconds=time.time() - t)
        row["policies"][name] = e
    row["seconds"] = time.time() - t0
    return row


def instances(R, K, n, pool, start=0, part=(0, 1)):
    """The first n non-trivial instances from index `start`; part=(k, P) keeps every P-th of them
    (offset k) so one size can be split over P processes without overlap."""
    i, got = start - 1, 0
    skipped = []
    while got < n:
        i += 1
        inst = make_instance(i, pool, R, K)
        T = Tables(inst)
        if T.stop(0):  # certificate decides with zero reads: trivial, skip (as optimal_small)
            skipped.append(i)
            continue
        got += 1
        if (got - 1) % part[1] != part[0]:
            continue
        yield inst, T, skipped


def _reads(row, name):
    e = row["policies"].get(name, {})
    if "same_as" in e:
        e = row["policies"].get(e["same_as"], {})
    return e


def summarise(rows, names):
    avg = dict(n=len(rows), n_optimal_exact=sum(bool(r.get("optimal_exact")) for r in rows))
    for key in ("optimal", "optimal_lo", "optimal_hi", "minimax", "hindsight"):
        vals = [r[key] for r in rows if r.get(key) is not None]
        if len(vals) == len(rows):
            avg[key] = float(np.mean(vals))
    if "optimal" in avg and "hindsight" in avg:
        avg["hindsight_over_optimal"] = avg["hindsight"] / avg["optimal"]
    if all(not r["lb_greedy"].get("overflow") for r in rows):
        avg["lb_greedy"] = float(np.mean([r["lb_greedy"]["reads"] for r in rows]))
    avg["policies"] = {}
    for name in names:
        ok = [r for r in rows if "reads" in _reads(r, name)]
        errs = [_reads(r, name)["error"] for r in rows if "error" in _reads(r, name)]
        e = dict(n_ok=len(ok), n_crash=len(errs), first_error=errs[0] if errs else None)
        if ok and len(ok) == len(rows) and "optimal_hi" in avg:
            lo = float(np.mean([_reads(r, name)["reads_lo"] for r in ok]))
            hi = float(np.mean([_reads(r, name)["reads_hi"] for r in ok]))
            e.update(reads=0.5 * (lo + hi), reads_lo=lo, reads_hi=hi,
                     # certified ratio interval: policy bounds over optimum bounds
                     ratio_lo=lo / avg["optimal_hi"], ratio_hi=hi / avg["optimal_lo"])
            if avg["n_optimal_exact"] == len(rows):
                e["ratio_to_optimal"] = e["reads"] / avg["optimal"]
                e["mean_of_ratios"] = float(np.mean([_reads(r, name)["reads"] / r["optimal"]
                                                     for r in ok]))
        avg["policies"][name] = e
    return avg


PARTS = Path(os.environ.get("OPTIMAL_SCALED_PARTS",
                            Path(tempfile.gettempdir()) / "optimal_scaled_parts"))


def save(key, payload):
    """Each run writes its own part file (parallel runs never share a file); `merge` assembles."""
    PARTS.mkdir(parents=True, exist_ok=True)
    tmp = PARTS / f"{key}.json.tmp"
    tmp.write_text(json.dumps(payload, indent=1, default=float))
    os.replace(tmp, PARTS / f"{key}.json")


def merge(keys=None):
    """Assemble part files into optimal_scaled.json; parts of one size (R{R}_K{K}_part{k}of{P})
    are concatenated (instances sorted by index) and the averages recomputed."""
    groups = {}
    for f in sorted(PARTS.glob("*.json")):
        key = f.stem.split("_part")[0]
        if keys and key not in keys:
            continue
        groups.setdefault(key, []).append(json.loads(f.read_text()))
    data = {}
    for key, ps in sorted(groups.items()):
        if len(ps) == 1:
            data[key] = ps[0]
            continue
        rows = sorted((r for p in ps for r in p["instances"]), key=lambda r: r["i"])
        st = dict(ps[0]["settings"])
        st.update(n_instances=len(rows), parts=[p["settings"].get("part") for p in ps],
                  skipped_trivial=sorted(set(x for p in ps for x in p["settings"]["skipped_trivial"])))
        st.pop("part", None)
        names = policy_variants(st["config"].get("policies", list(POLICIES)))
        data[key] = dict(settings=st, average=summarise(rows, names), instances=rows,
                         seconds=sum(p["seconds"] for p in ps))
    OUT.write_text(json.dumps(data, indent=1, default=float))
    return list(data)


def check_small():
    """Reproduce optimal_small.json (C=2, R=5, K=2, 24 instances) with the new code paths."""
    ref = json.loads((HERE / "optimal_small.json").read_text())
    pool = load_pool()
    names = ["seed", "p12_t9", "p5_t2", "p10_t6"]
    ref_name = {"seed": "seed", "p12_t9": "t9", "p5_t2": "t2", "p10_t6": "t6"}
    cfg = dict(dense=True, policies=names)
    rows, worst = [], dict(optimal=0.0, minimax=0, hindsight=0.0, bnb=0.0, policies=0.0)
    skipped = []
    for (inst, T, skipped), rr in zip(instances(5, 2, ref["settings"]["n_instances"], pool),
                                      ref["instances"]):
        assert inst["i"] == rr["i"], (inst["i"], rr["i"])
        row = run_instance(inst, T, cfg, checks=not rows)
        worst["optimal"] = max(worst["optimal"], abs(row["dense"]["optimal"] - rr["optimal"]))
        worst["bnb"] = max(worst["bnb"], abs(row["optimal"] - rr["optimal"]))
        worst["minimax"] = max(worst["minimax"], abs(row["minimax"] - rr["minimax"]))
        worst["hindsight"] = max(worst["hindsight"], abs(row["hindsight"] - rr["hindsight"]))
        for name, e in row["policies"].items():
            refname = name.replace(name.split("+")[0], ref_name[name.split("+")[0]], 1)
            re_ = rr["policies"].get(refname)
            if re_ is None:
                continue
            if "reads" in e and "reads" in re_:
                worst["policies"] = max(worst["policies"], abs(e["reads"] - re_["reads"]))
            elif ("error" in e) != ("error" in re_):
                worst["policies"] = float("inf")
        rows.append(row)
        print(f"[{inst['i']:2d}] dense {row['dense']['optimal']:.6f} bnb {row['optimal']:.6f} "
              f"ref {rr['optimal']:.6f} ({row['bnb_exact']['expanded']} exp) "
              f"mm {row['minimax']}/{rr['minimax']} hind {row['hindsight']:.6f}/{rr['hindsight']:.6f} "
              f"seed+shim {row['policies']['seed+shim'].get('reads', np.nan):.6f}/"
              f"{rr['policies']['seed+shim']['reads']:.6f} t={row['seconds']:.1f}s", flush=True)
    names_v = policy_variants(names)
    out = dict(settings=dict(C=C, R=5, K=2, seed=SEED, n_instances=len(rows), skipped=skipped,
                             note="t2/t6/t9 of optimal_small = colab/bundle/population p5_t2/p10_t6/"
                                  "p12_t9 (runs/a1 is not in this checkout)"),
               reproduction_max_abs_diff=worst, average=summarise(rows, names_v),
               reference_average=dict(optimal=ref["average"]["optimal"],
                                      minimax=ref["average"]["minimax"],
                                      hindsight=ref["average"]["hindsight"]),
               instances=rows)
    save("R5_K2_check", out)
    print("max |diff| vs optimal_small.json:", worst)
    return worst


def run_size(R, K, n, cfg, start=0, tag=None, part=(0, 1)):
    pool = load_pool()
    rows, skipped = [], []
    t0 = time.time()
    names_v = policy_variants(cfg.get("policies", list(POLICIES)))
    for inst, T, skipped in instances(R, K, n, pool, start, part):
        row = run_instance(inst, T, cfg, checks=not rows)
        rows.append(row)
        pol = " ".join(f"{k}={_reads(row, k).get('reads', float('nan')):.3f}" for k in row["policies"])
        print(f"[R{R}K{K} {inst['i']:3d}] {inst['kind']:14s} opt=[{row.get('optimal_lo', float('nan')):.4f},"
              f"{row.get('optimal_hi', float('nan')):.4f}] exact={row.get('optimal_exact')} "
              f"hind={row.get('hindsight', float('nan')):.4f} greedy={row['lb_greedy']['reads'] or float('nan'):.3f} {pol} "
              f"t={time.time() - t0:.0f}s", flush=True)
        out = dict(settings=dict(C=C, R=R, K=K, seed=SEED, n_instances=len(rows), start=start,
                                 part=list(part), skipped_trivial=skipped, config=cfg),
                   average=summarise(rows, names_v), instances=rows, seconds=time.time() - t0)
        save(tag or f"R{R}_K{K}", out)
    return rows


# ----------------------------------------------------------------------------- analysis
def decision_table(T):
    """decT[l0, l1] = 2 (certificate decides True), 1 (False), 0 (undecided)."""
    t0, t1 = T.tup
    out = np.zeros((len(t0), len(t1)), np.uint8)
    step = max(1, int(2e7 // max(1, len(t1) * T.K)))
    b = t1[None]
    for i in range(0, len(t0), step):
        a = t0[i:i + step, None]
        could = (a[..., 0] > 0) & (b[..., 0] > 0) & (np.maximum(a[..., 2], b[..., 2])
                                                   <= np.minimum(a[..., 3], b[..., 3]))
        guar = ((a[..., 1] == 0) & (b[..., 1] == 0) &
                (np.maximum(a[..., 4], b[..., 4]) <= np.minimum(a[..., 5], b[..., 5])))
        out[i:i + step] = np.where(guar.any(-1), 2, np.where(~could.any(-1), 1, 0))
    assert ((out > 0) == (T.stopT > 0)).all()
    return out


def split_optimal(T, D, max_entries=int(9e7)):
    tabs = lb_tables(T)
    none = np.full(T.N, -1, np.int32)  # keep an action at every state (no frozen shortcut)
    keep, args = _common(T)
    out = np.zeros(5)
    lib().bnb_split(*args, _p(tabs["W0"]), _p(tabs["F1"]), tabs["nA"], _p(tabs["W1"]),
                    _p(tabs["F0"]), tabs["nB"], _p(none), _p(none), _p(np.ascontiguousarray(D)),
                    int(max_entries), _p(out))
    return dict(reads=out[0], p_true=out[1], reads_true=out[2], reads_false=out[3], ok=bool(out[4]))


def split_policy(inst, T, D, choose):
    """Exact walk of a policy, splitting expected reads by the final decision."""
    R, K, N = inst["R"], inst["K"], T.N
    cd = case_dict(inst)
    memo = {}

    def rec(code):
        o0, o1 = code % N, code // N
        if T.stop(code):
            return (1.0 if D[T.lab[0][o0], T.lab[1][o1]] == 2 else 0.0), 0.0, 0.0
        if code in memo:
            return memo[code]
        view, d = make_view(inst, T, code, cd)
        c, r = (int(x) for x in choose(view))
        o = o0 if c == 0 else o1
        m = T.m[c]
        A = X = Y = 0.0
        for v in range(K + 1):
            step = (v + 1) * int(T.pw[r])
            p = m[o + step] / m[o]
            a, x, y = rec(code + (step if c == 0 else step * N))
            A += p * a
            X += p * (a + x)
            Y += p * ((1 - a) + y)
        memo[code] = (A, X, Y)
        return memo[code]

    A, X, Y = rec(0)
    return dict(reads=X + Y, p_true=A, reads_true=X, reads_false=Y)


def run_split(R, K, n, names=("seed",), tag=None):
    """Analysis: E[reads | final decision] for the optimal policy vs given policies."""
    pool = load_pool()
    rows = []
    for inst, T, _ in instances(R, K, n, pool):
        D = decision_table(T)
        row = dict(i=inst["i"], optimal=split_optimal(T, D), policies={})
        for nm in names:
            ch = load_policy(POLICIES[nm], nm, True, nm != "seed" and "KSEED" in POLICIES[nm].read_text(),
                             inst["rules"])
            row["policies"][nm] = split_policy(inst, T, D, ch)
        rows.append(row)
        print(R, K, inst["i"], row, flush=True)
    agg = {}
    for who in ["optimal"] + list(names):
        get = (lambda r: r["optimal"]) if who == "optimal" else (lambda r, w=who: r["policies"][w])
        pt = np.mean([get(r)["p_true"] for r in rows])
        agg[who] = dict(p_true=float(pt),
                        reads=float(np.mean([get(r)["reads"] for r in rows])),
                        reads_given_true=float(np.mean([get(r)["reads_true"] for r in rows]) / pt),
                        reads_given_false=float(np.mean([get(r)["reads_false"] for r in rows]) / (1 - pt)))
    save(tag or f"split_R{R}_K{K}", dict(settings=dict(C=C, R=R, K=K, n=len(rows), policies=list(names),
                                                        note="policies with KSEED use +shim+k0"),
                                         average=agg, instances=rows))
    return agg


def report(path=OUT):
    """Markdown table (ratios to the optimum) generated from optimal_scaled.json."""
    data = json.loads(Path(path).read_text())
    cols = [("seed", "seed+shim"), ("distilled", "distilled+shim"), ("p25_t21", "p25_t21+shim"),
            ("p25_t21+k0", "p25_t21+shim+k0"), ("p12_t9", "p12_t9+shim"),
            ("p12_t9+k0", "p12_t9+shim+k0")]
    head = ("| 크기 C×R×K | 사례 | 최적 평균 읽기 | 사후 하한/최적 | " +
            " | ".join(c for c, _ in cols) + " | 1단계 탐욕(LB) |")
    lines = [head, "|" + "---|" * (len(cols) + 5)]

    def order(k):
        st = data[k]["settings"]
        return (st["K"], st["R"], "check" in k)

    for key in sorted((k for k in data if not k.endswith("_check") or k == "R5_K2_check"), key=order):
        d = data[key]
        st, av = d["settings"], d["average"]
        n = av["n"]
        exact = av.get("n_optimal_exact") == n
        if exact:
            opt = f"{av['optimal']:.3f}"
        elif "optimal_lo" in av and "optimal_hi" in av:
            opt = f"[{av['optimal_lo']:.3f}, {av['optimal_hi']:.3f}]"
        else:
            opt = "실패"
        hs = f"{av['hindsight'] / av['optimal']:.3f}" if exact and "hindsight" in av else (
            f"{av['hindsight'] / av['optimal_hi']:.3f}~{av['hindsight'] / av['optimal_lo']:.3f}"
            if "hindsight" in av and "optimal_lo" in av else "-")
        cells = []
        for _, name in cols:
            e = av["policies"].get(name) or {}
            if "ratio_to_optimal" in e:
                cells.append(f"{e['ratio_to_optimal']:.3f}")
            elif "ratio_lo" in e:
                cells.append(f"{e['ratio_lo']:.3f}~{e['ratio_hi']:.3f}")
            elif e.get("n_crash"):
                cells.append(f"충돌 {e['n_crash']}/{n}")
            else:
                cells.append("-")
        g = av.get("lb_greedy")
        gs = (f"{g / av['optimal']:.3f}" if exact else
              f"{g / av['optimal_hi']:.3f}~{g / av['optimal_lo']:.3f}" if g and "optimal_lo" in av
              else "-") if g else "-"
        tag = " (재현)" if "check" in key else ""
        lines.append(f"| 2×{st['R']}×{st['K']}{tag} | {n} | {opt}{'' if exact else ' (구간)'} | {hs} | "
                     + " | ".join(cells) + f" | {gs} |")
    lines += ["", paired(data), "", validation(data)]
    return "\n".join(lines)


def paired(data):
    """Per size: evolved (+k0) vs seed on the same instances (wins / losses / undecided) and the
    mean per-instance ratio; intervals count as a win only if they do not overlap."""
    names = ["distilled+shim", "p25_t21+shim+k0", "p12_t9+shim+k0"]
    lines = ["| 크기 | " + " | ".join(f"{n} vs seed (이김/짐/미정, 평균 비)" for n in names) + " |",
             "|---|" + "---|" * len(names)]
    for key in sorted((k for k in data if not k.endswith("_check")),
                      key=lambda k: (data[k]["settings"]["K"], data[k]["settings"]["R"])):
        rows = data[key]["instances"]
        st = data[key]["settings"]
        cells = []
        for n in names:
            w = l = u = 0
            rat = []
            for r in rows:
                a, b = _reads(r, n), _reads(r, "seed+shim")
                if "reads_lo" not in a or "reads_lo" not in b:
                    continue
                if a["reads_hi"] < b["reads_lo"]:
                    w += 1
                elif a["reads_lo"] > b["reads_hi"]:
                    l += 1
                else:
                    u += 1
                rat.append(a["reads"] / b["reads"])
            cells.append(f"{w}/{l}/{u}, {np.mean(rat):.3f}" if rat else "-")
        lines.append(f"| 2×{st['R']}×{st['K']} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def validation(data):
    """Cross-checks recorded in optimal_scaled.json."""
    out = []
    if "R5_K2_check" in data:
        out.append(f"- R=5 K=2 재현 최대 |차이|: {data['R5_K2_check']['reproduction_max_abs_diff']}")
    exact = {}
    for k, d in data.items():
        if not k.endswith("_check"):
            st = d["settings"]
            for r in d["instances"]:
                if r.get("optimal_exact"):
                    exact[(st["R"], st["K"], r["i"])] = r["optimal"]
    for k, d in sorted(data.items()):
        st = d["settings"]
        if k.endswith("dense_check") or (not k.endswith("_check") and "dense" in d["instances"][0]):
            diffs = [abs(r["dense"]["optimal"] - r["optimal"]) for r in d["instances"] if "dense" in r]
            out.append(f"- {k}: 조밀 DP vs 분기한정 {len(diffs)}사례, 최대 |차이| {max(diffs):.1e}")
        if k.endswith("bounds_check"):
            res = []
            for r in d["instances"]:
                ex = exact.get((st["R"], st["K"], r["i"]))
                if ex is not None:
                    res.append((r["optimal_lo"] - 1e-12 <= ex <= r["optimal_hi"] + 1e-12,
                                r["optimal_hi"] - r["optimal_lo"]))
            if res:
                out.append(f"- {k}: 절단 구간이 정확해를 포함 {sum(a for a, _ in res)}/{len(res)}, "
                           f"최대 폭 {max(b for _, b in res):.1e}")
    return "\n".join(out)


def main():
    cmd = sys.argv[1] if len(sys.argv) > 1 else "check"
    if cmd == "check":
        check_small()
    elif cmd == "run":
        # run R K n [key=value ...]; values are JSON (e.g. taus=[1e-6,1e-7] dense=true)
        R, K, n = int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
        opts = {}
        for a in sys.argv[5:]:
            k, v = a.split("=", 1)
            try:
                opts[k] = json.loads(v)
            except json.JSONDecodeError:
                opts[k] = v
        start, tag = int(opts.pop("start", 0)), opts.pop("tag", None)
        part = tuple(int(x) for x in str(opts.pop("part", "0/1")).split("/"))
        if isinstance(opts.get("policies"), str):
            opts["policies"] = [x for x in opts["policies"].split(",") if x]
        if tag is None:
            tag = f"R{R}_K{K}" + (f"_part{part[0]}of{part[1]}" if part[1] > 1 else "")
        run_size(R, K, n, opts, start, tag, part)
    elif cmd == "merge":
        print(merge(sys.argv[2:] or None))
    elif cmd == "report":
        print(report())
    elif cmd == "split":
        R, K, n = int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
        names = tuple(sys.argv[5].split(",")) if len(sys.argv) > 5 else ("seed",)
        print(run_split(R, K, n, names))


if __name__ == "__main__":
    main()
