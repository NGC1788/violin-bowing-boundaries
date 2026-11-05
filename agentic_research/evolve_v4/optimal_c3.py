"""Exact Bayes-optimal adaptive querying at C=3 columns (extends optimal_scaled.py, which is C=2).

Run from the repo root (part files go to $OPTIMAL_C3_PARTS, then `merge` -> optimal_c3.json):
  python -m agentic_research.evolve_v4.optimal_c3 check_brute [n]     # R=3 K=2 vs plain-Python Bellman
  python -m agentic_research.evolve_v4.optimal_c3 check_c2 [n]        # C=2 embedded as C=3 vs optimal_scaled
  python -m agentic_research.evolve_v4.optimal_c3 run R K n [key=json ...]   e.g. run 4 2 8 dense=true
  python -m agentic_research.evolve_v4.optimal_c3 merge
  python -m agentic_research.evolve_v4.optimal_c3 report

Same instance generator, prior and frozen certificate formula as optimal_small / optimal_scaled,
with C=3 columns (the generator functions are the C=2 ones with the column count as a parameter;
at C=2 they reproduce optimal_scaled.make_instance draw for draw).  The column-decomposed bound and
the tau truncation become a sum over 3 columns:
  V(o0,o1,o2) >= sum_c E[ W_c(o_c | the other two columns fully known for free) ],
the expectation over the other columns' full labels (independent completions); W_c is a
single-column DP against the "virtual column" that combines the two known full labels.  A state
with two frozen columns (labels that can no longer change) is exactly that single-column DP.
Only the SEARCH split is touched (real-geometry sub-sampling); holdouts are never loaded.
"""
import ctypes
import gc
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
sys.setrecursionlimit(50000)

from agentic_research.evolve_v4 import optimal_scaled as S2  # noqa: E402  (C=2 code, reused read-only)

View, certificates, possible_starts, replay = S2.View, S2.certificates, S2.possible_starts, S2.replay
C = 3
SEED = S2.SEED
POLICIES = {
    "seed": HERE / "seed_a.py",
    "distilled": HERE / "distilled_a.py",
    "p25_t21": HERE / "colab/bundle/population/p25_t21.py",
    "best_policy": HERE / "improved/best_policy.py",
    "general_policy": HERE / "improved/general_policy.py",
    "general_v2": HERE / "improved2/general_v2.py",
}
OUT = HERE / "optimal_c3.json"
PARTS = Path(os.environ.get("OPTIMAL_C3_PARTS", Path(tempfile.gettempdir()) / "optimal_c3_parts"))


# ----------------------------------------------------------------------------- instances
def synth_geometry(rng, R, nc):
    """optimal_scaled.synth_geometry with the column count as a parameter."""
    logf = np.cumsum(rng.uniform(0.1, 0.6, (nc, R)), axis=1)
    c0 = logf[:, R // 2].mean()
    s = rng.integers(1, R - 1, nc)
    mid = np.array([(logf[i, s[i] - 1] + logf[i, s[i]]) / 2 for i in range(nc)])
    log_beta = (c0 - mid) / 2 + rng.normal(0, rng.choice([0.05, 0.15, 0.3]), nc)
    return np.exp(logf), log_beta, "synthetic"


def real_geometry(rng, pool, R, nc):
    case = pool[rng.integers(len(pool))]
    force, lb = case["public"]["force"], case["public"]["log_beta"]
    cols = rng.choice(len(force), nc, replace=False)
    ranks = np.sort(rng.choice(np.arange(10, 50), R, replace=False))
    f = np.maximum.accumulate(force[cols][:, ranks], axis=1) * (1 + 1e-6 * np.arange(R))
    return f, lb[cols], "real_subsample"


def make_instance(i, pool, R, K, nc=C):
    rng = np.random.default_rng(SEED + i)
    force, log_beta, kind = (real_geometry(rng, pool, R, nc) if (pool and i % 2 == 1)
                             else synth_geometry(rng, R, nc))
    eps = float(rng.choice([0.2, 0.4]))
    lower, upper = S2.intervals(force, log_beta, eps)
    priors = [S2.column_prior(rng, R, K) for _ in range(nc)]
    return dict(i=i, kind=kind, eps=eps, force=force, log_beta=log_beta, lower=lower, upper=upper,
                rules=S2.make_rules(K), priors=priors, R=R, K=K, C=nc,
                prior=[dict(onset=o.round(3).tolist(), level_p=l.round(3).tolist(), q=q)
                       for o, l, q in priors])


# ----------------------------------------------------------------------------- tables
def stop_pair(A, B):
    """Frozen certificates() verdict 'decision is not None' for label tuples A (n1,K,6) against
    B (n2,K,6) -> (n1, n2) bool (the formula of optimal_scaled.Tables._stop_table)."""
    a, b = A[:, None], B[None]
    could = (a[..., 0] > 0) & (b[..., 0] > 0) & (np.maximum(a[..., 2], b[..., 2])
                                               <= np.minimum(a[..., 3], b[..., 3]))
    guar = ((a[..., 1] == 0) & (b[..., 1] == 0)
            & (np.maximum(a[..., 4], b[..., 4]) <= np.minimum(a[..., 5], b[..., 5])))
    return guar.any(-1) | ~could.any(-1)


def combine(A, B):
    """Two columns' label tuples -> one 'virtual column' tuple with the same stop behaviour:
    stop3(l, a, b) == stop_pair(l, combine(a, b)) (candidate flags AND, invalid flags OR,
    interval ends combined with the same max/min the certificate takes over columns)."""
    a, b = A[:, None], B[None]
    out = np.empty(np.broadcast_shapes(a.shape, b.shape))
    out[..., 0] = (a[..., 0] > 0) & (b[..., 0] > 0)
    out[..., 1] = (a[..., 1] > 0) | (b[..., 1] > 0)
    out[..., 2] = np.maximum(a[..., 2], b[..., 2])
    out[..., 3] = np.minimum(a[..., 3], b[..., 3])
    out[..., 4] = np.maximum(a[..., 4], b[..., 4])
    out[..., 5] = np.minimum(a[..., 5], b[..., 5])
    return out


def others(c):
    return (1, 2) if c == 0 else (0, 2) if c == 1 else (0, 1)


class Tables3(S2.Tables):
    """Per-column tables (labels, label tuples, marginals) for 3 columns."""

    def __init__(self, inst=None, cols=None):
        if cols is not None:  # (R, K, dig, lab[3], tup[3], m[3]) given directly (C=2 embedding)
            self.R, self.K, self.dig, self.lab, self.tup, self.m = cols
            self.B = self.K + 2
            self.N = self.B ** self.R
            self.pw = self.B ** np.arange(self.R, dtype=np.int64)
            return
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
        masks = np.zeros((N, K), np.int64)
        bits = (1 << np.arange(R)).astype(np.int64)
        for a in range(0, N, 1 << 18):
            d = dig[a:a + (1 << 18)]
            part = np.full((len(d), K, R), -1, np.int8)
            for k in range(K):
                part[:, k, :] = np.where(d == 0, -1, (d - 1 >= k + 1))
            masks[a:a + len(d)] = (possible_starts(part).astype(np.int64) * bits).sum(-1)
        self.lab, self.tup, self.m = [], [], []
        for c in range(C):
            lab, tup = self._labels(masks, inst["lower"][c, 0], inst["upper"][c, 0])
            self.lab.append(lab)
            self.tup.append(np.ascontiguousarray(tup, dtype=np.float64))
            self.m.append(self._marginal(*inst["priors"][c]))

    def split(self, code):
        N = self.N
        return code % N, (code // N) % N, code // (N * N)

    def stop(self, code):
        o = self.split(code)
        a = np.stack([self.tup[c][self.lab[c][o[c]]] for c in range(C)])  # (3, K, 6)
        guar = (a[..., 1] == 0).all(0) & (a[..., 4].max(0) <= a[..., 5].min(0))
        could = (a[..., 0] > 0).all(0) & (a[..., 2].max(0) <= a[..., 3].min(0))
        return bool(guar.any() or not could.any())

    def digits(self, code):
        return np.stack([self.dig[o] for o in self.split(code)])  # (C, R)


# ----------------------------------------------------------------------------- C++ kernel
_LIB = None


def lib():
    global _LIB
    if _LIB is None:
        src = (HERE / "optimal_c3_kernel.cpp").read_text()
        tag = hashlib.sha256(src.encode()).hexdigest()[:12]
        so = Path(tempfile.gettempdir()) / f"optimal_c3_{tag}.so"
        if not so.exists():
            tmp = so.with_suffix(f".{os.getpid()}.so")
            subprocess.run(["g++", "-O3", "-march=native", "-std=c++17", "-shared", "-fPIC",
                            str(HERE / "optimal_c3_kernel.cpp"), "-o", str(tmp)], check=True)
            os.replace(tmp, so)
        L = ctypes.CDLL(str(so))
        P, I, I64, D = ctypes.c_void_p, ctypes.c_int, ctypes.c_int64, ctypes.c_double
        base = [I, I, I64, P, P, P, P]
        lbx = [P, P, P, P, P, P, P, P]
        L.stop_codes.argtypes = base + [I64, P, P]
        L.dense_solve.argtypes = base + [P]
        L.bnb_solve.argtypes = base + lbx + [D, I64, D, I, D, P]
        L.greedy_eval.argtypes = base + lbx + [I64, P]
        L.lb_codes.argtypes = base + lbx + [I64, P, P]
        L.hindsight.argtypes = base + [P, I64, P, P, P, P, I, P, P, P]
        _LIB = L
    return _LIB


def _p(a):
    return a.ctypes.data_as(ctypes.c_void_p)


class Args:
    """Keeps every array handed to the kernel alive; builds 3-pointer arrays."""

    def __init__(self, T):
        self.keep = []
        self.base = [T.R, T.K, T.N, self.parr(T.lab, np.int32), self.parr(T.tup, np.float64),
                     self.parr(T.m, np.float64), _p(self.arr(T.dig, np.int8))]

    def arr(self, a, dt):
        a = np.ascontiguousarray(a, dtype=dt)
        self.keep.append(a)
        return a

    def parr(self, arrs, dt):
        arrs = [self.arr(a, dt) for a in arrs]
        pa = (ctypes.c_void_p * 3)(*[a.ctypes.data for a in arrs])
        self.keep.append(pa)
        return ctypes.cast(pa, ctypes.c_void_p)

    def lb(self, tabs):
        return [self.parr(tabs["W"], np.float64), _p(self.arr(tabs["nA"], np.int32)),
                self.parr(tabs["cls"], np.int32), _p(self.arr(tabs["nf"], np.int32)),
                self.parr(tabs["gptr"], np.int64), self.parr(tabs["gidx"], np.int32),
                self.parr(tabs["gval"], np.float64), self.parr(tabs["fz"], np.int32)]


def stop_codes(T, codes):
    A = Args(T)
    codes = np.ascontiguousarray(codes, np.int64)
    out = np.zeros(len(codes), np.uint8)
    lib().stop_codes(*A.base, len(codes), _p(codes), _p(out))
    return out.astype(bool)


def dense_solve(T):
    A = Args(T)
    out = np.zeros(3)
    lib().dense_solve(*A.base, _p(out))
    return dict(optimal=float(out[0]), minimax=int(out[1]), nonstopped_states=int(out[2]))


def lb_tables(T):
    """Single-column relaxation tables for the 3-column decomposed lower bound."""
    R, K, N = T.R, T.K, T.N
    full = np.flatnonzero((T.dig > 0).all(1)).astype(np.int64)
    flab, fidx = [], []
    for c in range(C):
        u, inv = np.unique(T.lab[c][full], return_inverse=True)
        f = np.full(N, -1, np.int32)
        f[full] = inv.ravel()
        flab.append(u)
        fidx.append(f)
    nf = np.array([len(u) for u in flab], np.int32)
    W, cls, G, nA, fz = [], [], [], [], []
    L2 = S2.lib()
    for c in range(C):
        a, b = others(c)
        comb = combine(T.tup[a][flab[a]], T.tup[b][flab[b]]).reshape(-1, K, 6)
        rows = np.zeros((len(comb), len(T.tup[c])), bool)
        step = max(1, int(2e7 // max(1, len(T.tup[c]) * K)))
        for i in range(0, len(comb), step):
            rows[i:i + step] = stop_pair(comb[i:i + step], T.tup[c])
        urows, cl = np.unique(rows, axis=0, return_inverse=True)
        stopRow = np.ascontiguousarray(urows.T.astype(np.uint8))  # [l_c, class]
        Wc = np.zeros((N, len(urows)), np.float64)
        Fc = np.zeros((N, nf[c]), np.float32)
        labc = np.ascontiguousarray(T.lab[c], np.int32)
        mc = np.ascontiguousarray(T.m[c], np.float64)
        L2.column_tables(R, K, N, S2._p(labc), S2._p(mc), S2._p(T.dig), len(urows), S2._p(stopRow),
                         S2._p(fidx[c]), int(nf[c]), S2._p(Wc), S2._p(Fc))
        assert np.isfinite(Wc).all(), "a full column does not stop against full others"
        W.append(Wc)
        cls.append(cl.ravel().astype(np.int32).reshape(nf[a], nf[b]))
        nA.append(len(urows))
        G.append(Fc.astype(np.float64))
        lab2f = np.full(len(T.tup[c]), -1, np.int32)
        lab2f[flab[c]] = np.arange(nf[c], dtype=np.int32)
        z = np.where(T.frozen(c), lab2f[T.lab[c]], -1).astype(np.int32)
        assert (z[T.frozen(c)] >= 0).all()
        fz.append(z)
    gptr, gidx, gval = [], [], []
    for c in range(C):
        nz = G[c] > 0
        gptr.append(np.concatenate([[0], np.cumsum(nz.sum(1))]).astype(np.int64))
        r, f = np.nonzero(nz)
        gidx.append(f.astype(np.int32))
        gval.append(G[c][r, f])
    return dict(W=W, nA=np.array(nA, np.int32), cls=cls, nf=nf, gptr=gptr, gidx=gidx, gval=gval,
                fz=fz, G=G)


def bnb_solve(T, tabs, ub=np.inf, max_entries=int(9e7), tau=0.0, mode=0, gcut=1e-3):
    A = Args(T)
    out = np.zeros(12)
    lib().bnb_solve(*A.base, *A.lb(tabs), float(ub), int(max_entries), float(tau), int(mode),
                    float(gcut), _p(out))
    solved = bool(out[5]) and not out[3]
    return dict(value=float(out[0]), exact=solved and tau == 0.0, solved=solved,
                overflow=bool(out[3]), fail_high=(not out[3]) and not solved and out[0] >= ub,
                calls=int(out[1]), expanded=int(out[2]), memo=int(out[4]), root_lb=float(out[6]),
                frozen_hits=int(out[7]), truncated=int(out[8]), greedy_memo=int(out[9]),
                policy_ub=float(out[10]), policy_states=int(out[11]), tau=float(tau), mode=int(mode))


def greedy_eval(T, tabs, max_entries=int(6e7)):
    A = Args(T)
    out = np.zeros(3)
    lib().greedy_eval(*A.base, *A.lb(tabs), int(max_entries), _p(out))
    return dict(reads=float(out[0]), states=int(out[1]), overflow=bool(out[2]))


def lb_codes(T, tabs, codes):
    A = Args(T)
    codes = np.ascontiguousarray(codes, np.int64)
    out = np.zeros(len(codes))
    lib().lb_codes(*A.base, *A.lb(tabs), len(codes), _p(codes), _p(out))
    return out


def hindsight(T, n_sample=None, seed=11):
    """E over full labelings of the fewest cells whose reveal makes the certificate stop.  Exact
    over all triples of full column codes (product prior) unless n_sample is given; then 4 batches
    of n_sample/4 independent triples drawn from the prior (mean and standard error)."""
    R = T.R
    full = np.flatnonzero((T.dig > 0).all(1)).astype(np.int64)
    S = 1 << R
    part = np.zeros((len(full), S), np.int64)
    for s in range(S):
        for r in range(R):
            if s >> r & 1:
                part[:, s] += T.dig[full, r].astype(np.int64) * T.pw[r]
    A = Args(T)
    # full codes whose subset->label rows are identical are interchangeable here: merge them
    # (weights added), which shrinks the triple enumeration without changing the value
    uq, inv, pU = [], [], []
    for c in range(C):
        u, iv = np.unique(T.lab[c][part], axis=0, return_inverse=True)
        p = T.m[c][full] / T.m[c][full].sum()
        uq.append(u)
        inv.append(iv.ravel())
        pU.append(np.bincount(iv.ravel(), weights=p, minlength=len(u)))
    labsub = A.parr(uq, np.int32)
    allm = np.arange(1 << (3 * R), dtype=np.int64)
    pc = np.array([bin(x).count("1") for x in range(1 << (3 * R))], np.int8)
    order = np.argsort(pc, kind="stable")
    masks = A.arr(allm[order], np.int32)
    popc = A.arr(pc[order], np.int8)
    pF = [T.m[c][full] / T.m[c][full].sum() for c in range(C)]

    def call(f0, f1, f2, w):
        out = np.zeros(2)
        f = [A.arr(x, np.int32) for x in (f0, f1, f2)]
        w = A.arr(w, np.float64)
        lib().hindsight(*A.base, labsub, len(w), _p(f[0]), _p(f[1]), _p(f[2]), _p(w), len(masks),
                        _p(masks), _p(popc), _p(out))
        return float(out[0]), int(out[1])

    nfull = len(full)
    if not n_sample:
        tot, worst = 0.0, 0
        i1, i2 = np.meshgrid(np.arange(len(uq[1])), np.arange(len(uq[2])), indexing="ij")
        i1, i2 = i1.ravel(), i2.ravel()
        w12 = pU[1][i1] * pU[2][i2]
        for f0 in range(len(uq[0])):
            v, wst = call(np.full(len(i1), f0), i1, i2, pU[0][f0] * w12)
            tot += v
            worst = max(worst, wst)
        return dict(hindsight=tot, hindsight_worst=worst, method="exact",
                    merged_full_codes=[len(u) for u in uq])
    rng = np.random.default_rng(seed)
    nb = max(1, n_sample // 4)
    vals, worst = [], 0
    for _ in range(4):
        f = [inv[c][rng.choice(nfull, nb, p=pF[c])] for c in range(C)]
        v, wst = call(f[0], f[1], f[2], np.full(nb, 1.0 / nb))
        vals.append(v)
        worst = max(worst, wst)
    return dict(hindsight=float(np.mean(vals)), hindsight_se=float(np.std(vals, ddof=1) / 2),
                hindsight_worst_sampled=worst, method=f"sampled {4 * nb} triples in 4 batches")


# ----------------------------------------------------------------------------- policies
def load_policy(path, name, shim, k0, rules):
    return S2.load_policy(path, name, shim, k0, rules)  # runtime shim only; files untouched


def policy_variants(names):
    out = []
    for n in names:
        out += [n, n + "+shim"]
        if "KSEED" in POLICIES[n].read_text():
            out.append(n + "+shim+k0")
    return out


def case_dict(inst):
    K, R = inst["K"], inst["R"]
    zero = np.zeros((C, R), int)
    return dict(lower=inst["lower"], upper=inst["upper"], epsilon=inst["eps"],
                public=dict(force=inst["force"], log_beta=inst["log_beta"], rules=inst["rules"]),
                hidden=S2.hidden_for_levels(zero, inst["rules"]), labels=S2.labels_from_levels(zero, K),
                expected=False)


def make_view(inst, T, code, cd):
    d = T.digits(code)
    K = inst["K"]
    part = np.full((C, K, inst["R"]), -1, np.int8)
    for k in range(K):
        part[:, k, :] = np.where(d == 0, -1, (d - 1 >= k + 1))
    revealed = {k: np.where(d == 0, np.nan, v)
                for k, v in S2.hidden_for_levels(np.maximum(d - 1, 0), inst["rules"]).items()}
    return View(part, possible_starts(part), cd, revealed, int((d > 0).sum())), d


def eval_policy(inst, T, choose, stopf, tau=0.0, tabs=None, max_states=None):
    """Expected reads of a deterministic policy under the prior (decision-tree walk).  tau > 0:
    states with marginal probability < tau are not expanded and bounded by [LB, #unread]."""
    R, K, N = inst["R"], inst["K"], T.N
    cd = case_dict(inst)
    picks, vals = {}, {}
    colstep = [1, N, N * N]

    def rec(code):
        if stopf(code):
            return 0.0, 0.0
        if code in vals:
            return vals[code]
        o = T.split(code)
        if tau > 0 and T.m[0][o[0]] * T.m[1][o[1]] * T.m[2][o[2]] < tau:
            unread = int(sum((T.dig[x] == 0).sum() for x in o))
            vals[code] = (float(lb_codes(T, tabs, [code])[0]), float(unread))
            return vals[code]
        if max_states and len(picks) >= max_states:
            raise RuntimeError(f"policy tree exceeds {max_states} states")
        view, d = make_view(inst, T, code, cd)
        c, r = (int(x) for x in choose(view))
        if not (0 <= c < C and 0 <= r < R) or d[c, r] != 0:
            raise ValueError(f"invalid/repeated query {(c, r)}")
        picks[code] = (c, r)
        m = T.m[c]
        lo = hi = 1.0
        for v in range(K + 1):
            step = (v + 1) * int(T.pw[r])
            p = m[o[c] + step] / m[o[c]]
            a, b = rec(code + step * colstep[c])
            lo += p * a
            hi += p * b
        vals[code] = (lo, hi)
        return lo, hi

    lo, hi = rec(0)
    return lo, hi, len(picks)


def make_stopf(T):
    cache = {}

    def stopf(code):
        v = cache.get(code)
        if v is None:
            v = cache[code] = T.stop(code)
        return v
    return stopf


def stop_check(inst, T, n=3000):
    """Kernel stop and Python stop vs the frozen certificates() on random joint states."""
    rng = np.random.default_rng(3)
    K, R = inst["K"], inst["R"]
    codes = rng.integers(0, T.N ** 3, n)
    ker = stop_codes(T, codes)
    for code, kv in zip(codes, ker):
        d = T.digits(int(code))
        part = np.full((C, K, R), -1, np.int8)
        for k in range(K):
            part[:, k, :] = np.where(d == 0, -1, (d - 1 >= k + 1))
        ref = certificates(possible_starts(part), inst["lower"], inst["upper"])[0] is not None
        if ref != bool(kv) or ref != T.stop(int(code)):
            return False
    return True


def replay_check(inst, T, choose, n=30):
    """Tree-walk transitions/stops vs the frozen core.replay on sampled labelings."""
    rng = np.random.default_rng(1)
    K, R = inst["K"], inst["R"]
    cd = case_dict(inst)
    got, want = [], []
    for lv in _sample(inst, rng, n):
        lab = S2.labels_from_levels(lv, K)
        exp = certificates(possible_starts(lab), inst["lower"], inst["upper"])[0]
        case = dict(cd, labels=lab, hidden=S2.hidden_for_levels(lv, inst["rules"]), expected=exp)
        got.append(replay(case, choose))
        code, reads = 0, 0
        while not T.stop(code):
            view, _ = make_view(inst, T, code, cd)
            c, r = choose(view)
            code += (int(lv[c, r]) + 1) * int(T.pw[r]) * T.N ** int(c)
            reads += 1
        want.append(reads)
    return got == want


def _sample(inst, rng, n):
    R, K = inst["R"], inst["K"]
    out = np.zeros((n, C, R), int)
    for c, (onset, lvl_p, q) in enumerate(inst["priors"]):
        s = rng.choice(R, n, p=onset)
        l = rng.choice(np.arange(1, K + 1), n, p=lvl_p)
        base = np.where(np.arange(R)[None] >= s[:, None], l[:, None], 0)
        noise = rng.random((n, R)) < q
        out[:, c] = np.where(noise, rng.integers(0, K + 1, (n, R)), base)
    return out


# ----------------------------------------------------------------------------- validation
def brute_force(inst):
    """Plain-Python Bellman recursion over joint information states: probabilities by filtering
    the full joint enumeration of (K+1)^(C*R) level configurations (no column factorisation), stop
    test = frozen certificates(possible_starts(.)).  Only for tiny sizes (R=3)."""
    R, K = inst["R"], inst["K"]
    ncell = C * R
    configs = np.array(list(itertools.product(range(K + 1), repeat=ncell)), np.int8)
    w = np.ones(len(configs))
    for c, (onset, lvl_p, q) in enumerate(inst["priors"]):
        col = configs[:, c * R:(c + 1) * R]
        pc = np.zeros(len(configs))
        for s in range(R):
            for l in range(1, K + 1):
                f = np.full(len(configs), onset[s] * lvl_p[l - 1])
                for r in range(R):
                    base = l if r >= s else 0
                    f = f * ((1 - q) * (col[:, r] == base) + q / (K + 1))
                pc += f
        w *= pc
    lower, upper = inst["lower"], inst["upper"]
    memo = {}

    def stops(state):
        d = np.array(state).reshape(C, R)
        part = np.full((C, K, R), -1, np.int8)
        for k in range(K):
            part[:, k, :] = np.where(d < 0, -1, d >= k + 1)
        return certificates(possible_starts(part), lower, upper)[0] is not None

    def V(state, idx):
        if state in memo:
            return memo[state]
        if stops(state):
            memo[state] = 0.0
            return 0.0
        tot = w[idx].sum()
        best = np.inf
        for cell in range(ncell):
            if state[cell] >= 0:
                continue
            e = 1.0
            for v in range(K + 1):
                sub = idx[configs[idx, cell] == v]
                p = w[sub].sum() / tot
                if p > 0:
                    st = list(state)
                    st[cell] = v
                    e += p * V(tuple(st), sub)
            best = min(best, e)
        memo[state] = best
        return best

    return V(tuple([-1] * ncell), np.arange(len(configs))), len(memo)


def check_brute(n=4, R=3, K=2):
    pool = S2.load_pool()
    rows, worst = [], 0.0
    i, got = -1, 0
    while got < n:
        i += 1
        inst = make_instance(i, pool, R, K)
        T = Tables3(inst)
        if T.stop(0):
            continue
        got += 1
        tabs = lb_tables(T)
        t = time.time()
        bf, nst = brute_force(inst)
        tb = time.time() - t
        dn = dense_solve(T)
        bb = bnb_solve(T, tabs)
        g = greedy_eval(T, tabs)
        root_lb = bb["root_lb"]
        row = dict(i=i, kind=inst["kind"], brute=bf, brute_states=nst, brute_seconds=tb,
                   dense=dn["optimal"], bnb=bb["value"], bnb_exact=bb["exact"], root_lb=root_lb,
                   greedy=g["reads"], stop_check=stop_check(inst, T, 2000))
        worst = max(worst, abs(bf - dn["optimal"]), abs(bf - bb["value"]))
        row["lb_admissible"] = root_lb <= bf + 1e-12 and g["reads"] >= bf - 1e-12
        rows.append(row)
        print(row, flush=True)
    out = dict(settings=dict(C=C, R=R, K=K, seed=SEED, n=len(rows)), max_abs_diff=worst, instances=rows)
    save(f"check_brute_R{R}_K{K}", out)
    print("max |brute - kernel|:", worst)
    return out


def c2_reference(R, K, n, variants=None, part=(0, 1)):
    """Same-family C=2 baseline for the policies of this study (optimal_scaled machinery: its
    generator, exact branch and bound, decision-tree walk); optimal_scaled.json lacks
    best_policy / general_policy / general_v2 and R=4."""
    pool = S2.load_pool()
    names = variants or policy_variants(list(POLICIES))
    rows, t0 = [], time.time()
    for inst, T, skipped in S2.instances(R, K, n, pool, 0, part):
        tabs = S2.lb_tables(T)
        g = S2.greedy_eval(T, tabs)
        b = S2.bnb_solve(T, tabs, ub=g["reads"] + 1e-7, max_entries=int(4e7))
        row = dict(i=inst["i"], kind=inst["kind"], optimal=b["value"], optimal_exact=b["exact"],
                   optimal_lo=b["value"], optimal_hi=b["value"], lb_greedy=dict(reads=g["reads"]),
                   policies={})
        row.update(S2.hindsight(T))
        for name in names:
            base, shim, k0 = name.split("+")[0], "+shim" in name, name.endswith("+k0")
            try:
                ch = load_policy(POLICIES[base], base, shim, k0, inst["rules"])
                lo, hi, nst = S2.eval_policy(inst, T, ch)
                row["policies"][name] = dict(reads=lo, reads_lo=lo, reads_hi=hi, states=nst,
                                             ratio=lo / b["value"])
            except Exception as ex:
                row["policies"][name] = dict(error=f"{type(ex).__name__}: {str(ex)[:160]}")
        rows.append(row)
        print(f"[C2 R{R}K{K} {inst['i']}] opt={b['value']:.4f} exact={b['exact']} t={time.time() - t0:.0f}s",
              flush=True)
        save(f"c2ref_R{R}_K{K}" + (f"_part{part[0]}of{part[1]}" if part[1] > 1 else ""),
             dict(settings=dict(C=2, R=R, K=K, seed=SEED, n_instances=len(rows), part=list(part),
                                skipped_trivial=skipped, config=dict(variants=names)),
                  average=summarise(rows, names), instances=rows, seconds=time.time() - t0))
    return rows


def check_c2ref(R=5, K=2):
    """The C=2 reference rows must reproduce optimal_scaled.json for the policies both contain."""
    ref = {r["i"]: r for r in json.loads((HERE / "optimal_scaled.json").read_text())[f"R{R}_K{K}"]["instances"]}
    mine = json.loads((PARTS / f"c2ref_R{R}_K{K}.json").read_text())["instances"]
    worst, n = 0.0, 0
    for r in mine:
        q = ref.get(r["i"])
        if q is None:
            continue
        n += 1
        worst = max(worst, abs(q["optimal"] - r["optimal"]), abs(q["hindsight"] - r["hindsight"]))
        for k in ("seed+shim", "distilled+shim", "p25_t21+shim", "p25_t21+shim+k0"):
            worst = max(worst, abs(q["policies"][k]["reads"] - r["policies"][k]["reads"]))
    out = dict(R=R, K=K, n_compared=n, n_reference=len(ref), max_abs_diff=worst,
               compared=["optimal", "hindsight", "seed+shim", "distilled+shim", "p25_t21+shim",
                         "p25_t21+shim+k0"])
    save(f"check_c2ref_R{R}_K{K}", out)
    print(out)
    return out


def neutral_tables(T2):
    """Embed a C=2 instance as C=3: column 2 has one label that never constrains the certificate
    (a candidate in every rule, no invalid start, interval (-inf, inf)); it is frozen everywhere,
    so the C=3 optimum must equal the C=2 optimum."""
    K = T2.K
    neutral = np.tile(np.array([1.0, 0.0, -np.inf, np.inf, -np.inf, np.inf]), (1, K, 1))
    return Tables3(cols=(T2.R, K, T2.dig, [T2.lab[0], T2.lab[1], np.zeros(T2.N, np.int32)],
                         [np.ascontiguousarray(T2.tup[0], float), np.ascontiguousarray(T2.tup[1], float),
                          neutral], [T2.m[0], T2.m[1], T2.m[0]]))


def check_c2(n=8, R=5, K=2):
    """C=2 instances (optimal_scaled generator) embedded as C=3 vs optimal_scaled's exact solver;
    also checks that make_instance(nc=2) reproduces optimal_scaled.make_instance."""
    pool = S2.load_pool()
    rows, worst = [], 0.0
    for inst2, T2, _ in S2.instances(R, K, n, pool):
        mine = make_instance(inst2["i"], pool, R, K, nc=2)
        same_gen = all(np.array_equal(np.nan_to_num(mine[k]), np.nan_to_num(inst2[k]))
                       for k in ("force", "log_beta", "lower", "upper"))
        ref = S2.bnb_solve(T2, S2.lb_tables(T2), max_entries=int(5e7))
        T3 = neutral_tables(T2)
        tabs = lb_tables(T3)
        bb = bnb_solve(T3, tabs)
        g3 = greedy_eval(T3, tabs)
        g2 = S2.greedy_eval(T2, S2.lb_tables(T2))
        d = abs(bb["value"] - ref["value"])
        worst = max(worst, d)
        row = dict(i=inst2["i"], same_generator=same_gen, c2=ref["value"], c2_exact=ref["exact"],
                   c3_embedded=bb["value"], c3_exact=bb["exact"], abs_diff=d,
                   greedy_c2=g2["reads"], greedy_c3_embedded=g3["reads"])
        rows.append(row)
        print(row, flush=True)
    out = dict(settings=dict(R=R, K=K, n=len(rows)), max_abs_diff=worst, instances=rows)
    save(f"check_c2_R{R}_K{K}", out)
    print("max |C=3 embedded - C=2|:", worst)
    return out


# ----------------------------------------------------------------------------- driver
def optimum(T, tabs, cfg, greedy=None):
    out = {}
    ub = greedy + 1e-7 if greedy is not None else np.inf
    cap = cfg.get("cap", int(9e7))
    if cfg.get("exact", True):
        t = time.time()
        b = bnb_solve(T, tabs, ub=ub, max_entries=cap)
        b["seconds"] = time.time() - t
        out["bnb_exact"] = b
        if b["exact"]:
            out.update(optimal=b["value"], optimal_lo=b["value"], optimal_hi=b["value"],
                       optimal_exact=True)
            return out
    out["optimal_exact"] = False
    for tau in cfg.get("taus", (1e-5, 1e-6)):
        t = time.time()
        lo = bnb_solve(T, tabs, ub=ub, tau=tau, mode=0, max_entries=cap, gcut=cfg.get("gcut", 1e-3))
        rec = dict(tau=tau, lo=lo["value"], expanded=lo["expanded"], solved=lo["solved"],
                   overflow=lo["overflow"], policy_ub=lo["policy_ub"], seconds=time.time() - t)
        out.setdefault("bounds_runs", []).append(rec)
        if lo["overflow"]:
            break
        out["optimal_lo"] = max(out.get("optimal_lo", 0.0), lo["value"])
        if np.isfinite(lo["policy_ub"]):
            out["optimal_hi"] = min(out.get("optimal_hi", np.inf), lo["policy_ub"])
        if out.get("optimal_hi", np.inf) - out["optimal_lo"] < cfg.get("width", 1e-3):
            break
    if "optimal_lo" not in out:
        out["optimal_lo"] = bnb_solve(T, tabs, ub=0.0)["root_lb"]
    if greedy is not None:
        out["optimal_hi"] = min(out.get("optimal_hi", np.inf), greedy)
    if np.isfinite(out.get("optimal_hi", np.inf)):
        out["optimal"] = 0.5 * (out["optimal_lo"] + out["optimal_hi"])
    return out


def run_instance(inst, T, cfg, checks=False):
    t0 = time.time()
    row = dict(i=inst["i"], kind=inst["kind"], eps=inst["eps"], prior=inst["prior"],
               labels_per_column=[len(t) for t in T.tup])
    if checks:
        row["stop_matches_certificates"] = stop_check(inst, T)
    t = time.time()
    tabs = lb_tables(T)
    row["lb_tables"] = dict(classes=tabs["nA"].tolist(), full_labels=tabs["nf"].tolist(),
                            seconds=time.time() - t)
    if cfg.get("dense"):
        t = time.time()
        row["dense"] = dense_solve(T)
        row["dense"]["seconds"] = time.time() - t
        row["minimax"] = row["dense"]["minimax"]
    t = time.time()
    g = greedy_eval(T, tabs)
    row["lb_greedy"] = dict(g, seconds=time.time() - t)
    if g["overflow"]:
        row["lb_greedy"]["reads"] = None
    row.update(optimum(T, tabs, cfg, None if g["overflow"] else g["reads"]))
    if "dense" in row and row.get("optimal_exact"):
        assert abs(row["dense"]["optimal"] - row["optimal"]) < 1e-9, (row["dense"], row["optimal"])
    t = time.time()
    hs = cfg.get("hindsight_sample", "auto")
    if hs == "auto":
        hs = None if (T.K + 1) ** (3 * T.R) <= 2e7 else 40000
    row.update(hindsight(T, hs))
    row["hindsight_seconds"] = time.time() - t
    row["policies"] = {}
    stopf = make_stopf(T)
    n_pat = len(S2._core.response_patterns(inst["rules"]))
    names = cfg.get("variants") or policy_variants(cfg.get("policies", list(POLICIES)))
    for name in names:
        base, shim, k0 = name.split("+")[0], "+shim" in name, name.endswith("+k0")
        if shim and not k0 and n_pat == 10:
            row["policies"][name] = dict(same_as=base, note="10 response patterns: shim is a no-op")
            continue
        t = time.time()
        try:
            ch = load_policy(POLICIES[base], base, shim, k0, inst["rules"])
            lo, hi, nst = eval_policy(inst, T, ch, stopf, cfg.get("policy_tau", 0.0), tabs,
                                      cfg.get("policy_max_states"))
            e = dict(reads=0.5 * (lo + hi), reads_lo=lo, reads_hi=hi, states=nst,
                     exact=bool(lo == hi), seconds=time.time() - t)
            if row.get("optimal_exact"):
                e["ratio"] = e["reads"] / row["optimal"]
            if checks and base == "seed" and shim:
                row["replay_matches_tree_eval"] = replay_check(inst, T, ch)
        except Exception as ex:
            e = dict(error=f"{type(ex).__name__}: {str(ex)[:160]}", seconds=time.time() - t)
        row["policies"][name] = e
    row["seconds"] = time.time() - t0
    return row


def instances(R, K, n, pool, start=0, part=(0, 1)):
    i, got, skipped = start - 1, 0, []
    while got < n:
        i += 1
        inst = make_instance(i, pool, R, K)
        T = Tables3(inst)
        if T.stop(0):
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
    failed = [r["i"] for r in rows if not np.isfinite(r.get("optimal_hi", np.nan))]
    all_n = len(rows)
    rows = [r for r in rows if r["i"] not in failed]
    if not rows:
        return dict(n=0, n_total=all_n, n_bounds_failed=len(failed), failed_instances=failed)
    avg = dict(n=len(rows), n_total=all_n, n_bounds_failed=len(failed), failed_instances=failed,
               n_optimal_exact=sum(bool(r.get("optimal_exact")) for r in rows))
    for key in ("optimal", "optimal_lo", "optimal_hi", "minimax", "hindsight"):
        vals = [r[key] for r in rows if r.get(key) is not None]
        if len(vals) == len(rows):
            avg[key] = float(np.mean(vals))
    if "optimal" in avg and "hindsight" in avg:
        avg["hindsight_over_optimal"] = avg["hindsight"] / avg["optimal"]
    if all(r["lb_greedy"].get("reads") is not None for r in rows):
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
                     ratio_lo=lo / avg["optimal_hi"], ratio_hi=hi / avg["optimal_lo"])
            if avg["n_optimal_exact"] == len(rows) and lo == hi:
                e["ratio_to_optimal"] = e["reads"] / avg["optimal"]
                e["mean_of_ratios"] = float(np.mean([_reads(r, name)["reads"] / r["optimal"] for r in ok]))
        avg["policies"][name] = e
    return avg


def save(key, payload):
    PARTS.mkdir(parents=True, exist_ok=True)
    tmp = PARTS / f"{key}.json.tmp"
    tmp.write_text(json.dumps(payload, indent=1, default=float))
    os.replace(tmp, PARTS / f"{key}.json")


def run_size(R, K, n, cfg, start=0, tag=None, part=(0, 1)):
    pool = S2.load_pool()
    rows, skipped = [], []
    t0 = time.time()
    names_v = cfg.get("variants") or policy_variants(cfg.get("policies", list(POLICIES)))
    for inst, T, skipped in instances(R, K, n, pool, start, part):
        row = run_instance(inst, T, cfg, checks=not rows and part[0] == 0)
        del T
        gc.collect()
        rows.append(row)
        pol = " ".join(f"{k}={_reads(row, k).get('reads', float('nan')):.3f}" for k in row["policies"])
        print(f"[C3 R{R}K{K} {inst['i']:3d}] {inst['kind']:14s} opt=[{row.get('optimal_lo', np.nan):.4f},"
              f"{row.get('optimal_hi', np.nan):.4f}] exact={row.get('optimal_exact')} "
              f"hind={row.get('hindsight', np.nan):.4f} greedy={row['lb_greedy']['reads'] or np.nan:.3f} "
              f"{pol} t={time.time() - t0:.0f}s", flush=True)
        out = dict(settings=dict(C=C, R=R, K=K, seed=SEED, n_instances=len(rows), start=start,
                                 part=list(part), skipped_trivial=skipped, config=cfg),
                   average=summarise(rows, names_v), instances=rows, seconds=time.time() - t0)
        save(tag or f"R{R}_K{K}", out)
    return rows


def merge():
    groups, checks = {}, {}
    for f in sorted(PARTS.glob("*.json")):
        d = json.loads(f.read_text())
        if f.stem.startswith("check_"):
            checks[f.stem] = d
            continue
        groups.setdefault(f.stem.split("_part")[0], []).append(d)
    data = dict(checks=checks)
    for key, ps in sorted(groups.items()):
        if key.startswith("c2ref_"):
            rows = sorted((r for p in ps for r in p["instances"]), key=lambda r: r["i"])
            st = dict(ps[0]["settings"], n_instances=len(rows))
            st.pop("part", None)
            data.setdefault("c2_reference", {})[key] = dict(
                settings=st, average=summarise(rows, st["config"]["variants"]), instances=rows)
            continue
        rows = sorted((r for p in ps for r in p["instances"]), key=lambda r: r["i"])
        st = dict(ps[0]["settings"])
        st.update(n_instances=len(rows), parts=[p["settings"].get("part") for p in ps],
                  skipped_trivial=sorted(set(x for p in ps for x in p["settings"]["skipped_trivial"])))
        st.pop("part", None)
        names = st["config"].get("variants") or policy_variants(st["config"].get("policies", list(POLICIES)))
        data[key] = dict(settings=st, average=summarise(rows, names), instances=rows,
                         seconds=sum(p["seconds"] for p in ps))
    OUT.write_text(json.dumps(data, indent=1, default=float))
    return list(data)


COLS = [("seed", "seed+shim"), ("distilled", "distilled+shim"), ("p25_t21+k0", "p25_t21+shim+k0"),
        ("best_policy+k0", "best_policy+shim+k0"), ("general_policy+k0", "general_policy+shim+k0"),
        ("general_v2+k0", "general_v2+shim+k0")]


def report(path=OUT):
    data = json.loads(Path(path).read_text())
    lines = ["| 크기 C×R×K | 사례 | 최적 평균 읽기 | 사후 하한/최적 | " + " | ".join(c for c, _ in COLS)
             + " | 1단계 탐욕(LB) |", "|" + "---|" * (len(COLS) + 5)]
    blocks = [(k, data[k], 3) for k in data if k not in ("checks", "c2_reference")]
    blocks += [(k, v, 2) for k, v in data.get("c2_reference", {}).items()]
    for key, blk, nc in sorted(blocks, key=lambda x: (x[2], x[1]["settings"]["R"])):
        st, av = blk["settings"], blk["average"]
        n = av["n"]
        exact = av.get("n_optimal_exact") == n
        opt = f"{av['optimal']:.3f}" if exact else f"[{av['optimal_lo']:.3f}, {av['optimal_hi']:.3f}] (구간)"
        hs = (f"{av['hindsight'] / av['optimal']:.3f}" if exact else
              f"{av['hindsight'] / av['optimal_hi']:.3f}~{av['hindsight'] / av['optimal_lo']:.3f}")
        cells = []
        for _, name in COLS:
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
        gs = ("-" if not g else f"{g / av['optimal']:.3f}" if exact else
              f"{g / av['optimal_hi']:.3f}~{g / av['optimal_lo']:.3f}")
        ns = f"{n}" if not av.get("n_bounds_failed") else f"{n}/{av['n_total']}"
        lines.append(f"| {nc}×{st['R']}×{st['K']} | {ns} | {opt} | {hs} | " + " | ".join(cells) + f" | {gs} |")
    lines += ["", paired(blocks)]
    return "\n".join(lines)


def paired(blocks):
    """Per size: mean of per-instance ratios to the optimum (± standard error) and, for each
    evolved policy, wins/losses/ties against seed+shim on the same instances."""
    names = [n for _, n in COLS]
    lines = ["| 크기 | " + " | ".join(f"{c} (사례별 비 평균±표준오차; seed 대비 이김/짐/같음)" for c, _ in COLS)
             + " |", "|---|" + "---|" * len(COLS)]
    for key, blk, nc in sorted(blocks, key=lambda x: (x[2], x[1]["settings"]["R"])):
        # certified intervals (R=6): ratio of interval midpoints; a win/loss needs disjoint intervals
        rows = [r for r in blk["instances"] if np.isfinite(r.get("optimal_hi", np.nan))]
        cells = []
        for n in names:
            rat, w, l, u = [], 0, 0, 0
            for r in rows:
                a, b = _reads(r, n), _reads(r, "seed+shim")
                if "reads" not in a:
                    continue
                rat.append(a["reads"] / r["optimal"])
                if n != "seed+shim" and "reads" in b:
                    w += a["reads_hi"] < b["reads_lo"] - 1e-9
                    l += a["reads_lo"] > b["reads_hi"] + 1e-9
                    u += not (a["reads_hi"] < b["reads_lo"] - 1e-9 or a["reads_lo"] > b["reads_hi"] + 1e-9)
            if len(rat) < 2:
                cells.append("-")
                continue
            se = np.std(rat, ddof=1) / np.sqrt(len(rat))
            cells.append(f"{np.mean(rat):.3f}±{se:.3f}" + ("" if n == "seed+shim" else f"; {w}/{l}/{u}"))
        lines.append(f"| {nc}×{blk['settings']['R']}×{blk['settings']['K']} ({len(rows)}) | "
                     + " | ".join(cells) + " |")
    return "\n".join(lines)


def main():
    cmd = sys.argv[1] if len(sys.argv) > 1 else "check_brute"
    if cmd == "check_brute":
        check_brute(int(sys.argv[2]) if len(sys.argv) > 2 else 4)
    elif cmd == "check_c2":
        check_c2(int(sys.argv[2]) if len(sys.argv) > 2 else 8)
    elif cmd == "check_c2ref":
        check_c2ref()
    elif cmd == "c2ref":
        part = tuple(int(x) for x in (sys.argv[5] if len(sys.argv) > 5 else "0/1").split("/"))
        c2_reference(int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), part=part)
    elif cmd == "run":
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
        for key in ("policies", "variants"):
            if isinstance(opts.get(key), str):
                opts[key] = [x for x in opts[key].split(",") if x]
        if tag is None:
            tag = f"R{R}_K{K}" + (f"_part{part[0]}of{part[1]}" if part[1] > 1 else "")
        run_size(R, K, n, opts, start, tag, part)
    elif cmd == "merge":
        print(merge())
    elif cmd == "report":
        print(report())


if __name__ == "__main__":
    main()
