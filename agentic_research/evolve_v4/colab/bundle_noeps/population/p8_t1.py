# IDEA: Commit to cheapest refutation witnesses per rule (a "raiser" column whose early starts are zero-covered and a "capper" column with a positive pair), read the riskiest decisive waveform of those witnesses first, then fill the zero cover; when no witness is cheap, commit instead to one common intercept x and verify every column against its x-window.
# NOVELTY: Instead of one-step width/rule-count lookahead or entropy sampling, it plans explicit proof objects (raiser x capper per rule, or a common-x window for all columns) from the public interval geometry, prices them as probes/p + zero-cover fill with a per-column start posterior (with a censoring prior learned from rank-0 reads), and votes risk-first over the rules' plans (proof-number style) so failures surface after one or two reads.
import numpy as np

from agentic_research.evolve_v4.helpers import UNIFORM10, counterfactual_choose, response_patterns

INF = np.inf
LAM = 0.03       # penalty per known isolated positive below a candidate start
ECUT = 150.0     # kill plans dearer than this are "hopeless" -> confirm mode
A0, B0 = 1.0, 5.0  # Beta prior of the left-censoring rate (start 0), updated by rank-0 reads
NRW = 2.0        # prior weight of "no run in this column"
CONF_LAMBDA = 0.1
NOPT = 51
RHO = 0.1 + 0.8 * (np.arange(50) / 49.0) ** 2   # chance a zero at rank j is still consistent with an earlier start
LOGRHO = np.log(RHO)

H = 2.5
KMAT = np.exp(-0.5 * ((np.arange(50)[:, None] - np.arange(50)[None, :]) / H) ** 2)
np.fill_diagonal(KMAT, 0.0)
KAPPA = 0.5

_cache = {}
_state = {"seed": False}


def _cover_cost(need, unread):
    """cost[:, t + 1] = unread vertices needed to zero-cover every needed start <= t (greedy, right endpoint)."""
    a = np.zeros((9, NOPT))
    cost = np.zeros(9)
    last = np.full(9, -9)
    dead = np.zeros(9, bool)
    for k in range(49):
        nd = need[:, k]
        if nd.any():
            todo = nd & (last != k) & (last != k + 1) & ~dead
            if todo.any():
                if k == 0 and unread[0]:
                    last = np.where(todo, 0, last)
                    cost = cost + todo
                elif unread[k + 1]:
                    last = np.where(todo, k + 1, last)
                    cost = cost + todo
                elif unread[k]:
                    last = np.where(todo, k, last)
                    cost = cost + todo
                else:
                    dead = dead | todo
        a[:, k + 1] = np.where(dead, INF, cost)
    a[:, 50] = a[:, 49]
    return a


def _tables(pc, qc, lo, up):
    """Per-column proof tables for the 9 rules (independent of the other columns)."""
    valid = np.isfinite(lo)
    unread = pc[0] == -1
    pos = pc == 1
    zer = pc == 0
    cs = np.cumsum(pos, axis=1)
    posb = np.zeros((9, 50))
    posb[:, 2:] = cs[:, :-2]
    firstpos = np.where(pos.any(axis=1), pos.argmax(axis=1), 50)
    zz = zer & (np.arange(50)[None, :] < firstpos[:, None])
    lz = np.where(zz, LOGRHO[None, :], 0.0)
    rc = np.cumsum(lz[:, ::-1], axis=1)[:, ::-1]
    zpen = np.zeros((9, 50))
    zpen[:, :48] = rc[:, 2:]
    w0 = qc * np.exp(posb * np.log(LAM) + zpen)
    need = qc[:, :49] & valid[:49]
    need_all = qc[:, :49]
    a = _cover_cost(need, unread)
    a_all = _cover_cost(need_all, unread)
    lowv = np.where(qc & valid, lo, INF)
    sm = np.minimum.accumulate(lowv[:, ::-1], axis=1)[:, ::-1]
    lam = np.concatenate([sm, np.full((9, 1), INF)], axis=1)
    upv = np.where(qc & valid, up, -INF)
    cmax = np.maximum.accumulate(upv, axis=1)
    R = np.full((9, NOPT), INF)
    R[:, 0] = cmax[:, -1]
    R[:, 1:50] = cmax[:, :49]
    s0, s1 = pc[:, :49], pc[:, 1:]
    feas = (s0 != 0) & (s1 != 0)
    b = (s0 == -1).astype(float) + (s1 == -1)
    bn = np.where(feas, b, INF)
    bc = np.full((9, NOPT), INF)
    bc[:, 0] = 0.0
    bc[:, 1:50] = bn
    sp = [bn]
    for lv in range(1, 6):
        h = 1 << (lv - 1)
        prev = sp[-1]
        cur = np.full_like(prev, INF)
        cur[:, :49 - h] = np.minimum(prev[:, :49 - h], prev[:, h:])
        sp.append(cur)
    sp = np.stack(sp)    # (6, 9, 49)
    upm = np.maximum.accumulate(np.where(valid, up, -INF))
    lom = np.minimum.accumulate(np.where(valid, lo, INF)[::-1])[::-1]
    return dict(lam=lam, a=a, a_all=a_all, R=R, bc=bc, bn=bn, w0=w0, need=need, need_all=need_all,
                unread=unread, sp=sp, upm=upm, lom=lom, valid=valid,
                lov=lo[valid], upvv=up[valid])


def _cover(need_row, unread, t):
    """Vertices of the greedy zero cover of all needed starts <= t."""
    out = []
    last = -9
    for k in range(min(t, 48) + 1):
        if need_row[k] and last != k and last != k + 1:
            if k == 0 and unread[0]:
                last = 0
            elif unread[k + 1]:
                last = k + 1
            elif unread[k]:
                last = k
            else:
                return None
            out.append(last)
    return out


def _posterior(W0, partial, VF):
    """Start posterior per (column, rule): cumulative over all starts and over valid starts only."""
    known0 = partial[:, :, 0] >= 0
    n_read = known0.sum(axis=0)
    n_pos = (partial[:, :, 0] == 1).sum(axis=0)
    pi0 = (A0 + n_pos) / (A0 + B0 + n_read)          # left-censoring rate learned from rank-0 reads
    W = W0.copy()
    W[:, :, 0] *= (pi0 / (1.0 - pi0) * 48.0)[None, :]
    W[:, :, 49] *= NRW
    Z = np.maximum(W.sum(axis=2, keepdims=True), 1e-300)
    return np.cumsum(W, axis=2) / Z, np.cumsum(W * VF, axis=2) / Z


def _best_plan(lam, a, pr, R, bc, pcp):
    C = lam.shape[0]
    ok = np.isfinite(bc) & (pcp > 1e-9)
    ok[:, 1:] &= R[:, 1:] < R[:, :1]                    # dominated by "leave the column as is"
    flat_ok = np.flatnonzero(ok.ravel())
    if flat_ok.size == 0:
        return None
    Rf = R.ravel()[flat_ok]
    bcf = bc.ravel()[flat_ok]
    pcf = pcp.ravel()[flat_ok]
    g0 = bcf / pcf
    g1 = (1.0 + bcf) / pcf
    od = np.argsort(Rf, kind="stable")
    Rs = Rf[od]
    cols = flat_ok[od] // NOPT
    N = Rs.size
    ar = np.arange(N)
    M0 = np.full((C, N), INF)
    M1 = np.full((C, N), INF)
    M0[cols, ar] = g0[od]
    M1[cols, ar] = g1[od]
    M0 = np.minimum.accumulate(M0, axis=1)
    M1 = np.minimum.accumulate(M1, axis=1)
    qok = np.isfinite(a) & (pr > 1e-12)
    qok[:, 1:] &= lam[:, 1:] > lam[:, :-1]            # skip options that do not raise the column's level
    qf = np.flatnonzero(qok.ravel())
    if qf.size == 0:
        return None
    Lq = lam.ravel()[qf]
    Aq = a.ravel()[qf]
    Pq = pr.ravel()[qf]
    qc = qf // NOPT
    s = np.searchsorted(Rs, Lq, side="left") - 1
    good = s >= 0
    sc = np.where(good, s, 0)
    G0 = M0[:, sc]
    G1 = M1[:, sc]
    G0[:, ~good] = INF
    G1[:, ~good] = INF
    qar = np.arange(qf.size)
    G0[qc, qar] = INF
    G1[qc, qar] = INF
    Gm = np.where(Aq > 0, G1.min(axis=0), G0.min(axis=0))
    E = Gm / Pq + np.maximum(Aq - 1.0, 0.0)
    E = np.where(np.isfinite(E) & ((Aq > 0) | (Gm > 0)), E, INF)
    j = int(np.argmin(E))
    if not np.isfinite(E[j]):
        return None
    q = int(qf[j])
    c, i = divmod(q, NOPT)
    n = int(s[j])
    pre = od[:n + 1]
    vals = (g1 if Aq[j] > 0 else g0)[pre]
    vals = np.where(cols[:n + 1] != c, vals, INF)
    k = int(np.argmin(vals))
    c2, i2 = divmod(int(flat_ok[pre[k]]), NOPT)
    return float(E[j]), c, i - 1, c2, i2


def _confirm(r, tabs, cum, wl, wr):
    """Common-intercept plan for rule r: pick x, then every column must fit its x-window."""
    C = len(tabs)
    pool = []
    for t in tabs:
        pool.append(t["lov"])
        pool.append(t["upvv"])
    cand = np.unique(np.concatenate(pool))
    cand = cand[(cand >= wl) & (cand <= wr)]
    if cand.size == 0:
        return None
    if cand.size > 40:
        cand = cand[np.linspace(0, cand.size - 1, 40).astype(int)]
    J = np.zeros(cand.size)
    feas = np.ones(cand.size, bool)
    info = []
    for c in range(C):
        t = tabs[c]
        kmin = np.searchsorted(t["upm"], cand, side="left")
        kmax = np.searchsorted(t["lom"], cand, side="right") - 1
        bad = (kmin >= 50) | (kmax < 0) | (kmin > kmax)
        kmin = np.clip(kmin, 0, 49)
        kmax = np.clip(kmax, 0, 48)
        front = t["a_all"][r][kmin]
        ln = np.maximum(kmax - kmin + 1, 1)
        lv = np.minimum(np.floor(np.log2(ln)).astype(int), 5)
        spr = t["sp"][:, r, :]
        capc = np.minimum(spr[lv, kmin], spr[lv, np.maximum(kmax - (1 << lv) + 1, 0)])
        cc = cum[c, r]
        mass = cc[kmax] - np.where(kmin > 0, cc[np.maximum(kmin - 1, 0)], 0.0)
        cost = front + capc
        feas &= ~bad & np.isfinite(cost)
        J += np.log(np.maximum(mass, 1e-9)) - CONF_LAMBDA * np.where(np.isfinite(cost), cost, 0.0)
        info.append((kmin, kmax))
    if not feas.any():
        return None
    j = int(np.argmax(np.where(feas, J, -INF)))
    out = []
    for c, (kmin, kmax) in enumerate(info):
        lo_, hi_ = int(kmin[j]), int(kmax[j])
        bn = tabs[c]["bn"][r]
        seg = bn[lo_:hi_ + 1]
        if seg.size and np.isfinite(seg.min()):
            top = lo_ + int(len(seg) - 1 - np.argmin(seg[::-1]))     # highest pair with minimal cost
            out.append((lo_, top))
        else:
            out.append((lo_, -1))
    return float(J[j]), out


def _seed(view):
    patterns = response_patterns(view.rules)
    return counterfactual_choose(view.partial, view.possible, view.lower, view.upper,
                                 patterns, UNIFORM10)


def choose(view):
    partial, possible = view.partial, view.possible
    C = partial.shape[0]
    if view.queries_so_far == 0:
        _cache.clear()
        _state["seed"] = False
    if _state["seed"]:
        return _seed(view)
    tabs = []
    for c in range(C):
        key = (c, partial[c].tobytes(), view.lower[c, 0].tobytes(), view.upper[c, 0].tobytes())
        t = _cache.get(key)
        if t is None:
            t = _tables(partial[c], possible[c], view.lower[c, 0], view.upper[c, 0])
            _cache[key] = t
        tabs.append(t)
    LAMs = np.stack([t["lam"] for t in tabs])
    As = np.stack([t["a"] for t in tabs])
    Rs = np.stack([t["R"] for t in tabs])
    Bs = np.stack([t["bc"] for t in tabs])
    BN = np.stack([t["bn"] for t in tabs])
    VF = np.stack([t["valid"] for t in tabs])[:, None, :].astype(float)
    cum, cumv = _posterior(np.stack([t["w0"] for t in tabs]), partial, VF)
    PRs = np.ones((C, 9, NOPT))
    PRs[:, :, 1:51] = 1.0 - cumv
    PRs = np.where(PRs > 1e-12, PRs, 0.0)
    PCs = np.ones((C, 9, NOPT))
    PCs[:, :, 1:50] = np.where(BN == 2, cum[:, :, :49], np.where(BN == 1, 0.7, 1.0))
    PCs = np.where(PCs > 1e-12, PCs, 0.0)
    known = partial >= 0
    lab = np.where(known, partial, 0).astype(float)
    kn = known.astype(float)
    glob = (lab.sum(axis=(0, 2)) + 0.8) / (kn.sum(axis=(0, 2)) + 2.0)
    risk = (lab @ KMAT.T + KAPPA * glob[None, :, None]) / (kn @ KMAT.T + KAPPA)
    left = LAMs[:, :, 0]
    right = Rs[:, :, 0]
    wl_all, wr_all = left.max(axis=0), right.min(axis=0)
    live = wl_all <= wr_all
    votes = {}
    for r in np.flatnonzero(live):
        plan = _best_plan(LAMs[:, r], As[:, r], PRs[:, r], Rs[:, r], Bs[:, r], PCs[:, r])
        if plan is None or plan[0] > ECUT:
            continue
        _, c, t, c2, i2 = plan
        if t >= 0:
            tb = tabs[c]
            verts = _cover(tb["need"][r], tb["unread"], t)
            if verts:
                for v in verts:
                    votes[(c, v)] = votes.get((c, v), 0.0) + float(risk[c, r, v]) + 1e-3
        if i2 > 0:
            m = i2 - 1
            tb = tabs[c2]
            for v in (m, m + 1):
                if tb["unread"][v]:
                    votes[(c2, v)] = votes.get((c2, v), 0.0) + 1.0 - float(risk[c2, r, v]) + 1e-3
    if not votes:
        best = None
        for r in np.flatnonzero(live):
            res = _confirm(int(r), tabs, cum, wl_all[r], wr_all[r])
            if res is not None and (best is None or res[0] > best[0]):
                best = (res[0], int(r), res[1])
        if best is not None:
            _, r, info = best
            for c, (kmin, m) in enumerate(info):
                tb = tabs[c]
                verts = _cover(tb["need_all"][r], tb["unread"], kmin - 1)
                for v in verts or []:
                    votes[(c, v)] = votes.get((c, v), 0.0) + float(risk[c, r, v]) + 1e-3
                if m >= 0:
                    for v in (m, m + 1):
                        if tb["unread"][v]:
                            votes[(c, v)] = votes.get((c, v), 0.0) + 1.0 - float(risk[c, r, v]) + 1e-3
    if not votes:
        _state["seed"] = True
        _state["seed_at"] = view.queries_so_far
        return _seed(view)
    best = max(votes.items(), key=lambda kv: (kv[1], -kv[0][0], -kv[0][1]))[0]
    return int(best[0]), int(best[1])
