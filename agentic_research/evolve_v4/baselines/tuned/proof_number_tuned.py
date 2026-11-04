# Tuned copy of baselines/proof_number.py (search only, labels only, 28 scorings; see BASELINES_FAIR.md).
# Best of the tuning run: NPAIR=1, VOTE=none, NCOL=3 -> search 5386 (eps0.2 1724 / eps0.4 3662).
"""Proof-number-search-style policy, labels only (tunable copy of baselines/proof_number.py).

Goal tree: decision=False needs EVERY live rule refuted (AND); a rule is refuted
if SOME column has no valid start or SOME column pair cannot overlap (OR).
decision=True needs ONE rule certified to pass (OR over rules, AND over columns).
Proof number of a rule = fewest unread cells whose best-case outcomes refute it
(cheapest witness); disproof number = same toward a pass certificate.  We read a
cell of the cheapest witness of the most-proving rule.

Changes from the original: kill() is an exact faster rewrite (loops over S only, memoised per
call); argsort tie order is a knob (the original used numpy's default sort, whose tie order
depends on the CPU's SIMD sort). Knobs:
  ROUTE      "min": compare min PN with min DN; "sum": sum PN (AND over rules) with min DN;
             "pns": classic most-proving-node descent: root = OR over rules (pass), take the
             rule with the smallest pass cost, inside it (AND over columns) take the child with
             the smallest disproof = that rule's cheapest refutation witness
  NCOL       columns (fewest candidate starts) tried as single-column refutations
  NPAIR      top-left / bottom-right columns tried as separating pairs
  NTHETA     common points tried for pass witnesses
  PASS_BIAS  go for a pass certificate when min DN < PASS_BIAS * PN-side
  VOTE       "inv": cells shared by many cheap witnesses first; "flat": count only; "none"
  EARLY      tie break inside the witness: True = lowest rank first (original), False = highest
  SORT       "stable" or "quicksort" (original default)
  PICK       which witness cell first, after votes: "rank" (original), "zero" = cells whose
             hoped-for label is 0 first, "one" = cells hoped to read 1 first
"""
import numpy as np

INF = 10 ** 6
ROUTE = "min"
NCOL = 3
NPAIR = 1
NTHETA = 5
PASS_BIAS = 1.0
VOTE = "none"
EARLY = True
SORT = "stable"
PICK = "rank"


_MEMO = {}          # pure cache of kill(): key = every input, so it carries no information


def _argsort(a):
    return np.argsort(a, kind=SORT)


def kill(unread, poss, S, none):
    """Cheapest (cost, cells) to remove every start in S, survivors = poss & ~S.
    Exact rewrite of the original prefix-greedy cover: costs only change at elements of S."""
    idx = np.flatnonzero(S)
    if len(idx) == 0:
        return 0, []
    cells, cost, through = [], 0, -1
    marks, costs, lens = [], [], []          # state just before S element idx[t]
    for i in idx.tolist():
        marks.append(i); costs.append(cost); lens.append(len(cells))
        if i != none and i > through:
            if unread[i + 1]:
                cells.append((i + 1, 0)); through = i + 1; cost += 1
            elif unread[i]:
                cells.append((i, 0)); through = i; cost += 1
            elif i >= 1 and unread[i - 1]:
                cells.append((i - 1, 1)); through = i; cost += 1
            else:
                cost += INF; through = i
    best = (INF * 10, [])
    if not S[none]:
        best = (cost, cells)
    surv = np.flatnonzero(poss & ~S)
    if len(surv):
        pos = np.searchsorted(np.array(marks), surv, side="left")
        for j, t in zip(surv.tolist(), pos.tolist()):
            if j >= none:
                continue
            pc, pl = (costs[t], lens[t]) if t < len(marks) else (cost, len(cells))
            pair = [(j + u, 1) for u in (0, 1) if unread[j + u]]
            c = pc + len(pair)
            if c < best[0]:
                best = (c, cells[:pl] + pair)
    return best


def choose(view):
    partial, possible, lower, upper = view.partial, view.possible, view.lower, view.upper
    cols, nrules, ranks = partial.shape
    none = ranks - 1
    valid = np.isfinite(lower[:, 0])                  # (cols, ranks)
    lo, up = lower[:, 0], upper[:, 0]
    unread = partial[:, 0, :] == -1
    cand = possible & valid[:, None, :]
    left = np.where(cand, lo[:, None, :], np.inf).min(-1)
    right = np.where(cand, up[:, None, :], -np.inf).max(-1)
    live = cand.any(-1).all(0) & (left.max(0) <= right.min(0))
    if not live.any():
        r = int(np.flatnonzero(unread.ravel())[0]); return r // ranks, r % ranks
    if len(_MEMO) > 400000:
        _MEMO.clear()
    ub = [unread[c].tobytes() for c in range(cols)]

    def kc(c, S):
        key = (ub[c], S.tobytes(), possible[c, r_cur].tobytes())
        got = _MEMO.get(key)
        if got is None:
            got = _MEMO[key] = kill(unread[c], possible[c, r_cur], S, none)
        return got

    pn = np.full(nrules, INF * 10); pw = [None] * nrules
    dn = np.full(nrules, INF * 10); dw = [None] * nrules
    for r in np.flatnonzero(live):
        r_cur = r
        L, R = left[:, r].max(), right[:, r].min()
        # --- refutation witnesses
        order = _argsort(cand[:, r, :].sum(-1))[:NCOL]
        opts = [(c, cand[c, r].copy()) for c in order]
        top_l = _argsort(-left[:, r])[:NPAIR]
        bot_r = _argsort(right[:, r])[:NPAIR]
        for c1 in top_l:
            for c2 in bot_r:
                if c1 == c2:
                    continue
                opts.append((c1, cand[c1, r] & (lo[c1] <= right[c2, r])))
                opts.append((c2, cand[c2, r] & (up[c2] >= left[c1, r])))
        for c, S in opts:
            cost, cs = kc(c, S)
            if cost < pn[r] and cs:
                pn[r], pw[r] = cost, [(c, k, y) for k, y in cs]
        # --- pass witnesses: common point theta, every column keeps only starts covering it
        thetas = np.unique(np.linspace(L, R, NTHETA))
        for th in thetas:
            tot, wit = 0, []
            S_all = possible[:, r] & ~(valid & (lo <= th) & (th <= up))
            for c in range(cols):
                cost, cs = kc(c, S_all[c])
                tot += cost
                wit += [(c, k, y) for k, y in cs]
                if tot >= dn[r]:
                    break
            if tot < dn[r] and wit:
                dn[r], dw[r] = tot, wit
    psum = pn[live].sum() if ROUTE == "sum" else pn[live].min()
    use_pass = dn.min() < PASS_BIAS * psum
    if ROUTE == "pns":
        r = int(np.argmin(np.where(live, dn, INF * 100)))
        use_pass = pw[r] is None or dn[r] < PASS_BIAS * pn[r]
        wit = dw[r] if use_pass else pw[r]
    elif use_pass:
        r = int(np.argmin(dn)); wit = dw[r]
    else:
        r = int(np.argmin(np.where(live, pn, INF * 100))); wit = pw[r]
    if wit:
        votes = {}
        if VOTE != "none":
            for rr in np.flatnonzero(live):
                w = dw[rr] if use_pass else pw[rr]
                num = dn[rr] if use_pass else pn[rr]
                for c, k, _ in (w or []):
                    votes[(c, k)] = votes.get((c, k), 0) + (1.0 / (1 + num) if VOTE == "inv" else 1.0)
        sign = -1 if EARLY else 1
        want = {"rank": lambda y: 0, "zero": lambda y: 1 - y, "one": lambda y: y}[PICK]
        c, k = max(wit, key=lambda t: (votes.get((t[0], t[1]), 0), want(t[2]), sign * t[1], -t[0]))[:2]
        return int(c), int(k)
    flat = np.flatnonzero(unread.ravel())
    return int(flat[0] // ranks), int(flat[0] % ranks)
