"""Proof-number-search-style policy, labels only.

Goal tree: decision=False needs EVERY live rule refuted (AND); a rule is refuted
if SOME column has no valid start or SOME column pair cannot overlap (OR).
decision=True needs ONE rule certified to pass (OR over rules, AND over columns).
Proof number of a rule = fewest unread cells whose best-case outcomes refute it
(cheapest witness); disproof number = same toward a pass certificate.  We read a
cell of the cheapest witness of the most-proving rule.

Cell model: reading 0 at rank k kills starts k-1,k; reading 1 at k kills start
k+1; 1,1 at (j,j+1) fixes the first run at j and kills every later start.
"""
import numpy as np

INF = 10 ** 6
ROUTE = "min"      # "min": compare min PN with min DN; "sum": compare sum PN with min DN


def kill(unread, poss, S, none):
    """Cheapest (cost, cells) to remove every start in S, survivors = poss & ~S.
    unread: (R,) bool for this column, poss: (R,) bool, S: (R,) bool."""
    idx = np.flatnonzero(S)
    if len(idx) == 0:
        return 0, []
    # prefix greedy cover by zeros/ones; pre[i] = (cost, ncells) covering S elements < i
    cells, pre_cost, pre_len = [], {}, {}
    cost, through = 0, -1
    sl = set(idx.tolist())
    for i in range(none + 1):
        pre_cost[i], pre_len[i] = cost, len(cells)
        if i in sl and i != none and i > through:
            if i + 1 < none + 1 and unread[i + 1]:
                cells.append((i + 1, 0)); through = i + 1; cost += 1
            elif unread[i]:
                cells.append((i, 0)); through = i; cost += 1
            elif i >= 1 and unread[i - 1]:
                cells.append((i - 1, 1)); through = i; cost += 1
            else:
                cost += INF; through = i
    best = (INF * 10, [])
    if not S[none]:
        best = (cost, cells)           # no-run survivor: cover everything with zeros
    last = idx[idx != none].max() if (idx != none).any() else -1
    for j in np.flatnonzero(poss & ~S):
        if j >= none:
            continue
        pair = [(int(j) + t, 1) for t in (0, 1) if unread[j + t]]
        c = pre_cost[j] + len(pair)
        if c < best[0]:
            best = (c, cells[:pre_len[j]] + pair)
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
    pn = np.full(nrules, INF * 10); pw = [None] * nrules
    dn = np.full(nrules, INF * 10); dw = [None] * nrules
    for r in np.flatnonzero(live):
        L, R = left[:, r].max(), right[:, r].min()
        # --- refutation witnesses
        order = np.argsort(cand[:, r, :].sum(-1))[:4]
        opts = [(c, cand[c, r].copy()) for c in order]
        top_l = np.argsort(-left[:, r])[:3]
        bot_r = np.argsort(right[:, r])[:3]
        for c1 in top_l:
            for c2 in bot_r:
                if c1 == c2:
                    continue
                opts.append((c1, cand[c1, r] & (lo[c1] <= right[c2, r])))
                opts.append((c2, cand[c2, r] & (up[c2] >= left[c1, r])))
        for c, S in opts:
            cost, cs = kill(unread[c], possible[c, r], S, none)
            if cost < pn[r] and cs:
                pn[r], pw[r] = cost, [(c, k, y) for k, y in cs]
        # --- pass witnesses: common point theta, every column keeps only starts covering it
        thetas = np.unique(np.linspace(L, R, 5))
        for th in thetas:
            tot, wit = 0, []
            for c in range(cols):
                S = possible[c, r] & ~(valid[c] & (lo[c] <= th) & (th <= up[c]))
                cost, cs = kill(unread[c], possible[c, r], S, none)
                tot += cost
                wit += [(c, k, y) for k, y in cs]
                if tot >= dn[r]:
                    break
            if tot < dn[r] and wit:
                dn[r], dw[r] = tot, wit
    psum = pn[live].sum() if ROUTE == "sum" else pn[live].min()
    use_pass = dn.min() < psum
    if use_pass:
        r = int(np.argmin(dn)); wit = dw[r]
    else:
        r = int(np.argmin(np.where(live, pn, INF * 100))); wit = pw[r]
    if wit:
        # most proving: the witness cell whose column carries the most witness cells (shared path)
        votes = {}
        for rr in np.flatnonzero(live):
            w = dw[rr] if use_pass else pw[rr]
            for c, k, _ in (w or []):
                votes[(c, k)] = votes.get((c, k), 0) + 1.0 / (1 + (dn[rr] if use_pass else pn[rr]))
        c, k = max(wit, key=lambda t: (votes.get((t[0], t[1]), 0), -t[1], -t[0]))[:2]
        return int(c), int(k)
    flat = np.flatnonzero(unread.ravel())
    return int(flat[0] // ranks), int(flat[0] % ranks)
