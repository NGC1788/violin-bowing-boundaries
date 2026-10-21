"""Exact finite-choice interval-compatibility costs for a shared intercept.

Intervals are CLOSED, including point intervals. Infinite endpoints allowed.
Each column must choose one finite-cost state. No numerical minimizer/grid is
used: all endpoint cells are visited. Complexity O(M log M + M log C), where
M is the total number of states and C the number of columns.
"""
from __future__ import annotations
import heapq
import math


def _validate(columns):
    out=[]
    for col in columns:
        options=[]
        for j,o in enumerate(col):
            lo,hi,c=float(o['lo']),float(o['hi']),float(o['cost'])
            if math.isnan(lo) or math.isnan(hi) or math.isnan(c):
                raise ValueError('NaN is not a valid endpoint or cost')
            if lo>hi or lo==math.inf or hi==-math.inf or c<0:
                raise ValueError('Invalid interval/cost')
            if math.isfinite(c): options.append((lo,hi,c,j))
        out.append(options)
    return out


def _impossible(reason):
    return dict(feasible=False,cost=math.inf,intercept=None,choices=None,reason=reason)


def _inside(lo,hi):
    if lo==-math.inf and hi==math.inf:return 0.
    if lo==-math.inf:return hi-max(1.,abs(hi))
    if hi==math.inf:return lo+max(1.,abs(lo))
    return lo/2+hi/2


def optimize_intercept(options_by_column):
    """Minimize summed costs subject to all selected intervals intersecting."""
    cols=_validate(options_by_column); n=len(cols)
    if not n:return dict(feasible=True,cost=0.,intercept=0.,choices=[],cell={'lo':-math.inf,'hi':math.inf,'kind':'open'})
    if any(not c for c in cols):return _impossible('column_has_no_finite_cost_state')
    events={}; heaps=[[] for _ in cols];active=[set() for _ in cols]
    for i,col in enumerate(cols):
        for lo,hi,c,j in col:
            if lo==-math.inf:
                active[i].add(j);heapq.heappush(heaps[i],(c,j))
            else:events.setdefault(lo,[[],[]])[0].append((i,c,j))
            if hi!=math.inf:events.setdefault(hi,[[],[]])[1].append((i,j))
    current=[math.inf]*n;total=0.;missing=n
    def refresh(i):
        nonlocal total,missing
        old=current[i]
        if math.isfinite(old):total-=old
        else:missing-=1
        while heaps[i] and heaps[i][0][1] not in active[i]:heapq.heappop(heaps[i])
        new=heaps[i][0][0] if heaps[i] else math.inf
        current[i]=new
        if math.isfinite(new):total+=new
        else:missing+=1
    for i in range(n):refresh(i)
    best=math.inf;cell=None
    def consider(lo,hi,kind):
        nonlocal best,cell
        # A closed finite endpoint is at least as good as its adjacent open
        # optimum; prefer it on ties, avoiding overflow in extreme tail cells.
        if missing==0 and (total<best or (total==best and kind=='point' and cell is not None and cell['kind']=='open')):
            best=total;cell={'lo':lo,'hi':hi,'kind':kind}
    endpoints=sorted(events)
    consider(-math.inf,endpoints[0] if endpoints else math.inf,'open')
    for e,t in enumerate(endpoints):
        starts,ends=events[t];affected=set()
        for i,c,j in starts:
            active[i].add(j);heapq.heappush(heaps[i],(c,j));affected.add(i)
        for i in affected:refresh(i)
        consider(t,t,'point') # starts and ends both active at the endpoint
        affected=set()
        for i,j in ends:active[i].remove(j);affected.add(i)
        for i in affected:refresh(i)
        consider(t,endpoints[e+1] if e+1<len(endpoints) else math.inf,'open')
    if cell is None:return _impossible('no_common_intercept_for_any_admissible_states')
    a=cell['lo'] if cell['kind']=='point' else _inside(cell['lo'],cell['hi'])
    choices=[]
    for col in cols:
        covering=[(c,j) for lo,hi,c,j in col if lo<=a<=hi]
        if not covering:raise ArithmeticError('Unrepresentable open cell')
        choices.append(min(covering)[1])
    cost=sum(float(options_by_column[i][j]['cost']) for i,j in enumerate(choices))
    assert abs(cost-best)<1e-9
    return dict(feasible=True,cost=cost,intercept=a,choices=choices,cell=cell)


def optimize_incompatibility(options_by_column):
    """Minimum cost for empty common intersection (strict disjoint witness).

    In one dimension, empty intersection iff lo_i > hi_j for two distinct
    columns. Costs are additive. Subtract each column's cheapest-state cost,
    then sweep sorted lower endpoints while retaining cheaper upper endpoints
    from OTHER columns. This solves both directions of the operational verdict.
    """
    cols=_validate(options_by_column);n=len(cols)
    if n<2:return _impossible('at_least_two_columns_required')
    if any(not c for c in cols):return _impossible('column_has_no_finite_cost_state')
    cheapest=[min((c,j) for lo,hi,c,j in col) for col in cols]
    base=sum(x[0] for x in cheapest)
    options=[(lo,hi,c-cheapest[i][0],i,j) for i,col in enumerate(cols) for lo,hi,c,j in col]
    uppers=sorted(options,key=lambda x:x[1]);lowers=sorted(options,key=lambda x:x[0])
    per_col={};heap=[];cursor=0;best=math.inf;pair=None
    def clean():
        while heap and per_col.get(heap[0][1])!=(heap[0][0],heap[0][2]):heapq.heappop(heap)
    for lo,hi,inc,i,j in lowers:
        while cursor<len(uppers) and uppers[cursor][1]<lo:
            _,_,d,g,s=uppers[cursor];cursor+=1
            if g not in per_col or (d,s)<per_col[g]:
                per_col[g]=(d,s);heapq.heappush(heap,(d,g,s))
        clean();held=None
        if heap and heap[0][1]==i:
            held=heapq.heappop(heap);clean()
        if heap:
            d,g,s=heap[0];value=base+inc+d
            if value<best:best=value;pair=(i,j,g,s)
        if held is not None:heapq.heappush(heap,held)
    if pair is None:return _impossible('all_admissible_assignments_intersect')
    i,j,g,s=pair;choices=[x[1] for x in cheapest];choices[i]=j;choices[g]=s
    chosen=[options_by_column[c][z] for c,z in enumerate(choices)]
    lo=max(float(o['lo']) for o in chosen);hi=min(float(o['hi']) for o in chosen)
    assert lo>hi
    cost=sum(float(o['cost']) for o in chosen)
    assert abs(cost-best)<1e-9
    return dict(feasible=True,cost=cost,intercept=None,choices=choices,
                disjoint_pair=[i,g],intersection={'lo':lo,'hi':hi})
