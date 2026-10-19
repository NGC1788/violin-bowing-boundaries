"""Own-detector lower limits per acceleration column on the streamed subset, vs authors' labels on the SAME strokes."""
import sys, json, numpy as np
from collections import defaultdict
sys.path.insert(0, '/Users/jeongjihoon/violin-rne/novelty/attack_data/scripts')
import colbound as cb
rows = json.load(open('/Users/jeongjihoon/violin-rne/novelty/attack_data/work/own_labels_20231018.json'))
T0 = 1 / 97.86
cols = defaultdict(list)
for r in rows:
    if r['ok']:
        cols[(r['k'] - 1) % 30].append(r)
def succ_own(r, crit):
    if crit == 'any':
        return r['finalA'] is not None
    if crit == 'vstar':
        return r['finalA'] is not None and r['a'] * (r['t_slip'] + r['finalA'] * T0) <= 0.10
    return r['finalA'] is not None and r['finalA'] < crit
def succ_auth(r, crit, ts):
    if crit == 'any':
        return r['TT_auth'] != 30
    if crit == 'vstar':
        return r['TT_auth'] != 30 and r['a'] * (ts + r['TT_auth'] * T0) <= 0.10
    return r['TT_auth'] < crit
for crit in (10, 20, 'vstar', 'any'):
    out = []
    for c in sorted(cols):
        rs = sorted(cols[c], key=lambda r: r['F'])
        if len(rs) < 6: continue
        F = np.array([r['F'] for r in rs])
        so = np.array([succ_own(r, crit) for r in rs]); sa = np.array([succ_auth(r, crit, r['t_slip']) for r in rs])
        fo, st_o = cb.column_lower(F, so); fa, st_a = cb.column_lower(F, sa)
        out.append((np.median([r['a'] for r in rs]), len(rs), F.min(), F.max(), fo, st_o, fa, st_a))
    print(f'=== criterion {crit} (own final-entry vs authors TT, same strokes)')
    for o in out:
        print(f'  a={o[0]:.2f} n={o[1]:2d} F {o[2]:.2f}-{o[3]:.2f}: own F_low={o[4]:.2f}({o[5]})  authors F_low={o[6]:.2f}({o[7]})')
    for name, i in (('own', 4), ('authors', 6)):
        pts = [(np.log(o[0]), np.log(o[i])) for o in out if o[i + 1] == 'ok']
        if len(pts) >= 3:
            p = np.array(pts); print(f'  {name}: log-log slope over {len(p)} columns = {np.polyfit(p[:,0], p[:,1], 1)[0]:+.2f}')
