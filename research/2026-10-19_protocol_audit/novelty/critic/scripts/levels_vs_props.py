"""Critic sanity check: do published string properties (POMA Table 1 + digitized Fig.1 damping)
order the boundary LEVELS (given in the task; log offsets vs type-A law, t=0.2 s)?
Only already-scored specimens are used. C_T2 / D_T2 boundary levels are NOT read."""
import json, sys, numpy as np
zeta = json.load(open(sys.argv[1]))
lv = {  # lower, upper (natural log offsets) -- from handoff 10.16-10.19 / task text
 'A T1': (0.0, 0.0),            # A1 +0.115/+0.002, A2 -0.118/-0.002 -> mean ~0
 'B T1': (-0.357, 0.299), 'C T1': (-0.561, -0.004), 'D T1': (-0.248, 0.172),
 'A T2': (-0.539, 0.197), 'B T2': (-0.2285, 0.2445)}  # B_T2 mean of two reps
Z = {'A T1':1.0592,'B T1':1.0998,'C T1':1.0525,'D T1':1.0398,'A T2':1.1884,'B T2':1.1899,'C T2':1.1154,'D T2':1.2021}
EI = {'A T1':3.03,'B T1':4.40,'C T1':3.14,'D T1':5.25,'A T2':3.97,'B T2':7.30,'C T2':3.97,'D T2':5.77}
Zto = {'A T1':.039,'B T1':.055,'C T1':.039,'D T1':.064,'A T2':.027,'B T2':.060,'C T2':.040,'D T2':.039}
def lz(k, modes):
    v = [np.log(zeta[k][str(m)][0]) for m in modes if str(m) in zeta[k]]
    return np.mean(v), len(v)
keys = list(lv)
feat = {}
for k in Z:
    feat[k] = dict(logZ=np.log(Z[k]), logEI=np.log(EI[k]), logZto=np.log(Zto[k]),
                   lz1=lz(k,[1])[0], lz1_5=lz(k,[1,2,3,4,5])[0], lz7_11=lz(k,[7,8,9,10,11])[0])
print('string  lower  upper | ' + ' '.join(f'{n:>8s}' for n in feat['A T1']))
for k in Z:
    l = lv.get(k, (np.nan, np.nan))
    print(f'{k:6s} {l[0]:+.3f} {l[1]:+.3f} | ' + ' '.join(f'{feat[k][n]:+8.3f}' for n in feat[k]))
print('\nPearson r across the 6 scored strings (n=6; no multiplicity correction):')
for n in feat['A T1']:
    x = np.array([feat[k][n] for k in keys])
    for j, nm in ((0, 'lower'), (1, 'upper')):
        y = np.array([lv[k][j] for k in keys])
        print(f'  {nm:5s} vs {n:7s}: r = {np.corrcoef(x, y)[0,1]:+.2f}')
# Schelleng-type loss prediction for within-type tension pairs: dlogFmin = dlogZ + k*dlog zeta1 (R ~ Z/zeta)
print('\nWithin-type T1->T2 predictions from losses (R = 2 Z Q1/pi, Q1 = 1/(2 zeta1)); F_min ~ Z^2/R^k')
for t in 'ABCD':
    a, b = f'{t} T1', f'{t} T2'
    dZ = np.log(Z[b]/Z[a]); dzl = feat[b]['lz1'] - feat[a]['lz1']; dzm = feat[b]['lz1_5'] - feat[a]['lz1_5']
    for k in (1.0, 2.3):
        p1 = 2*dZ - k*(dZ - dzl); pm = 2*dZ - k*(dZ - dzm)
        obs = (lv[b][0]-lv[a][0]) if (a in lv and b in lv) else float('nan')
        print(f'  {t}: k={k}: pred dlogFmin (mode1) {p1:+.2f}, (modes1-5) {pm:+.2f} | observed {obs:+.2f}' )
