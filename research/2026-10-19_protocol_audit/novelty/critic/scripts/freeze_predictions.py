"""Critic proposal (2026-10-19): competing-hypothesis predictions for the UNSCORED tension pairs
C1->C_T2 and D1->D_T2, using only already-scored levels (handoff 10.16-10.19, final/tension_pairs.py statistic:
offset from the A1+A2 P-free law without Z, t = 0.2 s, K = 2) and published properties (POMA Table 1 + Fig. 1
digitized damping). No C_T2 / D_T2 boundary data are read by this script."""
import json, sys, datetime, numpy as np
zeta = json.load(open(sys.argv[1]))
# levels per specimen (lower, upper) as in tension_pairs.py output quoted in the task text
LV = {'A1': (+0.115, +0.002), 'A2': (-0.118, -0.002), 'B1': (-0.357, +0.299), 'C1': (-0.561, -0.004),
      'D1': (-0.248, +0.172), 'A_T2': (-0.539, +0.197), 'B_T2r1': (-0.318, +0.240), 'B_T2r2': (-0.139, +0.249)}
TYPE = {'A1': 'A T1', 'A2': 'A T1', 'B1': 'B T1', 'C1': 'C T1', 'D1': 'D T1', 'A_T2': 'A T2', 'B_T2r1': 'B T2', 'B_T2r2': 'B T2',
        'C_T2': 'C T2', 'D_T2': 'D T2'}
Z = {'A T1': 1.0592, 'B T1': 1.0998, 'C T1': 1.0525, 'D T1': 1.0398, 'A T2': 1.1884, 'B T2': 1.1899, 'C T2': 1.1154, 'D T2': 1.2021}
EI = {'A T1': 3.032, 'B T1': 4.397, 'C T1': 3.142, 'D T1': 5.246, 'A T2': 3.969, 'B T2': 7.295, 'C T2': 3.969, 'D T2': 5.766}
def lz(t, modes):
    return float(np.mean([np.log(zeta[t][str(m)][0]) for m in modes if str(m) in zeta[t]]))
X = {
  'Z (Schelleng slope fixed: upper 1, lower 2)': lambda t: np.log(Z[t]),
  'log EI (fitted slope)': lambda t: np.log(EI[t]),
  'high-mode damping log zeta 7-11 (fitted)': lambda t: lz(t, [7, 8, 9, 10, 11]),
  'low-mode damping log zeta 1-5 (fitted)': lambda t: lz(t, [1, 2, 3, 4, 5]),
}
spec = list(LV)
out = {'created': datetime.datetime.now().astimezone().isoformat(timespec='seconds'), 'statistic':
       'final/tension_pairs.py level (offset from A1+A2 P-free law without Z, t=0.2 s, K=2); prediction = level(X_T2) - level(X1)',
       'predictions': {}}
for j, bname in ((1, 'upper'), (0, 'lower')):
    y = np.array([LV[s][j] for s in spec])
    for name, f in X.items():
        x = np.array([f(TYPE[s]) for s in spec])
        if name.startswith('Z'):
            slope = 1.0 if j == 1 else 2.0
            resid = y - slope * x; icpt = resid.mean(); sd = resid.std(ddof=1)
            loo = sd
        else:
            A = np.vstack([np.ones_like(x), x]).T
            coef, *_ = np.linalg.lstsq(A, y, rcond=None); icpt, slope = coef
            res = y - A @ coef; sd = res.std(ddof=2)
            # leave-one-TYPE-out residual SD
            types = sorted({TYPE[s] for s in spec}); e = []
            for tt in types:
                m = np.array([TYPE[s] != tt for s in spec])
                c, *_ = np.linalg.lstsq(A[m], y[m], rcond=None)
                e += list(y[~m] - A[~m] @ c)
            loo = float(np.sqrt(np.mean(np.square(e))))
        for pair, (t1, t2) in {'C1->C_T2': ('C T1', 'C T2'), 'D1->D_T2': ('D T1', 'D T2')}.items():
            d = slope * (f(t2) - f(t1))
            out['predictions'].setdefault(pair, {}).setdefault(bname, {})[name] = dict(delta=round(float(d), 3), slope=round(float(slope), 3),
                                                                                 loo_rmse=round(float(loo), 3))
            print(f'{pair:9s} {bname:5s} | {name:45s}: delta {d:+.3f} (slope {slope:+.2f}, leave-one-type-out RMSE {loo:.3f})')
out['decision_rule'] = ('Upper boundary (primary): for each pair compute observed delta. A hypothesis is SUPPORTED for a pair if '
  '|obs - pred| <= 0.07 (approx. 2x the largest same-type between-sample/repeat upper difference seen so far, 0.009, plus '
  'cross-session allowance) and REJECTED if |obs - pred| > 0.15. The decisive pair is D: Z and EI predict a RISE, high-mode damping '
  'predicts a FALL. Lower boundary: report only, no decision (same-type repeat yardstick 0.18-0.23 log).')
out['note'] = ('Damping values are digitized from a raster figure (POMA Fig.1, +-3 px ~ +-4 % in zeta); C T2 lacks modes 7,10 and D T2 lacks mode 11 '
  'in the figure, so the 7-11 averages use the available modes. Same-type replicate controls to score with the same statistic once '
  'processed: A3, A4 (expect upper within +-0.03 of 0), D2 (within +-0.03 of D1 +0.172), B_T2 sample2 (within +-0.03 of +0.245).')
json.dump(out, open(sys.argv[2], 'w'), indent=1, ensure_ascii=False)
