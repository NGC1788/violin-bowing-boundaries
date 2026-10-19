"""Own flyback-counting detector on the streamed 2023-10-18 strokes; compare with the authors' labels."""
import sys, glob, math, numpy as np, scipy.io as sio, json
sys.path.insert(0, '/Users/jeongjihoon/violin-rne/novelty/attack_data/scripts')
import detect as dt
S = 'first_repetitions__2023-10-18'
b = '/Users/jeongjihoon/violin-rne/novelty/attack_data/raw/zenodo/raw_data/first_repetitions/2023-10-18/'
m = sio.loadmat(b + 'matx.mat')['matx']
rows = []
for p in sorted(glob.glob(f'/Users/jeongjihoon/violin-rne/novelty/attack_data/work/strokes/{S}/*.npz')):
    k = int(p[-8:-4]) - 1
    d = np.load(p); fb, y = d['fb'].astype(float), d['y'].astype(float)
    i_move = int(d['i_move'] - d['seg_start'])
    a, t0, res = dt.accel_fit(y); i0 = int(round(t0 * dt.FS))
    i_acc_end = i0 + int(np.sqrt(2 * 0.06 / a) * dt.FS)
    i_stroke_end = int(np.argmax(y))
    isl = dt.first_slip(fb, i_move)
    # own bow force: mean of 1 kHz bow force during acceleration (offset = pre-contact first second)
    bn = d['bn_1k'].astype(float) - float(d['bn_off'])
    g0 = int((d['seg_start'] + i0) / 50); g1 = int((d['seg_start'] + i_acc_end) / 50)
    F_own = float(np.mean(bn[g0:g1])) if g1 > g0 else math.nan
    if not math.isfinite(isl):
        rows.append(dict(k=k + 1, a=a, F=m[k, 1], F_own=F_own, TT_auth=m[k, 2], ok=False)); continue
    labA = dt.window_labels(fb, isl, i_acc_end)
    labS = dt.window_labels(fb, isl, i_stroke_end)
    hmA = [l[-1] for l in labA]; hmS = [l[-1] for l in labS]
    eA, fA = dt.entry_times(hmA); eS, fS = dt.entry_times(hmS)
    rows.append(dict(k=k + 1, a=a, F=m[k, 1], F_own=F_own, TT_auth=m[k, 2], ok=True, t_slip=(isl - i0) / dt.FS,
                     nA=len(hmA), nS=len(hmS), entryA=eA, finalA=fA, entryS=eS, finalS=fS,
                     fracA=float(np.mean(hmA)) if hmA else 0.0, labels=''.join('H' if h else '.' for h in hmS)))
json.dump(rows, open('/Users/jeongjihoon/violin-rne/novelty/attack_data/work/own_labels_20231018.json', 'w'))
ok = [r for r in rows if r['ok']]
print('strokes', len(rows), 'with slip', len(ok))
F_ratio = np.array([r['F_own'] / r['F'] for r in rows if r['F'] > 0.3])
print('own/author bow force ratio (F>0.3): median %.3f IQR %.3f-%.3f' % (np.median(F_ratio), *np.percentile(F_ratio, [25, 75])))
# agreement with authors
def own_success(r, N, kind):
    if kind == 'final':
        return r['finalA'] is not None and r['finalA'] < N
    return r['entryA'] is not None and r['entryA'] < N
for N in (10, 20, 1000):
    for kind in ('final', 'entry'):
        A = np.array([(r['TT_auth'] != 30) if N == 1000 else (r['TT_auth'] < N) for r in ok])
        B = np.array([own_success(r, N, kind) for r in ok])
        po = np.mean(A == B); pe = np.mean(A) * np.mean(B) + (1 - np.mean(A)) * (1 - np.mean(B))
        kap = (po - pe) / (1 - pe) if pe < 1 else math.nan
        print(f'N={"any" if N==1000 else N:>4} own-{kind:5s}: agree {po:.2f} kappa {kap:.2f}  (auth succ {A.sum()}, own succ {B.sum()}, n {len(A)})')
tt = np.array([(r['TT_auth'], r['finalA']) for r in ok if r['TT_auth'] != 30 and r['finalA'] is not None], float)
if len(tt) > 3:
    from scipy.stats import spearmanr
    print('TT authors vs own final (both sustained): n=%d spearman %.2f median diff (own-auth) %.1f periods' % (len(tt), spearmanr(tt[:, 0], tt[:, 1])[0], np.median(tt[:, 1] - tt[:, 0])))
for r in sorted(ok, key=lambda r: (round(r['a'], 1), r['F'])):
    print(f"{r['k']:4d} a={r['a']:.2f} F={r['F']:.2f} (own {r['F_own']:.2f}) slip {r['t_slip']*1000:4.0f} ms  TT_auth={r['TT_auth']:3.0f}  own entry/final(acc)={r['entryA']}/{r['finalA']} nA={r['nA']}  stroke entry/final={r['entryS']}/{r['finalS']} nS={r['nS']} {r['labels'][:100]}")
