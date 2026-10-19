import sys, glob, numpy as np, scipy.io as sio
sys.path.insert(0, '/Users/jeongjihoon/violin-rne/novelty/attack_data/scripts')
import detect as dt
S = sys.argv[1] if len(sys.argv) > 1 else 'first_repetitions__2023-10-18'
m = sio.loadmat('/Users/jeongjihoon/violin-rne/raw_placeholder' if False else '/Users/jeongjihoon/violin-rne/novelty/attack_data/raw/zenodo/raw_data/' + S.replace('__', '/') + '/matx.mat')['matx']
for p in sorted(glob.glob(f'/Users/jeongjihoon/violin-rne/novelty/attack_data/work/strokes/{S}/*.npz'))[:int(sys.argv[2]) if len(sys.argv) > 2 else 999]:
    k = int(p[-8:-4]) - 1
    d = np.load(p)
    fb, y = d['fb'], d['y']
    i_move = int(d['i_move'] - d['seg_start'])
    a, t0, res = dt.accel_fit(y)
    i0 = int(round(t0 * dt.FS))
    i_end = i0 + int(np.sqrt(2 * 0.06 / a) * dt.FS)
    isl = dt.first_slip(fb, i_move)
    lab = dt.window_labels(fb, isl, i_end)
    hm = [l[-1] for l in lab]
    fe, fs_ = dt.entry_times(hm)
    s = ''.join('H' if l[-1] else {'multiple_slip': 'm', 'aperiodic': 'x', 'no_oscillation': '_', 'ambiguous': '?', 'subharmonic': 's', 'helmholtz': 'h'}[l[7]] for l in lab)
    print(f'{k+1:4d} a={a:.3f} (auth {m[k,0]:.3f}) F_auth={m[k,1]:.2f} TT_auth={m[k,2]:.0f} slip@{(isl-i0)/dt.FS*1000:.0f}ms entry={fe} final={fs_} n={len(lab)} {s}')
