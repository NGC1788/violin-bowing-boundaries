import sys, glob, numpy as np, scipy.io as sio
sys.path.insert(0, '/Users/jeongjihoon/violin-rne/novelty/attack_data/scripts')
import detect as dt
S = 'first_repetitions__2023-10-18'
b = '/Users/jeongjihoon/violin-rne/novelty/attack_data/raw/zenodo/raw_data/first_repetitions/2023-10-18/'
m = sio.loadmat(b + 'matx.mat')['matx']; st = sio.loadmat(b + 'stat.mat')['stat']; tn = sio.loadmat(b + 'tns.mat')['tns']
beta, L = 5.5 / 70, 0.7
rows = []
for p in sorted(glob.glob(f'/Users/jeongjihoon/violin-rne/novelty/attack_data/work/strokes/{S}/*.npz')):
    k = int(p[-8:-4]) - 1
    d = np.load(p); fb, y = d['fb'].astype(float), d['y']
    i_move = int(d['i_move'] - d['seg_start'])
    a, t0, _ = dt.accel_fit(y); i0 = int(round(t0 * dt.FS))
    isl = dt.first_slip(fb, i_move)
    ts = (isl - i0) / dt.FS
    F = m[k, 1]; mus = st[k, 2]; T = tn[k, 2]
    xs = mus * F * beta * (1 - beta) * L / T
    tmod = np.sqrt(2 * xs / a)
    # peak bridge force before slip (quasi-static friction reaction) for reference
    pk = fb[i0:isl].max() if isl > i0 else np.nan
    rows.append((k + 1, a, F, mus, ts, tmod, pk))
    print(f'{k+1:4d} a={a:.2f} F={F:.2f} mu_s={mus:.2f} slip_meas={ts*1000:.0f} ms model={tmod*1000:.0f} ms ratio={ts/tmod:.2f} fb_peak={pk:.2f} N (mu_s*F*(1-beta)={mus*F*(1-beta):.2f})')
r = np.array(rows); ok = np.isfinite(r[:, 4]) & (r[:, 4] > 0)
print('median ratio meas/model', np.median(r[ok, 4] / r[ok, 5]))
