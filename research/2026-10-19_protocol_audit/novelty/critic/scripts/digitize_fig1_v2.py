"""v2: fixed geometry. Row gridlines from A panels (log-linear between 0.018 (top) and 0.0004 (bottom line)).
x: mode-5 gridline at axis_x+123, 122.5 px per 5 modes. Marker centre = median y of coloured pixels within +-3 px of the mode's x."""
import sys, json, numpy as np, matplotlib.image as mpimg
img = mpimg.imread(sys.argv[1])[:, :, :3]
sat = img.max(2) - img.min(2); colored = sat > 0.25
axis_x = [230, 770, 1310, 1850]
rows = {0: (19, 320, 0, 352), 1: (399, 707, 360, 732)}   # (y_0.018, y_0.0004, ymin, ymax)
labels = [['A T1', 'B T1', 'C T1', 'D T1'], ['A T2', 'B T2', 'C T2', 'D T2']]
pxm = 122.5 / 5
out = {}
for ri in (0, 1):
    yt, yb, ymin, ymax = rows[ri]
    for ci in range(4):
        x5 = axis_x[ci] + 123
        z = {}
        for m in range(1, 21):
            xc = x5 + (m - 5) * pxm
            x0, x1 = int(round(xc - 3)), int(round(xc + 3))
            sub = colored[ymin:ymax, x0:x1 + 1]
            ys = np.nonzero(sub.any(1))[0]
            if len(ys) < 4: continue
            # error bars are vertical lines through the centre: take the mid of the densest cluster (marker 'x' is widest)
            widths = colored[ymin:ymax, int(xc - 9):int(xc + 10)].sum(1)
            cand = np.nonzero(widths >= max(4, widths.max() * 0.6))[0]
            # restrict to rows within the vertical extent found at the centre
            cand = cand[(cand >= ys.min()) & (cand <= ys.max())]
            if len(cand) == 0: continue
            yc = np.median(cand) + ymin
            lz = np.log10(0.0004) + (yb - yc) / (yb - yt) * (np.log10(0.018) - np.log10(0.0004))
            lo = np.log10(0.0004) + (yb - (ys.max() + ymin)) / (yb - yt) * (np.log10(0.018) - np.log10(0.0004))
            hi = np.log10(0.0004) + (yb - (ys.min() + ymin)) / (yb - yt) * (np.log10(0.018) - np.log10(0.0004))
            z[m] = [round(10 ** lz, 6), round(10 ** lo, 6), round(10 ** hi, 6)]
        out[labels[ri][ci]] = z
json.dump(out, open(sys.argv[2], 'w'), indent=1)
for k, v in out.items():
    print(k, {m: v[m][0] for m in v})
