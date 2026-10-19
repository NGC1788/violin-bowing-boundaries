"""Digitize POMA 58 035013 Fig.1 (damping factor zeta vs mode, 8 strings).
Raster 2367x879. Detect gridlines (dotted gray) per panel, map log-y; detect coloured
marker pixels, cluster by x into modes (x gridlines at modes 5,10,15,20)."""
import sys, numpy as np, matplotlib.image as mpimg, json
img = mpimg.imread(sys.argv[1])[:, :, :3]
H, W, _ = img.shape
r, g, b = img[..., 0], img[..., 1], img[..., 2]
sat = img.max(2) - img.min(2)
colored = sat > 0.25
# axis lines: dark pixels long vertical runs
dark = img.max(2) < 0.25
# find panel left axes: columns with long dark vertical runs
colsum = dark.sum(0)
axes_x = [x for x in range(W) if colsum[x] > 0.55 * H]
# group
def groups(idx):
    out = []
    for i in idx:
        if out and i - out[-1][-1] <= 2: out[-1].append(i)
        else: out.append([i])
    return [int(np.mean(o)) for o in out]
ax_x = groups(axes_x)
rowsum = dark.sum(1)
axes_y = groups([y for y in range(H) if rowsum[y] > 0.35 * W])
print('vertical axes x:', ax_x, ' horizontal axes y:', axes_y, file=sys.stderr)
# light gray dotted gridlines: pixels gray (low sat) value 0.55-0.9
gray = (sat < 0.06) & (img.mean(2) > 0.5) & (img.mean(2) < 0.92)
res = {}
labels = [['A T1', 'B T1', 'C T1', 'D T1'], ['A T2', 'B T2', 'C T2', 'D T2']]
# panel bounds: top row y from 0 to axes_y[0], bottom from axes_y[0] to axes_y[1]
ybounds = [(0, axes_y[0]), (axes_y[0] + 5, axes_y[1])]
xb = ax_x[:4] + [W]
for ri, (y0, y1) in enumerate(ybounds):
    for ci in range(4):
        x0, x1 = xb[ci] + 3, (xb[ci + 1] - 60 if ci < 3 else W - 5)
        sub = gray[y0:y1, x0:x1]
        rs = sub.sum(1)
        gy = groups([y for y in range(sub.shape[0]) if rs[y] > 0.15 * sub.shape[1]])
        cs = gray[y0:y1, x0:x1].sum(0)
        gx = groups([x for x in range(sub.shape[1]) if cs[x] > 0.15 * sub.shape[0]])
        # gridlines y: expect 7 (0.018 ... 0.0004) ; take the lowest as 0.0004 and the top as 0.018
        gy = [y + y0 for y in gy]
        gx = [x + x0 for x in gx]
        csub = colored[y0:y1, x0:x1]
        ys, xs = np.nonzero(csub)
        ys = ys + y0; xs = xs + x0
        res[labels[ri][ci]] = dict(gy=gy, gx=gx, n=len(xs))
        # map x -> mode using gx (modes 5,10,15,20 gridlines)
        if len(gx) >= 2 and len(gy) >= 2:
            px_per_mode = (gx[-1] - gx[0]) / (5 * (len(gx) - 1))
            x_mode5 = gx[0]
            modes = np.round(5 + (xs - x_mode5) / px_per_mode).astype(int)
            ytop, ybot = gy[0], gy[-1]
            ltop, lbot = np.log10(0.018), np.log10(0.0004)
            nlines = len(gy)
            out = {}
            for m in sorted(set(modes)):
                sel = modes == m
                if sel.sum() < 15: continue
                # marker centre: use the 'x' marker -> median of y of pixels in a narrow x band around the mode centre
                xc = x_mode5 + (m - 5) * px_per_mode
                band = sel & (np.abs(xs - xc) < 0.18 * px_per_mode)
                yy = ys[band]
                # x-marker centre: rows where both diagonals present -> use rows with pixels near xc
                core = band & (np.abs(xs - xc) < 3)
                yc = np.median(ys[core]) if core.sum() > 0 else np.median(yy)
                lz = lbot + (ybot - yc) / (ybot - ytop) * (ltop - lbot) if nlines >= 7 else np.nan
                out[int(m)] = round(float(10 ** lz), 6)
            res[labels[ri][ci]]['zeta'] = out
print(json.dumps(res, indent=0))
