"""Digitize POMA 58, 035013 Fig. 1 (modal damping factor zeta_n of the eight cello strings) from the raster image
embedded in the PDF. Plan + calibration addendum: README.md. Claude, 2026-10-20.

Run:  uv run --python 3.12 --with pymupdf --with pillow --with numpy python -I digitize.py
Writes damping_digitized.json (numbers only) and local_overlay.png (contains the publisher's figure: keep local,
never commit)."""
import json
from pathlib import Path

import numpy as np
import pymupdf
from PIL import Image, ImageDraw

HERE = Path(__file__).resolve().parent
PDF = Path.home() / "violin-rne/papers/Lampis2025_POMA_58_035013.pdf"
LABELS = [0.018, 0.0095, 0.0051, 0.0027, 0.0014, 0.0008, 0.0004]
AXIS_X = [231, 771, 1311, 1851]                       # left axis columns (found from long dark vertical runs)
ROWS = [(0, 354), (372, 734)]                         # top / bottom panel rows (bottom axis lines at 355, 735)
NAMES = [["A T1", "B T1", "C T1", "D T1"], ["A T2", "B T2", "C T2", "D T2"]]


def load():
    doc = pymupdf.open(PDF)
    page = doc[4]
    xref = max(page.get_images(full=True), key=lambda im: im[2] * im[3])[0]
    pix = pymupdf.Pixmap(doc, xref)
    img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
    assert img.size == (2367, 879), img.size
    return img


def clusters(idx, gap=2):
    out, cur = [], []
    for i in idx:
        if cur and i - cur[-1] > gap:
            out.append(cur); cur = []
        cur.append(i)
    if cur:
        out.append(cur)
    return [float(np.mean(c)) for c in out]


def calibrate_y(grey, x0, y0, y1):
    r = grey[y0:y1, x0 + 5:x0 + 520].sum(1)
    rows = clusters([y0 + i for i in range(len(r)) if r[i] > 80])
    best = None
    for s in range(len(LABELS) - len(rows) + 1):
        lv = np.log10(LABELS[s:s + len(rows)])
        A = np.c_[np.ones(len(rows)), lv]; coef, *_ = np.linalg.lstsq(A, rows, rcond=None)
        res = float(np.sqrt(np.mean((A @ coef - rows) ** 2)))
        if best is None or res < best[0]:
            best = (res, coef, LABELS[s:s + len(rows)])
    res, coef, used = best
    return dict(rows=rows, labels=used, a=float(coef[0]), b=float(coef[1]), rms_px=res)


def calibrate_x(grey, x0, y0, y1):
    c = grey[y0:y1, x0 + 2:x0 + 530].sum(0)
    cols = [x for x in clusters([x0 + 2 + i for i in range(len(c)) if c[i] > 40]) if True]
    # keep the four strongest, evenly spaced lines (modes 5, 10, 15, 20); drop text fragments near mode 5
    cols = sorted(cols)
    cand = [x for x in cols if any(abs((x - x0) - k) < 25 for k in (122, 244, 367, 489))]
    picked = []
    for k in (122, 244, 367, 489):
        near = [x for x in cand if abs((x - x0) - k) < 25]
        picked.append(max(near, key=lambda x: c[int(round(x)) - x0 - 2]))
    modes = np.array([5, 10, 15, 20.0]); A = np.c_[np.ones(4), modes]; coef, *_ = np.linalg.lstsq(A, picked, rcond=None)
    return dict(cols=picked, a=float(coef[0]), b=float(coef[1]), rms_px=float(np.sqrt(np.mean((A @ coef - picked) ** 2))))


def x_template(hw=8.5, hh=9.0, th=1.6, size=14):
    yy, xx = np.mgrid[-size:size + 1, -size:size + 1].astype(float)
    k = hw / hh
    d = np.minimum(np.abs(xx - k * yy), np.abs(xx + k * yy)) / np.sqrt(1 + k * k)
    t = ((d <= th) & (np.abs(yy) <= hh) & (np.abs(xx) >= 2)).astype(float)
    return t - t.mean() * (np.abs(xx) >= 2)


def main():
    img = load(); a = np.asarray(img).astype(int)
    sat = a.max(2) - a.min(2)
    grey = (sat < 20) & (a.sum(2) > 300) & (a.sum(2) < 700)
    tmpl = x_template(); S = tmpl.shape[0] // 2
    out = {"source": "Lampis et al. 2025, POMA 58, 035013, Fig. 1 (raster 2367x879, page 5)", "panels": {}}
    ov = img.copy(); dr = ImageDraw.Draw(ov)
    for ri, (y0, y1) in enumerate(ROWS):
        for ci, x0 in enumerate(AXIS_X):
            name = NAMES[ri][ci]
            cy, cx = calibrate_y(grey, x0, y0, y1), calibrate_x(grey, x0, y0, y1)
            box = a[y0:y1, x0 + 3:x0 + 530]; bs = sat[y0:y1, x0 + 3:x0 + 530]
            col = np.median(box[bs > 80], axis=0)
            dist = np.sqrt(((box - col) ** 2).sum(2))
            mask = (dist < 90) & (bs > 40)
            pts = []
            for n in range(1, 21):
                xn = cx["a"] + cx["b"] * n - (x0 + 3)
                xi = int(round(xn))
                if xi - 13 < 0 or xi + 13 >= mask.shape[1]:
                    continue
                strip = mask[:, xi - 13:xi + 14]
                if strip[:, 9:18].sum() < 12:            # nothing plotted at this mode
                    continue
                pad = np.pad(mask.astype(float), S)
                best = (-1e9, None, None)
                for dx in range(-3, 4):
                    for yy in range(0, mask.shape[0]):
                        win = pad[yy:yy + 2 * S + 1, xi + dx:xi + dx + 2 * S + 1]
                        sc = float((win * tmpl).sum())
                        if sc > best[0]:
                            best = (sc, yy, xi + dx)
                sc, yc, xc = best
                # refine row: centroid of arm pixels (2 <= |dx| <= 9) inside +-10 rows of the template peak,
                # excluding cap rows (pixels at |dx| >= 10.5)
                arm = []
                for yy in range(max(0, yc - 10), min(mask.shape[0], yc + 11)):
                    row = np.nonzero(mask[yy, xc - 13:xc + 14])[0] - 13
                    if len(row) == 0 or np.any(np.abs(row) >= 11):
                        continue
                    arm += [yy] * int(np.sum((np.abs(row) >= 2) & (np.abs(row) <= 9)))
                y_c = float(np.mean(arm)) if len(arm) >= 6 else float(yc)
                # caps: rows with colour at |dx| >= 11 within the column
                capr = [yy for yy in range(mask.shape[0]) if np.sum(np.abs(np.nonzero(mask[yy, xc - 13:xc + 14])[0] - 13) >= 11) >= 2]
                capr = clusters(capr)
                up = [r for r in capr if r < y_c - 1]; dn = [r for r in capr if r > y_c + 1]
                Y = y0 + y_c
                zeta = 10 ** ((Y - cy["a"]) / cy["b"])
                z_hi = 10 ** ((y0 + max(up) - cy["a"]) / cy["b"]) if up else None
                z_lo = 10 ** ((y0 + min(dn) - cy["a"]) / cy["b"]) if dn else None
                pts.append(dict(mode=n, zeta=zeta, zeta_hi=z_hi, zeta_lo=z_lo, row_px=Y, score=sc, n_arm_px=len(arm),
                                at_axis_bottom=bool(y1 - Y < 8)))
                X = x0 + 3 + xc
                dr.ellipse([X - 3, Y - 3, X + 3, Y + 3], outline=(0, 0, 0), width=2)
                dr.text((X + 6, Y - 18), f"{n}", fill=(0, 0, 0))
            for r in cy["rows"]:
                dr.line([x0, r, x0 + 15, r], fill=(255, 0, 255), width=2)
            out["panels"][name] = dict(color_rgb=[float(v) for v in col], y_cal=cy, x_cal=cx, points=pts)
    (HERE / "damping_digitized.json").write_text(json.dumps(out, indent=1))
    ov.save(HERE / "local_overlay.png")
    for name, p in out["panels"].items():
        print(f"{name}: y rms {p['y_cal']['rms_px']:.2f}px ({len(p['y_cal']['rows'])} lines from {p['y_cal']['labels'][0]}), "
              f"{p['y_cal']['b']:.1f} px/dec, x rms {p['x_cal']['rms_px']:.2f}px | " +
              " ".join(f"{q['mode']}:{q['zeta']*1e4:.2f}" for q in p["points"]))


if __name__ == "__main__":
    main()
