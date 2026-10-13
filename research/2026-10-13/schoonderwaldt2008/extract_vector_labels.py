#!/usr/bin/env python3
"""Recover categorical markers from specified vector figures in one primary PDF.

No OCR, raster classification, waveform reconstruction, or missing-cell imputation.
The source document is not redistributed. Requires pdfplumber and numpy.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import pdfplumber

PDF_SHA256 = "9e7f018421ae5512a7fb3c4a54e2c53c9146393167721a3fe7c169eb6358387a"
SOURCE_URL = "https://www.knutsacoustics.com/files/schoonderwaldt-et-al.-empirical-schelleng.pdf"
DOI = "10.3813/AAA.918070"
DISTANCES_MM = [11, 13, 15, 18, 21, 24, 29, 34, 39, 46, 55]
FORCES_MN = [49, 58, 70, 84, 100, 120, 143, 171, 205, 245, 293, 350,
             419, 501, 599, 716, 857, 1025, 1226, 1466, 1753, 2097, 2508, 3000]
# Bounds are verified plot axes in PDF points, not a screenshot estimate.
# (figure, panel, PDF page, journal page, string, speed, left, top, right, bottom)
PANELS = [
    (4, "a", 7, 610, "D4", .05, 95.00649, 89.95403, 226.85649, 199.21403),
    (4, "b", 7, 610, "D4", .10, 241.37650, 89.95403, 373.22650, 199.21403),
    (4, "c", 7, 610, "D4", .15, 95.00649, 216.40404, 226.85649, 325.60399),
    (4, "d", 7, 610, "D4", .20, 241.37650, 216.40404, 373.22650, 325.60399),
    (7, "a", 10, 613, "E5", .10, 351.00650, 82.45404, 522.39650, 210.95403),
    (7, "b", 10, 613, "E5", .20, 351.22650, 231.20404, 522.57650, 359.73401),
]


def unique(values, tolerance=.02):
    result = []
    for x in sorted(values):
        if not result or abs(x-result[-1]) > tolerance:
            result.append(x)
    return np.array(result)


def axes(page, bounds):
    left, top, right, bottom = bounds
    xticks = unique([r["x0"] for r in page.lines if abs(r["bottom"]-bottom) < .03
                    and r["width"] < .01 and 1 < r["height"] < 2 and left < r["x0"] < right])
    yticks = unique([r["top"] for r in page.lines if abs(r["x0"]-left) < .03
                    and r["height"] < .01 and 1 < r["width"] < 2 and top < r["top"] < bottom])
    if len(xticks) != 6 or len(yticks) != 6:
        raise ValueError(f"axis tick count differs from six: {xticks}, {yticks}")
    # Six x ticks label columns 0,2,4,6,8,10. Six y ticks label force
    # ranks 23,19,15,11,7,3. Row-to-force mapping uses the printed force list.
    xfit = np.polyfit(np.arange(0, 11, 2), xticks, 1)
    yfit = np.polyfit(np.arange(23, 2, -4), yticks, 1)
    x = np.polyval(xfit, np.arange(11))
    y = np.polyval(yfit, np.arange(24))
    residual = max(abs(np.polyval(xfit, np.arange(0, 11, 2))-xticks).max(),
                   abs(np.polyval(yfit, np.arange(23, 2, -4))-yticks).max())
    if residual > .1:
        raise ValueError("tick positions are not consistent with the displayed regular log grid")
    return x, y, float(residual)


def stroke_gray(obj):
    value = obj.get("stroking_color", 0)
    return float(np.mean(value)) if isinstance(value, (tuple, list)) else float(value or 0.)


def primitive(obj, kind, figure):
    w, h = obj["width"], obj["height"]
    size = 2.55 if figure == 4 else 3.19
    if kind == "rect" and obj["stroke"] and not obj["fill"] and abs(w-size)<.15 and abs(h-size)<.15:
        return "gray_square" if stroke_gray(obj) > .25 else "square"
    if kind == "curve":
        if obj["fill"] and not obj["stroke"] and 1.2 < w < 1.5 and 1.2 < h < 1.5:
            return "dot"
        if obj["stroke"] and not obj["fill"] and 3.85 < w < 4.2 and 3.85 < h < 4.2:
            return "circle"
    if kind == "line" and obj["stroke"]:
        diameter = 4.04 if figure == 4 else 5.03
        if w < .02 and abs(h-diameter)<.2:
            return "vertical"
        if h < .02 and abs(w-diameter)<.2:
            return "horizontal"
        diagonal = diameter/np.sqrt(2)
        if abs(w-diagonal)<.2 and abs(h-diagonal)<.2:
            dy = obj["pts"][-1][1]-obj["pts"][0][1]
            dx = obj["pts"][-1][0]-obj["pts"][0][0]
            return "diagonal_positive" if dx*dy > 0 else "diagonal_negative"
    return None


def category(parts):
    p = Counter(parts)
    keys = set(p)
    allowed = {frozenset(["square"]): "helmholtz",
               frozenset(["gray_square"]): "helmholtz_not_systematically_analyzed",
               frozenset(["dot"]): "constant_slipping",
               frozenset(["circle"]): "anomalous_low_frequency",
               frozenset(["vertical", "horizontal"]): "multiple_slipping",
               frozenset(["diagonal_positive", "diagonal_negative"]): "raucous",
               frozenset(["vertical", "horizontal", "diagonal_positive", "diagonal_negative"]): "s_motion"}
    if not parts:
        return "unplotted_unknown"
    # Unexpected duplicate marker components could encode repeated or mixed
    # classes: fail closed rather than silently choosing a class.
    if any(v != 1 for v in p.values()):
        return "ambiguous_vector_components"
    return allowed.get(frozenset(keys), "ambiguous_vector_components")


def extract(pdf_path: Path, out: Path):
    digest = hashlib.sha256(pdf_path.read_bytes()).hexdigest()
    if digest != PDF_SHA256:
        raise ValueError("PDF hash mismatch: inspect the version and coordinates before reuse")
    out.mkdir(parents=True, exist_ok=True)
    pdf = pdfplumber.open(pdf_path)
    all_rows, audit = [], []
    for fig, panel, page_number, journal_page, string, speed, *bounds in PANELS:
        page = pdf.pages[page_number-1]
        xgrid, ygrid, ticks_residual = axes(page, bounds)
        left, top, right, bottom = bounds
        components = defaultdict(list)
        rejected = []
        for kind, objects in [("rect", page.rects), ("curve", page.curves), ("line", page.lines)]:
            for object_index, obj in enumerate(objects):
                center_x, center_y = (obj["x0"]+obj["x1"])/2, (obj["top"]+obj["bottom"])/2
                if not (left < center_x < right and top < center_y < bottom):
                    continue
                part = primitive(obj, kind, fig)
                if part is None:
                    continue
                col, row = int(np.argmin(abs(xgrid-center_x))), int(np.argmin(abs(ygrid-center_y)))
                residual = max(abs(xgrid[col]-center_x), abs(ygrid[row]-center_y))
                evidence = {"part": part, "pdf_object": f"{kind}:{object_index}",
                            "center_x": center_x, "center_y": center_y, "grid_residual_pt": float(residual)}
                if residual > .18:
                    rejected.append(evidence)
                else:
                    components[row, col].append(evidence)
        rows = []
        for row, force in enumerate(FORCES_MN):
            for col, distance in enumerate(DISTANCES_MM):
                evidence = components[row, col]
                label = category([p["part"] for p in evidence])
                known = label not in ("unplotted_unknown", "ambiguous_vector_components")
                strict = known and label != "helmholtz_not_systematically_analyzed"
                rows.append({"source_doi": DOI, "source_pdf_sha256": digest,
                    "figure": fig, "panel": panel, "pdf_page_1based": page_number,
                    "journal_page": journal_page, "instrument": "rigid_monochord",
                    "string": string, "bow_speed_nominal_m_s": speed,
                    "beta_column_0based": col, "force_rank_0based": row,
                    "bow_bridge_distance_reported_mm": distance, "string_length_reported_mm": 325,
                    "beta_from_reported_rounded_distance": distance/325,
                    "bow_force_nominal_N": force/1000,
                    "published_category": label,
                    "binary_helmholtz": (int(label.startswith("helmholtz")) if known else ""),
                    "strict_systematically_analyzed": strict,
                    "marker_parts": ";".join(p["part"] for p in evidence),
                    "pdf_object_indices": ";".join(p["pdf_object"] for p in evidence),
                    "grid_x_pdf_pt": round(float(xgrid[col]),6), "grid_y_pdf_pt": round(float(ygrid[row]),6),
                    "max_marker_grid_residual_pt": max((p["grid_residual_pt"] for p in evidence), default="")})
        all_rows.extend(rows)
        audit.append({"figure": fig, "panel": panel, "grid_cells": len(rows),
                      "axis_tick_max_fit_residual_pt": ticks_residual,
                      "category_counts": dict(Counter(r["published_category"] for r in rows)),
                      "off_grid_marker_primitives": rejected,
                      "maximum_marker_residual_pt": max([p["grid_residual_pt"] for a in components.values() for p in a],default=None)})
    with (out/"published_cells.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(all_rows[0]))
        writer.writeheader()
        writer.writerows(all_rows)
    manifest = {"source_doi": DOI, "source_url": SOURCE_URL, "source_pdf_sha256": digest,
                "source_pdf_bytes": pdf_path.stat().st_size,
                "extractor_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "pdfplumber_version": pdfplumber.__version__, "axes_units": "PDF points, top-down y",
                "label_scope": "published author classifications, not raw waveforms or independent mechanical ground truth",
                "missing_policy": "absence of a marker is unknown, never negative",
                "gray_square_policy": "separate non-systematically-analyzed category, excluded by strict flag",
                "beta_policy": "ordered reported rounded distances divided by reported 325 mm length; not exact measured beta",
                "position_accuracy_reported_mm": .5,
                "repetitions": "paper reports three repeated strokes; figure exposes only final author-selected category, not individual replicate labels",
                "panels": audit}
    (out/"extraction_audit.json").write_text(json.dumps(manifest, indent=2, allow_nan=False)+"\n")
    print(json.dumps(audit, indent=2))
    return all_rows, manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pdf", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path(__file__).parent)
    args = parser.parse_args()
    extract(args.pdf, args.output)
