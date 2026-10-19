"""Tension pairs T1 -> T2 and cross-string levels vs published string properties (post hoc; Claude, 2026-10-18).
Level of each string = offset of its boundaries from the A1+A2 P-free law without Z (t = 0.2 s, K = 2)."""
import numpy as np
import confirm_A_T2 as K

POMA = {  # Lampis et al. POMA 58, 035013 (2025) Table 1: T (N), mu (g/m), Z (kg/s), EI (1e-4 N m^2), d (mm), Zto
    "A1": (145.31, 7.721, 1.0592, 3.032, 0.947, 0.039), "A2": (145.31, 7.721, 1.0592, 3.032, 0.947, 0.039),
    "B1": (150.87, 8.017, 1.0998, 4.397, 0.927, 0.055), "C1": (144.40, 7.671, 1.0525, 3.142, 1.077, 0.039),
    "D1": (142.67, 7.578, 1.0398, 5.246, 1.123, 0.064), "A_T2": (163.03, 8.662, 1.1884, 3.969, 1.001, 0.027),
    "B_T2r1": (163.23, 8.674, 1.1899, 7.295, 0.973, 0.060), "B_T2r2": (163.23, 8.674, 1.1899, 7.295, 0.973, 0.060)}
tr = K.rows_for("A1") + K.rows_for("A2")
noz = K.C.physics_models(tr, {t: 1.0 for t in POMA})
lev = {}
for tag in POMA:
    rows = K.rows_for(tag)
    lev[tag] = tuple(K.offset_fit(rows, lambda r, j=j: noz["P-free"][0](r, tag)[j], w) for j, w in ((0, "lower"), (1, "upper")))
    T, mu, Z, EI, d, Zto = POMA[tag]
    print(f"{tag:7s} 최소 수준 {lev[tag][0]:+.3f}  최대 수준 {lev[tag][1]:+.3f} | Z {Z:.3f} T {T:.0f} EI {EI:.2f} d {d:.3f} Zto {Zto:.3f}")
for t1, t2 in (("A1", "A_T2"), ("A2", "A_T2"), ("B1", "B_T2r1"), ("B1", "B_T2r2")):
    zr = np.log(POMA[t2][2] / POMA[t1][2])
    dl, du = lev[t2][0] - lev[t1][0], lev[t2][1] - lev[t1][1]
    print(f"{t1}->{t2}: Z비 {np.exp(zr):.3f} | 최소 {dl:+.3f} (Z 지수 {dl / zr:+.1f}) | 최대 {du:+.3f} (Z 지수 {du / zr:+.1f})")
tags = ["A1", "A2", "B1", "C1", "D1", "A_T2", "B_T2r1", "B_T2r2"]
L = np.array([lev[t][0] for t in tags]); U = np.array([lev[t][1] for t in tags])
for i, name in enumerate(("T", "mu", "Z", "EI", "d", "Zto")):
    x = np.log([POMA[t][i] for t in tags])
    print(f"  log {name:3s}: 상관 최소 {np.corrcoef(x, L)[0, 1]:+.2f}, 최대 {np.corrcoef(x, U)[0, 1]:+.2f}")
