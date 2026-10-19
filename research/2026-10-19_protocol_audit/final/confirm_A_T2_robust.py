"""Post hoc robustness of the A_T2 lower-boundary shift: drop columns whose boundary interval lies below 0.2 N
(the force range with tens-of-mN sensor-offset uncertainty) and refit the offset; also per speed. (Claude, 2026-10-18)"""
import numpy as np
import confirm_A_T2 as K

tr = K.rows_for("A1") + K.rows_for("A2"); te = K.rows_for("A_T2")
noz = K.C.physics_models(tr, K.ONE)
base = lambda r: noz["P-free"][0](r, "A_T2")[0]
lg = np.log(0.2)
keep = [r for r in te if np.isfinite(r["lower"][0]) and r["lower"][0] >= lg]
print(f"0.2 N 이상 열 {len(keep)}개 (전체 {len(te)})")
for name, rows in (("전체", te), ("0.2 N 이상", keep)):
    d = K.offset_fit(rows, base, "lower")
    print(f"  {name}: 최소 활 힘 이동 {d:+.3f} log → Z 지수 {d / 0.1151:+.2f}")
for v in sorted({round(np.exp(r['lv']), 3) for r in te}):
    rows = [r for r in te if abs(np.exp(r["lv"]) - v) < 1e-3]
    d = K.offset_fit(rows, base, "lower"); dA1 = K.offset_fit([r for r in tr if r["tag"] == "A1" and abs(np.exp(r["lv"]) - v) < 1e-3], base, "lower")
    dA2 = K.offset_fit([r for r in tr if r["tag"] == "A2" and abs(np.exp(r["lv"]) - v) < 1e-3], base, "lower")
    print(f"  v={v}: A_T2 {d:+.3f}, A1 {dA1:+.3f}, A2 {dA2:+.3f}")
