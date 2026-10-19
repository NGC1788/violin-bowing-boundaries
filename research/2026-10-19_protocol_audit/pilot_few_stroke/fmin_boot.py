"""Bootstrap intervals and per-string consistency for the censored F_min law (run=2)."""
import warnings
import numpy as np
from fmin_censored import SOURCES, REPO, read, intervals, fit
warnings.filterwarnings("ignore")

data = []
for s, rel in SOURCES.items():
    data += intervals(read(REPO / rel), s, 2)[0]

print("현별 자유 거듭제곱 (검열 반영):")
for s in SOURCES:
    sub = [d for d in data if d[0] == s]
    r = fit(sub)
    print(f"  {s:3} 열 {len(sub):3}  속도 지수 {r['a']:+.2f}   β 지수 {r['p']:+.2f}   σ {r['sigma']:.2f}")

rng = np.random.default_rng(0)
by = {s: [d for d in data if d[0] == s] for s in SOURCES}
draws = []
for _ in range(150):
    sample = []
    for s, rows in by.items():                       # resample columns within each string
        idx = rng.integers(0, len(rows), len(rows))
        sample += [rows[i] for i in idx]
    r = fit(sample)
    draws.append((r["a"], r["p"]))
draws = np.array(draws)
lo, hi = np.percentile(draws, [2.5, 97.5], axis=0)
print(f"\n전체 5표본 부트스트랩 95% 구간 (150회): 속도 지수 [{lo[0]:+.2f}, {hi[0]:+.2f}]   β 지수 [{lo[1]:+.2f}, {hi[1]:+.2f}]")
print(f"속도 지수 > 1 인 비율 {np.mean(draws[:,0] > 1):.2f} | β 지수 < -1.5 인 비율 {np.mean(draws[:,1] < -1.5):.2f}")
