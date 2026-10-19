"""Robustness of the observed statistic: K, K_low, quantile, window length, floor tolerance, population, specimen;
plus a 'force-fall-only' prediction (every entered stroke assumed H to the end of the plateau)."""
import numpy as np
from common import load, column_ratios, entry_low


def med(out, f):
    r = out.get(f, [])
    return (len(r), np.median([x[1] for x in r]) if r else np.nan, np.mean([x[1] < 1 for x in r]) if r else np.nan,
            np.median([x[2] for x in r]) if r else np.nan)


def fmt(t):
    return f"cols {t[0]:2d} ratio {t[1]:.2f} (<1 {t[2]:.0%}) per-stroke low/entry {t[3]:.2f}"


cache = {}
for tag in ("A2", "A1"):
    for win in (0.2, 0.1, 0.4):
        cache[(tag, win)] = load(tag, win)

print("== baseline and K / quantile variants (0.2 s windows, floor strokes) ==")
for tag in ("A2", "A1"):
    S, speed = cache[(tag, 0.2)]
    fl = [s for s in S if s.floor]
    for f in sorted(speed):
        print(f"{tag} v={speed[f]:.2f}")
        for K in (2, 3, 4, 6):
            for q in (0.05, 0.1, 0.2, 0.5):
                print(f"   K={K} q={q:.2f}: " + fmt(med(column_ratios(fl, K=K, q=q), f)))
        for KL in (4, 8, 12):
            print(f"   K=2 K_low={KL} (low must sit at the end of a >= {KL}-window H run): " + fmt(med(column_ratios(fl, K=2, K_low=KL), f)))
        print(f"   K=2 lows only after the entry run: " + fmt(med(column_ratios(fl, K=2, skip_entry_run=True), f)))
        for mn in (10, 15):
            print(f"   K=2 min entered per column {mn}: " + fmt(med(column_ratios(fl, K=2, min_n=mn), f)))

print("\n== window length (K scaled to keep the run span ~0.25 s and also K=2) ==")
for tag in ("A2", "A1"):
    for win, Ks in ((0.1, (2, 4)), (0.2, (2,)), (0.4, (2, 4))):
        S, speed = cache[(tag, win)]
        fl = [s for s in S if s.floor]
        for K in Ks:
            o = column_ratios(fl, K=K)
            print(f"{tag} win={win} K={K}: " + " | ".join(f"v={speed[f]:.2f} " + fmt(med(o, f)) for f in sorted(speed)))

print("\n== population: floor tolerance and all strokes ==")
for tag in ("A2", "A1"):
    for tol in (1.05, 1.15, 1.5, 3.0, None):
        S, speed = load(tag, 0.2, floor_tol=tol if tol else 1e9)
        o = column_ratios(S if tol is None else [s for s in S if s.floor])
        print(f"{tag} floor_tol={tol if tol else 'all strokes'}: " + " | ".join(f"v={speed[f]:.2f} " + fmt(med(o, f)) for f in sorted(speed)))

print("\n== force-fall-only prediction: low replaced by the stroke's force at the end of the plateau (H assumed kept) ==")
for tag in ("A2", "A1"):
    S, speed = cache[(tag, 0.2)]
    fl = [s for s in S if s.floor]
    for f in sorted(speed):
        obs = {c: r for c, r, *_ in column_ratios(fl)[f]}
        from collections import defaultdict
        cells = defaultdict(list)
        for s in fl:
            if s.folder != f:
                continue
            el = entry_low(s.F, s.H)
            if el is None:
                continue
            cells[s.col].append((el[0], min(s.F[el[2]:].min(), el[0])))
        pred = {c: np.quantile([x[1] for x in g], 0.1) / np.quantile([x[0] for x in g], 0.1) for c, g in cells.items() if len(g) >= 5}
        cs = sorted(set(obs) & set(pred))
        o = np.array([obs[c] for c in cs]); p = np.array([pred[c] for c in cs])
        print(f"{tag} v={speed[f]:.2f}: observed median {np.median(o):.2f}, force-fall-only median {np.median(p):.2f}, "
              f"corr across columns {np.corrcoef(o, p)[0, 1]:.2f}, median obs/pred {np.median(o / p):.2f}")
