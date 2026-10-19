"""Synthetic tests for score_highforce.py (no real data). Claude, 2026-10-19."""
import numpy as np
import pytest

import score_highforce as S


def test_intervals_patterns():
    f = np.log(np.linspace(4, 12, 10))
    lo, up, nb = S.intervals(f, np.array([1, 1, 1, 1, 0, 0, 0, 0, 0, 0], bool))
    assert lo == (-np.inf, f[0]) and up == (f[3], f[4]) and nb == 1
    lo, up, nb = S.intervals(f, np.array([0, 0, 1, 1, 1, 1, 1, 1, 1, 1], bool))
    assert lo == (f[1], f[2]) and up == (f[-1], np.inf) and nb == 1
    lo, up, nb = S.intervals(f, np.array([0, 1, 0, 1, 0, 0, 1, 0, 0, 0], bool))      # no K=2 run
    assert lo == (f[-1], np.inf) and np.isnan(up[0]) and nb == 0
    lo, up, nb = S.intervals(f, np.array([1, 1, 0, 0, 1, 1, 1, 0, 0, 0], bool))      # two bands: upper from the last
    assert up == (f[6], f[7]) and nb == 2


def test_noband_left_censor_only_in_S1():
    f = np.log(np.linspace(4, 12, 6))
    cols = [dict(folder="X/v1", column=0, lv=np.log(0.05), lb=np.log(0.05), f=f, h=np.zeros(6, bool))]
    assert np.isnan(S.column_table(cols)[0]["upper"][0])
    assert S.column_table(cols, noband_as_left=True)[0]["upper"] == (-np.inf, f[0])


def test_assign_columns_rejects_wrong_design():
    beta = {("F/v1", i): 0.02 * 3.5 ** ((i // 50) / 29) * (1 + 1e-4 * (i % 50)) for i in range(1500)}
    col = S.assign_columns(beta)
    assert max(col.values()) == 29 and col[("F/v1", 0)] == 0 and col[("F/v1", 1499)] == 29
    with pytest.raises(AssertionError):
        S.assign_columns({k: v for k, v in beta.items() if k[1] < 1499})


def synth(e_v=1.0, e_b=-1.0, c=1.0, noise=0.0, seed=0, tgt_offsets=(0.0,)):
    rng = np.random.default_rng(seed)
    out = []
    for ti, off in enumerate(tgt_offsets):
        for j, v in enumerate((0.05, 0.1, 0.2)):
            for b in 0.02 * 3.5 ** (np.arange(30) / 29):
                lF = c + off + e_v * np.log(v) + e_b * np.log(b) + noise * rng.standard_normal()
                f = np.log(np.exp(np.linspace(np.log(4), np.log(12), 50)))
                h = f < lF
                out.append(dict(folder=f"T{ti}/v{j + 1}", column=0, lv=np.log(v), lb=np.log(b), f=f, h=h, tgt=f"T{ti}", vcat=j))
    return out


def test_p1_recovers_power_law():
    cols = synth(e_v=1.0, e_b=-1.0, c=0.6, noise=0.05)
    rows = S.column_table(cols, noband_as_left=True)    # unbiased when every no-band column lies above F_max
    lv, lb, lz, lo, hi, keep = S.upper_arrays(rows, 0.0)
    r = S.p1(lv, lb, lo, hi, nb=30)
    assert abs(r["e_v"] - 1.0) < 0.1 and abs(r["e_beta"] + 1.0) < 0.1


def test_p3_zero_curvature_for_power_law():
    cols = synth(e_v=0.8, e_b=-1.0, c=0.8, noise=0.05, tgt_offsets=(0.0, 0.1, -0.1, 0.05))
    rows = S.column_table(cols, noband_as_left=True)
    rows = [r for r in rows if np.isfinite(r["upper"][0]) or np.isfinite(r["upper"][1])]
    vcat = np.array([r["vcat"] for r in rows]); lb = np.array([r["lb"] for r in rows])
    lo = np.array([r["upper"][0] for r in rows]); hi = np.array([r["upper"][1] for r in rows])
    tg = np.array([r["tgt"] for r in rows])
    lvcat = np.log([0.05, 0.1, 0.2])
    r = S.p3(vcat, lvcat, lb, lo, hi, tg, nb=10)
    assert abs(r["curvature"]) < 0.15 and abs(r["e12"] - 0.8) < 0.15
