"""Synthetic tests for score_bt2s2 (no real labels)."""
import numpy as np
import pytest

import score_bt2s2 as S


def test_intervals_match_compare_convention():
    f = np.log(np.linspace(0.3, 7.5, 10))
    lo, up = S.intervals(f, [1, 1, 1, 0, 0, 0, 0, 0, 0, 0])
    assert lo == (-np.inf, f[0]) and up == (f[2], f[3])
    lo, up = S.intervals(f, [0, 1, 0, 1, 0, 0, 0, 0, 0, 0])
    assert lo == (f[-1], np.inf) and np.isnan(up[0])


def test_columns_40x40_and_speed_filter():
    rng = np.random.default_rng(0)
    win, beta, speed = {}, {}, {}
    for j, v in enumerate((0.05, 0.10, 0.20, 0.15)):
        fo = f"X/v{j + 1}"; speed[fo] = v
        for i in range(1600):
            b = 0.02 * 10 ** ((i // 40) / 39) * (1 + 1e-4 * (i % 40)); tr = i + 1
            F = 0.3 * 25 ** ((i % 40) / 39)
            Fmax = 2.0 * v / b
            beta[(fo, tr)] = b; win[(fo, tr)] = {0.2: (F, F < Fmax and F > 0.05)}
    cols = S.columns(win, beta, speed, 0.2)
    assert len(cols) == 160
    lv, lb, lo, hi, keep = S.upper_arrays(cols, S.STD_SPEEDS)
    assert set(np.round(np.exp(lv), 2)) == {0.05, 0.10, 0.20}
    p = S.free_fit(lv, lb, lo, hi)
    assert abs(p[1] - 1) < 0.15 and abs(p[2] + 1) < 0.15
    with pytest.raises(AssertionError):
        S.assign_columns({k: v for k, v in beta.items() if k[1] < 1600})
