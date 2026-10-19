"""Regression tests for support censoring, failures, and source-only calibration."""
import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest

spec = importlib.util.spec_from_file_location("compare_fixed", Path(__file__).with_name("compare_fixed.py"))
C = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = C
spec.loader.exec_module(C)


def row(**kw):
    out = dict(fmin=-2., fmax=2., lv=-2., lb=-2., lower=(-1., -.8), upper=(.8, 1.),
               idx=[0, 1, 2], tag="A1", col=0)
    out.update(kw)
    return out


def fun(rule):
    return lambda lf, lv, lb, t: rule(lf)


def test_inversion_never_calls_outside_measured_support():
    calls = []
    def pf(f, *args):
        calls.append(f)
        assert np.min(f) >= -.3 and np.max(f) <= .7
        return ((f >= 0) & (f <= .4)).astype(float)
    b = C.band_from_prob(pf, row(fmin=-.3, fmax=.7))
    assert b.status == "band" and 0 <= b.lower < .003 and .397 < b.upper <= .4
    assert len(calls) == 1


def test_no_band_is_empty_not_nan_hit():
    b = C.band_from_prob(fun(lambda f: np.zeros_like(f)), row())
    assert b.status == "empty"
    s = C.score_boundaries([row(lower=(1., np.inf), upper=(np.nan, np.nan))], [b])
    assert s["lower"]["counts"]["point_hits"] == 0
    assert s["lower"]["counts"]["prediction_failures_available"] == 1
    assert s["upper"]["counts"]["observation_unavailable"] == 1
    assert s["lower"]["median_abs_log_error_conditional"] is None


@pytest.mark.parametrize("rule,lkind,ukind", [
    (lambda f: (f < 1).astype(float), "left_censored", "point"),
    (lambda f: (f > -1).astype(float), "point", "right_censored"),
    (lambda f: np.ones_like(f), "left_censored", "right_censored"),
])
def test_edge_contact_is_support_censored(rule, lkind, ukind):
    b = C.band_from_prob(fun(rule), row())
    assert b.endpoint("lower")[0] == lkind and b.endpoint("upper")[0] == ukind
    if lkind == "left_censored":
        assert b.endpoint("lower")[1] == (-np.inf, -2.)
    if ukind == "right_censored":
        assert b.endpoint("upper")[1] == (2., np.inf)


def test_disconnected_probability_components_counted_as_hull():
    b = C.band_from_prob(fun(lambda f: ((abs(f) > .8) & (abs(f) < 1.5)).astype(float)), row())
    assert b.components == 2 and b.lower < -.8 and b.upper > .8


@pytest.mark.parametrize("rule", [lambda f: np.full_like(f, np.nan), lambda f: np.full_like(f, 1.1),
                                  lambda f: np.ones(len(f)-1)])
def test_invalid_probabilities_are_not_clipped_into_success(rule):
    assert C.band_from_prob(fun(rule), row()).status == "invalid"


@pytest.mark.parametrize("a,b", [(1., -1.), (np.inf, np.nan), (np.nan, 2.), (np.inf, 3.), (1., -np.inf)])
def test_bad_endpoints_marked_invalid(a, b):
    assert C.checked_band(a, b).status == "invalid"


def test_finite_and_one_sided_probability_are_mathematically_correct():
    f = np.array([-2., 0., 2.])
    np.testing.assert_allclose(C.band_prob(f, C.checked_band(-1., 1.), 2), C.expit(2*np.array([-1., 1., -1.])))
    np.testing.assert_allclose(C.band_prob(f, C.checked_band(-np.inf, 1.), 2), C.expit(2*(1-f)))
    np.testing.assert_allclose(C.band_prob(f, C.checked_band(-1., np.inf), 2), C.expit(2*(f+1)))
    np.testing.assert_allclose(C.band_prob(f, C.checked_band(-np.inf, np.inf), 2), 1.)
    np.testing.assert_allclose(C.band_prob(f, C.Band(np.nan, np.nan, "empty"), 2), 0.)
    assert np.all(np.isnan(C.band_prob(f, C.checked_band(np.nan, 1.), 2)))


def test_point_hit_and_censor_compatibility_not_conflated():
    r = row(lower=(-np.inf, -1.))
    p = C.checked_band(-np.inf, 1., support_min=-2., support_max=2.)
    s = C.score_boundaries([r], [p])["lower"]
    assert s["counts"]["point_hits"] == 0
    assert s["counts"]["censor_interval_overlaps"] == 1
    assert s["counts"]["censor_interval_contained"] == 1
    assert s["n_finite_observation_and_point_prediction"] == 0
    assert s["point_hit_rate_all_available"] == 0


def test_point_errors_have_real_denominator_and_no_arbitrary_three():
    rs = [row(), row(), row()]
    ps = [C.checked_band(-.85, .95), C.Band(np.nan, np.nan, "empty"), C.checked_band(np.nan, .9)]
    s = C.score_boundaries(rs, ps)["lower"]
    assert s["n_finite_observation_and_point_prediction"] == 1
    assert np.isclose(s["median_abs_log_error_conditional"], .05)
    assert s["counts"]["prediction_failures_available"] == 2
    assert s["point_hit_rate_all_available"] == 1/3
    assert s["point_hit_rate_point_predictions"] == 1


def source_fixture():
    data = {tg:{"force":np.exp([-2.,0.,2.]),"H":np.array([False,True,False])} for tg in C.TRAIN}
    rows = {tg:[row(tag=tg)] for tg in C.TRAIN}
    return data, rows


def test_slope_includes_no_band_constant_loss_but_not_nan():
    data, rows = source_fixture()
    pf = lambda r,t: C.Band(np.nan,np.nan,"empty") if t=="A1" else C.checked_band(-1.,1.)
    s = C.fit_slope(data, rows, pf)
    assert s["identified"] and np.all(np.isfinite(s["losses"]))
    assert s["k"] >= 13.75  # EPS clipping makes the high-k optimum a plateau
    assert s["counts"]["used_windows"] == 6 and s["counts"]["empty_columns"] == 1


def test_slope_invalid_excluded_and_allconstant_not_identified():
    data, rows = source_fixture()
    s = C.fit_slope(data,rows,lambda r,t:C.checked_band(np.nan,1.) if t=="A1" else C.Band(np.nan,np.nan,"empty"))
    assert s["k"] is None and not s["identified"]
    assert s["counts"]["excluded_windows"] == 3
    assert s["counts"]["used_windows"] == 3
    assert np.all(np.isfinite(s["losses"]))


def test_calibration_refuses_target_and_cannot_read_target_rows():
    data, rows = source_fixture()
    data["B1"] = data["A1"]
    with pytest.raises(ValueError,match="targets are forbidden"):
        C.fit_slope(data, rows, lambda r,t:C.checked_band(-1.,1.))
    data.pop("B1")
    class Forbidden:
        def __iter__(self): raise AssertionError("Target rows read")
    rows["B1"] = Forbidden()
    seen = []
    def pf(r,t):
        seen.append(t)
        return C.checked_band(-1.,1.)
    a = C.fit_slope(data,rows,pf)
    assert seen == ["A1","A2"] and a["source_tags"] == ["A1","A2"]


def test_classification_logloss_casting_and_failed_predictions():
    y = np.array([False,True,True])
    s = C.classification_score(y,[.1,.8,np.nan])
    assert np.isclose(s["logloss_conditional"], -np.mean(np.log([.9,.8])))
    assert s["n_valid"] == 2 and s["n_prediction_failures"] == 1 and s["iou_conditional"] == 1.


def test_triv_lookup_source_only_and_unsupported_speed_failure():
    rs = [row(tag=tg,lb=float(j),col=j,lower=(j,j+.2),upper=(j+2,j+2.2))
          for tg in C.TRAIN for j in range(6)]
    lookup = C.triv5_predictor(rs)
    b = lookup(row(lb=0.),"B1")
    assert np.isclose(b.lower,2.1) and np.isclose(b.upper,4.1)
    assert lookup(row(lv=99.),"B1").status == "invalid"
    with pytest.raises(ValueError,match="source rows only"):
        C.triv5_predictor(rs+[row(tag="B1")])


def test_safe_json_records_infinity_explicitly_and_no_nan_token():
    import json
    raw = C.safe_json({"a":np.inf,"b":-np.inf,"c":np.nan,"x":np.array([1.,np.nan])})
    encoded = json.dumps(raw,allow_nan=False)
    assert raw["a"] == "+inf" and raw["b"] == "-inf" and raw["c"] is None
    assert "NaN" not in encoded
