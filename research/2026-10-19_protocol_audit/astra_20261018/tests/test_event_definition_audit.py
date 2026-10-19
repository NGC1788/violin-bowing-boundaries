import itertools
from pathlib import Path
import sys
import numpy as np
import pytest
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import event_definition_audit as event


def test_exhaustive_prefix_stability_of_causal_rule():
    for n in range(2, 9):
        t = np.arange(n)*.05
        for values in itertools.product((False, True), repeat=n):
            for k in (1, 2, 3):
                for prefix in range(1, n):
                    assert not event.compare_prefix(t, values, prefix, k)['changed']


def test_suffix_counterexamples_both_directions():
    # Future non-H revokes a detection previously made at time zero.
    t = np.arange(6)*.05
    out = event.compare_prefix(t, [1, 1, 0, 0, 0, 0], 2)
    assert out['prefix_event_retracted_or_later']
    # Future H retrospectively qualifies an earlier time that had failed.
    out = event.compare_prefix(np.arange(10)*.05, [1, 0, 0, 1, 1, 1, 1, 1, 1, 1], 3)
    assert out['future_qualifies_earlier_event']


def test_suffix_matches_literal_historical_definition():
    for n in range(1, 10):
        for values in itertools.product((False, True), repeat=n):
            expected = next((i for i, h in enumerate(values) if h and np.mean(values[i:]) >= .8), None)
            assert event.suffix_index(values) == expected


def test_window_end_availability_and_censor_endpoint():
    t = np.arange(5)*.05
    index = event.causal_index(t, [1, 1, 1, 0, 1], 3)
    assert index == 2
    assert t[index]+.2 == pytest.approx(.3)
    assert event.causal_index(t, [0, 0, 0, 0, 0], 1) is None
    assert t[-1]+.2 == pytest.approx(.4)


def test_gap_resets_run_and_episode_span():
    t = [0, .05, .20, .25, .30]
    assert event.causal_index(t, [1]*5, 3) == 4
    episodes = event.episodes(t, [1]*5)
    assert [r['decisions'] for r in episodes] == [2, 3]
    assert episodes[0]['right_gap_edge'] and episodes[1]['left_gap_edge']
    assert episodes[0]['decision_span_s'] == pytest.approx(.05)
    assert episodes[1]['decision_span_s'] == pytest.approx(.1)


def test_recurrence_is_not_absorbing_detection():
    t = np.arange(8)*.05
    H = [0, 1, 1, 0, 0, 1, 1, 0]
    assert event.causal_index(t, H, 2) == 2
    assert len(event.episodes(t, H)) == 2
    assert any(not x for x in H[3:])


@pytest.mark.parametrize('t,h', [([], []), ([0, 0], [1, 1]), ([.1, 0], [1, 1]),
    ([0, float('nan')], [1, 1]), ([0, .05], [1, 2]), ([0], [1, 0])])
def test_invalid_sequence_rejected(t, h):
    with pytest.raises(ValueError):
        event.causal_index(t, h, 2)


def test_singleton_h_episode_has_zero_decision_span_not_zero_physical_dwell():
    episode = event.episodes([0, .05, .1], [0, 1, 0])[0]
    assert episode['decisions'] == 1 and episode['decision_span_s'] == 0
    assert not episode['left_observation_edge'] and not episode['right_observation_edge']
