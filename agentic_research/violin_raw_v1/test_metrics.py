import numpy as np
import pytest
from agentic_research.violin_raw_v1.metrics import row_scores, participant_means, summarize, coherence_rate, select_baseline, paired_bootstrap
from agentic_research.violin_raw_v1.models import patterns


def test_macro_weighting_is_not_recording_weighting():
    people, scores = participant_means([0, 0, 0, 2], ['a', 'a', 'a', 'b'])
    assert scores.mean() == 1 and scores.mean() != .5
    assert people.tolist() == ['a', 'b']
    p, s = participant_means([0, 0, 0, 2], ['a', 'a', 'a', 'b'], np.array([True, True, True, False]))
    assert p.tolist() == ['a'] and s.tolist() == [0]


def test_scores_clipping_guard_and_coherence():
    y = patterns()[[0, 2, 4, 8]]
    result = summarize(y, y, ['a', 'a', 'b', 'b'], np.array([True, False, True, False]))
    assert result['log_loss'] < 2e-6 and result['brier'] == 0
    assert result['coherence_rate'] == 1 and result['participants'] == 2
    assert result['invalid']['rows'] == 2
    mixture = np.arange(1, 11) / 55
    assert coherence_rate(np.tile(mixture @ patterns(), (3, 1))) == 1
    impossible = np.array([[1, 0, 0, 0, 0, 0, 0, 0, 0]])
    assert coherence_rate(impossible) == 0
    with pytest.raises(ValueError): row_scores(y, np.full_like(y, np.nan))


def test_fixed_baseline_tie_break_and_paired_uncertainty():
    assert select_baseline({'hgb': 1., 'frequency': 1., 'markov': 1.}) == 'frequency'
    assert select_baseline({'hgb': .9, 'frequency': 1., 'markov': 1.}) == 'hgb'
    result = paired_bootstrap([.8, .9, 1.], [1., 1.1, 1.2])
    assert abs(result['mean_difference'] + .2) < 1e-12
    np.testing.assert_allclose(result['percentile_95_interval'], [-.2, -.2])
    assert result['participants'] == 3
    with pytest.raises(ValueError): paired_bootstrap([1], [1])
