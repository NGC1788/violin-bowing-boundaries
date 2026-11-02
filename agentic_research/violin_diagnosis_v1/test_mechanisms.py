import numpy as np
from agentic_research.violin_diagnosis_v1.mechanisms import gain_admissible, gain_shift, rank_correlation
from agentic_research.violin_raw_v1.features import extract_example


def test_binary_gain_is_observer_preserving_when_admitted():
    sr=16000;t=np.arange(sr*2)/sr;x=.1*np.sin(2*np.pi*440*t)
    a=extract_example(x,sr,'DOUBLE')
    for gain in [.5,2.]:
        assert gain_admissible(x,sr,gain)
        b=extract_example(x*gain,sr,'DOUBLE')
        np.testing.assert_allclose(gain_shift(a['X'][None],gain)[0],b['X'],atol=1e-10)
        np.testing.assert_allclose(a['targets'],b['targets'],atol=1e-10)


def test_admission_rejects_clipping_silence_and_voicing_threshold_crossing():
    sr=16000;t=np.arange(sr*2)/sr
    assert not gain_admissible(.8*np.sin(2*np.pi*440*t),sr,2.)
    assert not gain_admissible(np.zeros(sr*2),sr,.5)
    assert not gain_admissible(.0002*np.sin(2*np.pi*440*t),sr,.5)
    assert rank_correlation([1,2,3],[3,2,1])==-1
