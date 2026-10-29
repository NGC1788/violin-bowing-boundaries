"""Safety orchestration tests: unattainable tasks cannot trigger learned search."""
from pathlib import Path
import sys
from unittest.mock import patch
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import stage


def test_impossible_gate_stops_before_any_llm():
    with tempfile.TemporaryDirectory() as t:
        p = {'state': Path(t)}
        m = {'protocol': {}}
        with patch.object(stage.control, 'paths', return_value=p), \
             patch.object(stage.control, 'integrity', return_value=m), \
             patch.object(stage.control, 'status', return_value={'stopped':False}), \
             patch.object(stage.control, 'expired', return_value=False), \
             patch.object(stage.control, 'headroom_gate', return_value={'ready':True,'passed':False}), \
             patch.object(stage.control, 'set_stop') as stop, \
             patch.object(stage.planner, 'one_round') as planner:
            result = stage.advance('test')
            assert result['status'] == 'stopped'
            stop.assert_called_once_with('test', True)
            planner.assert_not_called()


def test_user_stop_does_not_run_preflight_or_llm():
    with patch.object(stage.control, 'paths', return_value={}), \
         patch.object(stage.control, 'integrity', return_value={}), \
         patch.object(stage.control, 'status', return_value={'stopped':True}), \
         patch.object(stage.control, 'headroom_gate') as gate, \
         patch.object(stage.planner, 'one_round') as planner:
        assert stage.advance('test')['status'] == 'stopped'
        gate.assert_not_called()
        planner.assert_not_called()


def test_passing_gate_invokes_exactly_one_reviewed_round():
    with tempfile.TemporaryDirectory() as t:
        p = {'state': Path(t)}
        with patch.object(stage.control, 'paths', return_value=p), \
             patch.object(stage.control, 'integrity', return_value={'protocol':{}}), \
             patch.object(stage.control, 'status', return_value={'stopped':False}), \
             patch.object(stage.control, 'expired', return_value=False), \
             patch.object(stage.control, 'headroom_gate', return_value={'ready':True,'passed':True}), \
             patch.object(stage.control, 'report', return_value={'runs':[]}), \
             patch.object(stage.planner, 'one_round', return_value={'status':'evaluated'}) as planner:
            assert stage.advance('test', llm_timeout=90)['status'] == 'evaluated'
            planner.assert_called_once_with('test', execute=True, llm_timeout=90)
