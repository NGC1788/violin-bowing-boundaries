"""No network calls: proposal/review cannot bypass deterministic submission gates."""
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import planner


class PlannerTests(unittest.TestCase):
    def setup_context(self,tmp):
        r=Path(tmp);(r/'prompts').mkdir()
        for n in ('proposer','reviewer'):(r/'prompts'/f'{n}.md').write_text('Read only; JSON only.')
        state=r/'state';state.mkdir()
        p={'state':state}
        manifest={'manifest_sha':'manifest','protocol':{}}
        status={'stopped':False,'counts':{},'consumed_seconds':0,
                'budget':{'max_campaign_seconds':50}}
        selected={'config':{'id':'candidate'},'semantic_sha':'exactsha'}
        return r,p,manifest,status,{'candidate':selected}

    def call(self,ctx,replies):
        r,p,m,s,pool=ctx
        from contextlib import ExitStack
        stack=ExitStack()
        for obj,name,value in [(planner,'ROOT',r)]:stack.enter_context(patch.object(obj,name,value))
        for name,value in [('paths',p),('integrity',m),('status',s),('expired',False),('report',{})]:
            stack.enter_context(patch.object(planner.control,name,return_value=value))
        stack.enter_context(patch.object(planner,'available',return_value=pool))
        stack.enter_context(patch.object(planner,'llm',side_effect=replies))
        enqueue=stack.enter_context(patch.object(planner.control,'enqueue',return_value={'run_id':'exactsha'}))
        run=stack.enter_context(patch.object(planner.control,'run',return_value={}))
        return stack,enqueue,run

    def proposal(self):return {'candidate_id':'candidate','hypothesis':'A specific comparison of priors',
                               'falsifier':'No reduction against strong baseline','reason':'Separate structure from learned information'}

    def test_rejection_never_executes(self):
        with tempfile.TemporaryDirectory() as t:
            ctx=self.setup_context(t);stack,enq,run=self.call(ctx,[self.proposal(),{'verdict':'REJECT','candidate_sha':'exactsha'}])
            with stack:
                r=planner.one_round('x',True)
                self.assertEqual(r['status'],'rejected');enq.assert_not_called();run.assert_not_called()

    def test_hash_mismatch_never_executes(self):
        with tempfile.TemporaryDirectory() as t:
            ctx=self.setup_context(t);stack,enq,run=self.call(ctx,[self.proposal(),{'verdict':'APPROVE','candidate_sha':'different'}])
            with stack:
                with self.assertRaises(planner.control.ControlError):planner.one_round('x',True)
                enq.assert_not_called();run.assert_not_called()
            record=json.loads(next((ctx[1]['state']/'agent_rounds').glob('*/round.json')).read_text())
            self.assertEqual(record['status'],'failed')

    def test_approval_without_execute_only_queues(self):
        with tempfile.TemporaryDirectory() as t:
            ctx=self.setup_context(t);stack,enq,run=self.call(ctx,[self.proposal(),{'verdict':'APPROVE','candidate_sha':'exactsha'}])
            with stack:
                r=planner.one_round('x',False)
                self.assertEqual(r['status'],'submitted');enq.assert_called_once();run.assert_not_called()

    def test_round_failure_is_charged(self):
        with tempfile.TemporaryDirectory() as t:
            ctx=self.setup_context(t);stack,enq,run=self.call(ctx,[RuntimeError('backend failed')])
            with stack:
                with self.assertRaises(RuntimeError):planner.one_round('x',True)
                enq.assert_not_called();run.assert_not_called()
            self.assertEqual(len(list((ctx[1]['state']/'agent_rounds').glob('*/round.json'))),1)


if __name__=='__main__':unittest.main()
