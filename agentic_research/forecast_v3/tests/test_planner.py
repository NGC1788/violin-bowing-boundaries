"""Request-bound manual resume and backend routing tests; no network calls."""
from pathlib import Path
import json
import sys
from unittest.mock import Mock
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[3]))
from agentic_research.forecast_v3 import planner as P
from agentic_research.forecast_v3 import llm_backend as B


@pytest.fixture
def planning(tmp_path,monkeypatch):
    root=tmp_path/'engine';root.mkdir();(root/'prompts').mkdir()
    for name in ('proposer','reviewer'):(root/'prompts'/f'{name}.md').write_text('Read the evidence; return JSON only.')
    state=root/'state';state.mkdir()
    paths={'root':root,'state':state,'safety_halt':state/'SAFETY_HALT.json'}
    manifest={'manifest_sha':'frozen-manifest','protocol':{'planner':{'max_rounds':18}}}
    status={'stopped':False,'counts':{},'consumed_seconds':0,'budget':{'max_campaign_seconds':50,'max_trials':12}}
    pool={'candidate':{'config':{'id':'candidate','method':'mlp_joint'},'semantic_sha':'candidate-exact-hash'}}
    monkeypatch.setattr(P,'ROOT',root)
    for name,value in [('paths',paths),('integrity',manifest),('status',status),('expired',False),('report',{})]:
        monkeypatch.setattr(P.control,name,Mock(return_value=value))
    monkeypatch.setattr(P,'available',Mock(return_value=pool))
    enqueue=Mock(return_value={'run_id':'candidate-exact-hash'});run=Mock(return_value={'processed':[{}]});stop=Mock()
    monkeypatch.setattr(P.control,'enqueue',enqueue);monkeypatch.setattr(P.control,'run',run);monkeypatch.setattr(P.control,'set_stop',stop)
    monkeypatch.setattr(P,'PROPOSER_BACKEND','codex');monkeypatch.setattr(P,'REVIEWER_BACKEND','claude');monkeypatch.setattr(P,'MANUAL_DIR',None)
    return dict(root=root,state=state,manifest=manifest,status=status,pool=pool,enqueue=enqueue,run=run,stop=stop)


def response(role,schema,verdict='APPROVE'):
    bind=schema['properties']['request_sha']['enum'][0]
    if role=='proposer':
        return dict(candidate_id='candidate',hypothesis='Joint structure may improve future probabilities.',
                    falsifier='No improvement against the fixed strong baseline.',reason='A bounded comparison isolates this structural claim.',request_sha=bind)
    return dict(verdict=verdict,candidate_sha='candidate-exact-hash',reason='Compare the fixed matched-budget models.',
                limitations='Historically exposed development, not physical truth.',request_sha=bind)


def fake_replies(monkeypatch,verdict='APPROVE'):
    fake=Mock(side_effect=lambda prompt,schema,directory,role,timeout:response(role,schema,verdict))
    monkeypatch.setattr(P,'llm',fake);return fake


def test_reject_never_enqueues_or_executes(planning,monkeypatch):
    fake_replies(monkeypatch,'REJECT');result=P.one_round('x',True)
    assert result['status']=='rejected'
    planning['enqueue'].assert_not_called();planning['run'].assert_not_called()


def test_approve_without_execute_only_enqueues(planning,monkeypatch):
    fake_replies(monkeypatch);result=P.one_round('x',False)
    assert result['status']=='submitted'
    planning['enqueue'].assert_called_once();planning['run'].assert_not_called()


def test_wrong_request_hash_cannot_enqueue(planning,monkeypatch):
    monkeypatch.setattr(P,'llm',Mock(return_value={'request_sha':'wrong'}))
    with pytest.raises(P.control.ControlError,match='exact request'):P.one_round('x',True)
    planning['enqueue'].assert_not_called();planning['run'].assert_not_called()


def test_actual_manual_pause_resume_same_round_no_external_cli(planning,monkeypatch,tmp_path):
    monkeypatch.setattr(P,'PROPOSER_BACKEND','manual');monkeypatch.setattr(P,'REVIEWER_BACKEND','manual')
    responses=tmp_path/'responses';responses.mkdir();monkeypatch.setattr(P,'MANUAL_DIR',responses)
    spawn=Mock(side_effect=AssertionError('External CLI forbidden during manual test'));monkeypatch.setattr(B.subprocess,'Popen',spawn)
    first=P.one_round('x');assert first['status']=='awaiting_manual_input';rid=first['round_id']
    directory=planning['state']/'agent_rounds'/rid
    pending=P.one_round('x');assert pending['round_id']==rid
    proposal_schema=json.loads((directory/'proposer_schema.json').read_text())
    (responses/'proposer_response.json').write_text(json.dumps(response('proposer',proposal_schema)))
    second=P.one_round('x',resume_round=rid)
    assert second['status']=='awaiting_manual_input' and second['pending_role']=='reviewer'
    assert second['proposal']['candidate_id']=='candidate'
    # Saved proposal is reused. Removing the external proposer response must be safe.
    (responses/'proposer_response.json').unlink()
    reviewer_schema=json.loads((directory/'reviewer_schema.json').read_text())
    (responses/'reviewer_response.json').write_text(json.dumps(response('reviewer',reviewer_schema)))
    last=P.one_round('x',resume_round=rid)
    assert last['status']=='submitted' and last['round_id']==rid
    assert len(list((planning['state']/'agent_rounds').glob('*/round.json')))==1
    planning['enqueue'].assert_called_once();planning['run'].assert_not_called();spawn.assert_not_called()


def test_manual_tampered_prompt_rejected_before_accepting_response(planning,monkeypatch):
    monkeypatch.setattr(P,'PROPOSER_BACKEND','manual');monkeypatch.setattr(P,'REVIEWER_BACKEND','manual')
    first=P.one_round('x');directory=planning['state']/'agent_rounds'/first['round_id']
    (directory/'proposer_prompt.txt').write_text('tampered manual request')
    with pytest.raises(P.control.ControlError,match='prompt changed'):P.one_round('x',resume_round=first['round_id'])
    planning['enqueue'].assert_not_called()


def test_manual_wrong_candidate_review_hash_never_executes(planning,monkeypatch):
    def reply(prompt,schema,directory,role,timeout):
        answer=response(role,schema)
        if role=='reviewer':answer['candidate_sha']='other-candidate'
        return answer
    monkeypatch.setattr(P,'llm',reply)
    with pytest.raises((ValueError,P.control.ControlError)):P.one_round('x',True)
    planning['enqueue'].assert_not_called();planning['run'].assert_not_called()


def test_mixed_backends_are_routed_independently(planning,monkeypatch):
    calls=[]
    def backend(prompt,schema,directory,role,chosen,timeout,manual_dir):
        calls.append((role,chosen));return response(role,schema)
    monkeypatch.setattr(B,'run_structured',backend)
    assert P.one_round('x')['status']=='submitted'
    assert calls==[('proposer','codex'),('reviewer','claude')]


def test_failed_round_is_charged_and_exhaustion_stops(planning,monkeypatch):
    planning['manifest']['protocol']['planner']['max_rounds']=1
    fake=Mock(side_effect=RuntimeError('backend failure'));monkeypatch.setattr(P,'llm',fake)
    with pytest.raises(RuntimeError):P.one_round('x')
    result=P.one_round('x')
    assert result['status']=='stopped' and '1-round' in result['reason'] and fake.call_count==1
    planning['stop'].assert_called_once()


def test_budget_or_worker_exhaustion_never_invokes_llm(planning,monkeypatch):
    fake=Mock();monkeypatch.setattr(P,'llm',fake)
    planning['status']['counts']={'queued':2}
    planning['run'].return_value={'processed':[],'stopped':True,'reason':'runtime budget exhausted'}
    result=P.one_round('x',True)
    assert result['status']=='stopped' and 'runtime' in result['reason'];fake.assert_not_called()
    planning['status']['exhaustion_reason']='consecutive failure budget exhausted'
    assert P.one_round('x',True)['status']=='stopped';fake.assert_not_called()


def test_prompt_uses_actual_metrics_without_per_stroke_arrays(planning,monkeypatch):
    planning_report={'runs':[{'metrics':{'mean_log_loss':.31,'mean_brier':.12,'development_rows':24000,
        'probabilities':['PRIVATE_STROKE_ARRAY'],'ids':['PRIVATE_ID']}}]}
    P.control.report.return_value=planning_report
    fake=fake_replies(monkeypatch,'REJECT');P.one_round('x')
    prompt=fake.call_args_list[0].args[0]
    assert '"mean_log_loss":0.31' in prompt and '"development_rows":24000' in prompt
    assert 'PRIVATE_STROKE_ARRAY' not in prompt and 'PRIVATE_ID' not in prompt
