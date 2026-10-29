import json
from pathlib import Path
import sys
from unittest.mock import patch
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import llm_backend as backend

SCHEMA={'type':'object','properties':{'decision':{'type':'string','enum':['APPROVE','REJECT']}},
        'required':['decision'],'additionalProperties':False}


@pytest.mark.parametrize('bad',[{}, {'decision':'YES'}, {'decision':True},
                                {'decision':'APPROVE','code':'shell command'}, []])
def test_strict_response_rejection(bad):
    with pytest.raises(ValueError):
        backend.validate(bad,SCHEMA)


def test_manual_round_trip_and_missing_input(tmp_path):
    out=tmp_path/'round';responses=tmp_path/'responses';responses.mkdir()
    with pytest.raises(backend.ManualInputRequired):
        backend.run_structured('Review exact candidate abc',SCHEMA,out,'reviewer','manual',manual_dir=responses)
    (responses/'reviewer_response.json').write_text('{"decision":"APPROVE"}')
    assert backend.run_structured('Review exact candidate abc',SCHEMA,out,'reviewer','manual',manual_dir=responses)=={'decision':'APPROVE'}
    assert json.loads((out/'reviewer_call.json').read_text())['backend']=='manual'


def test_claude_structured_response_and_errors():
    assert backend.parse_claude({'subtype':'success','structured_output':{'decision':'APPROVE'}})=={'decision':'APPROVE'}
    with pytest.raises(RuntimeError):
        backend.parse_claude({'is_error':True,'result':'APPROVE'})
    with pytest.raises(ValueError):
        backend.parse_claude({'result':'```json\n{}\n```'})


def test_claude_no_tools_or_permission_bypass(tmp_path):
    with patch.object(backend.shutil,'which',return_value='/bin/claude'):
        cmd=backend.command_for('claude',SCHEMA,tmp_path/'out.json',tmp_path/'schema.json')
    assert cmd[cmd.index('--tools')+1]==''
    assert '--dangerously-skip-permissions' not in cmd
    assert '--strict-mcp-config' in cmd
    assert '--no-session-persistence' in cmd


def test_codex_read_only_and_ephemeral(tmp_path):
    with patch.object(backend.shutil,'which',return_value='/bin/codex'):
        cmd=backend.command_for('codex',SCHEMA,tmp_path/'out.json',tmp_path/'schema.json')
    assert cmd[cmd.index('--sandbox')+1]=='read-only'
    assert '--ephemeral' in cmd
    assert '--output-schema' in cmd
