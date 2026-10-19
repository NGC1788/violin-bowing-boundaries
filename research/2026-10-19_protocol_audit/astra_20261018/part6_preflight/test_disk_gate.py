import importlib.util
from pathlib import Path
import signal
from types import SimpleNamespace
import pytest

spec=importlib.util.spec_from_file_location('disk_gate',Path(__file__).with_name('disk_gate.py'))
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)


def pair(state='S'):
    p={'pid':41262,'ppid':1,'state':state,'started':'Sun Oct 18 12:17:34 2026','command':str(m.PYTHON)+' replicate.py'}
    c={'pid':41266,'ppid':41262,'state':'S','started':'Sun Oct 18 12:17:35 2026','command':f'{m.PYTHON} {m.REPO}/scripts/zenodo_catalog.py download --record 17795326 --file {m.NAME} --data-dir {m.ROOT}/data --connections 16'}
    return p,c


def test_ps_parse_and_identity():
    p=m.parse_process('41262 1 T Sun Oct 18 12:17:34 2026 /x/python replicate.py')
    assert p['state']=='T' and p['command']=='/x/python replicate.py'
    assert m.same_identity(p,dict(p,state='S'))
    assert not m.same_identity(p,dict(p,started='different'))


def test_parent_child_validation():
    p,c=pair();m.validate_pair(p,c)
    for wrong in [dict(c,ppid=3),dict(c,command=c['command'].replace('17795326','17782542')),dict(c,state='Z')]:
        with pytest.raises(ValueError):m.validate_pair(p,wrong)


def test_signal_rechecks_identity(monkeypatch):
    p,c=pair();calls=[]
    monkeypatch.setattr(m,'process',lambda pid:dict(p,started='changed'))
    monkeypatch.setattr(m.os,'kill',lambda *a:calls.append(a))
    with pytest.raises(RuntimeError):m.checked_signal(p,signal.SIGSTOP)
    assert not calls


def test_inventory_budget_and_collision():
    text='Path = rep1/v1/whole_1.csv\nSize = 38354000000\nPath = rep1/v1/beta_1.csv\nSize = 20\n'
    inv=m.inventory(text)
    assert inv['trials']==1
    assert not m.budget(95_470000000,inv)['passes']
    assert m.budget(100_354000020,inv)['passes']
    with pytest.raises(ValueError,match='collision'):m.inventory(text+text.replace('rep1','rep2'))
    with pytest.raises(ValueError,match='unsupported'):m.inventory('Path = 1/whole.csv\nSize = 10\n')


@pytest.mark.parametrize('initially_stopped,budget_passes,expected_signals,status',[
    (False,True,[signal.SIGSTOP,signal.SIGCONT],'passed_parent_resumed'),
    (True,True,[],'passed_parent_previously_stopped_no_resume'),
    (False,False,[signal.SIGSTOP],'held_attention_required'),
])
def test_gate_resume_ownership(tmp_path,monkeypatch,initially_stopped,budget_passes,expected_signals,status):
    p,c=pair('T' if initially_stopped else 'S');signals=[];child_reads=0
    def proc(pid):
        nonlocal child_reads
        if pid==p['pid']:return dict(p)
        child_reads+=1
        return dict(c,state='S' if child_reads==1 else 'Z')
    def kill(pid,sig):
        signals.append(sig);p['state']='T' if sig==signal.SIGSTOP else 'S'
    monkeypatch.setattr(m,'process',proc);monkeypatch.setattr(m.os,'kill',kill)
    monkeypatch.setattr(m,'safe_children',lambda *a:None)
    archive=tmp_path/'a.7z';archive.write_bytes(b'data');monkeypatch.setattr(m,'ARCHIVE',archive)
    inv={'max_folder_bytes':38_354000000}
    monkeypatch.setattr(m,'validate_archive',lambda p:(inv,{'md5':'verified'}))
    monkeypatch.setattr(m.shutil,'disk_usage',lambda p:SimpleNamespace(free=101_000000000 if budget_passes else 95_000000000))
    if budget_passes:result=m.run(p['pid'],c['pid'],tmp_path)
    else:
        with pytest.raises(RuntimeError,match='budget fails'):m.run(p['pid'],c['pid'],tmp_path)
        import json
        result=json.loads((tmp_path/'gate_receipt.json').read_text())
    assert signals==expected_signals and result['status']==status
    if not budget_passes:assert p['state']=='T'


def test_cancel_never_resumes(tmp_path,monkeypatch):
    p,c=pair();calls=[]
    monkeypatch.setattr(m,'process',lambda pid:dict(p) if pid==p['pid'] else dict(c))
    def kill(pid,sig):calls.append(sig);p['state']='T'
    monkeypatch.setattr(m.os,'kill',kill);monkeypatch.setattr(m,'safe_children',lambda *a:None)
    monkeypatch.setattr(m,'_CANCEL',signal.SIGTERM)
    with pytest.raises(RuntimeError,match='cancelled'):m.run(p['pid'],c['pid'],tmp_path)
    assert calls==[signal.SIGSTOP] and p['state']=='T'


def test_singleton(tmp_path):
    with m.singleton(tmp_path/'lock'):
        with pytest.raises(RuntimeError):
            with m.singleton(tmp_path/'lock'):pass


def test_zombie_command_can_change_but_birth_time_cannot():
    p,c=pair()
    zombie=dict(c,state='Z',command='(python)')
    assert m.same_downloader(c,zombie)
    assert not m.same_downloader(c,dict(zombie,started='different'))
    assert not m.same_downloader(c,dict(zombie,state='S'))
    assert m.same_downloader(c,None)
