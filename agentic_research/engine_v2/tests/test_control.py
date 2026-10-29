"""Control-path tests use a harmless synthetic harness, not the science dataset."""
from pathlib import Path
import json
import shutil
import sqlite3
import sys
import tempfile
import time
import unittest
from unittest.mock import patch
import types

BASE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(BASE))
import control as C

class ControlTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name)/'loop';self.root.mkdir()
        self.locks=Path(self.tmp.name)/'locks'
        for name in ('control.py','cli.py','worker.py'):shutil.copy2(BASE/name,self.root/name)
        (self.root/'harness').mkdir();(self.root/'harness/__init__.py').write_text('')
        (self.root/'harness/experiment.py').write_text('''import time
def run_experiment(config, data_path, output_dir):
    if config['prior']=='logreg': raise RuntimeError('intentional fixture failure')
    if config['prior']=='hgb': time.sleep(5)
    if config['prior']=='mlp_bce': raise AssertionError('synthetic verifier failure')
    if config['prior']=='mlp_boundary': return {'certificate_error_count':1,'all_certified':False}
    return {'synthetic':True,'score':0.5,'seed':config['seed'],'certificate_error_count':0,'all_certified':True}
''')
        (self.root/'stage.py').write_text('# immutable stage\n')
        (self.root/'candidates').mkdir()
        (self.root/'candidates/baseline.json').write_text('{"id":"baseline","prior":"uniform"}')
        self.data=Path(self.tmp.name)/'data.json';self.data.write_text('{}')
        self.proto=Path(self.tmp.name)/'protocol.json'
        self.proto.write_text(json.dumps({'benchmark':{'rule_bank':'nine','max_eval_columns':16,'max_train_columns':None,'eval_selection_seed':20261030},'budget':{'max_trials':4,'max_campaign_seconds':10,'max_trial_seconds':2,'end_date':'2099-01-01'}}))
        self.kw={'root':self.root,'lock_root':self.locks}
        C.init('test',self.data,self.proto,**self.kw);C.freeze('test',**self.kw)
        self.p=C.paths('test',**self.kw)
    def tearDown(self):self.tmp.cleanup()
    def test_identity_excludes_display_id_and_includes_parameters(self):
        a=C.enqueue('test',{'id':'first','seed':0},**self.kw)
        b=C.enqueue('test',{'id':'renamed','seed':0},**self.kw)
        c=C.enqueue('test',{'seed':1},**self.kw)
        self.assertTrue(b['duplicate']);self.assertEqual(a['run_id'],b['run_id']);self.assertNotEqual(a['run_id'],c['run_id'])
    def test_no_relock_and_source_data_protocol_tamper(self):
        with self.assertRaises(C.ControlError):C.freeze('test',**self.kw)
        for target in [self.root/'control.py',self.root/'stage.py',self.root/'candidates/baseline.json',self.data,self.proto]:
            original=target.read_bytes();target.write_bytes(original+b'\n ')
            if target==self.proto:
                target.write_text(json.dumps({'benchmark':{'max_eval_columns':32}}))
            with self.assertRaises(C.ControlError):C.enqueue('test',{},**self.kw)
            target.write_bytes(original)
        self.assertTrue(C.status('test',**self.kw)['integrity_ok'])
    def test_strict_candidate_contract(self):
        bad=[{'command':'anything'},{'data_path':'/tmp/x'},{'seed':True},{'prior':[]},{'learning_rate':float('nan')},
             {'max_eval_columns':32},{'rule_bank':'baseline'},{'id':'../escape'},{'hidden':0}]
        for candidate in bad:
            with self.subTest(candidate=candidate),self.assertRaises(C.ControlError):C.enqueue('test',candidate,**self.kw)
    def test_singleton_worker_lock(self):
        with C.exclusive_worker(self.p):
            with self.assertRaises(C.ControlError):
                with C.exclusive_worker(self.p):pass
    def test_completed_run_is_sealed_and_not_overwritten(self):
        job=C.enqueue('test',{},**self.kw);rid=job['run_id']
        outcome=C.run('test',max_jobs=2,**self.kw)
        self.assertEqual(outcome['processed'][0]['status'],'completed')
        report=C.report('test',**self.kw)
        self.assertEqual(report['runs'][0]['metrics']['score'],0.5)
        self.assertEqual(C.run('test',**self.kw)['processed'],[])
        result=self.p['runs']/rid/'worker_result.json';result.write_text('{"score":99}')
        self.assertEqual(C.report('test',**self.kw)['runs'][0]['status'],'invalid')
    def test_failure_is_preserved_and_charged(self):
        job=C.enqueue('test',{'prior':'logreg'},**self.kw)
        outcome=C.run('test',**self.kw)['processed'][0]
        self.assertEqual(outcome['status'],'failed');self.assertGreater(outcome['elapsed'],0)
        self.assertTrue((self.p['runs']/job['run_id']/'stdout.log').exists())
        self.assertGreater(C.status('test',**self.kw)['consumed_seconds'],0)
    def test_timeout_and_stop(self):
        C.enqueue('test',{'prior':'hgb'},**self.kw)
        t=time.monotonic();out=C.run('test',max_seconds=.25,**self.kw)['processed'][0]
        self.assertEqual(out['status'],'interrupted');self.assertLess(time.monotonic()-t,2)
        C.set_stop('test',True,**self.kw)
        with self.assertRaises(C.ControlError):C.enqueue('test',{'seed':1},**self.kw)
        C.set_stop('test',False,**self.kw)
        C.enqueue('test',{'seed':1},**self.kw)
    def test_stale_job_recovery_after_lock(self):
        rid=C.enqueue('test',{},**self.kw)['run_id']
        with C.database(self.p) as c:
            c.execute("UPDATE jobs SET status='running',started=?,pid=NULL WHERE run_id=?",(time.time()-1,rid))
        C.run('test',**self.kw)
        status=C.status('test',**self.kw)
        self.assertEqual(status['counts'].get('interrupted'),1)
        self.assertGreater(status['consumed_seconds'],0)
    def test_trial_and_runtime_global_budget(self):
        for seed in [0,1,2]:C.enqueue('test',{'seed':seed},**self.kw)
        C.enqueue('test',{'seed':0,'policy':'random'},**self.kw)
        with self.assertRaises(C.ControlError):C.enqueue('test',{'seed':1,'policy':'random'},**self.kw)
        with C.database(self.p) as c:c.execute("UPDATE jobs SET status='failed',elapsed=5")
        self.assertEqual(C.run('test',max_jobs=10,max_seconds=100,**self.kw)['processed'],[])
        with self.assertRaises(C.ControlError):C.enqueue('test',{'seed':1,'policy':'entropy'},**self.kw)
    def test_queued_identity_tamper_never_executes(self):
        rid=C.enqueue('test',{},**self.kw)['run_id']
        with C.database(self.p) as c:
            cfg=json.loads(c.execute('SELECT config_json FROM jobs WHERE run_id=?',(rid,)).fetchone()[0]);cfg['seed']=2
            c.execute('UPDATE jobs SET config_json=? WHERE run_id=?',(json.dumps(cfg),rid))
        C.run('test',**self.kw)
        self.assertEqual(C.status('test',**self.kw)['counts'].get('invalid'),1)
        self.assertFalse((self.p['runs']/rid).exists())

    def gated_campaign(self):
        protocol=json.loads(self.proto.read_text())
        protocol['headroom']={'required':True,'required_baselines':[{'prior':'uniform','policy':'ascending','seed':0}]}
        path=Path(self.tmp.name)/'gated.json';path.write_text(json.dumps(protocol))
        C.init('gated',self.data,path,**self.kw);C.freeze('gated',**self.kw)
        p=C.paths('gated',**self.kw);m=C.integrity(p)
        return p,m

    def fake_report(self,m,passed=True):
        return {'gate_pass':passed,'input_sha256':m['data_sha'],'manifest_sha':m['manifest_sha'],
                'protocol_sha':m['protocol_sha'],'eval_ids':['column-a'],'headroom_fraction':0.2 if passed else 0.05,
                'exact_joint_bounds':[{'id':'column-a','exact_joint_minimum':4}]}

    def test_candidate_directory_rejects_escape_and_symlink(self):
        for raw in ('../other','/tmp/candidates','state','candidates/../state'):
            with self.subTest(raw=raw),self.assertRaises(C.ControlError):
                C.resolve_candidate_dir(self.root,{'planner':{'candidate_dir':raw}})
        (self.root/'candidates/escape').symlink_to(self.tmp.name,target_is_directory=True)
        with self.assertRaises(C.ControlError):
            C.resolve_candidate_dir(self.root,{'planner':{'candidate_dir':'candidates/escape'}})
        self.assertEqual(C.resolve_candidate_dir(self.root,{}),(self.root/'candidates').resolve())

    def test_missing_gate_blocks_learned_enqueue_and_direct_worker_queue(self):
        p,m=self.gated_campaign()
        with self.assertRaisesRegex(C.ControlError,'blocked'):
            C.enqueue('gated',{'prior':'hgb'},**self.kw)
        C.enqueue('gated',{'prior':'uniform'},**self.kw)
        self.assertEqual(C.run('gated',**self.kw)['processed'][0]['status'],'completed')
        # Bypassing enqueue through a corrupted ledger must not bypass the worker gate.
        cfg=C.normalized_candidate({'prior':'hgb'},m['protocol']);rid=C.identity(cfg,m)
        with C.database(p) as db:
            db.execute('INSERT INTO jobs(run_id,config_json,status,submitted) VALUES(?,?,?,?)',
                       (rid,C.canonical(cfg),'queued',time.time()))
        C.run('gated',**self.kw)
        self.assertFalse((p['runs']/rid).exists())
        self.assertTrue(p['safety_halt'].exists())

    def test_headroom_trusted_recompute_and_tamper_stop(self):
        p,m=self.gated_campaign()
        baseline=C.enqueue('gated',{'prior':'uniform'},**self.kw)
        C.run('gated',**self.kw)
        report=self.fake_report(m)
        module=types.ModuleType('harness.headroom');module.make_report=lambda *args:report
        with patch.dict(sys.modules,{'harness.headroom':module}):
            with self.assertRaisesRegex(C.ControlError,'trusted recomputation'):
                C.register_headroom('gated',{'gate_pass':True},**self.kw)
            gate=C.register_headroom('gated',report,**self.kw)
            self.assertTrue(gate['passed']);self.assertNotIn('eval_ids',gate);self.assertNotIn('exact_joint_bounds',gate)
            with self.assertRaisesRegex(C.ControlError,'never overwrite'):
                C.register_headroom('gated',report,**self.kw)
        C.enqueue('gated',{'prior':'hgb'},**self.kw)
        result=p['runs']/baseline['run_id']/'worker_result.json';result.write_text('{}')
        self.assertFalse(C.headroom_gate(p,m)['passed']);self.assertTrue(p['stop'].exists())
        self.assertTrue(p['safety_halt'].exists())
        with self.assertRaisesRegex(C.ControlError,'safety halt'):C.set_stop('gated',False,**self.kw)

    def test_failed_headroom_sets_stop_and_cannot_be_bypassed_by_resume(self):
        p,m=self.gated_campaign()
        C.enqueue('gated',{'prior':'uniform'},**self.kw);C.run('gated',**self.kw)
        report=self.fake_report(m,False)
        module=types.ModuleType('harness.headroom');module.make_report=lambda *args:report
        with patch.dict(sys.modules,{'harness.headroom':module}):
            self.assertFalse(C.register_headroom('gated',report,**self.kw)['passed'])
        self.assertTrue(p['stop'].exists());C.set_stop('gated',False,**self.kw)
        with self.assertRaisesRegex(C.ControlError,'blocked'):C.enqueue('gated',{'prior':'logreg'},**self.kw)
        envelope=json.loads(p['headroom'].read_text());envelope['result']['gate_pass']=True
        p['headroom'].write_text(json.dumps(envelope))
        self.assertFalse(C.headroom_gate(p,m)['passed']);self.assertTrue(p['safety_halt'].exists())

    def test_verifier_exception_stops_remaining_jobs_permanently(self):
        first=C.enqueue('test',{'prior':'mlp_bce'},**self.kw)
        second=C.enqueue('test',{'prior':'uniform'},**self.kw)
        processed=C.run('test',max_jobs=2,**self.kw)['processed']
        self.assertEqual([x['status'] for x in processed],['invalid'])
        self.assertTrue((self.p['runs']/first['run_id']/'worker_failure.json').exists())
        self.assertFalse((self.p['runs']/second['run_id']).exists())
        self.assertTrue(self.p['safety_halt'].exists())
        with self.assertRaisesRegex(C.ControlError,'safety halt'):C.set_stop('test',False,**self.kw)
        with self.assertRaisesRegex(C.ControlError,'safety halt'):C.run('test',**self.kw)

    def test_failed_certificate_metrics_stop_campaign(self):
        C.enqueue('test',{'prior':'mlp_boundary'},**self.kw)
        self.assertEqual(C.run('test',**self.kw)['processed'][0]['status'],'invalid')
        self.assertTrue(self.p['safety_halt'].exists())

    def test_real_headroom_registration_with_synthetic_sealed_baselines(self):
        try:
            import numpy as np
            from harness import headroom
        except ImportError:
            self.skipTest('scientific Python dependencies not installed')
        # Exercise the real science report, controller seals, and worker receipts together.
        (self.root/'harness/experiment.py').write_text("""import hashlib
from pathlib import Path
def run_experiment(config,data_path,output_dir):
    return {'input_sha256':hashlib.sha256(Path(data_path).read_bytes()).hexdigest(),
            'rules':9,'eval_columns':4,'certificate_error_count':0,'all_certified':True,
            'mean_queries':3.,'columns':[{'id':f'dev-{i}','queries':3,'certificate_errors':0} for i in range(4)]}
""")
        protocol=json.loads(self.proto.read_text());protocol['benchmark']['max_eval_columns']=4
        protocol['headroom']={'required':True,'required_baselines':[
            {'prior':'uniform','policy':'ascending','seed':0},{'prior':'uniform','policy':'random','seed':0}]}
        path=Path(self.tmp.name)/'integration.json';path.write_text(json.dumps(protocol))
        C.init('integration',self.data,path,**self.kw);C.freeze('integration',**self.kw)
        p=C.paths('integration',**self.kw);m=C.integrity(p)
        for policy in ('ascending','random'):C.enqueue('integration',{'prior':'uniform','policy':policy},**self.kw)
        self.assertEqual(len(C.run('integration',max_jobs=2,**self.kw)['processed']),2)
        dev=[types.SimpleNamespace(id=f'dev-{i}',labels=np.tile([1,1]+[0]*48,(9,1))) for i in range(4)]
        receipt={'input_sha256':m['data_sha']}
        with patch.object(headroom,'load_dataset',return_value=([],dev,np.zeros((9,2)),receipt)):
            rows=list(reversed(C.report('integration',**self.kw)['runs']))
            report=headroom.make_report(m['data_path'],m['protocol'],rows,m['manifest_sha'])
            self.assertEqual(report,headroom.make_report(m['data_path'],m['protocol'],C._baseline_evidence(p,m),m['manifest_sha']))
            self.assertTrue(C.register_headroom('integration',report,**self.kw)['passed'])
        self.assertTrue(C.status('integration',**self.kw)['headroom']['passed'])

if __name__=='__main__':unittest.main()
