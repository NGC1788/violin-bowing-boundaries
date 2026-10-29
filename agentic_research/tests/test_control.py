"""Control-path tests use a harmless synthetic harness, not the science dataset."""
from pathlib import Path
import json
import shutil
import sqlite3
import sys
import tempfile
import time
import unittest

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
    return {'synthetic':True,'score':0.5,'seed':config['seed']}
''')
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
        for target in [self.root/'control.py',self.data,self.proto]:
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

if __name__=='__main__':unittest.main()
