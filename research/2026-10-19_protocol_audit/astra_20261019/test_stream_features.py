"""Synthetic solid archive parity/identity tests; no research outcomes or network."""
import io,sys,subprocess,tempfile,unittest,gzip,csv,json,os
from pathlib import Path
from unittest.mock import patch
import numpy as np
import stream_features as s


def ls(records):
    return '----------\n'+'\n\n'.join('Path = '+p+'\nSize = '+str(n) for p,n in records)+'\n'


class StreamTests(unittest.TestCase):
    def test_full_identity_and_order(self):
        a=s.inventory(ls([('a/v1/beta_1.csv',3),('b/v1/beta_1.csv',3),('a/v1/whole_1.csv',10),('b/v1/whole_1.csv',11)]))
        self.assertEqual([x[2] for x in a[-2:]],['a/v1','b/v1'])
        # 2026-10-19: whole immediately followed by its beta is accepted (bounded hold of MAX_HELD = 1)
        self.assertEqual(len(s.inventory(ls([('a/whole_1.csv',10),('a/beta_1.csv',3),('a/whole_2.csv',10),('a/beta_2.csv',3)]))),4)
        with self.assertRaises(ValueError):s.inventory(ls([('a/whole_1.csv',10),('a/whole_2.csv',10),('a/beta_1.csv',3),('a/beta_2.csv',3)]))
        with self.assertRaises(ValueError):s.inventory(ls([('a/beta_1.csv',3),('a/whole_1.csv',s.MAX_MEMBER+1)]))

    def test_short_reads_and_truncation(self):
        class Short(io.BytesIO):
            def read(self,n=-1):return super().read(min(n,2))
        self.assertEqual(s.read_exact(Short(b'abcdefgh'),8),b'abcdefgh')
        with self.assertRaises(EOFError):s.read_exact(io.BytesIO(b'ab'),3)

    def test_duplicate_or_nan_metadata(self):
        r={'v_max':.1,'onset_s':0,'plateau_on_s':.1,'plateau_off_s':.4,'total_s':.5,'ramp_s':.1,'accel':1,'force_at_onset':1}
        a=np.full((1,10,160),np.nan,dtype='f');a[0,:,:10]=0
        self.assertEqual(s.validate_metadata([('a',1)],{('a',1):.1},[r],a,{('a',1)})['trials'],1)
        a[0,0,0]=np.nan
        with self.assertRaises(ValueError):s.validate_metadata([('a',1)],{('a',1):.1},[r],a,{('a',1)})
        with self.assertRaises(ValueError):s.validate_metadata([('a',1),('a',1)],{('a',1):.1},[r,r],np.zeros((2,10,160)),{('a',1)})

    def test_pin_release_requires_validated_same_inode_replacement(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);archive=root/'example.7z';archive.write_bytes(b'original')
            base=root/'astra_20261018/archive_companion';(base/'pins').mkdir(parents=True);(base/'receipts').mkdir()
            pin=base/'pins'/archive.name;os.link(archive,pin)
            dest=root/'astra_20261019/highforce_stream'/archive.stem;dest.mkdir(parents=True)
            (dest/'features.csv.gz').write_bytes(b'verified');receipt={'status':'validated','archive_integrity':{'md5':'checked'},'outputs':{'features.csv.gz':s.digest(dest/'features.csv.gz')}}
            old=base/'receipts'/(archive.name+'.json')
            with patch.object(s,'ROOT',root):
                with self.assertRaises(ValueError):s.release_redundant_pin(archive,receipt)
                old.write_text(json.dumps({'status':'running'}))
                with self.assertRaises(ValueError):s.release_redundant_pin(archive,receipt)
                old.write_text(json.dumps({'status':'failed','error':'ambiguous folder basename'}))
                with self.assertRaises(ValueError):s.release_redundant_pin(archive,{**receipt,'archive_integrity':None})
                pin.unlink();pin.write_bytes(b'different')
                with self.assertRaises(ValueError):s.release_redundant_pin(archive,receipt)
                pin.unlink();os.link(archive,pin)
                self.assertTrue(s.release_redundant_pin(archive,receipt));self.assertFalse(pin.exists());self.assertEqual(archive.read_bytes(),b'original')

    def test_solid_stream_order_and_existing_worker_parity(self):
        sys.path.insert(0,str(s.ROOT/'timeres'));import replicate
        sys.path.insert(0,str(s.ROOT/'claude_extra'));import ramp_meta,stroke_blocks_full
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);t=np.arange(30000)/50000
            speed=.1*np.minimum(np.minimum(t/.05,(.6-t)/.05),1)
            sig=np.column_stack([1-.1*t,speed,.2*np.sin(2*np.pi*196*t),.1*np.sin(2*np.pi*196*t+.3)])
            expected={}
            for folder in ('a/v1','b/v1'):
                p=root/folder;p.mkdir(parents=True)
                np.savetxt(p/'whole_1.csv',sig,delimiter=',',fmt='%.8g');(p/'beta_1.csv').write_text('0.1')
                expected[folder]=[1]
            archive=root/'test.7z'
            subprocess.run([str(s.SEVEN),'a',str(archive),'a','b','-ms=on','-bso0','-bsp0'],cwd=root,check=True)
            text=subprocess.check_output([str(s.SEVEN),'l','-slt',str(archive)],text=True)
            items=s.inventory(text)
            proc=subprocess.Popen([str(s.SEVEN),'x','-so',str(archive),'-ir!*/whole_*.csv','-ir!*/beta_*.csv'],stdout=subprocess.PIPE,stderr=subprocess.PIPE)
            for path,size,*_ in items:self.assertEqual(s.read_exact(proc.stdout,size),(root/path).read_bytes())
            self.assertEqual(proc.stdout.read(),b'');proc.stdout.close();self.assertEqual(proc.wait(),0);proc.stderr.close()
            task=(str(root/'a/v1/whole_1.csv'),str(root/'a/v1/beta_1.csv'),'a/v1',1)
            rows,ramp,blocks=s.worker(task);legacy=replicate.windows_of(task)
            self.assertEqual(len(rows),len(legacy))
            for a,b in zip(rows,legacy):
                for key in a:
                    if isinstance(a[key],float):np.testing.assert_equal(a[key],b[key])
                    else:self.assertEqual(a[key],b[key])
            raw=(root/'a/v1/whole_1.csv').read_bytes();self.assertEqual(ramp,ramp_meta.stroke(raw))
            parsed=np.fromstring(raw.replace(b'\n',b','),sep=',').reshape(-1,4)
            np.testing.assert_equal(blocks,stroke_blocks_full.blocks(parsed))
            # Exercise actual concurrent staging/publication then idempotent checkpoint reuse.
            out=root/'result.csv.gz'
            with patch.object(s,'ROOT',root),patch.object(replicate,'wait_for_budget',lambda *_:None):
                receipt=s.run(archive,out,expected,log=lambda *_:None)
                self.assertEqual(receipt['validation']['trials'],2)
                with gzip.open(out,'rt') as f:actual=list(csv.DictReader(f))
                self.assertEqual({(r['folder'],int(r['trial'])) for r in actual},{('a/v1',1),('b/v1',1)})
                self.assertEqual(len(actual),len(legacy)*2)
                self.assertEqual(s.run(archive,out,expected,log=lambda *_:None),receipt)

    def test_whole_before_beta_bounded_hold_end_to_end(self):
        """2026-10-19 22h (Claude): archives like D2 store whole_n then beta_n; run() must hold one whole only."""
        sys.path.insert(0,str(s.ROOT/'timeres'));import replicate
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);t=np.arange(30000)/50000
            speed=.1*np.minimum(np.minimum(t/.05,(.6-t)/.05),1)
            sig=np.column_stack([1-.1*t,speed,.2*np.sin(2*np.pi*196*t),.1*np.sin(2*np.pi*196*t+.3)])
            p=root/'d/v1';p.mkdir(parents=True)
            for n in (1,2):
                np.savetxt(p/f'whole_{n}.csv',sig*(1+.01*n),delimiter=',',fmt='%.8g');(p/f'beta_{n}.csv').write_text(str(.05*n))
            archive=root/'pair.7z'
            for f in ('d/v1/whole_1.csv','d/v1/beta_1.csv','d/v1/whole_2.csv','d/v1/beta_2.csv'):
                subprocess.run([str(s.SEVEN),'a',str(archive),f,'-ms=on','-bso0','-bsp0'],cwd=root,check=True)
            text=subprocess.check_output([str(s.SEVEN),'l','-slt',str(archive)],text=True)
            self.assertEqual([x[3] for x in s.inventory(text)],['whole','beta','whole','beta'])
            out=root/'pair.csv.gz'
            with patch.object(s,'ROOT',root),patch.object(replicate,'wait_for_budget',lambda *_:None):
                receipt=s.run(archive,out,{'d/v1':[1,2]},log=lambda *_:None)
            self.assertEqual(receipt['validation']['trials'],2)
            with gzip.open(out,'rt') as f:actual=list(csv.DictReader(f))
            for n in (1,2):
                legacy=replicate.windows_of((str(p/f'whole_{n}.csv'),str(p/f'beta_{n}.csv'),'d/v1',n))
                mine=[r for r in actual if int(r['trial'])==n]
                self.assertEqual(len(mine),len(legacy))
                self.assertAlmostEqual(float(mine[0]['beta']),.05*n)
                for a,b in zip(mine,legacy):self.assertAlmostEqual(float(a['force']),float(b['force']),places=6)

if __name__=='__main__':unittest.main()
