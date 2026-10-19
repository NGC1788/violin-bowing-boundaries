import csv
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

SPEC = importlib.util.spec_from_file_location('archive_companion', Path(__file__).parents[1] / 'archive_companion.py')
m = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(m)


def fixture_outputs(tmp_path, keys=(('archive/v1', 1), ('archive/v2', 1))):
    ramp = tmp_path / 'ramp.csv'
    blocks = tmp_path / 'blocks.npz'
    fields = ['folder', 'trial', 'beta', 'v_max', 'onset_s', 'plateau_on_s', 'plateau_off_s', 'total_s', 'ramp_s', 'accel', 'force_at_onset', 'error']
    rows = [dict(zip(fields, [f, t, .1, .1, .1, .2, .25, .3, .1, 1, .5, ''])) for f, t in keys]
    with ramp.open('w', newline='') as f:
        w = csv.DictWriter(f, fields); w.writeheader(); w.writerows(rows)
    data = np.full((len(keys), 10, 160), np.nan, dtype=np.float32)
    data[:, :9, :6] = .1
    np.savez_compressed(blocks, data=data, fields=np.array(m.BLOCK_FIELDS), folder=np.array([x[0].split('/')[-1] for x in keys]), trial=np.array([x[1] for x in keys]), beta=np.full(len(keys), .1), block_s=.05)
    return ramp, blocks, rows


def rewrite_blocks(path, **changes):
    with np.load(path, allow_pickle=False) as z:
        arrays = {k: z[k] for k in z.files}
    arrays.update(changes)
    np.savez_compressed(path, **arrays)


def test_hardlink_survives_original_unlink(tmp_path):
    source = tmp_path / 'source.7z'; source.write_bytes(b'archive')
    work = tmp_path / 'work'
    pin = m.pin_archive(source, work)
    assert pin.stat().st_ino == source.stat().st_ino
    source.unlink()
    assert pin.read_bytes() == b'archive'


def test_pin_never_overwrites_different_inode(tmp_path):
    source = tmp_path / 'source.7z'; source.write_bytes(b'a')
    m.pin_archive(source, tmp_path / 'work')
    source.unlink(); source.write_bytes(b'b')
    with pytest.raises(RuntimeError, match='different inode'):
        m.pin_archive(source, tmp_path / 'work')


def test_singleton_lock(tmp_path):
    with m.singleton(tmp_path):
        with pytest.raises(RuntimeError, match='holds the lock'):
            with m.singleton(tmp_path):
                pass


def test_inventory_exact_pairs_and_ambiguous_leaf():
    text = 'Path = a/v1/whole_1.csv\nSize = 22\nPath = a/v1/beta_1.csv\nSize = 2\n'
    assert set(m.parse_inventory(text)) == {('a/v1', 1)}
    with pytest.raises(ValueError, match='incomplete'):
        m.parse_inventory(text.split('Path = a/v1/beta')[0])
    with pytest.raises(ValueError, match='ambiguous'):
        m.parse_inventory(text + text.replace('a/v1', 'b/v1'))


def test_valid_output_and_nan_coherence(tmp_path):
    ramp, block, _ = fixture_outputs(tmp_path)
    report = m.validate_outputs(ramp, block, {('archive/v1', 1), ('archive/v2', 1)})
    assert report['trials'] == 2 and report['blocks'] == 12


@pytest.mark.parametrize('defect', ['missing', 'duplicate', 'beta', 'allnan', 'tail', 'field', 'infinite'])
def test_reject_block_corruption(tmp_path, defect):
    ramp, block, _ = fixture_outputs(tmp_path)
    with np.load(block) as z: d = {k: z[k] for k in z.files}
    if defect == 'missing': d['data'] = d['data'][:1]
    elif defect == 'duplicate': d['folder'][1] = d['folder'][0]
    elif defect == 'beta': d['beta'][0] = .2
    elif defect == 'allnan': d['data'][0] = np.nan
    elif defect == 'tail': d['data'][0, 0, 7] = .1
    elif defect == 'field': d['fields'][0] = 'wrong'
    elif defect == 'infinite': d['data'][0, 0, 1] = np.inf
    np.savez_compressed(block, **d)
    with pytest.raises(ValueError):
        m.validate_outputs(ramp, block, {('archive/v1', 1), ('archive/v2', 1)})


def test_reject_error_ramp_and_truncation(tmp_path):
    ramp, block, rows = fixture_outputs(tmp_path)
    fields = list(rows[0])
    for change in ({'error': 'failed'}, {'total_s': 8.05}):
        rows[0].update(change)
        with ramp.open('w', newline='') as f:
            w=csv.DictWriter(f, fields);w.writeheader();w.writerows(rows)
        with pytest.raises(ValueError):
            m.validate_outputs(ramp, block, {('archive/v1', 1), ('archive/v2', 1)})
        rows[0]['error'] = ''


def test_publish_refuses_overwrite(tmp_path):
    stage=tmp_path/'stage';stage.mkdir();(stage/'result').write_bytes(b'a')
    final=tmp_path/'final'
    m.publish(stage, final)
    assert (final/'result').read_bytes()==b'a' and not stage.exists()
    stage.mkdir();(stage/'result').write_bytes(b'b')
    with pytest.raises(FileExistsError):m.publish(stage, final)
    assert (final/'result').read_bytes()==b'a'


def test_failed_child_retains_pin_and_partial_output(tmp_path, monkeypatch):
    work=tmp_path/'work';work.mkdir()
    source=tmp_path/'source.7z';source.write_bytes(b'archive')
    pin=m.pin_archive(source,work)
    script=tmp_path/'script.py';script.write_text('# dummy')
    monkeypatch.setattr(m,'SCRIPTS',{'ramp_meta':script,'stroke_blocks':script})
    monkeypatch.setattr(m,'metadata_spec',lambda name:{'bytes':7,'md5':m.digest(pin,'md5')})
    monkeypatch.setattr(m,'archive_inventory',lambda pin:{('archive/v1',1):{}})
    monkeypatch.setattr(m.shutil,'disk_usage',lambda path:SimpleNamespace(free=100_000_000_000))
    def fail(cmd, **kwargs):
        Path(cmd[-1]).write_text('partial')
        return SimpleNamespace(returncode=7)
    monkeypatch.setattr(m.subprocess,'run',fail)
    receipt=m.run_archive(pin,work)
    assert receipt['status']=='failed' and pin.exists() and source.exists()
    assert receipt['commands'][0]['exit_code']==7
    assert Path(receipt['stage']).exists()
    assert not (work/'source').exists()


def test_valid_legacy_reused_only_after_coverage(tmp_path,monkeypatch):
    work=tmp_path/'work';work.mkdir()
    source=tmp_path/'source.7z';source.write_bytes(b'archive');pin=m.pin_archive(source,work)
    script=tmp_path/'script.py';script.write_text('# dummy')
    monkeypatch.setattr(m,'SCRIPTS',{'ramp_meta':script,'stroke_blocks':script})
    monkeypatch.setattr(m,'metadata_spec',lambda name:{'bytes':7,'md5':m.digest(pin,'md5')})
    expected={('archive/v1',1),('archive/v2',1)}
    monkeypatch.setattr(m,'archive_inventory',lambda pin:expected)
    monkeypatch.setattr(m.shutil,'disk_usage',lambda path:SimpleNamespace(free=100_000_000_000))
    ramp,block,_=fixture_outputs(tmp_path)
    monkeypatch.setattr(m,'LEGACY',{'source.7z':(ramp,block)})
    monkeypatch.setattr(m.subprocess,'run',lambda *a,**kw:pytest.fail('must reuse valid output'))
    receipt=m.run_archive(pin,work)
    assert receipt['status']=='complete' and not pin.exists() and source.exists()
    assert receipt['commands']==[] and receipt['validation']['trials']==2
    assert (work/'source'/'receipt.json').exists()


def test_disk_reserve_failure_preserves_archive(tmp_path,monkeypatch):
    work=tmp_path/'work';work.mkdir();source=tmp_path/'source.7z';source.write_bytes(b'archive');pin=m.pin_archive(source,work)
    script=tmp_path/'script.py';script.write_text('# dummy')
    monkeypatch.setattr(m,'SCRIPTS',{'ramp_meta':script,'stroke_blocks':script})
    monkeypatch.setattr(m,'metadata_spec',lambda name:{'bytes':7,'md5':m.digest(pin,'md5')})
    monkeypatch.setattr(m,'archive_inventory',lambda pin:{('a/v1',1)})
    monkeypatch.setattr(m.shutil,'disk_usage',lambda path:SimpleNamespace(free=60_500_000_000))
    receipt=m.run_archive(pin,work)
    assert receipt['status']=='failed' and pin.exists()
    assert 'reserve' in receipt['error']
