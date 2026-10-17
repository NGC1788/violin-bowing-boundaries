"""Release integrity checks do not replace raw waveform reproduction."""
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil

import pytest

ROOT=Path(__file__).resolve().parents[1]
SPEC=importlib.util.spec_from_file_location('lockbox_release',ROOT/'research/2026-10-14/lockbox_release.py')
release=importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(release)


def copy_release(tmp_path):
    path=tmp_path/'release'
    shutil.copytree(release.DEFAULT,path)
    return path


def rewrite(path,name,change):
    p=path/name
    d=json.loads(p.read_text())
    change(d)
    p.write_text(json.dumps(d,indent=2)+'\n')
    if name=='release_manifest.json': return
    manifest_path=path/'release_manifest.json'
    m=json.loads(manifest_path.read_text())
    m['files'][name]=hashlib.sha256(p.read_bytes()).hexdigest()
    if name in m['original_file_hashes']:
        m['original_file_hashes'][name]=m['files'][name]
    manifest_path.write_text(json.dumps(m,indent=2)+'\n')


def test_public_release_loads_expected_eligible_rows():
    target,_,_=release.load_release(release.DEFAULT)
    assert target['counts']=={'raw':6000,'eligible':5892,'off_plateau':0,'nonpositive_force':108}
    assert target['X'].shape==(5892,3)


def test_missing_checksum_cannot_skip_payload_verification(tmp_path):
    path=copy_release(tmp_path)
    rewrite(path,'release_manifest.json',lambda d:d['files'].pop('regimes.csv.gz'))
    with pytest.raises(ValueError,match='inventory'):
        release.load_release(path)


def test_changed_payload_bytes_rejected_before_loading(tmp_path):
    path=copy_release(tmp_path)
    with (path/'regimes.csv.gz').open('ab') as f: f.write(b'x')
    with pytest.raises(ValueError,match='checksum'):
        release.load_release(path)


@pytest.mark.parametrize('name,change,match',[
    ('preprocessing_provenance.json',lambda d:d['configuration'].__setitem__('workers',17),'binding'),
    ('preprocessing_provenance.json',lambda d:d['output_sha256'].__setitem__('extra.csv','0'*64),'binding'),
    ('transport_manifest.json',lambda d:d.__setitem__('main_results_sha256','0'*64),'cross-link'),
    ('transport_manifest.json',lambda d:d.__setitem__('evaluated_utc','2020-01-01T00:00:00+00:00'),'chronology'),
    ('original_manifest.json',lambda d:d.__setitem__('counts',{'raw':1}),'provenance'),
])
def test_rehashed_but_inconsistent_provenance_rejected(tmp_path,name,change,match):
    path=copy_release(tmp_path)
    rewrite(path,name,change)
    with pytest.raises(ValueError,match=match):
        release.load_release(path)


def rebind_main_result_hash(path):
    digest=release.sha(path/'original_results.json')
    rewrite(path,'transport_manifest.json',lambda d:d.__setitem__('main_results_sha256',digest))


def test_extra_original_result_model_rejected(tmp_path):
    path=copy_release(tmp_path)
    rewrite(path,'original_results.json',lambda d:d.__setitem__('extra_model',{}))
    rebind_main_result_hash(path)
    with pytest.raises(ValueError,match='inventory'):
        release.load_release(path)


def test_transport_derived_delta_must_match_score_arithmetic(tmp_path):
    path=copy_release(tmp_path)
    # Secondary outcome avoids the redundant primary-result field; checksum
    # updates cannot conceal a wrong arithmetic difference in a released table.
    rewrite(path,'transport_results.json',lambda d:d['A__0.05_0.2'].__setitem__('affine_minus_k0',.4))
    with pytest.raises(ValueError,match='difference'):
        release.load_release(path)


def test_redundant_primary_outcome_must_match_results(tmp_path):
    path=copy_release(tmp_path)
    rewrite(path,'transport_manifest.json',lambda d:d['primary_result'].__setitem__('affine_minus_k0',.4))
    with pytest.raises(ValueError,match='primary result'):
        release.load_release(path)


def test_recursive_metric_comparison_rejects_extra_values_and_nan():
    with pytest.raises(AssertionError,match='keys'):
        release.compare_numbers({'x':1.},{'x':1.,'extra':1.})
    with pytest.raises(AssertionError,match='value'):
        release.compare_numbers({'x':float('nan')},{'x':1.})
