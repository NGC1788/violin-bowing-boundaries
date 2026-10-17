#!/usr/bin/env python3
"""Publish/replay completed A2 numerical evidence; never refit or rewrite freezes.

Replay checks released derived features, not extraction from the 13.76 GiB raw
archive. The original whole-archive verification receipt is historical evidence.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gzip
import hashlib
import json
from pathlib import Path
import shutil
import sys

import numpy as np
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts'))
import affine_transport as af
import benchmark_data as bd
import benchmark_sets as bs
import evaluate_lockbox as el
import evaluate_transport_lockbox as et
import multispeed_scaling as ms

DEFAULT = Path(__file__).parent/'lockbox_A2'
ORIGINAL_FILES = {'original_manifest.json','original_results.json','transport_manifest.json',
                  'transport_results.json','preprocessing_provenance.json'}
PAYLOAD_FILES = ORIGINAL_FILES | {'regimes.csv.gz','feature_summary.json'}


def dump(path, value):
    with Path(path).open('x') as f:
        json.dump(value, f, indent=2, allow_nan=False)
        f.write('\n')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def public_paths(value):
    if isinstance(value, dict):
        return {k: public_paths(v) for k,v in value.items()}
    if isinstance(value, list):
        return [public_paths(v) for v in value]
    if isinstance(value, str):
        return value.replace(str(ROOT)+'/', '')
    return value


def export_release(out):
    out = Path(out)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError('preserve the existing release')
    spec,frozen,evaluation,preprocessing = et.check_spec(et.DEFAULT_SPEC)
    target,_ = et.completed_target(et.DEFAULT_STATE,spec,frozen,evaluation,preprocessing)
    state = json.loads(et.DEFAULT_STATE.read_text())
    main_results = et.resolve_path(state['results'])
    binding_path = et.resolve_path(state['preprocessing_provenance'])
    binding = json.loads(binding_path.read_text())
    paths = {Path(p).name: binding_path.parent/p for p in binding['output_sha256']}
    additional = ROOT/'reports/research_20261013/transport_lockbox_A2'
    if json.loads((additional/'manifest.json').read_text())['ids_sha256'] != bd.id_hash(target['ids']):
        raise ValueError('original and additional evaluation IDs differ')
    out.mkdir(parents=True,exist_ok=True)
    raw = paths['regimes.csv'].read_bytes()
    (out/'regimes.csv.gz').write_bytes(gzip.compress(raw,mtime=0))
    summary = json.loads(paths['summary.json'].read_text())
    dump(out/'feature_summary.json',public_paths(summary))
    origins = {'original_manifest.json': main_results.with_name('manifest.json'),
               'original_results.json': main_results,
               'transport_manifest.json': additional/'manifest.json',
               'transport_results.json': additional/'results.json',
               'preprocessing_provenance.json': binding_path}
    for name,p in origins.items():
        if public_paths(json.loads(p.read_text())) != json.loads(p.read_text()):
            raise ValueError('unexpected private path in supposedly verbatim manifest')
        shutil.copyfile(p,out/name)
    manifest = {'released_utc': datetime.now(timezone.utc).isoformat(),
        'scope': 'Derived numerical replay of first completed fixed A2 evaluations; not a new untouched test.',
        'root_outcome_inspection_recorded_utc': '2026-10-14T10:46:48+00:00',
        'counts': target['counts'], 'ids_sha256': bd.id_hash(target['ids']),
        'regimes_decompressed_sha256': hashlib.sha256(raw).hexdigest(),
        'original_feature_summary_sha256': sha(paths['summary.json']),
        'summary_edit': 'Repository absolute prefix removed from source_trials and four plots paths; scientific values unchanged.',
        'historical_archive_receipt_is_not_local_raw_reverification': True,
        'license': 'CC-BY-4.0', 'attribution': 'Alessio Lampis, A collection of bowed responses of cello strings (part 2), record17782542; project-derived features and labels.',
        'source_url': 'https://doi.org/10.5281/zenodo.17782542',
        'files': {p.name: sha(p) for p in sorted(out.iterdir()) if p.is_file()},
        'original_file_hashes': {name: sha(p) for name,p in origins.items()}}
    dump(out/'release_manifest.json',manifest)
    return manifest


def load_release(directory):
    directory=Path(directory)
    manifest=json.loads((directory/'release_manifest.json').read_text())
    if set(manifest['files'])!=PAYLOAD_FILES or set(manifest['original_file_hashes'])!=ORIGINAL_FILES:
        raise ValueError('complete fixed release file inventory required')
    for name,expected in manifest['files'].items():
        path=(directory/name).resolve()
        if not path.is_relative_to(directory.resolve()) or sha(path)!=expected:
            raise ValueError('released file checksum/path mismatch: '+name)
    if any(manifest['original_file_hashes'][k]!=manifest['files'][k] for k in ORIGINAL_FILES):
        raise ValueError('verbatim original-file checksum mismatch')
    raw=gzip.decompress((directory/'regimes.csv.gz').read_bytes())
    if hashlib.sha256(raw).hexdigest()!=manifest['regimes_decompressed_sha256']:
        raise ValueError('decompressed feature CSV changed')
    spec,frozen,evaluation,preprocessing=et.check_spec(et.DEFAULT_SPEC)
    target=el.load_target(directory/'regimes.csv.gz',json.loads((directory/'feature_summary.json').read_text()),
                         frozen,evaluation,frozen['lockbox']['file'].removesuffix('.7z'))
    main=json.loads((directory/'original_manifest.json').read_text())
    receipt=el.validate_receipt(main['archive_verification'],frozen['lockbox'])
    if (target['counts']!=manifest['counts'] or bd.id_hash(target['ids'])!=manifest['ids_sha256']
            or target['counts']!=main['counts'] or bd.id_hash(target['ids'])!=main['ids_sha256']
            or target['input_csv_sha256']!=main['input_csv_sha256']
            or main['frozen_model_fingerprint']!=frozen['freeze_sha256']
            or main['evaluation_spec_sha256']!=evaluation['spec_sha256']
            or main['feature_summary_sha256']!=manifest['original_feature_summary_sha256']
            or main['selected_simple_baseline']!=frozen['selected_simple_baseline']
            or main['primary_comparisons']!=frozen['primary_comparisons']
            or main['legacy_label_disagreements']!=target['legacy_label_disagreements']
            or main.get('status')!='PASS' or main.get('no_fit_or_adaptation') is not True):
        raise ValueError('released inputs and frozen evaluation provenance differ')
    binding=json.loads((directory/'preprocessing_provenance.json').read_text())
    expected_config={'archive': {k:receipt[k] for k in ('record','archive_filename','actual_md5','bytes')},
        'frozen_model_sha256': frozen['freeze_sha256'], 'evaluation_spec_sha256': evaluation['spec_sha256'],
        'preprocessing_spec_sha256': preprocessing['preprocessing_spec_sha256'],
        'feature_pipeline_sha256': evaluation['feature_pipeline_sha256'],
        'collection_sha256': preprocessing['implementation_sha256']['collection.py'],'workers': 2}
    outputs=binding['output_sha256']
    if (binding.get('status')!='complete' or binding['configuration']!=expected_config or len(outputs)!=2
            or {Path(k).name for k in outputs}!={'regimes.csv','summary.json'}
            or len({Path(k).parent for k in outputs})!=1
            or any(Path(k).is_absolute() or '..' in Path(k).parts for k in outputs)):
        raise ValueError('preprocessing binding mismatch')
    if {Path(k).name:v for k,v in outputs.items()}!={'regimes.csv':target['input_csv_sha256'],
                                                    'summary.json':main['feature_summary_sha256']}:
        raise ValueError('preprocessing output checksum cross-link mismatch')
    additional=json.loads((directory/'transport_manifest.json').read_text())
    expected_additional={'status':'PASS','additive_spec_sha256':spec['spec_sha256'],
        'spec_created_utc':spec['created_utc'],'main_protocol_unchanged':True,'no_fit_or_adaptation':True,
        'model_count':sum(len(s['models']) for s in spec['sources'].values()),
        'primary_source':spec['primary_source'],'primary_endpoint':spec['primary_endpoint'],
        'counts':target['counts'],'ids_sha256':bd.id_hash(target['ids']),
        'main_manifest_sha256':sha(directory/'original_manifest.json'),
        'main_results_sha256':sha(directory/'original_results.json'),
        'preprocessing_provenance_sha256':sha(directory/'preprocessing_provenance.json'),
        'archive_verification':receipt,'input_csv_sha256':target['input_csv_sha256'],
        'feature_summary_sha256':main['feature_summary_sha256'],'scope':spec['statistical_scope']}
    if any(additional.get(k)!=v for k,v in expected_additional.items()):
        raise ValueError('transport manifest cross-link mismatch')
    times=[datetime.fromisoformat(v) for v in (spec['created_utc'],receipt['verified_utc'],
                                               main['evaluated_utc'],additional['evaluated_utc'])]
    if not times[0]<times[1]<=times[2]<=times[3]:
        raise ValueError('freeze/archive/evaluation chronology mismatch')
    originals=json.loads((directory/'original_results.json').read_text())
    additions=json.loads((directory/'transport_results.json').read_text())
    if set(originals)!=set(frozen['all_declared_models']) or set(additions)!=set(spec['sources']):
        raise ValueError('result model or source-pair inventory mismatch')
    if additional['primary_result']!=additions[spec['primary_source']]:
        raise ValueError('transport primary result cross-link mismatch')
    for key,source in spec['sources'].items():
        row=additions[key]
        if (set(row['methods'])!=set(source['models']) or row['training_speeds']!=source['training_speeds']
                or row['selected_simple_baseline']!=source['selected_simple_baseline']):
            raise ValueError('transport source/model selection differs')
        endpoint={m:r['macro_panels']['family_min_iou'] for m,r in row['methods'].items()}
        if (row['primary_endpoint_by_model']!=endpoint
                or row['affine_minus_k0']!=endpoint['affine_flow']-endpoint['translation_k0']
                or row['affine_minus_selected_simple']!=endpoint['affine_flow']-endpoint[source['selected_simple_baseline']]):
            raise ValueError('transport derived endpoint/difference mismatch')
    return target,spec,frozen


def compare_numbers(actual,expected, *, tolerance=1e-12):
    errors=[]
    def visit(a,b):
        if isinstance(a,dict):
            if set(a)!=set(b): raise AssertionError('metric keys differ')
            for k in a: visit(a[k],b[k])
        elif isinstance(a,list):
            if len(a)!=len(b): raise AssertionError('metric lengths differ')
            for x,y in zip(a,b): visit(x,y)
        elif isinstance(a,(int,float)) and not isinstance(a,bool):
            error=abs(float(a)-float(b))
            if not np.isfinite(error) or error>tolerance: raise AssertionError('metric value differs')
            errors.append(error)
        elif a!=b:
            raise AssertionError('metric metadata differs')
    visit(actual,expected)
    return len(errors),max(errors,default=0.)


def replay(directory):
    directory=Path(directory)
    target,spec,frozen=load_release(directory)
    el.check_freeze(el.DEFAULT_FREEZE)  # no implicit refitting; restore first if absent
    import joblib
    originals=json.loads((directory/'original_results.json').read_text())
    additions=json.loads((directory/'transport_results.json').read_text())
    checks=[]
    def score(p):
        return bs.evaluate_panels(p,target['nominal'],target['family'],target['groups'],np.ones(len(p),bool))
    for name,item in frozen['all_declared_models'].items():
        model=joblib.load(ROOT/item['path'])
        checks.append(compare_numbers(score(model.predict_proba(target['X'])[:,1]),originals[name]))
    for key,source in spec['sources'].items():
        for name,model in source['models'].items():
            p=af.predict_frozen(model,target['X']) if name in ('affine_flow','translation_k0') else ms.predict_frozen(model,target['X'])
            checks.append(compare_numbers(score(p),additions[key]['methods'][name]))
    return {'models_replayed': len(checks),'numeric_values': sum(c for c,_ in checks),
            'maximum_absolute_error': max(e for _,e in checks),'counts': target['counts'],
            'no_fit_or_adaptation': True,'raw_waveform_extraction_repeated': False}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=('export','replay'))
    p.add_argument('--directory',type=Path,default=DEFAULT)
    a=p.parse_args()
    with threadpool_limits(limits=1):
        print(json.dumps(export_release(a.directory) if a.command=='export' else replay(a.directory),indent=2))
