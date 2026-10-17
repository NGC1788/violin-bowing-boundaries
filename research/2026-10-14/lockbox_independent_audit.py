#!/usr/bin/env python3
"""Independent numerical/provenance audit of completed A2; no model fitting.

Labels, metrics, coordinate transforms and transport formulas are reconstructed
here rather than calling the production evaluator or its metric helpers. The
original standard classifiers retain their frozen sklearn prediction methods.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import gzip
import hashlib
import io
import json
import math
from pathlib import Path
import sys

import joblib
import numpy as np
from scipy.special import expit
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts'))  # frozen joblib classes only
HERE = Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(path):
    return json.loads(Path(path).read_text())


def compare(a, b, path='', errors=None):
    errors = [] if errors is None else errors
    if isinstance(a, dict):
        assert isinstance(b, dict) and set(a) == set(b), path
        for key in a:
            compare(a[key], b[key], path + '/' + key, errors)
    elif isinstance(a, list):
        assert isinstance(b, list) and len(a) == len(b), path
        for i, (x, y) in enumerate(zip(a, b)):
            compare(x, y, f'{path}/{i}', errors)
    elif isinstance(a, (int, float)) and not isinstance(a, bool):
        error = abs(float(a) - float(b))
        assert np.isfinite(error) and error <= 1e-12, (path, a, b)
        errors.append(error)
    else:
        assert a == b, (path, a, b)
    return errors


def ratio(n, d):
    return n / d if d else 1.


def score(probability, nominal, family):
    """Set operations independent of the production vectorized metric code."""
    prediction = set(np.flatnonzero(probability >= .5).tolist())
    truths = [set(np.flatnonzero(family[:, j]).tolist()) for j in range(family.shape[1])]
    values = [ratio(len(prediction & truth), len(prediction | truth)) for truth in truths]
    lower, upper = set.intersection(*truths), set.union(*truths)
    central = set(np.flatnonzero(nominal).tolist())
    return {'n': len(probability), 'nominal_iou': ratio(len(prediction & central), len(prediction | central)),
            'family_min_iou': min(values), 'family_mean_iou': float(np.mean(values)), 'family_max_iou': max(values),
            'box_lower_iou': ratio(len(prediction & lower), len(prediction | upper)),
            'box_upper_iou': ratio(len(prediction & upper), len(prediction | lower)),
            'brier_nominal': float(np.mean((probability - nominal) ** 2)),
            'predicted_positive': len(prediction), 'family_ious': values}


def panels_score(probability, nominal, family, groups):
    panels = {}
    for group in sorted(set(groups)):
        keep = groups == group
        panels[group] = score(probability[keep], nominal[keep], family[keep])
    names = ('nominal_iou', 'family_min_iou', 'family_mean_iou', 'family_max_iou',
             'box_lower_iou', 'box_upper_iou', 'brier_nominal')
    average = {key: float(np.mean([row[key] for row in panels.values()])) for key in names}
    macro = {**average, 'worst_panel_family_min_iou': min(p['family_min_iou'] for p in panels.values())}
    return {'panels': panels, 'macro_panels': macro, 'specimen_panel_means': {'A2': average},
            'global_same_rule': score(probability, nominal, family)}


def original_prediction(model, X):
    raw = np.log(X / np.array([.1, 1., .1]))
    if hasattr(model, 'estimator'):
        coordinates = np.column_stack((raw[:, 0], raw[:, 1] - raw[:, 2])) if model.covariant else raw
        probabilities = model.estimator.predict_proba(coordinates)
        classes = list(model.estimator.classes_)
        return probabilities[:, classes.index(1)] if 1 in classes else np.zeros(len(X))
    coordinates = (np.column_stack((raw[:, 0], raw[:, 1] - model.speed_exponent * raw[:, 2]))
                   if model.covariant else raw)
    normalized = (coordinates - model.center_) / model.scale_
    d2 = ((normalized[:, None, :] - model.anchors_[None, :, :]) ** 2).sum(axis=2)
    basis = np.column_stack((np.ones(len(X)), normalized, np.exp(-d2 / (2 * model.length_scale ** 2))))
    return expit(basis @ model.coef_)


def transport_prediction(model, X):
    b, u, t = np.log(X / np.array([.1, 1., .1])).T
    if 'theta' in model:
        theta = np.asarray(model['theta'])
        a0, a1, k = theta[6:]
        integral = t if k == 0 else -np.expm1(-k * t) / k
        z = np.exp(-k * t) * u - (a0 + a1 * b) * integral
        norm = model['normalization']
        B, Z = (b - norm['mean'][0]) / norm['scale'][0], (z - norm['mean'][1]) / norm['scale'][1]
        design = np.column_stack((np.ones(len(X)), B, Z, B * B, B * Z, Z * Z))
        return expit(design @ theta[:6])
    z = np.column_stack((b, u, t)) if model['alpha'] is None else np.column_stack((b, u - model['alpha'] * t))
    terms = np.prod(z[:, None, :] ** np.asarray(model['polynomial_powers'])[None, :, :], axis=2)
    normalized = (terms - model['polynomial_feature_mean']) / model['polynomial_feature_scale']
    return expit(normalized @ model['logistic_coefficient'] + model['logistic_intercept'])


def audit(verify_archive=False):
    directory = HERE / 'lockbox_A2'
    release = load(directory / 'release_manifest.json')
    for name, expected in release['files'].items():
        assert sha(directory / name) == expected, name
    raw = gzip.decompress((directory / 'regimes.csv.gz').read_bytes())
    raw_hash = hashlib.sha256(raw).hexdigest()
    assert raw_hash == release['regimes_decompressed_sha256']
    frozen_dir = ROOT / 'research/2026-10-13'
    specs = {}
    all_code = {}
    fields = {'lockbox_frozen_manifest_portable.json': 'freeze_sha256',
              'lockbox_evaluation_spec.json': 'spec_sha256',
              'lockbox_preprocessing_spec.json': 'preprocessing_spec_sha256',
              'transport_lockbox_spec.json': 'spec_sha256'}
    for name, field in fields.items():
        item = load(frozen_dir / name)
        canonical = {k: v for k, v in item.items() if k != field}
        actual = hashlib.sha256(json.dumps(canonical, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()
        assert actual == item[field], name
        specs[name] = item
        for filename, expected in item.get('implementation_sha256', {}).items():
            assert sha(ROOT / 'scripts' / filename) == expected, filename
            all_code[filename] = expected
    frozen, evaluation, preprocessing, transport = [specs[k] for k in fields]
    assert sha(ROOT / 'scripts/evaluate_lockbox.py') == evaluation['evaluator_sha256']
    all_code['evaluate_lockbox.py'] = evaluation['evaluator_sha256']
    for name, expected in evaluation['feature_pipeline_sha256'].items():
        assert sha(ROOT / 'scripts' / name) == expected
        all_code[name] = expected
    main = load(directory / 'original_manifest.json')
    additional = load(directory / 'transport_manifest.json')
    binding = load(directory / 'preprocessing_provenance.json')
    assert main['frozen_model_fingerprint'] == frozen['freeze_sha256'] == transport['main_model_sha256']
    assert main['evaluation_spec_sha256'] == evaluation['spec_sha256'] == transport['main_evaluation_spec_sha256']
    assert preprocessing['preprocessing_spec_sha256'] == transport['main_preprocessing_spec_sha256']
    assert additional['additive_spec_sha256'] == transport['spec_sha256']
    assert main['input_csv_sha256'] == additional['input_csv_sha256'] == raw_hash
    assert main['no_fit_or_adaptation'] and additional['no_fit_or_adaptation']
    receipt = main['archive_verification']
    assert receipt == additional['archive_verification']
    assert receipt['status'] == 'verified' and receipt['whole_archive_verified']
    assert receipt['actual_md5'] == frozen['lockbox']['md5'] == 'ec520fecf6c5011ef8491b335edc1cee'
    assert receipt['record'] == frozen['lockbox']['record']
    assert receipt['archive_filename'] == frozen['lockbox']['file'] and receipt['bytes'] == 14775691753
    original_dir = ROOT / 'reports/research_20261013/lockbox_A2'
    transport_dir = ROOT / 'reports/research_20261013/transport_lockbox_A2'
    original_paths = {'original_manifest.json': original_dir / 'manifest.json',
                      'original_results.json': original_dir / 'results.json',
                      'transport_manifest.json': transport_dir / 'manifest.json',
                      'transport_results.json': transport_dir / 'results.json',
                      'preprocessing_provenance.json': ROOT / 'reports/research_20261013/lockbox_A2_preprocess/lockbox_provenance.json'}
    for name, path in original_paths.items():
        assert path.read_bytes() == (directory / name).read_bytes(), name
        assert sha(path) == release['original_file_hashes'][name]
    assert additional['main_manifest_sha256'] == sha(original_paths['original_manifest.json'])
    assert additional['main_results_sha256'] == sha(original_paths['original_results.json'])
    assert additional['preprocessing_provenance_sha256'] == sha(original_paths['preprocessing_provenance.json'])
    expected_config = {'archive': {k: receipt[k] for k in ('record', 'archive_filename', 'actual_md5', 'bytes')},
                       'frozen_model_sha256': frozen['freeze_sha256'], 'evaluation_spec_sha256': evaluation['spec_sha256'],
                       'preprocessing_spec_sha256': preprocessing['preprocessing_spec_sha256'],
                       'feature_pipeline_sha256': evaluation['feature_pipeline_sha256'],
                       'collection_sha256': preprocessing['implementation_sha256']['collection.py'], 'workers': 2}
    assert binding['status'] == 'complete' and binding['configuration'] == expected_config
    for name, expected in binding['output_sha256'].items():
        output = original_paths['preprocessing_provenance.json'].parent / name
        assert sha(output) == expected, name
        if output.name == 'regimes.csv':
            assert output.read_bytes() == raw
        elif output.name == 'summary.json':
            assert sha(output) == main['feature_summary_sha256'] == additional['feature_summary_sha256']
            summary = load(output)
            public_summary = load(directory / 'feature_summary.json')
            transformed = dict(summary)
            transformed['source_trials'] = summary['source_trials'].replace(str(ROOT) + '/', '')
            transformed['plots'] = [p.replace(str(ROOT) + '/', '') for p in summary['plots']]
            assert len(summary['plots']) == 4 and transformed == public_summary
    assert summary['thresholds_provisional']['flyback_frac'] == .5
    assert summary['thresholds_provisional']['hysteresis_sigma'] == 3.
    assert summary['f0_search_hz'] == [40., 160.]
    assert frozen['lockbox']['file'].removesuffix('.7z') in summary['source_trials']
    rows = sorted(csv.DictReader(io.StringIO(raw.decode(), newline='')), key=lambda r: (r['condition'], int(r['trial'])))
    assert len(rows) == 6000
    all_ids = [(r['condition'], int(r['trial'])) for r in rows]
    cells = [(r['condition'], int(r['beta_level']), int(r['force_rank'])) for r in rows]
    assert len(set(all_ids)) == len(rows) == len(set(cells))
    speeds = {'v1': .05, 'v2': .1, 'v3': .2}
    assert {key: sum(r['condition'] == key for r in rows) for key in speeds} == {key: 2000 for key in speeds}
    assert all(r['window_inside_plateau'] in ('True', 'False') for r in rows)
    eligible = [r for r in rows if r['window_inside_plateau'] == 'True' and float(r['c1_window_mean']) > 0]
    counts = {'raw': len(rows), 'eligible': len(eligible),
              'off_plateau': sum(r['window_inside_plateau'] == 'False' for r in rows),
              'nonpositive_force': sum(float(r['c1_window_mean']) <= 0 for r in rows)}
    assert counts == main['counts'] == additional['counts'] == release['counts']
    X = np.array([[float(r[k]) for k in ('beta', 'c1_window_mean', 'c2_window_mean')] for r in eligible])
    assert np.isfinite(X).all() and (X > 0).all()
    assert all(math.isclose(float(r['c2_window_mean']), speeds[r['condition']], rel_tol=.011) for r in eligible)
    observation = frozen['observation_family']
    family = []
    for row in eligible:
        periodicity, f0, flyback, count = [float(row[k]) for k in ('periodicity', 'f0_hz', 'second_flyback_ratio', 'flybacks_per_period')]
        valid = all(math.isfinite(x) for x in (periodicity, f0, flyback)) and count >= 1
        family.append([valid and periodicity >= gate and f0 >= .75 * observation['reference_f0_hz'] and flyback < fraction
                       for fraction in observation['flyback_fraction'] for gate in observation['periodicity']])
    family = np.array(family, dtype=bool)
    nominal = family[:, 7]
    diagram = frozen['lockbox']['file'].removesuffix('.7z')
    ids = np.array([f"{diagram}/{r['condition']}/{int(r['trial']):04d}" for r in eligible])
    groups = np.array([f"A2/{speeds[r['condition']]:g}" for r in eligible])
    levels = np.array([int(r['beta_level']) for r in eligible])
    ids_hash = hashlib.sha256(('\n'.join(sorted(ids)) + '\n').encode()).hexdigest()
    assert ids_hash == main['ids_sha256'] == additional['ids_sha256'] == release['ids_sha256']
    disagreements = int(np.count_nonzero(nominal != np.array([r['label'] == 'helmholtz' for r in eligible])))
    assert disagreements == main['legacy_label_disagreements']
    # Explicitly verify there is no development/lockbox ID overlap.
    import benchmark_data as bd
    development = bd.load_dataset(include_ambiguous=True)
    assert not set(ids) & set(development['ids'])
    source_single = (development['string'] == 'A') & (development['speed'] == .1)
    assert hashlib.sha256(('\n'.join(sorted(development['ids'][source_single])) + '\n').encode()).hexdigest() == frozen['training_ids_sha256']
    assert development['manifest']['manifest_sha256'] == frozen['data_manifest_sha256']
    benchmark_dir = ROOT / 'reports/research_20261013/benchmark_A_portable'
    assert sha(benchmark_dir / 'selection.json') == frozen['source_cv_selection_sha256']
    assert sha(benchmark_dir / 'manifest.json') == frozen['benchmark_manifest_sha256']
    source_selection = load(benchmark_dir / 'selection.json')
    for name, model in frozen['all_declared_models'].items():
        picked = source_selection[name]
        winner = max(range(len(picked['cv'])), key=lambda i: picked['cv'][i]['metrics']['family_min_iou'])
        assert winner == picked['primary_index']
        selected = picked['cv'][winner]
        assert selected['params'] == model['params']
        assert selected['metrics']['family_min_iou'] == model['source_cv_family_min_iou']
    standards = [name for name in frozen['all_declared_models'] if name.endswith(('_logistic', '_svc', '_extra_trees', '_hgb'))]
    assert max(standards, key=lambda name: frozen['all_declared_models'][name]['source_cv_family_min_iou']) == frozen['selected_simple_baseline']
    original_results = load(directory / 'original_results.json')
    addition_results = load(directory / 'transport_results.json')
    arrays = {}
    for label, path in (('original', original_dir), ('transport', transport_dir)):
        arrays[label] = np.load(path / 'predictions.npz', allow_pickle=False)
        for key, value in {'ids': ids, 'family': family, 'nominal': nominal, 'groups': groups, 'beta_level': levels}.items():
            assert np.array_equal(arrays[label][key], value), (label, key)
    error_values, probability_errors, entries = [], [], {}
    for name, info in frozen['all_declared_models'].items():
        model_path = ROOT / info['path']
        assert sha(model_path) == info['sha256'], name
        p = original_prediction(joblib.load(model_path), X)
        err = float(np.max(abs(p - arrays['original'][name])))
        assert err <= 1e-12, (name, err)
        probability_errors.append(err)
        actual = panels_score(p, nominal, family, groups)
        compare(actual, original_results[name], name, error_values)
        entries[name] = actual['macro_panels']['family_min_iou']
    assert set(arrays['original'].files) == set(frozen['all_declared_models']) | {'ids', 'family', 'nominal', 'groups', 'beta_level'}
    transport_entries = {}
    for key, source in transport['sources'].items():
        scores = {}
        source_mask = (development['string'] == 'A') & np.isin(development['speed'], source['training_speeds'])
        source_ids = development['ids'][source_mask]
        assert len(source_ids) == source['source']['n']
        assert hashlib.sha256(('\n'.join(sorted(source_ids)) + '\n').encode()).hexdigest() == source['source']['ids_sha256']
        assert not set(source_ids) & set(ids)
        fingerprint = hashlib.sha256()
        source_features = development['output_features']
        source_family = np.array([[bool(np.isfinite(source_features['periodicity'][i])
            and np.isfinite(source_features['f0_hz'][i]) and np.isfinite(source_features['second_flyback_ratio'][i])
            and source_features['flybacks_per_period'][i] >= 1 and source_features['periodicity'][i] >= gate
            and source_features['f0_hz'][i] >= .75 * observation['reference_f0_hz']
            and source_features['second_flyback_ratio'][i] < fraction)
            for fraction in observation['flyback_fraction'] for gate in observation['periodicity']]
            for i in np.flatnonzero(source_mask)], dtype=bool)
        source_nominal = development['y_fixed'][source_mask] == 1
        assert np.array_equal(source_family[:, 7], source_nominal)
        for array in (development['X'][source_mask].astype('<f8'), source_nominal.astype('u1'), source_family.astype('u1')):
            fingerprint.update(json.dumps(array.shape).encode())
            fingerprint.update(np.ascontiguousarray(array).tobytes())
        assert fingerprint.hexdigest() == source['source']['predictors_and_labels_sha256']
        for method, select_name in (('affine_flow', 'flow'), ('translation_k0', 'k0')):
            candidates = source['source_selection'][select_name]
            selected = max(candidates, key=lambda c: c['metrics']['family_min_iou'])
            assert all(source['models'][method][k] == selected[k] for k in ('lambda', 'warp_lambda'))
            assert selected['metrics']['family_min_iou'] == source['source_cv_family_min_iou'][method]
        assert source['models']['translation_k0']['theta'][8] == 0
        for method in ('learned_alpha', 'raw_3d'):
            candidates = [c for c in source['source_selection']['simple'] if (c['alpha'] is None) == (method == 'raw_3d')]
            selected = max(candidates, key=lambda c: c['metrics']['family_min_iou'])
            assert all(source['models'][method][k] == selected[k] for k in ('alpha', 'C'))
            assert selected['metrics']['family_min_iou'] == source['source_cv_family_min_iou'][method]
        assert source['selected_simple_baseline'] == max(('learned_alpha', 'raw_3d'), key=source['source_cv_family_min_iou'].__getitem__)
        for name, model in source['models'].items():
            p = transport_prediction(model, X)
            err = float(np.max(abs(p - arrays['transport'][key + '__' + name])))
            assert err <= 1e-12, (key, name, err)
            probability_errors.append(err)
            scores[name] = panels_score(p, nominal, family, groups)
            compare(scores[name], addition_results[key]['methods'][name], key + '/' + name, error_values)
        endpoints = {name: value['macro_panels']['family_min_iou'] for name, value in scores.items()}
        record = {'training_speeds': source['training_speeds'], 'methods': scores, 'primary_endpoint_by_model': endpoints,
                  'selected_simple_baseline': source['selected_simple_baseline'],
                  'affine_minus_k0': endpoints['affine_flow'] - endpoints['translation_k0'],
                  'affine_minus_selected_simple': endpoints['affine_flow'] - endpoints[source['selected_simple_baseline']]}
        # Include the outer endpoint copies and contrasts, not just per-model metrics.
        compare(record, addition_results[key], key, error_values)
        transport_entries[key] = {k: v for k, v in record.items() if k != 'methods'}
    assert len(probability_errors) == 26
    expected_keys = {key + '__' + name for key, source in transport['sources'].items() for name in source['models']}
    assert set(arrays['transport'].files) == expected_keys | {'ids', 'family', 'nominal', 'groups', 'beta_level'}
    compare(additional['primary_result'], addition_results[transport['primary_source']], 'manifest_primary', error_values)
    chronology = {'original_model_freeze': frozen['frozen_utc'], 'evaluation_freeze': evaluation['created_utc'],
                  'preprocessing_freeze': preprocessing['created_utc'], 'transport_freeze': transport['created_utc'],
                  'archive_verified': receipt['verified_utc'], 'features_completed': summary['created_utc'],
                  'original_evaluated': main['evaluated_utc'], 'transport_evaluated': additional['evaluated_utc']}
    times = [datetime.fromisoformat(value) for value in chronology.values()]
    assert times == sorted(times), chronology
    archive_check = {'performed': False, 'historical_receipt_md5': receipt['actual_md5']}
    if verify_archive:
        archive = ROOT / 'data/raw/zenodo' / receipt['record'] / receipt['archive_filename']
        before = archive.stat()
        digest = hashlib.md5()
        with archive.open('rb') as handle:
            for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b''):
                digest.update(chunk)
        after = archive.stat()
        assert before.st_size == after.st_size == receipt['bytes']
        assert before.st_mtime_ns == after.st_mtime_ns
        assert digest.hexdigest() == receipt['actual_md5']
        archive_check = {'performed': True, 'bytes': after.st_size, 'md5': digest.hexdigest()}
    primary = transport_entries[transport['primary_source']]
    return {'audited_utc': datetime.now(timezone.utc).isoformat(), 'status': 'PASS',
            'audit_script_sha256': sha(__file__), 'frozen_implementation_files_checked': len(all_code),
            'independent_method': 'csv.DictReader eligibility; Python scalar label gates; Python set IoU/bounds; independent coordinate/RBF/transport formulas; existing frozen sklearn estimators only for original standard classifiers.',
            'no_fit_or_adaptation': True, 'counts': counts,
            'eligible_by_panel': {group: int(np.sum(groups == group)) for group in sorted(set(groups))},
            'labels_matched_saved_arrays': True, 'development_ids_overlap': 0,
            'models_checked': 26, 'probability_values_checked': 26 * len(eligible),
            'max_probability_absolute_error': max(probability_errors),
            'numeric_metric_occurrences_checked': len(error_values), 'max_metric_absolute_error': max(error_values),
            'original_model_endpoints': entries, 'transport_endpoints_and_contrasts': transport_entries,
            'original_family_minus_selected_baseline': entries['covariant_family_robust'] - entries[frozen['selected_simple_baseline']],
            'transport_primary': primary, 'chronology': chronology, 'archive_reverification': archive_check,
            'scope': 'Independent arithmetic/provenance reconstruction from saved derived features. No raw waveform extraction rerun; no proof about external human exposure, physical Helmholtz truth, or independent specimen replication.',
            'release_verbatim_json_count': 5, 'summary_transformation_verified': 'Repository prefix removed from source_trials and four plots paths; all other values identical.',
            'all_frozen_specs_and_code_unchanged': True}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--verify-archive', action='store_true')
    parser.add_argument('--out', type=Path, default=HERE / 'lockbox_independent_audit.json')
    args = parser.parse_args()
    with threadpool_limits(limits=1):
        result = audit(args.verify_archive)
    args.out.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps({key: result[key] for key in ('status', 'models_checked', 'probability_values_checked',
                      'max_probability_absolute_error', 'numeric_metric_occurrences_checked', 'max_metric_absolute_error',
                      'archive_reverification')}, indent=2))
