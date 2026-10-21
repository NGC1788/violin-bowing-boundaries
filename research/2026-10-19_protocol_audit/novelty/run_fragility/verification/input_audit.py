"""Independent shape, identity, order, and finite-bracket input audit."""
from pathlib import Path
from collections import Counter, defaultdict
import argparse
import gzip
import hashlib
import json
import numpy as np
from exhaustive import first_run

HERE = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--external-dependencies', choices=('optional', 'required', 'skip'),
                        default='optional', help='Control external provenance hashes only; compact input checks are always required.')
    args = parser.parse_args()
    path = HERE.parent / 'input_columns.json.gz'
    with gzip.open(path, 'rt') as file:
        data = json.load(file)
    identities = set()
    column_ids = set()
    groups = defaultdict(int)
    duplicated_force_columns = nonpositive_force_windows = 0
    k_counts = {k: Counter() for k in (1, 2, 3)}
    for col in data['columns']:
        key = (col['session'], col['folder'], col['column'])
        assert key not in column_ids, key
        column_ids.add(key)
        forces = np.asarray(col['force'], dtype=float)
        labels = np.asarray(col['labels'])
        trials = col['trial']
        assert len(forces) == len(labels) == len(trials)
        assert np.isfinite(forces).all() and np.all(np.diff(forces) >= 0)
        assert np.isin(labels, [0, 1]).all()
        assert np.isfinite(col['beta']) and col['beta'] > 0
        assert np.isfinite(col['speed']) and col['speed'] > 0
        duplicated_force_columns += bool(np.any(np.diff(forces) == 0))
        nonpositive_force_windows += int(np.sum(forces <= 0))
        groups[(col['session'], col['folder'])] += 1
        for trial in trials:
            ident = (col['session'], col['folder'], trial)
            assert ident not in identities, ident
            identities.add(ident)
        for k, counts in k_counts.items():
            start = first_run(labels, k)
            if start is None:
                counts['no_original_run'] += 1
            elif start == 0:
                counts['original_left_censored'] += 1
            elif not 0 < forces[start - 1] < forces[start]:
                counts['original_nonpositive_or_zero_width'] += 1
            else:
                counts['eligible_columns'] += 1
                counts['eligible_windows'] += len(labels)
                counts['positive_endpoint_below_legacy_1mN_floor'] += int(forces[start - 1] < .001)
    assert 'D_T2' not in {col['session'] for col in data['columns']}
    report = dict(status='passed', input_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                  sessions=len({key[0] for key in groups}), folders=len(groups),
                  columns=len(column_ids), unique_windows=len(identities),
                  duplicated_force_columns=duplicated_force_columns,
                  nonpositive_force_windows=nonpositive_force_windows,
                  columns_per_folder=dict(Counter(groups.values())),
                  eligibility={str(k): dict(counts) for k, counts in k_counts.items()},
                  scope='Checks exported derived inputs; classifier truth and original signal extraction are not independently revalidated.')
    root = HERE.parents[2]
    repo = Path.home() / 'violin-bowing-boundaries'
    dependencies = [('violin-rne/' + name, root / name) for name in (
        'final/data.py', 'final/compare.py', 'claude_extra/verify_astra_boundaries.py',
        'novelty/label_sensitivity/label_sweep.py', 'novelty/tilt_principle/tilt.py')]
    dependencies.append(('violin-bowing-boundaries/scripts/regime_map.py', repo / 'scripts/regime_map.py'))
    threshold_rel = 'reports/research_20261013/lockbox_A2_preprocess/2024-04-19_TypeA_sample2/regime_map'
    report.update(dependency_sha256={}, threshold_summary_candidates=None,
                  external_dependency_mode=args.external_dependencies, missing_external_dependencies=[])
    if args.external_dependencies != 'skip':
        threshold_files = sorted((repo / threshold_rel).glob('*/summary.json'))
        report['threshold_summary_candidates'] = len(threshold_files)
        dependencies.extend(('violin-bowing-boundaries/' + str(p.relative_to(repo)), p) for p in threshold_files)
        if not threshold_files:
            report['missing_external_dependencies'].append('violin-bowing-boundaries/' + threshold_rel + '/*/summary.json')
        for name, dependency in dependencies:
            if dependency.is_file():
                report['dependency_sha256'][name] = hashlib.sha256(dependency.read_bytes()).hexdigest()
            else:
                report['missing_external_dependencies'].append(name)
        if args.external_dependencies == 'required' and report['missing_external_dependencies']:
            raise FileNotFoundError('Required external provenance files missing: ' + ', '.join(report['missing_external_dependencies']))
    (HERE / 'input_audit_results.json').write_text(json.dumps(report, indent=2) + '\n')
    (HERE / 'input_audit.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
