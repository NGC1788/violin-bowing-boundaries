"""Apply saved primary witnesses to frozen input labels, without DP helpers."""
from pathlib import Path
from collections import Counter, defaultdict
import gzip
import hashlib
import json
import math
import numpy as np
from exhaustive import first_run

HERE = Path(__file__).resolve().parent


def eligible(group, k):
    result = {}
    for column in group:
        f = column['force']
        start = first_run(column['labels'], k)
        if start is not None and start > 0 and 0 < f[start - 1] < f[start]:
            result[column['column']] = column
    return result


def intersection(cols, labels, k, p, epsilon):
    lower, upper, starts = [], [], {}
    for cid, column in cols.items():
        f = column['force']
        start = first_run(labels[cid], k)
        assert start is not None and start > 0
        assert 0 < f[start - 1] < f[start]
        lower.append(math.log(f[start - 1]) - p * math.log(column['beta']) - epsilon)
        upper.append(math.log(f[start]) - p * math.log(column['beta']) + epsilon)
        starts[cid] = start
    return [max(lower), min(upper)], starts


def main():
    input_path, result_path = HERE.parent / 'input_columns.json.gz', HERE.parent / 'results.json'
    with gzip.open(input_path, 'rt') as file:
        data = json.load(file)
    results = json.loads(result_path.read_text())
    assert hashlib.sha256(input_path.read_bytes()).hexdigest() == results['source_sha256']['input_columns.json.gz']
    groups = defaultdict(list)
    for col in data['columns']:
        groups[col['session'], col['folder']].append(col)
    counts = Counter()
    primary = Counter()
    restore_widths = []
    target_widths = []
    for record in results['records']:
        if not (record['k'] == 2 and record['epsilon'] == .2):
            continue
        session, folder, k, p, epsilon, mode = (record[name] for name in ('session', 'folder', 'k', 'p', 'epsilon', 'mode'))
        key = f'{session}|{folder}|K{k}|p{p}|eps{epsilon}|{mode}'
        columns = eligible(groups[session, folder], k)
        original = {cid: np.array(column['labels'], dtype=np.uint8) for cid, column in columns.items()}
        original_interval, original_starts = intersection(columns, original, k, p, epsilon)
        baseline = original_interval[0] <= original_interval[1]
        assert record['baseline_compatible'] == baseline
        assert len(columns) == record['n_columns']
        assert sum(len(c['labels']) for c in columns.values()) == record['n_windows']
        counts['primary_directional_records'] += 1
        for action, should_compatible in [('restore', True), ('exclude', False)]:
            solution = record[action]
            saved = results['witnesses'][key][action]
            if not solution['feasible']:
                assert saved is None
                counts['infeasible_witness_entries'] += 1
                continue
            assert saved is not None
            edited = {cid: labels.copy() for cid, labels in original.items()}
            touched = set()
            edits = Counter()
            for change in saved['changes']:
                cid, rank = change['column'], change['force_rank']
                col = columns[cid]
                assert (cid, rank) not in touched
                touched.add((cid, rank))
                assert col['trial'][rank] == change['trial']
                assert col['force'][rank] == change['force']
                assert col['beta'] == change['beta']
                assert edited[cid][rank] == change['before']
                assert change['before'] in (0, 1) and change['after'] == 1 - change['before']
                if mode == 'delete':
                    assert change['before'] == 1
                if mode == 'add':
                    assert change['before'] == 0
                edited[cid][rank] = change['after']
                edits['additions' if change['after'] else 'deletions'] += 1
            final_interval, starts = intersection(columns, edited, k, p, epsilon)
            assert (final_interval[0] <= final_interval[1]) == should_compatible
            np.testing.assert_array_equal(final_interval, saved['intersection'])
            actual = sum(int(np.sum(edited[cid] != original[cid])) for cid in columns)
            changed_columns = sum(bool(np.any(edited[cid] != original[cid])) for cid in columns)
            assert actual == solution['cost'] == saved['n_flips'] == len(saved['changes'])
            assert changed_columns == saved['changed_columns']
            assert edits['additions'] == saved['additions']
            assert edits['deletions'] == saved['deletions']
            for move in saved['boundary_moves']:
                cid = move['column']
                assert original_starts[cid] == move['old_start']
                assert starts[cid] == move['new_start']
            counts['feasible_witnesses_checked'] += 1
            counts['listed_flips_checked'] += actual
            if should_compatible:
                width = final_interval[1] - final_interval[0]
                restore_widths.append(width)
                assert final_interval[0] <= solution['intercept'] <= final_interval[1]
                if mode == 'both' and not baseline:
                    target_widths.append(width)
            if mode == 'both' and action == record['flip_target']:
                assert actual > 0
                assert record['flip_cost'] == actual
                primary['finite_decision_changes'] += 1
                primary['restorations' if should_compatible else 'exclusions'] += 1
    report = dict(status='passed', counts=dict(counts), primary_both=dict(primary),
                  primary_restore_witness_intersection=dict(
                      count=len(target_widths), endpoint_only=sum(x == 0 for x in target_widths),
                      minimum_width=min(target_widths, default=None)),
                  all_directional_restore_intersections=dict(
                      count=len(restore_widths), endpoint_only=sum(x == 0 for x in restore_widths),
                      minimum_width=min(restore_widths, default=None)),
                  source_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in (input_path, result_path)},
                  interpretation='Witnesses establish attainable exact stated decisions; minimality relies on separately verified combinatorial solvers.')
    (HERE / 'empirical_witness_results.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
