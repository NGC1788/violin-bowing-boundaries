#!/usr/bin/env python3
"""Restrict the saved width-law cost envelopes to the training bound k in [-2,2].

This sensitivity analysis retains nominal controls and independent free band
centers. It does not fit or evaluate a predictor and is not a new methodology.
Original experiment artifacts and their protocol remain unchanged.
"""
from __future__ import annotations
import argparse
import copy
from datetime import datetime, timezone
import gzip
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import affine_transport as af
import benchmark_data as bd
import set_envelope as se
import width_law_certificate as cert


def restricted_minimum(cells, lower=-2., upper=2.):
    """Minimize on a nondegenerate closed bound using open envelope cells.

    Each physical pair candidate is feasible on an open k interval. Any
    candidate feasible exactly at a finite domain endpoint is also feasible on
    its interior side nearby. Endpoint values therefore cannot improve the
    minimum over interior cells. They can be worse (an isolated cost spike), so
    choose a strictly interior constructive representative, never the endpoint.
    """
    if not np.isfinite([lower, upper]).all() or not lower < upper:
        raise ValueError('finite nondegenerate ordered bounds required')
    clipped = []
    for cell in cells:
        a = -np.inf if cell['lower'] is None else float(cell['lower'])
        b = np.inf if cell['upper'] is None else float(cell['upper'])
        if np.isnan([a, b]).any() or not a < b:
            raise ValueError('ordered non-NaN cell endpoints required')
        cost = cell['cost']
        if not np.isfinite(cost) or cost < 0 or cost != int(cost):
            raise ValueError('nonnegative integer disagreement count required')
        lo, hi = max(lower, a), min(upper, b)
        if lo < hi:
            k = lo+(hi-lo)/2
            if not lo < k < hi:
                raise ArithmeticError('no representable strict interior point')
            clipped.append({'lower': lo, 'upper': hi, 'cost': int(cost), 'representative': k})
    if not clipped:
        raise ValueError('no open cell intersects the bounded domain')
    best = min(clipped, key=lambda cell: (cell['cost'], abs(cell['representative']), cell['lower']))
    return {'cost': best['cost'], 'k': best['representative'],
            'selected_cell': {key: best[key] for key in ('lower', 'upper', 'cost')},
            'clipped_cells': [{key: cell[key] for key in ('lower', 'upper', 'cost')} for cell in clipped]}


def load_report():
    spec = importlib.util.spec_from_file_location('width_law_independent_report', HERE / 'width_law_report.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run(bundle_path):
    blob = Path(bundle_path).read_bytes()
    bundle = json.loads(gzip.decompress(blob))
    report = load_report()
    # The independent original checker also checks task completeness, original
    # input and experiment-source hashes, and all stored endpoint witnesses.
    original_audit = report.verify_bundle(bundle)
    data = {'cello': bd.load_dataset(include_ambiguous=True), 'published': af.load_published()}
    family = se.label_family(data['cello']['output_features'], data['cello']['reference_f0_hz'])
    output, new_witnesses, column_cache = [], {}, {}
    for task, original in bundle['tasks'].items():
        domain, source, speeds, rule = (original[k] for k in ('domain', 'source', 'source_speeds', 'rule_index'))
        selected = restricted_minimum(original['cost_cells'])
        old_cost, new_cost = original['shared_width_cost'], selected['cost']
        if not old_cost <= new_cost <= original['constant_width_cost']:
            raise AssertionError('bounded cost must lie between free k and feasible k=0 costs')
        outside = not -2. <= original['witness_k'] <= 2.
        result = copy.deepcopy(original)
        if outside:
            ds = data[domain]
            labels = family[:, rule] if domain == 'cello' else ds['y_fixed'] == 1
            k = selected['k']
            dt = float(np.log(speeds[1]/speeds[0]))
            witnesses = []
            for column in original['columns']:
                pair = []
                for speed in speeds:
                    key = (domain, source, speed, column['beta_level'], rule)
                    if key not in column_cache:
                        mask = (ds['string'] == source) & (ds['speed'] == speed) & (ds['beta_level'] == column['beta_level'])
                        column_cache[key] = cert.enumerate_bands(ds['X'][mask, 1], labels[mask])
                    pair.append(column_cache[key])
                candidates, _ = cert.pair_intervals(*pair, dt)
                feasible = [candidate for candidate in candidates if candidate['lower'] < k < candidate['upper']]
                chosen = min(feasible, key=lambda candidate: candidate['cost'])
                witnesses.append(cert.pair_witness(*pair, chosen, k, dt))
            result.update(witness_k=k, witness_ratio=float(np.exp(k*dt)), witnesses=witnesses,
                          shared_width_cost=new_cost, extra_disagreements=new_cost-original['independent_band_cost'],
                          cost_cells=selected['clipped_cells'])
            verification = report.verify_task(result, ds, labels)
            new_witnesses[task] = {'witness_k': k, 'witness_ratio': result['witness_ratio'],
                                  'selected_cell': selected['selected_cell'], 'witnesses': witnesses,
                                  'independent_verification': verification}
        else:
            if new_cost != old_cost:
                raise AssertionError('a bounded original optimum proves unchanged cost')
        if not -2. <= result['witness_k'] <= 2.:
            raise AssertionError('selected witness remains outside bounds')
        output.append({'task': task, 'domain': domain, 'source': source, 'source_speeds': speeds,
                       'rule_index': rule, 'rule_parameters': original['rule_parameters'],
                       'source_rows': original['source_rows'], 'independent_band_cost': original['independent_band_cost'],
                       'original_witness_k': original['witness_k'], 'original_witness_outside_bounds': outside,
                       'unbounded_minimum_cost': old_cost, 'bounded_minimum_cost': new_cost,
                       'bounded_minus_unbounded': new_cost-old_cost,
                       'bounded_extra_disagreements': new_cost-original['independent_band_cost'],
                       'bounded_witness_k': result['witness_k'], 'bounded_witness_verified': True,
                       'witness_location': 'new_witnesses' if outside else 'original_bundle'})
    groups = {'all': output,
              'cello_all_rules': [r for r in output if r['domain'] == 'cello'],
              'cello_nominal': [r for r in output if r['domain'] == 'cello' and r['rule_index'] == 7],
              'published_D': [r for r in output if r['domain'] == 'published']}
    aggregate = {}
    for name, rows in groups.items():
        aggregate[name] = {'tasks': len(rows),
            'saved_witness_outside_bounds': sum(r['original_witness_outside_bounds'] for r in rows),
            'cost_increased_tasks': sum(r['bounded_minus_unbounded'] > 0 for r in rows),
            'maximum_cost_increase': max(r['bounded_minus_unbounded'] for r in rows),
            'sum_cost_increase_with_observation_reuse': sum(r['bounded_minus_unbounded'] for r in rows),
            'maximum_bounded_extra_disagreements': max(r['bounded_extra_disagreements'] for r in rows),
            'maximum_bounded_extra_disagreement_fraction': max(r['bounded_extra_disagreements']/r['source_rows'] for r in rows)}
    return {'created_utc': datetime.now(timezone.utc).isoformat(),
            'scope': 'Post-hoc sensitivity of the same nominal-control, free-center Hamming diagnostic. No predictor fit, new data, A2 evaluation, or novelty claim.',
            'bounds_closed': [-2., 2.], 'input_bundle_sha256': hashlib.sha256(blob).hexdigest(),
            'source_sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in
                (Path(__file__), HERE / 'width_law_report.py', Path(cert.__file__))},
            'original_tasks_verified': original_audit['tasks_verified'],
            'new_constructive_witnesses': len(new_witnesses), 'aggregate': aggregate,
            'tasks': output, 'new_witnesses': new_witnesses}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle', type=Path, default=HERE / 'width_law_artifacts.json.gz')
    parser.add_argument('--output', type=Path, default=HERE / 'width_bound_sensitivity.json')
    args = parser.parse_args()
    result = run(args.bundle)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps(result['aggregate'], indent=2))
