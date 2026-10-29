"""Audit only the pilot's 16 receipt-pinned columns; no full-development analysis."""
from pathlib import Path
import gzip, hashlib, json, sys
import numpy as np
from joint_certificate_reference import minimum_joint_certificate

ROOT = Path('/Users/jeongjihoon/violin-bowing-boundaries/agentic_research')
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from harness.run_distribution import certificate_lower_bound


def main():
    state = json.loads((ROOT/'state/pilot_20261030_v1/campaign.json').read_text())
    runs = ROOT/'runs/pilot_20261030_v1'
    completed = sorted(runs.glob('*/metrics.json'))
    receipt_path = completed[0].parent/'data_receipt.json'
    receipt = json.loads(receipt_path.read_text())
    ids = receipt['dev_ids']
    assert len(ids) == len(set(ids)) == 16
    thresholds = np.asarray(receipt['rules'])
    blob = Path(state['data']).read_bytes()
    assert hashlib.sha256(blob).hexdigest() == receipt['input_sha256']
    raw = json.loads(gzip.decompress(blob))
    selected = {c['id']: c for c in raw['columns'] if c['id'] in set(ids)}
    # No labels, bounds or policy scores are computed for any unselected column.
    assert set(selected) == set(ids)
    rows = []
    for cid in ids:
        c = selected[cid]
        per = np.asarray([np.nan if v is None else v for v in c['periodicity']])
        ratio = np.asarray([np.nan if v is None else v for v in c['ratio']])
        guard, fixed = np.asarray(c['guard'], bool), np.asarray(c['fixed'], bool)
        labels = fixed[None, :] | (guard[None, :] & (per[None, :] >= thresholds[:, 0, None])
                                   & (ratio[None, :] < thresholds[:, 1, None]))
        exact, witness = minimum_joint_certificate(labels)
        old, _ = certificate_lower_bound(labels)
        rows.append({'id': cid, 'max_individual_bound': old, 'exact_joint_bound': exact,
                     'witness_positions': witness})
    bound = {r['id']: r['exact_joint_bound'] for r in rows}
    policy_rows = []
    for file in completed:
        metric = json.loads(file.read_text())
        assert set(c['id'] for c in metric['columns']) == set(ids)
        gap = [c['queries']-bound[c['id']] for c in metric['columns']]
        assert min(gap) >= 0
        policy_rows.append({'run_id': file.parent.name, 'config': metric['config'],
                            'mean_queries': metric['mean_queries'], 'mean_gap_to_exact_joint_bound': float(np.mean(gap))})
    report = {'status': 'PASS', 'scope': 'Exactly 16 receipt-pinned pilot columns; no full-development labels analyzed',
              'receipt': str(receipt_path), 'input_sha256': receipt['input_sha256'],
              'columns': rows, 'mean_max_individual_bound': float(np.mean([r['max_individual_bound'] for r in rows])),
              'mean_exact_joint_bound': float(np.mean([r['exact_joint_bound'] for r in rows])),
              'completed_runs_at_audit': policy_rows}
    (HERE/'pilot_joint_bound_results.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({k: v for k, v in report.items() if k != 'columns'}, indent=2))


if __name__ == '__main__':
    main()
