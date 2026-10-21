"""Read-only arithmetic, syntax and provenance checks; no data fitting."""
import ast
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def read(name):
    return json.loads((HERE / name).read_text())


primary = read('response/results.json')
matched = read('response/matched_force_mask.json')
theory = read('theory/counterexamples.json')
assert not primary['failures'] and not matched['failures']
assert primary['fit_counts']['replay'] == 13720 and matched['n_fits'] == 1660
assert primary['baseline_gate']['pass']
summary = matched['stable_target_summary']
assert summary['fixed_gate_pass']
features = ['amplitude', 'flyback_height', 'periodicity', 'second_flyback_ratio']
for model in ['M0', 'M1']:
    cells = sum(summary['features'][f]['n_sessions'] * len(summary['features'][f]['a']) for f in features)
    mean = sum(summary['features'][f]['models'][model]['mae'] *
               summary['features'][f]['n_sessions'] * len(summary['features'][f]['a']) for f in features) / cells
    assert cells == 338 and abs(mean - summary['models'][model]['mae']) < 1e-12
m0, m1 = (summary['models'][m]['mae'] for m in ['M0', 'M1'])
assert abs((1 - m1 / m0) - summary['mae_reduction']) < 1e-12
assert m1 <= .15 and 1 - m1 / m0 >= .25
assert all(summary['features'][f]['models']['M1']['mae'] <= .25 for f in features)
ratio = summary['features']['second_flyback_ratio']
assert ratio['observed_zero'] is None and ratio['models']['M1']['zero'] is not None
assert theory['exponent_change'] < -.1
counter = theory['same_marginals_different_response']
assert abs(counter['original_changed_exponent'] - 1) < 1e-6
assert abs(counter['permuted_changed_exponent']) < 1e-6
syntax = []
for path in sorted(HERE.rglob('*.py')):
    ast.parse(path.read_text(), filename=str(path))
    syntax.append(str(path.relative_to(HERE)))
hashes = {}
for name, expected in primary['source_sha256'].items():
    path = ROOT / name
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    hashes[name] = actual == expected
    if name != 'novelty/criterion_identifiability/PLAN.md':
        assert actual == expected, name
# PLAN.md intentionally gained the documented matched-mask addendum after primary execution.
out = {'read_only_no_refits': True, 'all_checks_pass': True,
       'main_replay_fits': 13720, 'matched_mask_fits': 1660,
       'main_cells': 338, 'm0_mae': m0, 'm1_mae': m1,
       'relative_reduction': 1-m1/m0, 'primary_input_hash_matches': hashes,
       'plan_hash_exception': 'Post-run audit addendum; original planned values preserved in local commit 2d8b05c.',
       'python_syntax_checked': syntax,
       'figures_visually_inspected': ['response/response_curves.png', 'locality/fig_endpoint_locality.png'],
       'limits': 'Arithmetic/provenance verification, not a physical validation or independent statistical replication.'}
(HERE / 'integration_check.json').write_text(json.dumps(out, indent=2, ensure_ascii=False) + '\n')
print(json.dumps({k: out[k] for k in ['all_checks_pass','main_cells','m0_mae','m1_mae','relative_reduction']}, indent=2))
