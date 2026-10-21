"""Export a physical guard for the joint rule from existing feature caches."""
from pathlib import Path
import gzip, hashlib, json, math, sys
from collections import Counter
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / 'novelty/tilt_principle'))
import tilt as T


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    prior = ROOT / 'novelty/coherent_fragility/input_features.json.gz'
    with gzip.open(prior, 'rt') as f:
        old = json.load(f)
    by_session = {}
    for c in old['columns']:
        by_session.setdefault(c['session'], []).append(c)
    columns = []
    counts = Counter()
    sources = {'prior_fixture': sha(prior), 'PLAN.md': sha(HERE / 'PLAN.md')}
    for session, cs in by_session.items():
        spec = T.LS.STEM[session]
        stem, rep = spec if isinstance(spec, tuple) else (spec, None)
        raw, _ = T.read2(stem, rep)
        records = {row[0]: row for row in raw[0]}
        source = T.LS.D.FEAT / f'{stem}.csv.gz'
        sources[str(source.relative_to(ROOT))] = sha(source)
        assert sources[str(source.relative_to(ROOT))] == old['source_sha256'][str(source.relative_to(ROOT))]
        for c in cs:
            pvals, rvals, guard, fixed = [], [], [], []
            for j, trial in enumerate(c['trial']):
                _, force, speed, amplitude, p, f0, fpp, ratio = records[c['folder'], trial]
                assert force == c['force'][j]
                baseline = bool(c['labels'][j])
                assert baseline == T.LS.helm(amplitude, p, f0, fpp, ratio, T.BASE)
                finite = math.isfinite(p) and math.isfinite(ratio)
                physical = (finite and p >= T.LS.AP['max_aperiodic']
                            and not (math.isfinite(f0) and f0 < T.LS.AP['subharmonic_ratio'] * T.LS.REF_F0)
                            and math.isfinite(fpp) and fpp >= 1)
                lock = baseline and not finite
                reconstructed = lock or (physical and p >= .8 and ratio < .5)
                assert reconstructed == baseline
                # The two old baseline slices must preserve their original gate.
                oldp = c['features']['periodicity']; oldr = c['features']['second_flyback_ratio']
                for u in (.7, .8, .9):
                    lhs = lock or (physical and p >= u and ratio < .5)
                    rhs = bool(oldp['fixed'][j] or (oldp['guard'][j] and p >= u))
                    assert lhs == rhs
                for v in (.3, .5, .7):
                    lhs = lock or (physical and p >= .8 and ratio < v)
                    rhs = bool(oldr['fixed'][j] or (oldr['guard'][j] and ratio < v))
                    assert lhs == rhs
                pvals.append(p if math.isfinite(p) else None)
                rvals.append(ratio if math.isfinite(ratio) else None)
                guard.append(int(physical and not lock)); fixed.append(int(lock))
                counts['windows'] += 1
                counts['locked_missing_positive'] += int(lock)
                counts['joint_only_admissible_windows'] += int(physical and .7 <= p < .8 and .5 <= ratio < .7)
            first = next((i for i in range(len(c['labels']) - 1) if c['labels'][i] and c['labels'][i+1]), None)
            eligible = first is not None and first > 0 and 0 < c['force'][first-1] < c['force'][first]
            columns.append({k: c[k] for k in ('id', 'group', 'session', 'folder', 'column', 'trial', 'speed', 'beta', 'force')} | {
                'baseline': c['labels'], 'eligible': eligible, 'guard': guard, 'fixed': fixed,
                'periodicity': pvals, 'ratio': rvals})
            counts['columns'] += 1; counts['eligible_columns'] += int(eligible)
        print(session, flush=True)
    assert counts['windows'] == 78000 and counts['eligible_columns'] == 1132
    sources['export_inputs.py'] = sha(Path(__file__))
    out = {'description': 'Joint physical guard and two features; same original force order and labels. CC BY 4.0 derived IWK data.',
           'source_records': old['source_records'], 'source_sha256': sources, 'columns': columns}
    payload = json.dumps(out, separators=(',', ':'), allow_nan=False).encode()
    with (HERE / 'input_joint.json.gz').open('wb') as f:
        with gzip.GzipFile(fileobj=f, mode='wb', mtime=0) as z:
            z.write(payload)
    audit = dict(counts) | {'baseline_mismatches': 0, 'slice_label_mismatches': 0,
                            'input_sha256': sha(HERE / 'input_joint.json.gz'), 'source_sha256': sources}
    # Size-only preflight; no compatibility or cost selection.
    groups = sorted({c['group'] for c in columns})
    audit['grid_preflight'] = []
    for group in groups:
        cc = [c for c in columns if c['group'] == group]
        u = {p for c in cc for p, q, g in zip(c['periodicity'], c['ratio'], c['guard']) if g and .7 <= p < .9 and q < .7}
        v = {q for c in cc for p, q, g in zip(c['periodicity'], c['ratio'], c['guard']) if g and .3 <= q < .7 and p >= .7}
        audit['grid_preflight'].append({'group': group, 'u_events': len(u), 'v_events': len(v), 'max_cells': (len(u)+1)*(len(v)+1)})
    (HERE / 'input_audit.json').write_text(json.dumps(audit, indent=2) + '\n')
    print(dict(counts), 'local_cells_bound', sum(x['max_cells'] for x in audit['grid_preflight']), flush=True)


if __name__ == '__main__':
    main()
