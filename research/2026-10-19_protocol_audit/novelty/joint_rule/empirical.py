"""Complete local profiles and streamed exact global joint-threshold search."""
from pathlib import Path
import gzip, hashlib, json, math, time
import numpy as np
from solver import analyze_group, threshold_atoms, map_atoms

HERE = Path(__file__).resolve().parent
SLOPES = (-2., -1.2)
EPS = (0., .1, .2, .4)
BUDGETS = (.01, .05, .1, .15, .3, 1.)
BASE = (.8, .5)


def clean(x):
    if isinstance(x, dict): return {k: clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)): return [clean(v) for v in x]
    if isinstance(x, np.ndarray): return clean(x.tolist())
    if isinstance(x, np.generic): return clean(x.item())
    if isinstance(x, float) and not math.isfinite(x): return None
    return x


def write(path, x):
    path.write_text(json.dumps(clean(x), indent=2, allow_nan=False) + '\n')


def closest(atoms, base):
    return np.where(base > atoms['hi'], atoms['hi'],
                    np.where(base <= atoms['lo'],
                             np.where(atoms['lo_closed'], atoms['lo'], np.nextafter(atoms['lo'], np.inf)), base))


def best_index(indices, *arrays):
    if not len(indices): return None
    key = np.lexsort(tuple(np.asarray(a)[indices] for a in (np.arange(len(arrays[0])),) + arrays[::-1]))
    return int(indices[key[0]])


def replace_best(table, key, rank, ui, vi):
    old = table.get(key)
    if old is None or rank < old[0]: table[key] = (rank, ui, vi)


def main():
    start = time.monotonic()
    with gzip.open(HERE / 'input_joint.json.gz', 'rt') as f: raw = json.load(f)
    groups = sorted({c['group'] for c in raw['columns']})
    profiles = []
    (HERE / 'profiles').mkdir(exist_ok=True)
    for gi, group in enumerate(groups):
        p = analyze_group([c for c in raw['columns'] if c['group'] == group], slopes=SLOPES, epsilons=EPS)
        arrays = {k: v for k, v in p.items() if isinstance(v, np.ndarray)}
        for axis in ('u', 'v'):
            arrays.update({axis + '_' + k: v for k, v in p[axis + '_atoms'].items()})
        np.savez_compressed(HERE / 'profiles' / f'{gi:02d}.npz', **arrays)
        meta = {k: p[k] for k in ('group', 'u_domain', 'v_domain', 'metadata', 'k')}
        write(HERE / 'profiles' / f'{gi:02d}.json', meta)
        profiles.append(p)
        print('profile', gi, group.split('|')[0], p['n_all_edits'].shape, flush=True)
    ua = threshold_atoms(np.concatenate([p['u_atoms']['hi'] for p in profiles]), (.7, .9))
    va = threshold_atoms(np.concatenate([p['v_atoms']['hi'] for p in profiles]), (.3, .7))
    um = [map_atoms(p['u_atoms'], ua['hi']) for p in profiles]
    vm = [map_atoms(p['v_atoms'], va['hi']) for p in profiles]
    uclose, vclose = closest(ua, BASE[0]), closest(va, BASE[1])
    udist = ((uclose - BASE[0]) / .2) ** 2
    vdist = ((vclose - BASE[1]) / .4) ** 2
    ub = int(np.searchsorted(ua['hi'], BASE[0])); vb = int(np.searchsorted(va['hi'], BASE[1]))
    U, V = len(ua['hi']), len(va['hi'])
    selected, local, tolerance = {}, {}, {}
    baseline_compatible = None
    for ui in range(U):
        costs = np.array([p['n_all_edits'][um[g][ui], vm[g]] for g, p in enumerate(profiles)])
        ecosts = np.array([p['n_eligible_edits'][um[g][ui], vm[g]] for g, p in enumerate(profiles)])
        bad = np.array([p['invalid_eligible_columns'][um[g][ui], vm[g]] for g, p in enumerate(profiles)])
        valid_all = bad.sum(axis=0) == 0
        total = costs.sum(axis=0)
        distance = udist[ui] + vdist
        ok = np.array([p['compatible'][um[g][ui], vm[g]] for g, p in enumerate(profiles)])
        if ui == ub:
            baseline_compatible = ok[:, vb].copy()
            assert total[vb] == 0 and valid_all[vb]
        for si in range(len(SLOPES)):
            for ei in range(len(EPS)):
                coverage = ok[:, :, si, ei].sum(axis=0)
                for policy, permitted in [('unrestricted', np.ones(V, bool)), ('preserve_all', valid_all)]:
                    for budget in BUDGETS:
                        ii = np.flatnonzero(permitted & (total <= math.floor(78000 * budget)))
                        vi = best_index(ii, -coverage, total, distance)
                        if vi is not None:
                            rank = (-int(coverage[vi]), int(total[vi]), float(distance[vi]), ui, vi)
                            replace_best(selected, (si, ei, policy, budget), rank, ui, vi)
                for g in range(len(groups)):
                    for mode, mask in [('joint', np.ones(V, bool)), ('periodicity_only', np.arange(V) == vb),
                                       ('ratio_only', np.full(V, ui == ub, bool)), ('preserve_all', valid_all)]:
                        ii = np.flatnonzero(ok[g, :, si, ei] & mask)
                        vi = best_index(ii, ecosts[g], total, distance)
                        if vi is not None:
                            rank = (int(ecosts[g, vi]), int(total[vi]), float(distance[vi]), ui, vi)
                            replace_best(local, (g, si, ei, mode), rank, ui, vi)
            for g, p in enumerate(profiles):
                gap = p['gap'][um[g][ui], vm[g], si]
                delta = np.maximum(0., gap / 2)
                for policy, mask in [('unrestricted', bad[g] == 0), ('preserve_all', valid_all)]:
                    ii = np.flatnonzero(mask & np.isfinite(delta))
                    vi = best_index(ii, delta, total, distance)
                    if vi is not None:
                        rank = (float(delta[vi]), int(total[vi]), float(distance[vi]), ui, vi)
                        replace_best(tolerance, (g, si, policy), rank, ui, vi)
        if ui % 200 == 0: print('global row', ui, '/', U, 'v_cells', V, flush=True)

    def witness(ui, vi, si, ei):
        rows = []
        for g, p in enumerate(profiles):
            a, b = um[g][ui], vm[g][vi]
            rows.append({'group': groups[g], 'compatible': bool(p['compatible'][a, b, si, ei]),
                         'all_edits': int(p['n_all_edits'][a, b]), 'eligible_edits': int(p['n_eligible_edits'][a, b]),
                         'changed_eligible_columns': int(p['n_changed_columns'][a, b]),
                         'invalid_columns': int(p['invalid_eligible_columns'][a, b]),
                         'lower': p['lower'][a, b, si], 'upper': p['upper'][a, b, si],
                         'lower_witness': int(p['lower_witness'][a, b, si]),
                         'upper_witness': int(p['upper_witness'][a, b, si]),
                         'invalid_witness': int(p['invalid_witness'][a, b])})
        return {'u': float(uclose[ui]), 'v': float(vclose[vi]), 'global_cell': [ui, vi],
                'u_cell': [ua['lo'][ui], ua['hi'][ui], ua['lo_closed'][ui]],
                'v_cell': [va['lo'][vi], va['hi'][vi], va['lo_closed'][vi]],
                'all_edits': sum(r['all_edits'] for r in rows), 'eligible_edits': sum(r['eligible_edits'] for r in rows),
                'invalid_columns': sum(r['invalid_columns'] for r in rows),
                'compatible_groups': [r['group'] for r in rows if r['compatible']],
                'invalid_groups': [r['group'] for r in rows if r['invalid_columns']], 'groups': rows}

    analyses = []
    for si, slope in enumerate(SLOPES):
        for ei, epsilon in enumerate(EPS):
            a = {'p': slope, 'epsilon': epsilon,
                 'baseline_compatible_groups': [g for g, good in zip(groups, baseline_compatible[:, si, ei]) if good],
                 'policies': {}, 'local': []}
            for policy in ('unrestricted', 'preserve_all'):
                choices = []
                for budget in BUDGETS:
                    b = selected.get((si, ei, policy, budget))
                    choices.append({'budget_fraction': budget, 'witness': None if b is None else witness(b[1], b[2], si, ei)})
                a['policies'][policy] = choices
            for g, group in enumerate(groups):
                row = {'group': group, 'modes': {}}
                for mode in ('joint', 'periodicity_only', 'ratio_only', 'preserve_all'):
                    b = local.get((g, si, ei, mode))
                    row['modes'][mode] = None if b is None else {'eligible_edits': b[0][0], 'witness': witness(b[1], b[2], si, ei)}
                row['joint_only'] = row['modes']['joint'] is not None and row['modes']['periodicity_only'] is None and row['modes']['ratio_only'] is None
                a['local'].append(row)
            analyses.append(a)
    trows = []
    for (g, si, policy), b in sorted(tolerance.items()):
        trows.append({'group': groups[g], 'p': SLOPES[si], 'policy': policy,
                      'minimum_epsilon': b[0][0], 'witness': witness(b[1], b[2], si, EPS.index(.2))})
    out = {'protocol_commit': '406d44f', 'post_hoc': True, 'groups': groups,
           'n_global_cells': U * V, 'global_shape': [U, V],
           'n_local_cells': sum(p['n_all_edits'].size for p in profiles),
           'analyses': analyses, 'minimum_tolerance': trows,
           'duration_seconds': time.monotonic() - start,
           'source_sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in [HERE / 'PLAN.md', HERE / 'solver.py', Path(__file__), HERE / 'input_joint.json.gz']}}
    write(HERE / 'results.json', out)
    np.savez_compressed(HERE / 'global_atoms.npz', **{'u_' + k: v for k, v in ua.items()}, **{'v_' + k: v for k, v in va.items()})
    for a in analyses:
        print('RESULT', a['p'], a['epsilon'], 'local', sum(x['modes']['joint'] is not None for x in a['local']),
              'joint_only', sum(x['joint_only'] for x in a['local']),
              [(k, len(v[-1]['witness']['compatible_groups']), v[-1]['witness']['all_edits'], len(v[-1]['witness']['invalid_groups'])) for k,v in a['policies'].items()], flush=True)
    print('seconds', out['duration_seconds'], flush=True)


if __name__ == '__main__': main()
