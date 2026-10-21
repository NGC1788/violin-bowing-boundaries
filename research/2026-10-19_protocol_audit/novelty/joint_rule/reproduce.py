"""Reproduce joint-threshold results in a new, isolated output directory.

Reads only this compact artifact, never re-exports waveforms, and never writes
an original result. No network, installation, GPU, or pickled array is needed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time

HERE = Path(__file__).resolve().parent
VERIFICATION = (
    'oracle.py', 'exhaustive.py', 'validation_cases.py', 'fixture_audit.py',
    'real_profiles.py', 'global_optima.py', 'lower_bounds.py',
    'tolerance_cutoff_checks.py', 'obstruction_verify.py',
)
SOURCES = (
    'solver.py', 'empirical.py', 'input_joint.json.gz',
    'prior_input_features.json.gz', 'prior_arbitrary_reference.json',
    'PLAN.md', 'theory/rectangle_oracle.py',
    'theory/test_rectangle_oracle.py', 'theory/diagnostics.py',
)
STEPS = (
    'verification/exhaustive.py', 'verification/validation_cases.py',
    'theory/test_rectangle_oracle.py', 'verification/fixture_audit.py',
    'empirical.py', 'verification/real_profiles.py', 'verification/global_optima.py',
    'verification/lower_bounds.py',
    'verification/tolerance_cutoff_checks.py', 'theory/diagnostics.py',
    'verification/obstruction_verify.py',
)
JSON_RESULTS = (
    'verification/exhaustive_results.json', 'verification/validation_cases.json',
    'theory/rectangle_test_results.json', 'verification/fixture_audit.json',
    'results.json', 'verification/real_profiles.json',
    'verification/global_optima.json', 'verification/lower_bounds.json',
    'verification/tolerance_cutoff_checks.json', 'theory/obstructions.json',
    'theory/obstruction_certificates.json', 'verification/obstruction_verify.json',
)
# Only top-level runtime/provenance metadata is excluded. No scientific field,
# threshold, integer count, inferred verdict, or nested value is excluded.
METADATA_KEYS = {'seconds', 'elapsed_seconds', 'duration_seconds', 'source_sha256'}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def new_output(requested):
    if requested is None:
        return Path(tempfile.mkdtemp(prefix='joint-rule-reproduce-'))
    raw = Path(requested).expanduser()
    if raw.is_symlink():
        raise ValueError('Output directory must not be a symlink')
    path = raw.resolve()
    if path == HERE or HERE in path.parents:
        raise ValueError('Output must be outside the source artifact directory')
    if path.exists() and (not path.is_dir() or any(path.iterdir())):
        raise ValueError('Output directory must be new or empty')
    path.mkdir(parents=True, exist_ok=True)
    return path


def compare_values(left, right, path='$', counts=None, errors=None):
    if counts is None:
        counts = {'numbers': 0, 'other_values': 0}
    if errors is None:
        errors = []
    if isinstance(left, dict) and isinstance(right, dict):
        if set(left) != set(right):
            errors.append(f'{path}: dictionary keys differ')
        for key in sorted(set(left) & set(right)):
            compare_values(left[key], right[key], f'{path}.{key}', counts, errors)
    elif isinstance(left, list) and isinstance(right, list):
        if len(left) != len(right):
            errors.append(f'{path}: list lengths differ {len(left)} != {len(right)}')
        for i, (a, b) in enumerate(zip(left, right)):
            compare_values(a, b, f'{path}[{i}]', counts, errors)
    elif (isinstance(left, (int, float)) and not isinstance(left, bool)
          and isinstance(right, (int, float)) and not isinstance(right, bool)):
        counts['numbers'] += 1
        same = left == right if isinstance(left, int) and isinstance(right, int) else (
            math.isfinite(left) and math.isfinite(right)
            and math.isclose(left, right, rel_tol=1e-12, abs_tol=1e-12))
        if not same:
            errors.append(f'{path}: numeric mismatch {left!r} != {right!r}')
    else:
        counts['other_values'] += 1
        if type(left) is not type(right) or left != right:
            errors.append(f'{path}: value mismatch {left!r} != {right!r}')
    return counts, errors


def compare_json(reference, generated):
    left = json.loads(reference.read_text())
    right = json.loads(generated.read_text())
    ignored = sorted((set(left) | set(right)) & METADATA_KEYS)
    counts, errors = compare_values(
        {k: v for k, v in left.items() if k not in METADATA_KEYS},
        {k: v for k, v in right.items() if k not in METADATA_KEYS})
    return dict(passed=not errors, compared=counts,
                ignored_top_level_metadata=ignored,
                provenance_differences=[k for k in ignored if left.get(k) != right.get(k)],
                mismatch_count=len(errors), first_mismatches=errors[:20])


def compare_npz(reference, generated):
    import numpy as np
    errors, arrays, elements = [], 0, 0
    with np.load(reference, allow_pickle=False) as left, np.load(generated, allow_pickle=False) as right:
        if set(left.files) != set(right.files):
            errors.append('array keys differ')
        for key in sorted(set(left.files) & set(right.files)):
            a, b = left[key], right[key]
            arrays += 1
            elements += int(a.size)
            if a.dtype.hasobject or b.dtype.hasobject:
                raise ValueError(f'Object arrays are forbidden: {key}')
            if a.shape != b.shape or a.dtype != b.dtype:
                errors.append(f'{key}: shape/dtype mismatch')
                continue
            if a.dtype.kind in 'fc':
                same = np.isclose(a, b, atol=1e-12, rtol=1e-12, equal_nan=True)
            else:
                same = a == b
            if not np.all(same):
                errors.append(f'{key}: {int(np.count_nonzero(~same))} elements differ')
    return dict(passed=not errors, arrays_compared=arrays,
                elements_compared=elements, mismatch_count=len(errors),
                first_mismatches=errors[:20])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', help='New or empty directory outside the artifact; default creates a temporary directory')
    parser.add_argument('--python', default=sys.executable, help='Interpreter with NumPy; Matplotlib is needed only for --figures')
    parser.add_argument('--figures', action='store_true', help='Also run figures.py in the copy; compare its scientific selection if present')
    parser.add_argument('--timeout-per-step', type=float, default=600, help='Per-subprocess timeout in seconds (default 600)')
    args = parser.parse_args(argv)
    if not math.isfinite(args.timeout_per_step) or args.timeout_per_step <= 0:
        parser.error('--timeout-per-step must be finite and positive')
    try:
        output = new_output(args.output_dir)
    except ValueError as exc:
        parser.error(str(exc))
    work, reference, logs = output / 'work', output / 'reference', output / 'logs'
    for directory in (work, reference, logs):
        directory.mkdir()
    started = time.monotonic()
    report_path = output / 'reproduction_report.json'
    report = dict(status='running', source=str(HERE), output=str(output),
                  raw_export_performed=False, steps=[], comparisons={})

    def save():
        report['duration_seconds'] = time.monotonic() - started
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + '\n')

    try:
        inventory = {p.name for p in (HERE / 'verification').glob('*.py')}
        if inventory != set(VERIFICATION):
            raise RuntimeError(f'Verifier inventory changed; explicitly update runner: {sorted(inventory)}')
        sources = list(SOURCES) + [f'verification/{name}' for name in VERIFICATION]
        json_results = list(JSON_RESULTS)
        profile_npzs = sorted(p.name for p in (HERE / 'profiles').glob('*.npz'))
        profile_jsons = sorted(p.name for p in (HERE / 'profiles').glob('*.json'))
        if not profile_npzs or {Path(p).stem for p in profile_npzs} != {Path(p).stem for p in profile_jsons}:
            raise RuntimeError('Every local profile must have one NPZ and one metadata JSON')
        npz_results = ['global_atoms.npz'] + [f'profiles/{name}' for name in profile_npzs]
        json_results.extend(f'profiles/{name}' for name in profile_jsons)
        steps = list(STEPS)
        reference_pngs = []
        if args.figures:
            sources.append('figures.py')
            steps.append('figures.py')
            for figure_json in ('figure_selection.json', 'figure_data.json'):
                if (HERE / figure_json).is_file():
                    json_results.append(figure_json)
            reference_pngs = sorted(p.name for p in HERE.glob('fig_*.png'))
            if not reference_pngs:
                raise RuntimeError('No original fig_*.png files to verify against')
        expected_results = json_results + npz_results
        manifest = {}
        for relative in sources + expected_results:
            original = HERE / relative
            if original.is_symlink() or not original.is_file():
                raise ValueError(f'Missing or symlinked required artifact: {original}')
            digest = sha(original)
            destination = (work if relative in sources else reference) / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(original, destination)
            if sha(destination) != digest:
                raise RuntimeError(f'Snapshot changed while copying: {relative}')
            manifest[relative] = digest
        report['snapshot_sha256'] = manifest
        report['runner_sha256'] = sha(Path(__file__))
        # Keep source/source-result files out of Python's import path. Every
        # scientific subprocess sees only its fresh copied artifact tree.
        env = os.environ.copy()
        env.pop('PYTHONPATH', None)
        env.update(PYTHONNOUSERSITE='1', PYTHONDONTWRITEBYTECODE='1',
                   OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1',
                   VECLIB_MAXIMUM_THREADS='1', MPLBACKEND='Agg',
                   MPLCONFIGDIR=str(output / 'matplotlib-cache'))
        probe = ("import json,sys,importlib.metadata as m; "
                 "print(json.dumps({'python':sys.version,'executable':sys.executable,"
                 "'packages':{d.metadata['Name']:d.version for d in m.distributions()}}))")
        runtime = subprocess.run([args.python, '-c', probe], cwd=work, env=env,
                                 text=True, capture_output=True, check=True, timeout=30)
        report['runtime'] = json.loads(runtime.stdout)
        for index, script in enumerate(steps, 1):
            logfile = logs / f'{index:02d}_{Path(script).stem}.log'
            print(f'[{index}/{len(steps)}] {script}', flush=True)
            tick = time.monotonic()
            with logfile.open('w') as stream:
                result = subprocess.run([args.python, str(work / script)], cwd=work,
                                        env=env, text=True, stdout=stream,
                                        stderr=subprocess.STDOUT,
                                        timeout=args.timeout_per_step)
            report['steps'].append(dict(script=script, exit_code=result.returncode,
                                       duration_seconds=time.monotonic() - tick,
                                       log=str(logfile.relative_to(output))))
            save()
            if result.returncode:
                raise RuntimeError(f'{script} exited {result.returncode}; see {logfile}')
        for relative in expected_results:
            generated = work / relative
            if not generated.is_file():
                raise RuntimeError(f'Required fresh result missing: {relative}')
            function = compare_npz if relative.endswith('.npz') else compare_json
            report['comparisons'][relative] = function(reference / relative, generated)
        if args.figures:
            for name in reference_pngs:
                if not (work / name).is_file() or not (work / name).stat().st_size:
                    raise RuntimeError(f'Figure step did not create a nonempty {name}')
            report['regenerated_figures_sha256'] = {name: sha(work / name) for name in reference_pngs}
            report['png_pixels_compared'] = False
        report['source_files_unchanged'] = all(
            (HERE / rel).is_file() and sha(HERE / rel) == digest for rel, digest in manifest.items())
        fixtures = ('input_joint.json.gz', 'prior_input_features.json.gz',
                    'prior_arbitrary_reference.json')
        report['copied_fixtures_unchanged'] = all(sha(work / name) == manifest[name] for name in fixtures)
        report['numerical_tolerance'] = dict(integer='exact', float_absolute=1e-12,
                                              float_relative=1e-12, equal_nan=True)
        report['generated_files_sha256'] = {rel: sha(work / rel) for rel in expected_results}
        report['log_sha256'] = {str(p.relative_to(output)): sha(p) for p in sorted(logs.glob('*.log'))}
        report['summary_counts'] = dict(
            json_files=sum(not rel.endswith('.npz') for rel in expected_results),
            npz_files=sum(rel.endswith('.npz') for rel in expected_results),
            arrays=sum(r.get('arrays_compared', 0) for r in report['comparisons'].values()),
            array_elements=sum(r.get('elements_compared', 0) for r in report['comparisons'].values()),
            json_numbers=sum(r.get('compared', {}).get('numbers', 0) for r in report['comparisons'].values()))
        report['all_scientific_comparisons_passed'] = all(v['passed'] for v in report['comparisons'].values())
        report['status'] = 'passed' if (report['all_scientific_comparisons_passed']
                                      and report['source_files_unchanged']
                                      and report['copied_fixtures_unchanged']) else 'failed'
        save()
        print(json.dumps(dict(status=report['status'], report=str(report_path),
                              steps=len(steps), compared_files=len(expected_results)), indent=2), flush=True)
        return 0 if report['status'] == 'passed' else 1
    except Exception as exc:
        report.update(status='failed', error=f'{type(exc).__name__}: {exc}')
        save()
        print(f'Reproduction failed: {exc}\nReport: {report_path}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
