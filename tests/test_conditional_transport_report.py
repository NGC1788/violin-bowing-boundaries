"""Audit report tests cover exact weak-label bounds and public-only replay."""
import importlib.util
import itertools
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT/'research/2026-10-13/conditional_transport_assessment.py'
spec = importlib.util.spec_from_file_location('conditional_transport_assessment', REPORT)
report = importlib.util.module_from_spec(spec)
spec.loader.exec_module(report)


def test_independent_box_extrema_exhaustive():
    n = 4
    for truth in itertools.product(range(3), repeat=n):
        lower, upper = np.array(truth) == 2, np.array(truth) > 0
        family = np.column_stack((lower, upper))
        for prediction in itertools.product((False, True), repeat=n):
            p = np.array(prediction)
            outcomes = [np.array(t) for t in itertools.product((False,True), repeat=n)
                        if np.all(np.array(t) >= lower) and np.all(np.array(t) <= upper)]
            exact = [report.iou(p, t) for t in outcomes]
            scored = report.score(p.astype(float), lower, family)
            assert scored['box_lower_iou'] == min(exact)
            assert scored['box_upper_iou'] == max(exact)


def test_complete_audit_from_public_bundle_without_reports(tmp_path):
    # A fresh filesystem tree has no reports, raw waveforms, local checkpoints,
    # A2 archive, symlink to project data, or original absolute source paths.
    checkout = tmp_path/'checkout'
    shutil.copytree(ROOT/'scripts', checkout/'scripts', ignore=shutil.ignore_patterns('__pycache__'))
    shutil.copytree(ROOT/'research/inputs', checkout/'research/inputs')
    destination = checkout/'research/2026-10-13'
    destination.mkdir(exist_ok=True)
    for name in ('conditional_transport_assessment.py', 'conditional_transport_artifacts.json.gz'):
        shutil.copy2(ROOT/'research/2026-10-13'/name, destination/name)
    (destination/'schoonderwaldt2008').mkdir()
    shutil.copy2(ROOT/'research/2026-10-13/schoonderwaldt2008/published_cells.csv',
                 destination/'schoonderwaldt2008/published_cells.csv')
    env = {**os.environ, 'OPENBLAS_NUM_THREADS':'1', 'OMP_NUM_THREADS':'1',
           'MPLCONFIGDIR':str(tmp_path/'mpl'), 'PYTHONDONTWRITEBYTECODE':'1'}
    result = subprocess.run([sys.executable, str(destination/'conditional_transport_assessment.py')],
                            cwd=checkout, env=env, capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stdout+'\n'+result.stderr
    audit = json.loads((destination/'conditional_transport_independent_audit.json').read_text())
    original = json.loads((ROOT/'research/2026-10-13/conditional_transport_independent_audit.json').read_text())
    assert audit == original
    assert audit['tasks'] == 18
    assert audit['optimizer_starts'] == {'successful':2484}
    assert not audit['broad_improvement_gate']
    assert not (checkout/'reports').exists()
    assert not (checkout/'data').exists()
