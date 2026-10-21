"""Re-run the compact research artifact in a fresh output directory.

No raw export, network request, package installation, or original result write.
Only the Python standard library is imported by this driver itself.
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
    "exhaustive.py", "validation_cases.py", "global_oracle.py",
    "global_validation.py", "end_to_end.py", "input_audit.py",
    "empirical_witness_audit.py",
)
RESULTS = (
    "verification/exhaustive_results.json", "verification/validation_results.json",
    "verification/global_results.json", "verification/global_validation_results.json",
    "verification/end_to_end_results.json", "verification/input_audit_results.json",
    "verification/empirical_witness_results.json", "synthetic_results.json",
    "results.json", "diagnostics.json",
)
# Explicitly documented top-level provenance/runtime fields only. Every
# scientific key and all nested scientific values remain in the comparison.
METADATA_KEYS = {
    "elapsed_seconds", "duration_seconds", "source_sha256", "dependency_sha256",
    "threshold_summary_candidates", "external_dependency_mode",
    "missing_external_dependencies",
}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def new_output(requested):
    if requested is None:
        return Path(tempfile.mkdtemp(prefix="run-fragility-reproduce-"))
    raw = Path(requested).expanduser()
    if raw.is_symlink():
        raise ValueError("Output directory must not be a symlink")
    path = raw.resolve()
    if path == HERE or HERE in path.parents:
        raise ValueError("Output must be outside the source artifact directory")
    if path.exists() and (not path.is_dir() or any(path.iterdir())):
        raise ValueError("Output directory must be new or empty")
    path.mkdir(parents=True, exist_ok=True)
    return path


def compare_values(expected, actual, path="$", counts=None, errors=None):
    if counts is None:
        counts = {"numbers": 0, "other_values": 0}
    if errors is None:
        errors = []
    if isinstance(expected, dict) and isinstance(actual, dict):
        if set(expected) != set(actual):
            errors.append(f"{path}: keys differ; missing={sorted(set(expected)-set(actual))}, extra={sorted(set(actual)-set(expected))}")
        for key in sorted(set(expected) & set(actual)):
            compare_values(expected[key], actual[key], f"{path}.{key}", counts, errors)
    elif isinstance(expected, list) and isinstance(actual, list):
        if len(expected) != len(actual):
            errors.append(f"{path}: list lengths differ {len(expected)} != {len(actual)}")
        for i, (left, right) in enumerate(zip(expected, actual)):
            compare_values(left, right, f"{path}[{i}]", counts, errors)
    elif (isinstance(expected, (int, float)) and not isinstance(expected, bool)
          and isinstance(actual, (int, float)) and not isinstance(actual, bool)):
        counts["numbers"] += 1
        if isinstance(expected, int) and isinstance(actual, int):
            same = expected == actual
        else:
            same = math.isfinite(expected) and math.isfinite(actual) and math.isclose(expected, actual, rel_tol=1e-12, abs_tol=1e-12)
        if not same:
            errors.append(f"{path}: numeric mismatch {expected!r} != {actual!r}")
    else:
        counts["other_values"] += 1
        if type(expected) is not type(actual) or expected != actual:
            errors.append(f"{path}: value mismatch {expected!r} != {actual!r}")
    return counts, errors


def compare_file(reference, generated):
    expected, actual = json.loads(reference.read_text()), json.loads(generated.read_text())
    ignored = sorted((set(expected) | set(actual)) & METADATA_KEYS)
    left = {k: v for k, v in expected.items() if k not in METADATA_KEYS}
    right = {k: v for k, v in actual.items() if k not in METADATA_KEYS}
    counts, errors = compare_values(left, right)
    return {
        "passed": not errors, "compared": counts,
        "ignored_top_level_metadata": ignored,
        "provenance_differences": [k for k in ignored if expected.get(k) != actual.get(k)],
        "mismatch_count": len(errors), "first_mismatches": errors[:20],
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", help="New/empty directory, outside this source directory; default: new temporary directory")
    parser.add_argument("--python", default=sys.executable, help="Interpreter with NumPy (and Matplotlib for --figures)")
    parser.add_argument("--figures", action="store_true", help="Also regenerate the two PNGs; scientific figure selection is compared, not PNG bytes")
    parser.add_argument("--timeout-per-step", type=float, default=300, help="Maximum seconds per subprocess (default 300)")
    args = parser.parse_args(argv)
    if args.timeout_per_step <= 0:
        parser.error("--timeout-per-step must be positive")
    try:
        output = new_output(args.output_dir)
    except ValueError as exc:
        parser.error(str(exc))
    work, reference, logs = output / "work", output / "reference", output / "logs"
    for path in (work, reference, logs):
        path.mkdir()
    started = time.monotonic()
    report = {"status": "running", "source": str(HERE), "output": str(output), "steps": [], "comparisons": {}}
    report_path = output / "reproduction_report.json"
    def save():
        report["duration_seconds"] = time.monotonic() - started
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    try:
        observed_scripts = {p.name for p in (HERE / "verification").glob("*.py")}
        if observed_scripts != set(VERIFICATION):
            raise RuntimeError(f"Verification script inventory changed: {sorted(observed_scripts)}; update execution order explicitly")
        sources = ["core.py", "global_profile.py", "synthetic.py", "empirical.py", "diagnose_results.py", "PLAN.md", "SYNTHETIC.md", "input_columns.json.gz"]
        sources += [f"verification/{name}" for name in VERIFICATION]
        expected_results = list(RESULTS)
        if args.figures:
            sources.append("figures.py")
            expected_results.append("figure_selection.json")
        # Snapshots are taken before any process runs. No old result is copied
        # into work/, so a failed step cannot leave a plausible stale result.
        manifest = {}
        for rel in sources + expected_results:
            original = HERE / rel
            if original.is_symlink() or not original.is_file():
                raise ValueError(f"Missing or symlinked required artifact: {original}")
            manifest[rel] = sha(original)
            target = (work if rel in sources else reference) / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(original, target)
            assert sha(target) == manifest[rel]
        report["snapshot_sha256"] = manifest
        report["raw_export_performed"] = False
        env = os.environ.copy()
        env.pop("PYTHONPATH", None)
        env.update(PYTHONNOUSERSITE="1", PYTHONDONTWRITEBYTECODE="1", OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", VECLIB_MAXIMUM_THREADS="1", MPLBACKEND="Agg", MPLCONFIGDIR=str(output / "matplotlib-cache"))
        probe = "import json,sys,importlib.metadata as m; print(json.dumps({'python':sys.version,'executable':sys.executable,'packages':{d.metadata['Name']:d.version for d in m.distributions()}}))"
        runtime = subprocess.run([args.python, "-c", probe], cwd=work, env=env, text=True, capture_output=True, check=True, timeout=30)
        report["runtime"] = json.loads(runtime.stdout)
        steps = [(f"verification/{name}", []) for name in VERIFICATION if name not in ("input_audit.py", "empirical_witness_audit.py")]
        steps += [("verification/input_audit.py", ["--external-dependencies", "skip"]), ("synthetic.py", []), ("empirical.py", []), ("diagnose_results.py", []), ("verification/empirical_witness_audit.py", [])]
        if args.figures:
            steps.append(("figures.py", []))
        for index, (script, extra) in enumerate(steps, 1):
            log = logs / f"{index:02d}_{Path(script).stem}.log"
            command = [args.python, str(work / script), *extra]
            print(f"[{index}/{len(steps)}] {script}", flush=True)
            step_started = time.monotonic()
            with log.open("w") as stream:
                process = subprocess.run(command, cwd=work, env=env, text=True, stdout=stream, stderr=subprocess.STDOUT, timeout=args.timeout_per_step)
            report["steps"].append({"script": script, "arguments": extra, "exit_code": process.returncode, "duration_seconds": time.monotonic() - step_started, "log": str(log.relative_to(output))})
            save()
            if process.returncode:
                raise RuntimeError(f"{script} exited {process.returncode}; see {log}")
        for rel in expected_results:
            if not (work / rel).is_file():
                raise RuntimeError(f"A required fresh result was not generated: {rel}")
            report["comparisons"][rel] = compare_file(reference / rel, work / rel)
        if args.figures:
            names = ("fig_costs.png", "fig_witness.png")
            if not all((work / name).is_file() and (work / name).stat().st_size > 0 for name in names):
                raise RuntimeError("Figure step did not create both nonempty PNG files")
            report["regenerated_figure_sha256"] = {name: sha(work / name) for name in names}
        report["source_files_unchanged"] = all((HERE / rel).is_file() and sha(HERE / rel) == digest for rel, digest in manifest.items())
        report["copied_fixture_unchanged"] = sha(work / "input_columns.json.gz") == manifest["input_columns.json.gz"]
        report["numerical_tolerance"] = {"integer": "exact", "float_absolute": 1e-12, "float_relative": 1e-12}
        report["all_scientific_comparisons_passed"] = all(c["passed"] for c in report["comparisons"].values())
        report["status"] = "passed" if report["all_scientific_comparisons_passed"] and report["source_files_unchanged"] and report["copied_fixture_unchanged"] else "failed"
        save()
        print(json.dumps({"status": report["status"], "report": str(report_path), "steps": len(report["steps"]), "compared_files": len(report["comparisons"])}, indent=2), flush=True)
        return 0 if report["status"] == "passed" else 1
    except Exception as exc:
        report.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        save()
        print(f"Reproduction failed: {exc}\nReport: {report_path}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
