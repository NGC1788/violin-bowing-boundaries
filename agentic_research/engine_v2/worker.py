"""Fixed subprocess entrypoint with a machine-readable verifier failure receipt."""
import argparse
import json
from pathlib import Path
import traceback
from control import atomic_json, ControlError

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--request',required=True);ap.add_argument('--run-id',required=True)
    a=ap.parse_args();request=json.loads(Path(a.request).read_text())
    if request['run_id']!=a.run_id:raise ControlError('request identity mismatch')
    output=Path(request['output_dir'])
    try:
        from harness.experiment import run_experiment
        config={**request['config'],'id':a.run_id}
        metrics=run_experiment(config,request['data_path'],request['output_dir'])
        if not isinstance(metrics,dict):raise ControlError('run_experiment must return a dictionary')
        if metrics.get('certificate_error_count')!=0 or metrics.get('all_certified') is not True:
            raise AssertionError('returned certificate metrics failed verification')
        atomic_json(output/'worker_result.json',metrics)
    except Exception as exc:
        atomic_json(output/'worker_failure.json',{'type':type(exc).__name__,'error':str(exc),
                    'verification_failed':isinstance(exc,AssertionError) or 'certificate' in str(exc).lower() or 'verif' in str(exc).lower()})
        traceback.print_exc()
        return 3 if isinstance(exc,AssertionError) else 1
    return 0

if __name__=='__main__':raise SystemExit(main())
