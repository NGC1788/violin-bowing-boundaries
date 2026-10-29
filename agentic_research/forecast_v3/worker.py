"""Trusted forecast worker: predictive checks, never an exact-label certificate."""
import argparse
import traceback
import json
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
from forecast_v3.control import atomic_json,ControlError,file_sha


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--request',required=True)
    ap.add_argument('--run-id',required=True)
    a=ap.parse_args()
    request=json.loads(Path(a.request).read_text())
    if request['run_id']!=a.run_id:
        raise ControlError('Request identity mismatch')
    output=Path(request['output_dir'])
    try:
        from forecast_v3.data import load_prepared_dataset
        from forecast_v3.experiment import evaluate
        if file_sha(request['data_path'])!=request['data_sha'] or file_sha(Path(request['data_path']).with_suffix('.json'))!=request['data_receipt_sha']:
            raise AssertionError('Frozen data/receipt verification failed in worker')
        dataset=load_prepared_dataset(request['data_path'])
        metrics=evaluate({**request['config'],'id':a.run_id},dataset,output)
        if metrics.get('verification_ok') is not True or metrics.get('verification_error_count')!=0:
            raise AssertionError('Forecast probability/data verification failed')
        atomic_json(output/'worker_result.json',metrics)
        return 0
    except Exception as exc:
        atomic_json(output/'worker_failure.json',{'verification_failed':isinstance(exc,AssertionError) or 'verif' in str(exc).lower(),
            'type':type(exc).__name__,'error':str(exc)})
        traceback.print_exc()
        return 2


if __name__=='__main__':
    raise SystemExit(main())
