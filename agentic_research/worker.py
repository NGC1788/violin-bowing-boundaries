"""Fixed subprocess entrypoint: no dynamic model code or shell commands from candidates."""
import argparse
import json
from pathlib import Path
from control import atomic_json, ControlError

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--request',required=True);ap.add_argument('--run-id',required=True)
    a=ap.parse_args();request=json.loads(Path(a.request).read_text())
    if request['run_id']!=a.run_id:raise ControlError('request identity mismatch')
    from harness.experiment import run_experiment
    config={**request['config'],'id':a.run_id}
    metrics=run_experiment(config,request['data_path'],request['output_dir'])
    if not isinstance(metrics,dict):raise ControlError('run_experiment must return a dictionary')
    atomic_json(Path(request['output_dir'])/'worker_result.json',metrics)

if __name__=='__main__':main()
