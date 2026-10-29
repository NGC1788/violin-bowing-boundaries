"""Command line for a bounded, content-addressed research campaign."""
import argparse
import json
from pathlib import Path
import sys
import control

def main(argv=None):
    ap=argparse.ArgumentParser();sub=ap.add_subparsers(dest='command',required=True)
    for name in ('init','freeze','enqueue','run','status','stop','resume','report'):
        p=sub.add_parser(name);p.add_argument('--campaign',required=True);p.add_argument('--json',action='store_true',help='JSON is always the output format')
        if name=='init':p.add_argument('--data',required=True);p.add_argument('--protocol',required=True)
        if name=='enqueue':p.add_argument('--config',required=True)
        if name=='run':p.add_argument('--max-jobs',type=int,default=1);p.add_argument('--max-seconds',type=float,default=2700)
    a=ap.parse_args(argv)
    try:
        if a.command=='init':r=control.init(a.campaign,a.data,a.protocol)
        elif a.command=='freeze':r=control.freeze(a.campaign)
        elif a.command=='enqueue':r=control.enqueue(a.campaign,json.loads(Path(a.config).read_text()))
        elif a.command=='run':r=control.run(a.campaign,a.max_jobs,a.max_seconds)
        elif a.command in ('stop','resume'):r=control.set_stop(a.campaign,a.command=='stop')
        elif a.command=='status':r=control.status(a.campaign)
        else:r=control.report(a.campaign)
        print(json.dumps(r,ensure_ascii=False,indent=2,allow_nan=False))
        return 0
    except (control.ControlError,ValueError,OSError) as e:
        print(f'ERROR: {e}',file=sys.stderr);return 2

if __name__=='__main__':raise SystemExit(main())
