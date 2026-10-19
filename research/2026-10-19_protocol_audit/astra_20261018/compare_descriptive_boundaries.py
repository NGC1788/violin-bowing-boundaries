"""Same-time, measured-current-force associations; no isolation of settling effects."""
from pathlib import Path
import sys,json
import numpy as np
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'timeres/audit_20261017'))
import window_audit as wa
import boundary_snapshot_audit as bs
from history_prediction_audit import STEMS

def main():
    output={'scope':'Descriptive operational force-grid boundaries, conditional on the full protocol; not equilibrium or causal exponents',
            'censoring_assumption':'Absent K2 run is treated as right-censored; no proof of a physical single threshold above the grid.',
            'settings':[]}
    for tag,stem in STEMS.items():
        path=HERE.parent/f'timeres/features/{stem}.csv.gz'
        data=wa.load_features(path);levels,grid=wa.recover_grid(data)
        H=wa.vector_labels(data,.4063327834010124,{str(f):98.08 for f in data['folder_names']})
        wm=np.isclose(data['window_s'],.2)
        ref=bs.extract(data,levels,H,wm&np.isclose(data['t'],0))
        for t in (0.,.1,.2,.3):
            values=bs.common_complete(bs.extract(data,levels,H,wm&np.isclose(data['t'],t)),ref,3)
            if len(values)!=120:raise ValueError('incomplete common 40-position support across three speeds')
            r={'specimen':tag,'window_start_s':t,'decision_end_s':t+.2,
               'summary':bs.summarize(values),'fit':bs.fit(values),
               'position_cluster_sensitivity':bs.bootstrap_columns(values,200),
               'feature_sha256':wa.sha(path)}
            output['settings'].append(r)
            print(tag,t,r['fit']['speed_exp'],r['fit']['beta_exp'],flush=True)
    output['source_sha256']=wa.sha(__file__)
    output['likelihood_implementation_sha256']=wa.sha(bs.__file__)
    (HERE/'results/descriptive_boundaries.json').write_text(json.dumps(output,indent=2,allow_nan=False)+'\n')

if __name__=='__main__':main()
