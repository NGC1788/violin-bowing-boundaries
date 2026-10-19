"""Compute a frozen target descriptor without loading bridge features or labels."""
from pathlib import Path
import numpy as np
import pyarrow.csv as pc

FIELDS = ['folder','trial','window_s','t','force','speed','error']


def from_columns(d, *, reverse_rank=False):
    if any(x not in (None,'') for x in d['error']):
        raise ValueError('Feature errors must be resolved before descriptor extraction')
    chosen = np.isclose(d['window_s'],.1,atol=1e-10,rtol=0)
    folders = sorted(set(d['folder'][chosen]))
    if len(folders)!=3: raise ValueError('Expected three speed folders for one diagram')
    result = {}
    for folder in folders:
        m = chosen & (d['folder']==folder)
        ids = np.unique(d['trial'][m])
        if not np.array_equal(ids,np.arange(1,2001)):
            raise ValueError('Expected acquired trial IDs 1..2000 in every folder')
        delta=[];selected_starts=[];nominal_speed=[]
        for trial in ids:
            rank=(trial-1)%50+1
            if reverse_rank: rank=51-rank
            if rank>20: continue
            idx=np.flatnonzero(m & (d['trial']==trial))
            t=d['t'][idx];force=d['force'][idx];speed=d['speed'][idx]
            if not np.isfinite(np.column_stack((t,force,speed))).all():
                raise ValueError('Nonfinite controls')
            start=np.flatnonzero(t==0)
            if len(start)!=1 or len(np.unique(t))!=len(t):
                raise ValueError('Missing or duplicate descriptor window')
            eligible=np.flatnonzero((t>0)&(speed*t>=.05))
            if not len(eligible): raise ValueError('Five-centimetre support missing')
            j=eligible[np.argmin(t[eligible])]
            delta.append(force[j]-force[start[0]])
            selected_starts.append(float(t[j]));nominal_speed.append(float(speed[start[0]]))
        if len(delta)!=800: raise ValueError('Expected 40 positions times 20 nominal ranks')
        result[str(folder)]={'dec5_pl_N':float(np.median(delta)),'n':len(delta),
            'selected_start_range_s':[min(selected_starts),max(selected_starts)],
            'speed_m_s':float(np.median(nominal_speed))}
    return {'dec5_pl_N':float(np.mean([v['dec5_pl_N'] for v in result.values()])),
            'folders':result,
            'scope':'부호 있는 힘 변화. 음수는 감소. 0.1초 창의 평균값 차이이며, 정확한 순간 5cm 힘 차이가 아니다.'}


def load_descriptor(path, *, reverse_rank=False):
    table=pc.read_csv(Path(path),convert_options=pc.ConvertOptions(include_columns=FIELDS))
    return from_columns({k:table[k].to_numpy(zero_copy_only=False) for k in FIELDS},reverse_rank=reverse_rank)
