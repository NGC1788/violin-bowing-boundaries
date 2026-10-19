import sys, os, glob
import numpy as np
from scipy.io import loadmat
root = sys.argv[1]
def show(path):
    d = loadmat(path, squeeze_me=True, struct_as_record=False)
    out = {}
    for k, v in d.items():
        if k.startswith('__'): continue
        if hasattr(v, '_fieldnames'):
            out[k] = {f: getattr(v, f) for f in v._fieldnames}
        else:
            out[k] = v
    return out
for sub in ['string_properties', 'pluck_best_params']:
    for p in sorted(glob.glob(os.path.join(root, sub, '*.mat')), key=lambda s: int(os.path.basename(s).split('.')[0])):
        print(sub, os.path.basename(p), show(p))
for p in sorted(glob.glob(os.path.join(root, 'bowed_best_params', 'helicoreT2_*.mat')), key=lambda s: int(s.split('_')[-1].split('.')[0]))[:5]:
    print(os.path.basename(p), show(p))
