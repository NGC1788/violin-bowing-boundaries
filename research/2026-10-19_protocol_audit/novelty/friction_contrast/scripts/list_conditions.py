import sys, os, glob, csv
import numpy as np
root = sys.argv[1]
rows = []
for p in glob.glob(os.path.join(root, 'bowing_parameters_*_*.csv')):
    n, k = os.path.basename(p)[:-4].split('_')[-2:]
    with open(p) as f:
        r = list(csv.DictReader(f))[0]
    fr = os.path.join(root, f'friction_velocity_reconstruction_{n}_{k}.csv')
    a = np.loadtxt(fr, delimiter=',', skiprows=1)
    rows.append((int(n), int(k), float(r['beta']), float(r['Fb']), float(r['v0']), a.shape[0], a[0, -1], a[-1, -1]))
rows.sort()
for r in rows:
    print('%3d %d beta=%.4f Fb=%.3f v0=%.3f N=%6d t=[%.2f, %.2f]' % r)
