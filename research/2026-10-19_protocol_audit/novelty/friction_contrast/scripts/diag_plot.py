import sys, os, csv
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
root, out = sys.argv[1], sys.argv[2]
cases = [(30, 1), (30, 0), (14, 1), (10, 1), (17, 1), (39, 1), (47, 1), (41, 1)]
fig, axes = plt.subplots(len(cases), 3, figsize=(15, 2.6 * len(cases)))
for i, (n, k) in enumerate(cases):
    a = np.loadtxt(os.path.join(root, f'friction_velocity_reconstruction_{n}_{k}.csv'), delimiter=',', skiprows=1)
    with open(os.path.join(root, f'bowing_parameters_{n}_{k}.csv')) as f:
        p = list(csv.DictReader(f))[0]
    Fb, v0, beta = float(p['Fb']), float(p['v0']), float(p['beta'])
    t, Fr, V = a[:, 4], a[:, 0], a[:, 1]
    sel = t < t[0] + 60
    axes[i, 0].plot(t[sel], V[sel]); axes[i, 0].axhline(v0, color='r', lw=0.5); axes[i, 0].set_ylabel('V_rec')
    axes[i, 0].set_title(f'{n}_{k} beta={beta:.3f} Fb={Fb:.2f} v0={v0}')
    axes[i, 1].plot(t[sel], Fr[sel] / Fb); axes[i, 1].set_ylabel('Fr/Fb')
    axes[i, 2].plot(v0 - V, Fr / Fb, lw=0.3); axes[i, 2].set_xlabel('v0 - V (sliding)'); axes[i, 2].set_ylabel('Fr/Fb')
plt.tight_layout(); plt.savefig(out, dpi=80)
