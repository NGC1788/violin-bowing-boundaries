"""Coverage audit of the downloaded friction-reconstruction dataset (Zenodo 15879128, Lampis/IWK), offline only.
(Claude, 2026-10-19). Reads the untrusted data folder as data only (numbers from .csv/.mat; MATLAB code read as text).

Questions: which string(s) and which (beta, v0, Fb) conditions do the friction reconstructions cover, and which of
our boundary diagrams does that string correspond to?
Usage: python -I coverage.py <data_untrusted> <prev_results_dir> <out.json>
"""
import csv
import glob
import json
import os
import re
import sys

import numpy as np
from scipy.io import loadmat

POMA_T2 = {  # Lampis et al. POMA 58, 035013 (2025) Table 1, tension group T2: T (N), Z (kg/s), EI (1e-4 N m^2), d (mm)
    "A_T2": (163.03, 1.1884, 3.969, 1.001), "B_T2": (163.23, 1.1899, 7.295, 0.973),
    "C_T2": (153.04, 1.1154, 3.969, 1.083), "D_T2": (164.82, 1.2021, 5.766, 1.201)}
POMA_T1 = {"A1": (145.31, 1.0592, 3.032, 0.947), "B1": (150.87, 1.0998, 4.397, 0.927),
           "C1": (144.40, 1.0525, 3.142, 1.077), "D1": (142.67, 1.0398, 5.246, 1.123)}


def mat_struct(path, name):
    d = loadmat(path, squeeze_me=True, struct_as_record=False)[name]
    return {f: (getattr(d, f).tolist() if hasattr(getattr(d, f), 'tolist') else getattr(d, f)) for f in d._fieldnames}


def main():
    root, prev, out = sys.argv[1:4]
    om = os.path.join(root, 'output_measurements')
    rec = os.path.join(om, 'reconstructions')
    res = {}
    # 1. reconstruction strokes
    rows = []
    for p in glob.glob(os.path.join(rec, 'bowing_parameters_*_*.csv')):
        n, k = map(int, os.path.basename(p)[:-4].split('_')[-2:])
        with open(p) as f:
            r = list(csv.DictReader(f))[0]
        rows.append(dict(n=n, k=k, beta=float(r['beta']), Fb=float(r['Fb']), v0=float(r['v0'])))
    rows.sort(key=lambda r: (r['n'], r['k']))
    others = sorted(set(os.listdir(rec)) - {os.path.basename(p) for p in glob.glob(os.path.join(rec, '*_*_*.csv'))
                                             if re.search(r'(bowing_parameters|friction_velocity_reconstruction)_\d+_\d\.csv$', p)})
    m15 = {(r['n'], r['k']): r for r in json.load(open(os.path.join(prev, 'friction_metrics.json')))}
    m30 = {(r['n'], r['k']): r for r in json.load(open(os.path.join(prev, 'friction_metrics_thr0.30.json')))}
    for r in rows:
        key = (r['n'], r['k'])
        r['helmholtz_both_thr'] = bool(m15[key].get('helmholtz') and m30[key].get('helmholtz'))
    strokes = sorted({r['n'] for r in rows})
    orphan = os.path.join(rec, 'friction_velocity_reconstruction_1.csv')
    if os.path.exists(orphan):
        with open(orphan) as f:
            hdr = f.readline().strip()
        a = np.loadtxt(orphan, delimiter=',', skiprows=1)
        res['orphan_file'] = dict(header=hdr, n_samples=int(a.shape[0]), note='no time column, no bowing_parameters file -> Fb, v0, beta unknown; not usable for mu')
        print(f"orphan file friction_velocity_reconstruction_1.csv: header {hdr}, {a.shape[0]} samples, no bowing parameters -> unusable for mu = Fr/Fb")
    res['n_files'] = len(rows); res['n_strokes'] = len(strokes); res['other_files'] = others
    # conditions (beta rounded to the pluck positions, v0)
    cond = {}
    for r in rows:
        if r['k'] != 1:
            continue
        key = f"{r['beta']:.4f}_{r['v0']:.2f}"
        c = cond.setdefault(key, dict(beta=r['beta'], v0=r['v0'], Fb=[], H=[]))
        c['Fb'].append(round(r['Fb'], 3)); c['H'].append(r['helmholtz_both_thr'])
    for c in cond.values():
        o = np.argsort(c['Fb']); c['Fb'] = [c['Fb'][i] for i in o]; c['H'] = [c['H'][i] for i in o]
    res['conditions'] = cond
    res['strokes'] = rows
    print(f"files {len(rows)} ({len(strokes)} strokes x k in {{0: attack ~316 ms, 1: steady ~100 ms}}); extra files: {others}")
    print(f"{'beta':>7s} {'v0':>5s}  n  Fb (steady window; H = Helmholtz at both slip thresholds 0.15/0.30)")
    for key in sorted(cond, key=lambda s: (float(s.split('_')[1]), float(s.split('_')[0]))):
        c = cond[key]
        print(f"{c['beta']:7.4f} {c['v0']:5.2f} {len(c['Fb']):2d}  " + ' '.join(f"{f:.2f}{'H' if h else 'x'}" for f, h in zip(c['Fb'], c['H'])))
    nH = sum(r['helmholtz_both_thr'] for r in rows if r['k'] == 1)
    print(f"steady windows: {sum(r['k'] == 1 for r in rows)}, Helmholtz at both thresholds: {nH}")
    Fs = np.array([r['Fb'] for r in rows if r['k'] == 1])
    print(f"bow force range {Fs.min():.2f}-{Fs.max():.2f} N; speeds {sorted({r['v0'] for r in rows})}; "
          f"beta values {sorted({round(r['beta'], 4) for r in rows})}")

    # 2. string identity
    sp = {}
    for p in sorted(glob.glob(os.path.join(om, 'string_properties', '*.mat')), key=lambda s: int(os.path.basename(s)[:-4])):
        i = int(os.path.basename(p)[:-4])
        sp[i] = mat_struct(p, 'stringProperties')
    pl = {}
    for p in sorted(glob.glob(os.path.join(om, 'pluck_best_params', '*.mat')), key=lambda s: int(os.path.basename(s)[:-4])):
        i = int(os.path.basename(p)[:-4])
        pl[i] = mat_struct(p, 'pluckProp')
    T2 = [i for i in sp if 14 <= i <= 24]
    d_mm = sorted({round(sp[i]['diam'] * 1000, 5) for i in T2})
    EI = np.array([sp[i]['curr_B'] for i in T2]) * 1e4  # 1e-4 N m^2 (detection_inharmonicity: 2 c T L^2/(a pi^2))
    EIp = np.array([pl[i]['B_est'] for i in T2 if i in pl]) * 1e4
    Tp = np.array([pl[i]['T_est'] for i in T2 if i in pl])
    Zs = np.array([sp[i]['Z'] for i in T2])
    f0 = np.array([sp[i]['f0_meas'] for i in T2])
    ident = dict(diam_mm=d_mm, EI_pluck_detect=[float(np.median(EI)), float(EI.min()), float(EI.max())],
                 EI_pluck_fit=[float(np.median(EIp)), float(EIp.min()), float(EIp.max())],
                 T_assumed=sorted({sp[i]['T'] for i in T2}), T_pluck_fit=[float(np.median(Tp)), float(Tp.min()), float(Tp.max())],
                 Z=[float(np.median(Zs)), float(Zs.min()), float(Zs.max())], f0=[float(np.median(f0)), float(f0.min()), float(f0.max())])
    print('\nstring identity (string_properties/pluck_best_params 14-24, i.e. the positions of all reconstructions):')
    print(f"  diameter {d_mm} mm; EI {ident['EI_pluck_detect'][0]:.2f} [{ident['EI_pluck_detect'][1]:.2f}, {ident['EI_pluck_detect'][2]:.2f}] e-4 N m^2 (inharmonicity), "
          f"{ident['EI_pluck_fit'][0]:.2f} [{ident['EI_pluck_fit'][1]:.2f}, {ident['EI_pluck_fit'][2]:.2f}] (pluck fit); T assumed {ident['T_assumed']} N, "
          f"pluck-fit T {ident['T_pluck_fit'][0]:.1f} [{ident['T_pluck_fit'][1]:.1f}, {ident['T_pluck_fit'][2]:.1f}]; Z {ident['Z'][0]:.4f}; f0 {ident['f0'][0]:.2f} Hz")
    dist = {}
    for tag, (T, Z, ei, d) in {**POMA_T2, **POMA_T1}.items():
        dist[tag] = dict(d_rel=abs(d_mm[0] - d) / d, EI_rel=abs(ident['EI_pluck_detect'][0] - ei) / ei, T_rel=abs(ident['T_pluck_fit'][0] - T) / T)
        print(f"  vs POMA {tag:5s}: |d| {100 * dist[tag]['d_rel']:5.1f} %  |EI| {100 * dist[tag]['EI_rel']:5.1f} %  |T(pluck fit)| {100 * dist[tag]['T_rel']:5.1f} %")
    ident['distance_to_POMA'] = dist
    # MATLAB main script: recording folder / session id (text only)
    txt = open(os.path.join(os.path.dirname(root.rstrip('/')), 'raw', 'main_measurements_reconstruction.m'), errors='replace').read()
    folders = sorted(set(re.findall(r"raw_data\\([^\\]+)\\", txt)))
    sess = sorted(set(re.findall(r"SChelleng_(\d{6})_(\d{6})", txt)))
    ident['recording_folders'] = folders; ident['bowing_sessions'] = sess
    print(f"  recording folders named in main script: {folders}; bowing session ids (ddmmyy_hhmmss): {sess}")
    other = {i: dict(diam_mm=sp[i]['diam'] * 1000, T=sp[i]['T'], EI=sp[i]['curr_B'] * 1e4, beta=sp[i]['beta']) for i in sp if i not in T2}
    ident['other_string_properties'] = other
    print(f"  other string_properties entries (no reconstruction CSVs): {json.dumps(other)}")
    res['identity'] = ident
    json.dump(res, open(out, 'w'), indent=1)


if __name__ == '__main__':
    main()
