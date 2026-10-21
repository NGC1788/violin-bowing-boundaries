"""Fixed-plan Gaussian feature replay. All data already exposed; explanatory only.

Does not change source labels or source results. Reference tilt curves are outcomes,
never regression inputs. One thread is recommended for this small matrix workload.
"""
from pathlib import Path
import csv, hashlib, importlib.util, json, sys, time
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / 'novelty/tilt_principle'))
import tilt as T
LS = T.LS
spec = importlib.util.spec_from_file_location('lower_law_audit_statistics', ROOT / 'verify_lower_law/statistics.py')
ST = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ST)

GRIDS = {'amplitude': T.A_LOW, 'flyback_height': T.A_LOW,
         'periodicity': [-.6, -.4, -.2, 0., .2, .4, .6],
         'second_flyback_ratio': [-1., -.5, 0., .5, 1.],
         'B1_corner': T.A_UP}
UPPER = {'second_flyback_ratio', 'B1_corner'}
N_REPLAY = 20
SEED = 20261021


def source_targets():
    main = json.loads((ROOT / 'novelty/tilt_principle/results.json').read_text())
    gen = json.loads((ROOT / 'novelty/tilt_principle/results_generalize.json').read_text())
    edge = json.loads((ROOT / 'novelty/tilt_principle/results_edge.json').read_text())
    out = {}
    for s in LS.STEM:
        if s == 'D_T2':
            continue
        out[s] = {'baseline': main['sessions'][s]['base']['window'], 'features': {}}
        for f in ('amplitude', 'flyback_height'):
            out[s]['features'][f] = [main['sessions'][s]['delta'][f'{f}|0.15|{a}']['window'] for a in GRIDS[f]]
        out[s]['features']['periodicity'] = [edge['sessions'][s]['delta'][f'0.15|{a}']['window'] for a in GRIDS['periodicity']]
        out[s]['features']['second_flyback_ratio'] = [gen['sessions'][s]['g1'][f'0.15|{a}']['window'] for a in GRIDS['second_flyback_ratio']]
    out['B1']['features']['B1_corner'] = [main['B1_corner']['delta'][f'0.15|{a}']['window'] for a in GRIDS['B1_corner']]
    return out


class BoundaryCache:
    def __init__(self, d):
        self.template = LS.C.intervals(d, .2)
        self.ids = [np.array(r['idx'], int) for r in self.template]
        self.force = [np.log(np.maximum(d['force'][ix], .001)) for ix in self.ids]

    def rows(self, h):
        rows = []
        for r, ix, f in zip(self.template, self.ids, self.force):
            hh = h[ix]
            run = np.flatnonzero(hh[:-1] & hh[1:])
            if len(run):
                k = run[0]
                lo, hi = (-np.inf, f[0]) if k == 0 else (f[k - 1], f[k])
            else:
                lo, hi = f[-1], np.inf
            rows.append({'lv': r['lv'], 'lb': r['lb'], 'lower': (lo, hi)})
        return rows


def safe_fit(rows, init=None):
    lo, hi = np.array([r['lower'] for r in rows]).T
    bad = lo >= hi
    if bad.any():
        return {'success': False, 'reason': 'zero_or_reversed_interval', 'n_bad': int(bad.sum())}
    try:
        f = ST.fit(rows, init)
    except (ValueError, AssertionError, np.linalg.LinAlgError) as e:
        return {'success': False, 'reason': 'design_or_fit_error', 'message': str(e)}
    if not f['success'] or not np.isfinite(f['params']).all() or f['grad_max'] > 2e-3:
        return dict(f, success=False, reason='optimizer_or_gradient_failure')
    return f


def regression(d, x, model):
    # Missing/nonpositive original features stay ineligible for simulated trimming.
    mask = d['H'] & np.isfinite(x) & (x > 0)
    mask_force = np.isfinite(d['force']) & (d['force'] > 0)
    excluded_force = int((mask & ~mask_force).sum()) if model == 'M1' else 0
    if model == 'M1':
        mask &= mask_force
    names = [f'folder_{g}' for g in np.unique(d['folder'])] + ['log_beta']
    X = np.column_stack([(d['folder'] == g).astype(float) for g in np.unique(d['folder'])] + [np.log(d['beta'])])
    if model == 'M1':
        X = np.column_stack([X, np.log(np.maximum(d['force'], np.finfo(float).tiny))])
        names.append('log_window_force')
    y = np.log(x[mask]); XX = X[mask]
    coef, _, rank, _ = np.linalg.lstsq(XX, y, rcond=None)
    assert rank == X.shape[1] and len(y) > rank
    residual = y - XX @ coef
    sigma = float(np.sqrt(np.dot(residual, residual) / (len(y) - rank)))
    return X @ coef, mask, {'coef': dict(zip(names, coef.tolist())), 'sigma': sigma, 'n': len(y),
                          'rank': int(rank), 'excluded_nonpositive_force': excluded_force,
                          'r_squared': float(1 - np.dot(residual, residual) / np.sum((y - y.mean())**2))}


def trim_log(d, logx, valid, a, upper):
    h = d['H'].copy(); score = logx - a * np.log(d['beta'])
    records = []
    for g in np.unique(d['folder']):
        mask = (d['folder'] == g) & d['H'] & valid & np.isfinite(score)
        if not mask.any():
            continue
        q = np.quantile(score[mask], .85 if upper else .15)
        remove = mask & ((score > q) if upper else (score < q))
        h[remove] = False
        records.append({'folder': int(g), 'eligible': int(mask.sum()), 'removed': int(remove.sum()),
                        'at_threshold': int((mask & (score == q)).sum()), 'fraction': float(remove.sum() / mask.sum())})
    return h, records


def checks():
    # Independently check the likelihood gradient and known finite-interval recovery.
    rng = np.random.default_rng(47)
    X = np.c_[np.ones(400), rng.normal(size=(400, 2))]
    beta = np.array([.3, .8, -1.2]); y = X @ beta + rng.normal(0, .15, len(X))
    rows = [{'lv': xx[1], 'lb': xx[2], 'lower': (yy-.01, yy+.01)} for xx, yy in zip(X,y)]
    f = ST.fit(rows)
    assert f['success'] and max(abs(np.array(f['params'][:3])-beta)) < .04
    p = np.r_[beta, np.log(.15)]; lo,hi = np.array([r['lower'] for r in rows]).T
    g = ST.objective(p, X, lo, hi)[1]
    num = np.array([(ST.objective(p + np.eye(4)[j]*1e-5, X, lo, hi)[0] - ST.objective(p - np.eye(4)[j]*1e-5, X, lo, hi)[0])/2e-5 for j in range(4)])
    assert max(abs(g-num)) < 1e-4
    d = {'H': np.ones(10,bool), 'folder': np.zeros(10,int), 'beta': np.ones(10)}
    _, tied = trim_log(d, np.zeros(10), np.ones(10,bool), 0., False)
    assert tied[0]['removed'] == 0  # Quantile trimming does not promise exactly 15% with ties.
    return {'synthetic_max_coefficient_error': float(max(abs(np.array(f['params'][:3])-beta))),
            'gradient_max_error': float(max(abs(g-num))), 'ties_remove_zero_of_ten': True}


def aggregate(out):
    names = ['amplitude', 'flyback_height', 'periodicity', 'second_flyback_ratio']
    res = {'features': {}, 'models': {}}
    all_errors = {m: [] for m in ('M0','M1')}
    for feat in names + ['B1_corner']:
        ss = [s for s,e in out['sessions'].items() if feat in e['features']]
        if not ss:
            continue
        observed = np.array([out['sessions'][s]['features'][feat]['observed'] for s in ss])
        ff = {'n_sessions': len(ss), 'a': GRIDS[feat], 'observed_mean': observed.mean(0).tolist(),
              'observed_zero': T.zero_cross(GRIDS[feat], observed.mean(0)), 'models': {}}
        for model in ('M0', 'M1'):
            predicted = np.array([out['sessions'][s]['features'][feat]['models'][model]['mean_delta'] for s in ss])
            err = predicted - observed
            mean = np.mean(predicted, axis=0)
            z = T.zero_cross(GRIDS[feat], mean)
            ff['models'][model] = {'mae': float(np.mean(np.abs(err))), 'rmse': float(np.sqrt(np.mean(err**2))),
                                   'mean_delta': mean.tolist(), 'zero': z,
                                   'zero_error': None if z is None or ff['observed_zero'] is None else abs(z-ff['observed_zero'])}
            if feat != 'B1_corner':
                all_errors[model].extend(err.ravel().tolist())
        res['features'][feat] = ff
    for m, ee in all_errors.items():
        e = np.array(ee)
        res['models'][m] = {'n_cells': len(e), 'mae': float(np.mean(abs(e))), 'rmse': float(np.sqrt(np.mean(e**2)))}
    if len(res['features']) >= 4:
        reduction = 1-res['models']['M1']['mae']/res['models']['M0']['mae']
        pass_all = res['models']['M1']['mae'] <= .15 and reduction >= .25 and all(res['features'][f]['models']['M1']['mae'] <= .25 for f in names)
        res['mae_reduction'] = reduction
        res['fixed_gate_pass'] = bool(pass_all)
        res['gate_including_B1_supplement'] = bool(pass_all and res['features'].get('B1_corner',{}).get('models',{}).get('M1',{}).get('mae',np.inf) <= .25)
    return res


def main():
    start = time.monotonic(); targets = source_targets()
    out = {'post_hoc': True, 'seed': SEED, 'replays': N_REPLAY, 'excluded': {'D_T2': 'zero-width 1 mN-clamped intervals in window-force baseline'},
           'checks': checks(), 'sessions': {}, 'failures': [], 'fit_counts': {'baseline':0,'observed_verification':0,'replay':0},
           'quantile_rule': 'numpy linear quantile; strict < (lower) or > (upper); rounded/tied counts can differ from 15%; no exact-rate claim'}
    rng = np.random.default_rng(SEED)
    source_files = [Path(__file__), ROOT/'novelty/criterion_identifiability/PLAN.md',
                    ROOT/'verify_lower_law/statistics.py', ROOT/'novelty/label_sensitivity/label_sweep.py',
                    ROOT/'novelty/tilt_principle/tilt.py', ROOT/'final/compare.py']
    source_files += [ROOT/'novelty/tilt_principle'/f for f in ['results.json','results_edge.json','results_generalize.json']]
    for sess, target in targets.items():
        st = LS.STEM[sess]; stem, rep = st if isinstance(st,tuple) else (st,None)
        raw, fh = T.read2(stem,rep); d = LS.build(raw,T.BASE); cache = BoundaryCache(d)
        source_files.append(LS.D.FEAT/f'{stem}.csv.gz')
        baseline = safe_fit(cache.rows(d['H'])); out['fit_counts']['baseline'] += 1
        assert baseline['success'], (sess,baseline)
        diff = abs(baseline['params'][2] - target['baseline'])
        assert diff < .02, (sess,diff)
        e = {'baseline': baseline, 'stored_baseline': target['baseline'], 'baseline_difference': diff, 'features': {}}
        out['sessions'][sess] = e
        XX = {'amplitude': np.array([r[3] for r in raw[0]]), 'flyback_height': fh,
              'periodicity': np.array([r[4] for r in raw[0]]), 'second_flyback_ratio': np.array([r[7] for r in raw[0]])}
        if sess == 'B1':
            path = ROOT/'novelty/label_sensitivity/b1_flyback/b1_flyback.csv'; source_files.append(path)
            rec = {(r['folder'],int(r['trial'])): LS.fnum(r['flyback_ms']) for r in csv.DictReader(path.open())}
            XX['B1_corner'] = np.array([rec.get(r[0],np.nan) for r in raw[0]])
        for feature, x in XX.items():
            aa = GRIDS[feature]; upper = feature in UPPER
            ef = {'observed': target['features'][feature], 'observed_refit': [], 'observed_quantiles': [], 'models': {}}
            e['features'][feature] = ef
            valid = np.isfinite(x) & (x>0)
            logx = np.log(np.where(valid,x,1.))
            for ai,a in enumerate(aa):
                h, qr = trim_log(d,logx,valid,a,upper)
                assert np.array_equal(h,T.tilted(x,d['beta'],d['folder'],d['H'],a,.15,upper))
                # Independent parity check against original boundary extraction for every observed variant.
                rs = cache.rows(h); original = LS.C.intervals(dict(d,H=h),.2)
                assert all(np.array_equal(r['lower'],old['lower']) for r,old in zip(rs,original))
                fitted = safe_fit(rs,baseline['params']); out['fit_counts']['observed_verification'] += 1
                if not fitted['success']:
                    out['failures'].append({'session':sess,'feature':feature,'kind':'observed','a':a,**fitted})
                    ef['observed_refit'].append(None)
                else:
                    ef['observed_refit'].append(fitted['params'][2]-baseline['params'][2])
                ef['observed_quantiles'].append(qr)
            # Pair Monte Carlo residuals between M0/M1 and throughout each tilt grid.
            zz = rng.standard_normal((N_REPLAY,len(x)))
            for model in ('M0','M1'):
                mu, mask, reg = regression(d,x,model)
                curves = []; quants = []; diagnostics = {'n':0,'success':0,'fallbacks':0,'max_gradient':0.,'failure_reasons':{}}
                for j in range(N_REPLAY):
                    sim = mu + reg['sigma']*zz[j]
                    vals = []; qs = []
                    for a in aa:
                        h,qr = trim_log(d,sim,mask,a,upper)
                        f = safe_fit(cache.rows(h),baseline['params']); out['fit_counts']['replay'] += 1
                        diagnostics['n'] += 1
                        if f['success']:
                            diagnostics['success'] += 1; diagnostics['fallbacks'] += int(f['fallback'])
                            diagnostics['max_gradient'] = max(diagnostics['max_gradient'],f['grad_max'])
                            vals.append(f['params'][2]-baseline['params'][2])
                        else:
                            diagnostics['failure_reasons'][f['reason']] = diagnostics['failure_reasons'].get(f['reason'],0)+1
                            out['failures'].append({'session':sess,'feature':feature,'model':model,'replay':j,'a':a,**f})
                            vals.append(np.nan)
                        qs.extend(qr)
                    curves.append(vals); quants.extend(qs)
                curves = np.array(curves)
                ef['models'][model] = {'regression':reg,'delta_replays':curves.tolist(), 'mean_delta':np.nanmean(curves,0).tolist(),
                    'mc_se':(np.nanstd(curves,0,ddof=1)/np.sqrt(np.sum(np.isfinite(curves),0))).tolist(), 'diagnostics':diagnostics,
                    'trim_rate_range':[min(q['fraction'] for q in quants),max(q['fraction'] for q in quants)],
                    'n_quantile_ties':sum(q['at_threshold']>0 for q in quants)}
            print(sess, feature, 'MAE',*[round(float(np.nanmean(abs(np.array(ef['models'][m]['mean_delta'])-ef['observed']))),4) for m in ('M0','M1')],flush=True)
        out['partial_summary'] = aggregate(out)
        (HERE/'results.json').write_text(json.dumps(out,indent=2,allow_nan=True))
        print('SESSION',sess,'seconds',round(time.monotonic()-start,1),flush=True)
    out['summary'] = aggregate(out)
    out['duration_seconds'] = time.monotonic()-start
    out['source_sha256'] = {str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(set(source_files))}
    out['baseline_gate'] = {'max_abs_difference':max(e['baseline_difference'] for e in out['sessions'].values()),'pass':True}
    out['observed_curve_refit_max_difference'] = max(abs(v-t) for e in out['sessions'].values() for f in e['features'].values() for v,t in zip(f['observed_refit'],f['observed']) if v is not None)
    (HERE/'results.json').write_text(json.dumps(out,indent=2,allow_nan=True))
    print(json.dumps(out['summary'],indent=2),flush=True)


if __name__ == '__main__':
    main()
