"""Audit legacy objective discrepancies and render the fixed-plan replay, without tuning."""
import hashlib, json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import replay as R

HERE=R.HERE


def main():
    o=json.loads((HERE/'results.json').read_text())
    audit={'legacy_objective_comparisons':[], 'corrected_target_metrics':{}, 'observed_trim':{}, 'nonpositive_force':{}}
    diffs=[]
    for s,e in o['sessions'].items():
        for feature,f in e['features'].items():
            for a,old,new in zip(R.GRIDS[feature],f['observed'],f['observed_refit']):
                if new is not None and abs(old-new)>.02:
                    diffs.append((s,feature,a,old,new))
            for qset in f['observed_quantiles']:
                for q in qset:
                    z=audit['observed_trim'].setdefault(feature,{'fractions':[],'multiple_values_at_threshold':0,'threshold_hits':0})
                    z['fractions'].append(q['fraction'])
                    z['multiple_values_at_threshold']+=int(q['at_threshold']>1)
                    z['threshold_hits']+=int(q['at_threshold']>0)
    for s,feature,a,old,new in diffs:
        st=R.LS.STEM[s];stem,rep=st if isinstance(st,tuple) else (st,None)
        raw,_=R.T.read2(stem,rep);d=R.LS.build(raw,R.T.BASE)
        x=np.array([r[4 if feature=='periodicity' else 7] for r in raw[0]])
        h=R.T.tilted(x,d['beta'],d['folder'],d['H'],a,.15,feature in R.UPPER)
        rows=R.LS.C.intervals(dict(d,H=h),.2)
        p=R.LS.C.censored_fit(rows,lambda r:(np.array([1.,r['lv'],r['lb']]),0.),'lower')
        sf=R.ST.fit(rows);warm=R.ST.fit(rows,p)
        X,lo,hi=R.ST.arrays(rows)
        old_stable_nll,grad=R.ST.objective(p,X,lo,hi)
        audit['legacy_objective_comparisons'].append({'session':s,'feature':feature,'a':a,
            'stored_delta':old,'stable_delta':new,'legacy_params_recomputed':p.tolist(),
            'legacy_params_evaluated_stable_nll':float(old_stable_nll),'legacy_stable_gradient_max':float(max(abs(grad))),
            'stable_fit':sf,'stable_fit_initialized_at_legacy':warm,
            'stable_nll_improvement':float(old_stable_nll-sf['nll'])})
    corrected={m:[] for m in ('M0','M1')}
    for feature in R.GRIDS:
        selected=[e['features'][feature] for e in o['sessions'].values() if feature in e['features']]
        yy=np.array([f['observed_refit'] for f in selected],float)
        fr={'observed_mean':yy.mean(0).tolist(),'observed_zero':R.T.zero_cross(R.GRIDS[feature],yy.mean(0)),'models':{}}
        for m in ('M0','M1'):
            pred=np.array([f['models'][m]['mean_delta'] for f in selected])
            err=pred-yy
            fr['models'][m]={'mae':float(abs(err).mean()),'rmse':float(np.sqrt(np.mean(err**2)))}
            if feature!='B1_corner':corrected[m].extend(err.ravel().tolist())
        audit['corrected_target_metrics'][feature]=fr
    audit['corrected_target_overall']={m:{'mae':float(np.mean(np.abs(e))),'rmse':float(np.sqrt(np.mean(np.array(e)**2)))} for m,e in corrected.items()}
    red=1-audit['corrected_target_overall']['M1']['mae']/audit['corrected_target_overall']['M0']['mae']
    audit['corrected_target_gate']={'mae_reduction':red,'pass':bool(audit['corrected_target_overall']['M1']['mae']<=.15 and red>=.25 and all(z['models']['M1']['mae']<=.25 for k,z in audit['corrected_target_metrics'].items() if k!='B1_corner'))}
    for feature,z in audit['observed_trim'].items():
        f=z.pop('fractions');z['fraction_min']=min(f);z['fraction_max']=max(f)
    # Unique invalid-force windows are counted once per session, not repeatedly per feature.
    for s in o['sessions']:
        st=R.LS.STEM[s];stem,rep=st if isinstance(st,tuple) else (st,None)
        raw,_=R.T.read2(stem,rep);d=R.LS.build(raw,R.T.BASE)
        mask=d['H'] & (~np.isfinite(d['force']) | (d['force']<=0))
        audit['nonpositive_force'][s]={'n_base_H':int(d['H'].sum()),'n_force_invalid':int(mask.sum()),
            'keys':[(str(d['folders'][d['folder'][i]]),int(d['trial'][i]),float(d['force'][i])) for i in np.flatnonzero(mask)]}
    audit['provenance']={str(p.relative_to(R.ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [__import__('pathlib').Path(__file__),HERE/'results.json']}
    (HERE/'audit.json').write_text(json.dumps(audit,indent=2))

    names={'amplitude':'Amplitude std.','flyback_height':'Flyback height','periodicity':'Periodicity',
           'second_flyback_ratio':'Second-flyback ratio','B1_corner':'Corner width (B1 only)'}
    fig,axs=plt.subplots(2,3,figsize=(14,8.3));styles={'M0':('#828282','--'),'M1':('#2166ac','-')}
    for ax,(feature,name) in zip(axs.flat,names.items()):
        z=o['summary']['features'][feature];a=np.array(z['a']);n=z['n_sessions']
        ax.plot(a,z['observed_mean'],'o-',color='black',lw=1.7,ms=4,label='Stored observed curve')
        if max(abs(np.array(z['observed_mean'])-np.array(audit['corrected_target_metrics'][feature]['observed_mean'])))>.001:
            ax.plot(a,audit['corrected_target_metrics'][feature]['observed_mean'],'x:',color='#b2182b',lw=1.1,label='Stable-likelihood refit')
        for m,(color,ls) in styles.items():
            ff=[e['features'][feature]['models'][m] for e in o['sessions'].values() if feature in e['features']]
            mc=np.sqrt(np.sum(np.array([f['mc_se'] for f in ff])**2,axis=0))/n
            y=np.array(z['models'][m]['mean_delta'])
            ax.plot(a,y,ls,color=color,lw=1.8,label=m+(' + force' if m=='M1' else ' beta only'))
            ax.fill_between(a,y-1.96*mc,y+1.96*mc,color=color,alpha=.15)
        ax.axhline(0,color='#aaa',lw=.7);ax.set_title(name+f' (n={n})',fontsize=12)
        ax.set_xlabel('Threshold tilt a');ax.set_ylabel('Change in fitted beta exponent')
        ax.grid(alpha=.15)
    ax=axs.flat[-1];xx=np.arange(4);width=.36
    for j,(m,(color,_)) in enumerate(styles.items()):
        yy=[o['summary']['features'][f]['models'][m]['mae'] for f in list(names)[:4]]
        ax.bar(xx+(j-.5)*width,yy,width,color=color,label=m)
    ax.set_xticks(xx,['Amplitude','Height','Periodicity','Ratio'],rotation=20)
    ax.set_ylabel('MAE across session × tilt cells');ax.set_title('Error against stored observed curves')
    ax.axhline(.25,color='#b2182b',ls=':',lw=1)
    ax.text(.5,.253,'Per-feature gate = 0.25',color='#b2182b',fontsize=9)
    ax.set_ylim(0,.27)
    handles,labels=axs.flat[2].get_legend_handles_labels();fig.legend(handles,labels,loc='lower center',ncol=4,bbox_to_anchor=(.5,.045),frameon=False)
    fig.suptitle('Gaussian feature replay explains much of criterion-induced exponent drift',fontsize=16)
    fig.text(.5,.013,'Post hoc, window-force axis; 20 replays. Shading: ±1.96 Monte Carlo SE of the mean, not a population confidence interval.',ha='center',fontsize=10)
    fig.tight_layout(rect=(0,.09,1,.95));fig.savefig(HERE/'response_curves.png',dpi=170);fig.savefig(HERE/'response_curves.pdf');plt.close(fig)
    print(json.dumps({'legacy_differences':audit['legacy_objective_comparisons'],'corrected':audit['corrected_target_overall'],'gate':audit['corrected_target_gate'],'observed_trim':audit['observed_trim']},indent=2))


if __name__=='__main__':main()
