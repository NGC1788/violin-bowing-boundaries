"""Descriptive scientific figures; every primary folder is shown."""
from pathlib import Path
import json,gzip,math
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from empirical import prepare
HERE=Path(__file__).resolve().parent
plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False})


def main():
    result=json.loads((HERE/'results.json').read_text())
    primary=[r for r in result['records'] if r['k']==2 and r['epsilon']==.2 and r['mode']=='both']
    sessions=list(dict.fromkeys(r['session'] for r in primary))
    fig,axs=plt.subplots(1,2,figsize=(14,5.6),sharey=True)
    speeds=[.05,.10,.20];colors=['#315ca8','#c06424','#27815c'];markers=['o','s','^']
    for ax,p in zip(axs,[-2.,-1.2]):
        for v,color,marker in zip(speeds,colors,markers):
            rr=[r for r in primary if r['p']==p and abs(r['speed']-v)<.01]
            offset=(speeds.index(v)-1)*.23
            good=[r for r in rr if r['restore']['feasible']]
            bad=[r for r in rr if not r['restore']['feasible']]
            ax.scatter([sessions.index(r['session'])+offset for r in good],
                       [100*r['restore']['cost']/r['n_windows'] for r in good],s=49,c=color,marker=marker,label=f'{v:.2f} m/s')
            ax.scatter([sessions.index(r['session'])+offset for r in bad],[-.8]*len(bad),s=44,c=color,marker='x')
        ax.axhline(0,c='0.7',lw=.7);ax.axhline(1,c='0.7',lw=.7,ls=':')
        ax.set_xticks(range(len(sessions)),sessions,rotation=48,ha='right')
        ax.set_title(f'Fixed beta exponent p = {p:g}')
        ax.grid(axis='y',alpha=.15);ax.set_ylim(-1.5,16.5)
    axs[0].set_ylabel('Minimum restoration edits / eligible windows (%)')
    axs[0].legend(frameon=False,loc='upper left')
    axs[1].text(.02,.94,'x below zero: force-grid support insufficient\n(not a large or infinite robustness estimate)',
                transform=axs[1].transAxes,va='top',fontsize=10)
    fig.suptitle('Exact label-edit costs depend on both the boundary labels and the measured force range',fontsize=14)
    fig.text(.5,.015,'K = 2; log-force tolerance 0.2; originally finite positive boundary columns only. All 39 folders initially incompatible.\nArbitrary independent label edits; descriptive sensitivity, not classification error rates or physical model selection.',ha='center',fontsize=10)
    fig.tight_layout(rect=(0,.10,1,.94));fig.savefig(HERE/'fig_costs.png',dpi=190);plt.close(fig)

    # A prescribed central example: lower median restoration fraction among
    # feasible p=-2 folders, selected for illustration rather than validation.
    feasible=sorted([r for r in primary if r['p']==-2 and r['restore']['feasible']],key=lambda r:(r['flip_fraction_eligible'],r['session'],r['folder']))
    selected=feasible[(len(feasible)-1)//2]
    with gzip.open(HERE/'input_columns.json.gz','rt') as f:data=json.load(f)
    cols=[c for c in data['columns'] if c['session']==selected['session'] and c['folder']==selected['folder']]
    kept,_=prepare(cols,2);cols=[c['col'] for c in kept]
    key=f"{selected['session']}|{selected['folder']}|K2|p-2.0|eps0.2|both"
    w=result['witnesses'][key]['restore'];lookup={(c['column'],c['trial']):c['after'] for c in w['changes']}
    a=sum(w['intersection'])/2 # interior witness for the same optimal states
    fig,axs=plt.subplots(1,2,figsize=(12,5.3),sharex=True,sharey=True)
    for ax,changed in zip(axs,[False,True]):
        for col in cols:
            for trial,force,h in zip(col['trial'],col['force'],col['labels']):
                if force<=0:continue
                edited=lookup.get((col['column'],trial),h) if changed else h
                ax.scatter(col['beta'],force,s=13,c='#2468a2' if edited else '#c9c9c9',zorder=2)
                if (col['column'],trial) in lookup:
                    ax.scatter(col['beta'],force,s=68,facecolor='none',edgecolor='#d43d3d',lw=1.2,zorder=4)
        beta=np.geomspace(min(c['beta'] for c in cols),max(c['beta'] for c in cols),150)
        force=np.exp(a)*beta**-2
        ax.plot(beta,force,c='black',lw=1.4,zorder=3)
        ax.fill_between(beta,force*np.exp(-.2),force*np.exp(.2),color='black',alpha=.08)
        ax.set_xscale('log');ax.set_yscale('log');ax.set_xlabel('Relative bow position beta');ax.grid(alpha=.12)
        ax.set_title('Edited labels: compatible' if changed else 'Original labels: incompatible')
    axs[0].set_ylabel('Window-mean bow force (N)')
    fig.suptitle(f"{selected['session']}, {selected['speed']:.2f} m/s: exactly {w['n_flips']} edits ({w['changed_columns']} columns)",fontsize=14)
    fig.text(.5,.01,'Blue = H; gray = non-H; red rings = edited windows. Black curve/tube = one compatible fixed-slope law.\nIllustration chosen by the median edit fraction among supported p=-2 folders; labels are hypothetical, not corrected truth.',ha='center',fontsize=9)
    fig.tight_layout(rect=(0,.1,1,.94));fig.savefig(HERE/'fig_witness.png',dpi=180);plt.close(fig)
    (HERE/'figure_selection.json').write_text(json.dumps({'method':'lower median edit fraction among feasible p=-2 primary folders',
        'session':selected['session'],'folder':selected['folder'],'speed':selected['speed'],'n_flips':w['n_flips'],
        'changed_columns':w['changed_columns'],'interior_intercept':a},indent=2)+'\n')

if __name__=='__main__':main()
