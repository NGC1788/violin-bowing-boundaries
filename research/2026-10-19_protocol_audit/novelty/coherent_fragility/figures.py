"""Scientific figures from saved exact profiles; no parameter refitting."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
HERE=Path(__file__).resolve().parent
plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False})

def main():
    data=json.loads((HERE/'results.json').read_text());rules={r['name']:r for r in data['rules']}
    cases=[]
    for name,p in [('ratio_primary',-2.),('amplitude_stress',-1.2),('height_stress',-1.2)]:
        rule=rules[name];a=next(x for x in rule['analyses'] if x['p']==p and x['epsilon']==.2)
        local=next(x for x in a['local'] if x['feasible']);g=a['global_best']
        cases.append(dict(name=name,p=p,local=local,global_best=g))
    fig,axs=plt.subplots(1,2,figsize=(14,6.4),gridspec_kw={'width_ratios':[1,1.3]})
    y=np.arange(3);labels=['Flyback ratio / p = -2\nrepeat 1, 0.10 m/s','Amplitude / p = -1.2\nrepeat 1, 0.05 m/s','Flyback height / p = -1.2\nrepeat 1, 0.05 m/s']
    a=axs[0];a.barh(y-.17,[c['local']['arbitrary_minimum'] for c in cases],height=.3,color='#aaa',label='Arbitrary label minimum')
    a.barh(y+.17,[c['local']['eligible_edits'] for c in cases],height=.3,color='#2466a4',label='One feature threshold')
    for i,c in enumerate(cases):
        for off,key in [(-.17,'arbitrary_minimum'),(.17,'eligible_edits')]:
            v=c['local'][key];a.text(v+1,i+off,str(int(v)),va='center')
    a.set_yticks(y,labels);a.set_ylim(3.3,-.65);a.set_xlim(0,54);a.set_xlabel('Edits in the target folder’s eligible windows');a.set_title('Same target and denominator');a.legend(loc='lower right',fontsize=9)
    a=axs[1];parts=np.array([[c['local']['eligible_edits'],c['local']['folder_all_edits']-c['local']['eligible_edits'],c['global_best']['all_edits']-c['local']['folder_all_edits']] for c in cases])
    left=np.zeros(3)
    for j,(label,color) in enumerate([('Target: eligible windows','#2466a4'),('Target: excluded columns','#eea34b'),('Other folders','#d0d7de')]):
        a.barh(y,parts[:,j],left=left,color=color,height=.55,label=label);left+=parts[:,j]
    for i,c in enumerate(cases):a.text(left[i]+140,i,f"{int(left[i]):,} edits\n{c['global_best']['n_invalid_groups']} folders unresolved",va='center',fontsize=10)
    a.set_yticks(y,['','','']);a.set_ylim(3.3,-.65);a.set_xlim(0,19500);a.set_xlabel('Edits across all 78,000 windows');a.set_title('Collateral changes are much larger');a.legend(loc='lower right',fontsize=9)
    fig.suptitle('Small arbitrary edit distances do not imply small coherent-criterion changes',fontsize=15,y=.98)
    fig.subplots_adjust(left=.21,right=.98,bottom=.20,top=.81,wspace=.28)
    fig.text(.5,.04,'All primary rule–slope cases with any restored folder; K=2, log-force tolerance 0.2.\nAmplitude/height are uncalibrated stress tests. Hypothetical labels; no physical-truth claim.',ha='center',fontsize=10)
    fig.savefig(HERE/'fig_coherence_gap.png',dpi=170);plt.close(fig)
    r=rules['ratio_primary'];a=next(x for x in r['analyses'] if x['p']==-2 and x['epsilon']==.2);t=a['global_best']['atom']['threshold']
    with np.load(HERE/r['profile_file'],allow_pickle=False) as z:
        x=z['atom_hi'];c=z['compatible'][:,:,0,2].sum(axis=1);cost=z['n_all_edits'].sum(axis=1)/780;bad=(z['invalid_eligible_columns']>0).sum(axis=1)
    fig,axs=plt.subplots(3,1,figsize=(10,8),sharex=True)
    for ax,v,color,label in zip(axs,[c,cost,bad],['#21895c','#2466a4','#bd4937'],['Compatible folders / 39','Changed windows (%)','Folders losing any\neligible finite boundary']):
        ax.step(x,v,where='pre',color=color,lw=1.8);ax.axvline(.5,color='#777',ls='--',label='Original threshold 0.5');ax.axvline(t,color='#bd4937',ls=':',label=f'Selected threshold {t:.6f}')
        ax.set_ylabel(label);ax.grid(alpha=.15)
    axs[0].set_ylim(-.05,1.2);axs[0].set_yticks([0,1]);axs[0].legend(loc='upper right',fontsize=10)
    axs[-1].set_xlabel('One shared second-flyback-ratio cutoff: H requires ratio < cutoff');axs[-1].set_xlim(.3,.7)
    fig.suptitle('One folder becomes compatible, while other boundaries disappear\nFixed p = -2; same threshold across all folders',fontsize=14)
    fig.tight_layout(rect=(0,.04,1,.92));fig.text(.5,.015,'Full declared primary range, exact threshold profile. Invalid folders never count as compatible.',ha='center',fontsize=10)
    fig.savefig(HERE/'fig_threshold_profile.png',dpi=170);plt.close(fig)
    (HERE/'figure_data.json').write_text(json.dumps({'case_selection':'All primary rule-slope cases with at least one restored folder','cases':cases,'profile_rule':'ratio_primary'},indent=2)+'\n')
if __name__=='__main__':main()
