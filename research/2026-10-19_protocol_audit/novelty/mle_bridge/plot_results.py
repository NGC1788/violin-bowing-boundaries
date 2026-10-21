"""Scientific figures from saved audit outputs only; no new fitting."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
HERE=Path(__file__).resolve().parent
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'savefig.dpi':180})

def main():
    cases=json.loads((HERE/'production/selected.json').read_text())['cases']
    chosen=[c for c in cases if c['arbitrary_geometry_feasible']]
    fig,ax=plt.subplots(1,2,figsize=(11.6,4.6),layout='constrained')
    for p,color in [(-2.,'#265c98'),(-1.2,'#ba4d30')]:
        cc=[c for c in chosen if c['p']==p]
        ax[0].scatter([c['toward_target'] for c in cc],[c['T_reduction'] for c in cc],c=color,s=39,label=f'target p={p:g}',edgecolors='white',linewidths=.5)
        ranks=sorted(c['controls']['all']['abs_delta_beta']['midrank_percentile'] for c in cc)
        ax[1].plot(ranks,np.arange(1,len(ranks)+1)/len(ranks),drawstyle='steps-post',color=color,label=f'p={p:g} (n={len(cc)})')
    for a in ax:a.grid(alpha=.15)
    ax[0].axhline(0,c='gray',lw=.8);ax[0].axvline(0,c='gray',lw=.8)
    ax[0].set(xlabel='Movement of fitted beta exponent toward target\npositive = closer after deleting the certificate',ylabel='Reduction of profiled likelihood contrast T\npositive = less contrast after deletion',title='A  Minimum obstruction does not imply MLE leverage')
    worst=min(chosen,key=lambda c:c['T_reduction'])
    ax[0].annotate('B repeat 2, 0.05 m/s\np = -1.2',xy=(worst['toward_target'],worst['T_reduction']),xytext=(-.105,-28),arrowprops={'arrowstyle':'->','color':'#555'},fontsize=9)
    ax[0].legend(loc='lower right',frameon=False)
    ax[1].plot([0,1],[0,1],'--',color='gray',lw=1,label='visual reference; not a null test')
    ax[1].axvline(.95,c='gray',lw=.7)
    ax[1].set(xlim=(0,1),ylim=(0,1.03),xlabel='Rank of |beta change| among ALL size-matched deletions\nfrom the same eligible speed folder',ylabel='Fraction of selected certificates',title='B  Fixed witnesses span the control influence ranks')
    ax[1].legend(frameon=False,fontsize=8,loc='upper left')
    fig.suptitle('52 classifier-specific cases; original 3-speed censored MLE retained',fontsize=13)
    fig.savefig(HERE/'fig_mle_influence.png');fig.savefig(HERE/'fig_mle_influence.pdf');plt.close(fig)
    rr=json.loads((HERE/'coherent_replay.json').read_text())['rules']
    base=rr['baseline']['sessions']['A4'];new=rr['near_cutoff_no_loss']['sessions']['A4']
    values=[[.40916524926568076,.2055524847973138],
            [abs(base['fit']['params'][2]+2),abs(new['fit']['params'][2]+2)],
            [base['T']['-2.0'],new['T']['-2.0']]]
    fig,axs=plt.subplots(1,3,figsize=(11.4,4.1),layout='constrained')
    titles=['A  Hard compatibility improves','B  MLE exponent moves away','C  Likelihood contrast increases']
    ylabs=['Required log-force tolerance\none speed, originally finite columns','Distance |fitted beta exponent - (-2)|\nall 3 speeds and 120 censored columns','T = 2 (NLL fixed - NLL free)\nall 3 speeds; descriptive, not a p-value']
    for a,v,title,ylab in zip(axs,values,titles,ylabs):
        a.bar([0,1],v,color=['#63758b','#dc8b39'],width=.55)
        a.set_xticks([0,1],['Baseline\n(.8, .5)','Shared rule\n(.8, .493264)'])
        a.set(ylabel=ylab,title=title,ylim=(0,max(v)*1.2))
        for i,y in enumerate(v):a.text(i,y+max(v)*.03,f'{y:.4f}',ha='center')
        a.yaxis.grid(alpha=.15)
    fig.suptitle('A4: 150 global label changes, no loss of any originally finite boundary',fontsize=13)
    fig.savefig(HERE/'fig_geometric_vs_mle.png');fig.savefig(HERE/'fig_geometric_vs_mle.pdf');plt.close(fig)
    print('Saved two PNG/PDF scientific figures')
if __name__=='__main__':main()
