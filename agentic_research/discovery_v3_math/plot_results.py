"""Render already-sealed toy results; no decoder evaluation or model fitting."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE=Path(__file__).resolve().parent


def main():
    d=json.loads((HERE/'RESULTS.json').read_text())
    shared=d['cases']['shared']['orders']['512'];q=np.array(shared['joint']);m=np.array(shared['marginals_with_guard']);ind=np.outer(*m)
    mask=np.array(d['cases']['mask']['orders']['512']['joint']);wrong=np.array(d['counterexamples']['wrong_fullmask_joint'])
    panels=[[q,ind,q-ind],[mask,wrong,mask-wrong]]
    titles=[['Joint factor integration','Product of marginal CDFs','Joint minus product'],
            ['Mask distribution integrated','Assumed full voicing','Mixture minus full mask']]
    fig,axes=plt.subplots(2,3,figsize=(11,6),layout='constrained')
    for row in range(2):
        thresholds=[[.4,.8,1.2],[.4,.8,1.2]] if row==0 else [[.6,.9,1.3],[.2,.3,.5]]
        for col in range(3):
            values=panels[row][col];ax=axes[row,col]
            if col==2:
                bound=max(.01,float(np.abs(values).max()));im=ax.imshow(values,cmap='RdBu_r',vmin=-bound,vmax=bound,origin='lower')
            else:im=ax.imshow(values,cmap='viridis',vmin=0,vmax=1,origin='lower')
            ax.set_title(titles[row][col],fontsize=11)
            ax.set_xticks(range(3),thresholds[1]);ax.set_yticks(range(3),thresholds[0])
            ax.set_xlabel('dBFS std threshold');ax.set_ylabel('Pitch std threshold (cents)')
            for i in range(3):
                for j in range(3):
                    color='white' if (col<2 and values[i,j]<.6) else 'black'
                    ax.text(j,i,f'{values[i,j]:.3f}',ha='center',va='center',color=color,fontsize=10)
            fig.colorbar(im,ax=ax,shrink=.8)
    fig.suptitle('Synthetic fixtures: probability differences, not violin performance',fontsize=13)
    fig.savefig(HERE/'toy_joint_mask.png',dpi=160)
    fig.savefig(HERE/'toy_joint_mask.svg')
    plt.close(fig)


if __name__=='__main__':main()
